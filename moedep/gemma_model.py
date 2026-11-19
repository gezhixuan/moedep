"""Gemma-4 26B-A4B wrapper: same interface as MoeVLM (moedep/model.py).

Architecture notes (transformers 5.12, model_type gemma4): every decoder layer has a dense MLP plus a MoE block
(128 experts, top-8, softmax router with a per-expert output scale). Images are encoded to a fixed 280 soft tokens.
The router is patched to (a) return raw logits as its first output so that output_router_logits captures logits,
(b) mask logits before the softmax (re-route) or zero the weights of masked experts (zero)."""
import types
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText
from moedep.model import MoeVLM, YES, NO


def _patched_gemma_router_forward(self, hidden_states):
    hidden_states = self.norm(hidden_states)
    hidden_states = hidden_states * self.scale * self.scalar_root_size
    logits = self.proj(hidden_states)                       # (B*S, E)
    if getattr(self, "_capture", False):
        self._last_logits = logits.detach()
    sel = logits
    m = getattr(self, "_abl_mask", None)
    if m is not None and getattr(self, "_abl_scope_kind", None) is not None:   # token-scoped: (N, E) mask
        N = logits.shape[0]; rows = getattr(self, "_abl_rows", None)
        if rows is None or rows.numel() != N:
            rows = torch.full((N,), self._abl_scope_kind == "text", dtype=torch.bool, device=logits.device)
        m = rows.to(logits.device)[:, None] & m[None, :]
    if m is not None and self._abl_mode == "reroute":
        sel = logits.masked_fill(m, float("-inf"))
    probs = torch.softmax(sel.float(), dim=-1)
    top_w, top_i = torch.topk(probs, k=self.config.top_k_experts, dim=-1)
    top_w = top_w / top_w.sum(dim=-1, keepdim=True)
    if m is not None and self._abl_mode == "zero":
        hit = m[top_i] if m.dim() == 1 else torch.gather(m, 1, top_i)
        top_w = top_w.masked_fill(hit, 0.0)
    top_w = top_w.to(logits.dtype) * self.per_expert_scale[top_i]
    return logits, top_w, top_i


class GemmaMoeVLM(MoeVLM):
    def __init__(self, model_path, attn="sdpa", max_pixels=None):
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tok = self.processor.tokenizer
        dm = "auto" if torch.cuda.device_count() > 1 else "cuda"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation=attn, device_map=dm).eval()
        print(f"device_map={dm} n_gpus={torch.cuda.device_count()}", flush=True)
        self.cfg = self.model.config
        tc = self.cfg.text_config
        self.lm = self.model.model.language_model
        self.layers = list(self.lm.layers)
        self.moe_layers = [i for i, l in enumerate(self.layers) if getattr(l, "enable_moe_block", False)]
        self.routers = [self.layers[i].router for i in self.moe_layers]
        self.E, self.K, self.L = tc.num_experts, tc.top_k_experts, len(self.routers)
        self.image_token_id = self.cfg.image_token_id
        self.vspecial = {self.cfg.boi_token_id, self.cfg.eoi_token_id}
        self.max_pixels = self.min_pixels = None
        self.end_token = "<end_of_turn>"
        ids = {ch: self.tok.encode(ch, add_special_tokens=False) for ch in "ABCDEFGH"}
        assert all(len(v) == 1 for v in ids.values()), ids
        self.letter_ids = {k: v[0] for k, v in ids.items()}
        self.yes_ids = sorted({self.tok.encode(v, add_special_tokens=False)[0] for v in YES})
        self.no_ids = sorted({self.tok.encode(v, add_special_tokens=False)[0] for v in NO})
        for r in self.routers:
            r._abl_mask, r._abl_mode = None, "reroute"
            r.forward = types.MethodType(_patched_gemma_router_forward, r)
        self._logits_to_keep_ok = True
        print(f"gemma4: L={self.L} E={self.E} K={self.K} image_token={self.image_token_id}", flush=True)

    @torch.no_grad()
    def forward(self, inputs, capture=True):
        """Gemma4's output class has no router_logits field, so the patched routers store their logits."""
        self._arm_scope(inputs["input_ids"][0])
        for r in self.routers:
            r._capture = capture; r._last_logits = None
        kw = dict(use_cache=False)
        if self._logits_to_keep_ok:
            try:
                out = self.model(**inputs, logits_to_keep=1, **kw)
            except TypeError:
                self._logits_to_keep_ok = False
                out = self.model(**inputs, **kw)
        else:
            out = self.model(**inputs, **kw)
        lp0 = torch.log_softmax(out.logits[0, -1].float(), -1)
        res = {"lp_last": lp0, "top1": self.tok.decode([int(lp0.argmax())]), "aux_loss": None}
        if capture:
            idx, w = [], []
            for r in self.routers:
                x = r._last_logits; assert x is not None
                probs = torch.softmax(x.float(), -1)
                tv, ti = torch.topk(probs, self.K, -1)
                tv = tv / tv.sum(-1, keepdim=True)
                idx.append(ti.to(torch.uint8).cpu()); w.append(tv.to(torch.float16).cpu())
                r._last_logits = None
            res["topk_idx"] = torch.stack(idx).numpy()   # (L, S, K)
            res["topk_w"] = torch.stack(w).numpy()
        return res

    @torch.no_grad()
    def teacher_forced_nll(self, user_text, images, assistant_text):
        """Score assistant_text as a continuation of the generation prompt (which, for Gemma-4, ends with an empty
        thought channel), followed by the turn-end token. apply_chat_template's assistant form omits the thought
        channel and is off-distribution for this model."""
        content = [{"type": "image"} for _ in (images or [])] + [{"type": "text", "text": user_text}]
        prompt = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                    add_generation_prompt=True)
        kw = {"images": [list(images)]} if images else {}
        pre = self.processor(text=[prompt], return_tensors="pt", **kw)["input_ids"][0]
        full_inputs = self.processor(text=[prompt + assistant_text + "<turn|>"], return_tensors="pt", **kw).to(self.model.device)
        ids = full_inputs["input_ids"][0]; P = int(pre.shape[0]); S = int(ids.shape[0])
        if not torch.equal(ids[:P].cpu(), pre):
            P = int((ids[:P].cpu() == pre).cumprod(0).sum())
        end_id = self.tok.convert_tokens_to_ids("<turn|>")
        span = list(range(P, S))
        for j in span:
            if int(ids[j]) == end_id:
                span = list(range(P, j)); break
        keep = S - P + 1
        self._arm_scope(ids)
        out = self.model(**full_inputs, use_cache=False, logits_to_keep=keep)
        lp = torch.log_softmax(out.logits[0].float(), -1)
        tok_lp = torch.stack([lp[j - P, ids[j]] for j in span])
        return {"nll": float(-tok_lp.mean()), "n_tokens": len(span), "sum_nll": float(-tok_lp.sum())}

    def build_inputs(self, user_text, images=None, assistant_text=None):
        content = [{"type": "image"} for _ in (images or [])] + [{"type": "text", "text": user_text}]
        messages = [{"role": "user", "content": content}]
        if assistant_text is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})
        prompt = self.processor.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=assistant_text is None)
        kw = {"images": [list(images)]} if images else {}
        inputs = self.processor(text=[prompt], return_tensors="pt", **kw)
        return inputs.to(self.model.device)

    def set_ablation(self, masks=None, mode="reroute", scope=None):
        self._scope_kind = scope if masks is not None else None
        for li, r in enumerate(self.routers):
            if masks is None or not np.any(masks[li]):
                r._abl_mask = None
            else:
                r._abl_mask = torch.as_tensor(np.asarray(masks[li], dtype=bool), device=r.proj.weight.device)
            r._abl_mode = mode; r._abl_scope_kind = self._scope_kind; r._abl_rows = None
