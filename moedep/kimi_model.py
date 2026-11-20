"""Kimi-VL-A3B runner with the same interface as moedep.model.MoeVLM.
Language model = DeepSeek-V3-style MoE (layer 0 dense; layers 1..26: 64 routed experts top-6 with noaux_tc bias
balancing + 2 always-on shared experts). Routing is captured and ablated by replacing MoEGate.forward.
Needs the second environment of the README (requirements-kimi.txt: transformers 4.50.3, tiktoken)."""
import types
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM
from moedep.model import MoeVLM, TAG_IMAGE, TAG_VSPECIAL, YES, NO


def _patched_gate_forward(self, hidden_states):
    bsz, seq_len, h = hidden_states.shape
    x = hidden_states.view(-1, h)
    logits = F.linear(x.type(torch.float32), self.weight.type(torch.float32), None)
    scores = logits.sigmoid()
    m = getattr(self, "_abl_mask", None)
    if m is not None and getattr(self, "_abl_scope_kind", None) is not None:   # token-scoped: (N, E) mask
        N = x.shape[0]; rows = getattr(self, "_abl_rows", None)
        if rows is None or rows.numel() != N:
            rows = torch.full((N,), self._abl_scope_kind in ("text", "textonly"), dtype=torch.bool, device=x.device)
        m = rows.to(x.device)[:, None] & m[None, :]
    if self.topk_method == "noaux_tc":
        choice = scores + self.e_score_correction_bias.unsqueeze(0) * getattr(self, "_bias_scale", 1.0)
        if m is not None and self._abl_mode == "reroute":
            choice = choice.masked_fill(m, float("-inf"))
        # n_group == 1 in Kimi-VL-A3B, so the group selection is the identity
        if self.n_group > 1:
            gs = choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
            gi = torch.topk(gs, k=self.topk_group, dim=-1, sorted=False)[1]
            gm = torch.zeros_like(gs); gm.scatter_(1, gi, 1)
            sm = gm.unsqueeze(-1).expand(bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group).reshape(bsz * seq_len, -1)
            choice = choice.masked_fill(~sm.bool(), float("-inf"))
        _, topk_idx = torch.topk(choice, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)
    else:
        sel = scores.masked_fill(m, float("-inf")) if (m is not None and self._abl_mode == "reroute") else scores
        topk_weight, topk_idx = torch.topk(sel, k=self.top_k, dim=-1, sorted=False)
    if self.top_k > 1 and self.norm_topk_prob:
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weight = topk_weight * self.routed_scaling_factor
    if m is not None and self._abl_mode == "zero":
        hit = m[topk_idx] if m.dim() == 1 else torch.gather(m, 1, topk_idx)
        topk_weight = topk_weight.masked_fill(hit, 0.0)
    if getattr(self, "_capture", False):
        self._last = (topk_idx.detach(), topk_weight.detach())
    return topk_idx, topk_weight, None


class KimiMoeVLM(MoeVLM):
    def __init__(self, model_path, attn="sdpa", max_pixels=None, min_pixels=None):
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tok = self.processor.tokenizer
        self.cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        kw = dict(torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation=attn, **kw).eval()
        except Exception as e:                                   # remote code may reject the kwarg
            print(f"attn_implementation={attn} rejected ({type(e).__name__}); loading with the default", flush=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **kw).eval()
        print(f"device_map=cuda n_gpus={torch.cuda.device_count()} attn={getattr(self.model.config, '_attn_implementation', '?')}", flush=True)
        self.lm = self.model.language_model
        self.layers = list(self.lm.model.layers)
        self.moe_layers = [i for i, l in enumerate(self.layers) if hasattr(l.mlp, "gate")]
        self.routers = [self.layers[i].mlp.gate for i in self.moe_layers]
        tc = self.cfg.text_config
        self.E, self.K, self.L = tc.n_routed_experts, tc.num_experts_per_tok, len(self.routers)
        self.image_token_id = self.cfg.media_placeholder_token_id
        self.vspecial = {self.tok.convert_tokens_to_ids(t) for t in ("<|media_start|>", "<|media_content|>", "<|media_end|>")}
        self.max_pixels, self.min_pixels = max_pixels, min_pixels
        ids = {ch: self.tok.encode(ch, add_special_tokens=False) for ch in "ABCDEFGH"}
        assert all(len(v) == 1 for v in ids.values()), ids
        self.letter_ids = {k: v[0] for k, v in ids.items()}
        self.yes_ids = sorted({self.tok.encode(v, add_special_tokens=False)[0] for v in YES})
        self.no_ids = sorted({self.tok.encode(v, add_special_tokens=False)[0] for v in NO})
        for r in self.routers:
            r._abl_mask, r._abl_mode, r._capture, r._last = None, "reroute", False, None
            r.forward = types.MethodType(_patched_gate_forward, r)
        self._logits_to_keep_ok = False                           # remote forward has no logits_to_keep

    def set_bias_scale(self, scale):
        """Scale Kimi's native aux-loss-free balancing bias (1 = native routing, 0 = bias removed)."""
        for r in self.routers:
            r._bias_scale = float(scale)
        print(f"kimi balancing bias scale = {scale}", flush=True)

    def build_inputs(self, user_text, images=None, assistant_text=None):
        content = [{"type": "image"} for _ in (images or [])] + [{"type": "text", "text": user_text}]
        messages = [{"role": "user", "content": content}]
        if assistant_text is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=assistant_text is None)
        if images:
            inputs = self.processor(images=images, text=[prompt], return_tensors="pt", padding=True, truncation=True)
        else:
            inputs = self.tok([prompt], return_tensors="pt")
        return inputs.to(self.model.device)

    @torch.no_grad()
    def forward(self, inputs, capture=True):
        self._arm_scope(inputs["input_ids"][0])
        for r in self.routers:
            r._capture = capture; r._last = None
        out = self.model(**inputs, use_cache=False)
        lp0 = torch.log_softmax(out.logits[0, -1].float(), -1)
        res = {"lp_last": lp0, "top1": self.tok.decode([int(lp0.argmax())]), "aux_loss": None}
        if capture:
            idx, w = [], []
            for r in self.routers:
                ti, tv = r._last
                idx.append(ti.to(torch.uint8).cpu()); w.append(tv.to(torch.float16).cpu()); r._last = None
            res["topk_idx"] = torch.stack(idx).numpy(); res["topk_w"] = torch.stack(w).numpy()
        for r in self.routers:
            r._capture = False
        return res


def load_runner(model_path, attn="sdpa", max_pixels=1280 * 32 * 32):
    mt = AutoConfig.from_pretrained(model_path, trust_remote_code=True).model_type
    if mt == "kimi_vl":
        return KimiMoeVLM(model_path, attn=attn)
    if mt == "gemma4":
        from moedep.gemma_model import GemmaMoeVLM
        return GemmaMoeVLM(model_path, attn=attn)
    return MoeVLM(model_path, attn=attn, max_pixels=max_pixels)
