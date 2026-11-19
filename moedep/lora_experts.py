"""Vectorized (grouped-bmm) expert forward with per-expert LoRA for HF Qwen3-VL-MoE experts.

Replaces Qwen3VLMoeTextExperts.forward (a Python loop over hit experts) by a padded grouped batched matmul, and adds
LoRA deltas  W_gu + B_gu A_gu * s  and  W_d + B_d A_d * s  per expert (fp32 LoRA params, bf16 base).  Experts whose
`enabled` flag is 0 get no delta (frozen).  Routers can be upcast to fp32 and trained; the patched router can also
report masked logits so the HF load-balancing loss sees the blocked experts (deprivation-without-recruitment condition).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertLoRA(nn.Module):
    def __init__(self, E, D, I, r=8, alpha=16.0, device="cuda"):
        super().__init__()
        self.r, self.scale = r, alpha / r
        self.A_gu = nn.Parameter(torch.empty(E, D, r, device=device)); self.B_gu = nn.Parameter(torch.zeros(E, r, 2 * I, device=device))
        self.A_d = nn.Parameter(torch.empty(E, I, r, device=device)); self.B_d = nn.Parameter(torch.zeros(E, r, D, device=device))
        nn.init.kaiming_uniform_(self.A_gu, a=math.sqrt(5)); nn.init.kaiming_uniform_(self.A_d, a=math.sqrt(5))
        self.register_buffer("enabled", torch.ones(E, device=device))

    def delta_norms(self):
        """||B A||_F per expert for gate_up and down (fp32), shape (E,)."""
        with torch.no_grad():
            g = torch.bmm(self.A_gu, self.B_gu) * self.scale; d = torch.bmm(self.A_d, self.B_d) * self.scale
            return (g.flatten(1).norm(dim=1) + d.flatten(1).norm(dim=1)) * self.enabled


def grouped_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Drop-in replacement for Qwen3VLMoeTextExperts.forward. self.lora (ExpertLoRA or None) is read if present."""
    N, D = hidden_states.shape; K = top_k_index.shape[1]; E = self.num_experts
    flat_e = top_k_index.reshape(-1); flat_t = torch.arange(N, device=hidden_states.device).repeat_interleave(K)
    flat_w = top_k_weights.reshape(-1)
    order = torch.argsort(flat_e, stable=True); e_s = flat_e[order]; t_s = flat_t[order]; w_s = flat_w[order]
    counts = torch.bincount(e_s, minlength=E); max_n = int(counts.max())
    starts = torch.cumsum(counts, 0) - counts; pos = torch.arange(len(order), device=hidden_states.device) - starts[e_s]
    X = hidden_states.new_zeros(E, max_n, D); X[e_s, pos] = hidden_states[t_s]
    lora = getattr(self, "lora", None)
    GU = torch.bmm(X, self.gate_up_proj.transpose(1, 2))                                  # (E, max_n, 2I)
    if lora is not None:
        GU = GU + (torch.bmm(torch.bmm(X.float(), lora.A_gu), lora.B_gu) * lora.scale * lora.enabled[:, None, None]).to(GU.dtype)
    g, u = GU.chunk(2, dim=-1); H = self.act_fn(g) * u                                      # (E, max_n, I)
    Y = torch.bmm(H, self.down_proj.transpose(1, 2))                                        # (E, max_n, D)
    if lora is not None:
        Y = Y + (torch.bmm(torch.bmm(H.float(), lora.A_d), lora.B_d) * lora.scale * lora.enabled[:, None, None]).to(Y.dtype)
    out = Y[e_s, pos] * w_s.to(Y.dtype)[:, None]
    final = hidden_states.new_zeros(N, D)
    return final.index_add(0, t_s, out)


def patched_router_forward(self, hidden_states):
    """Router forward with optional ablation mask (reroute / zero), fp32 weights, and masked-logit reporting."""
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states.to(self.weight.dtype), self.weight)
    sel = router_logits
    bias = getattr(self, "_bias", None)                       # aux-loss-free balancing: bias affects selection only
    if bias is not None:
        sel = sel + bias
    m = getattr(self, "_abl_mask", None)
    if m is not None and self._abl_mode == "reroute":
        sel = sel.masked_fill(m, float("-inf"))
    probs = torch.softmax(sel, dtype=torch.float, dim=-1)
    if bias is None:
        top_val, top_idx = torch.topk(probs, self.top_k, dim=-1)
    else:
        _, top_idx = torch.topk(probs, self.top_k, dim=-1)
        top_val = torch.softmax(router_logits, dtype=torch.float, dim=-1).gather(1, top_idx)
    top_val = top_val / top_val.sum(dim=-1, keepdim=True)
    if m is not None and self._abl_mode == "zero":
        top_val = top_val.masked_fill(m[top_idx], 0.0)
    reported = router_logits
    if bias is not None or (m is not None and self._abl_mode == "reroute" and getattr(self, "_report_masked", False)):
        reported = sel                                        # recorded logits reflect the realized selection
    return reported, top_val.to(hidden_states.dtype), top_idx


def install(model_wrapper, r=8, alpha=16.0, train_routers=True, device="cuda"):
    """Patch experts (vectorized + LoRA) and routers of a MoeVLM wrapper. Returns the list of ExpertLoRA modules."""
    import types
    loras = []
    for i in model_wrapper.moe_layers:
        ex = model_wrapper.layers[i].mlp.experts
        E, twoI, D = ex.gate_up_proj.shape
        ex.lora = ExpertLoRA(E, D, twoI // 2, r=r, alpha=alpha, device=ex.gate_up_proj.device)
        ex.forward = types.MethodType(grouped_experts_forward, ex); loras.append(ex.lora)
        ex.gate_up_proj.requires_grad_(False); ex.down_proj.requires_grad_(False)
    for rt in model_wrapper.routers:
        rt.forward = types.MethodType(patched_router_forward, rt)
        if train_routers:
            rt.weight.data = rt.weight.data.float(); rt.weight.requires_grad_(True)
        else:
            rt.weight.requires_grad_(False)
    return loras


def set_enabled(loras, mask_LE):
    """mask_LE: bool (L, E) of experts whose LoRA is enabled (True) or frozen (False)."""
    for li, lo in enumerate(loras):
        lo.enabled.copy_(torch.as_tensor(np.asarray(mask_LE[li], dtype=np.float32), device=lo.enabled.device))


def lora_state(loras):
    return {f"layer{li}.{k}": v.detach().cpu() for li, lo in enumerate(loras) for k, v in lo.state_dict().items()}


@torch.no_grad()
def update_balance_bias(routers, router_logits_per_layer, gamma):
    """DeepSeek-V3-style aux-loss-free balancing: after each step, lower the bias of over-loaded experts and raise it for
    under-loaded ones by gamma, using the realized (biased) top-k selection of the batch."""
    for rt, lg in zip(routers, router_logits_per_layer):
        sel = lg.detach().float()          # recorded logits already include the bias (see patched_router_forward)
        _, idx = torch.topk(torch.softmax(sel, dim=-1), rt.top_k, dim=-1)
        load = torch.bincount(idx.reshape(-1), minlength=rt.num_experts).float()
        rt._bias += gamma * torch.sign(load.mean() - load)
