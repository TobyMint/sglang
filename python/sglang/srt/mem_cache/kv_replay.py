#!/usr/bin/env python
"""Real-KV replay: score-space diagnostics for KV quantisers (gate 1 of the pre-registration).

Why this exists
---------------
Storage-space MSE does not rank quantisers the way the task does. In our serving runs the
ordering by K reconstruction error was  affine INT4 0.072 < symmetric 0.097 < E2M1+free 0.101
< E2M1+E8M0 0.115, while end-to-end accuracy ordered 85 / 83 / 79 / 77 -- not the same order.
So we need the error in the coordinates attention actually reads:

    score error     dz_j = q^T (khat_j - k_j) / sqrt(d)
    attention KL    p = softmax(z),  KL(p || phat) ~= 0.5 * sum_j p_j (dz_j - <dz>_p)^2

with the mean taken over the *visible* tokens under p (a Fisher-weighted, attention-weighted
mean), not a plain subtraction of a constant.

The script captures one fixed set of post-RoPE Q/K/V per layer from the BF16 model, then
evaluates every quantiser on exactly the same tensors, so differences cannot come from
different inputs, different sampling, or different batching.

Sanity contract
---------------
The captured K must equal the K the model actually hands to the KV cache for that layer.
`--selftest` asserts this; if the assertion fails the recomputation is wrong and nothing
downstream can be trusted.

Usage
-----
    python kv_replay.py --selftest                      # 1 prompt, verify the captures
    python kv_replay.py --prompts 8 --out /work/results/replay.json
"""
import argparse
import json
import math
import os
import sys

import torch

MODEL = "/data/models/Qwen3-4B-Instruct-2507"

# ------------------------------------------------------------------ quantisers
# Each takes [T, H, D] and returns the dequantised tensor. Group G=32 along the head dim,
# which is the setting used in the serving runs being explained.


def _e2m1_round(x):
    levels = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    a = x.abs().clamp(max=6.0)
    out = torch.zeros_like(a)
    prev = torch.zeros_like(a)
    for lv in levels[1:]:
        out = torch.where(a >= 0.5 * (prev + lv), torch.full_like(a, lv), out)
        prev = torch.full_like(a, lv)
    return torch.sign(x) * out


def q_affine_int4(x, G=32):
    xf = x.float().view(*x.shape[:-1], x.shape[-1] // G, G)
    lo = xf.amin(-1, keepdim=True)
    hi = xf.amax(-1, keepdim=True)
    sc = ((hi - lo) / 15).clamp(min=1e-8)
    return (torch.round((xf - lo) / sc).clamp_(0, 15).mul_(sc).add_(lo)).view_as(x).to(x.dtype)


def q_sym_int4(x, G=32):
    xf = x.float().view(*x.shape[:-1], x.shape[-1] // G, G)
    sc = (xf.abs().amax(-1, keepdim=True) / 7).clamp(min=1e-8)
    return (torch.round(xf / sc).clamp_(-8, 7).mul_(sc)).view_as(x).to(x.dtype)


def q_e2m1_free(x, G=32):
    xf = x.float().view(*x.shape[:-1], x.shape[-1] // G, G)
    sc = (xf.abs().amax(-1, keepdim=True) / 6).clamp(min=1e-8)
    return (_e2m1_round(xf / sc) * sc).view_as(x).to(x.dtype)


def q_e2m1_e8m0(x, G=32):
    xf = x.float().view(*x.shape[:-1], x.shape[-1] // G, G)
    amax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    sc = torch.pow(2.0, torch.floor(torch.log2(amax)) - 2)     # OCP MX v1.0 §6.3
    return (_e2m1_round(xf / sc) * sc).view_as(x).to(x.dtype)


def q_e2m1_e8m0_best(x, G=32):
    """Same format, but each block picks the best of {e-1, e, e+1}. This is the baseline the
    "finer scale" claim has to beat -- otherwise the gap was a selection artefact."""
    xf = x.float().view(*x.shape[:-1], x.shape[-1] // G, G)
    amax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    e0 = torch.floor(torch.log2(amax)) - 2
    best_q, best_err = None, None
    for dk in (-1, 0, 1):
        sc = torch.pow(2.0, e0 + dk)
        q = _e2m1_round(xf / sc) * sc
        err = (q - xf).pow(2).mean(-1, keepdim=True)
        if best_q is None:
            best_q, best_err = q, err
        else:
            take = err < best_err
            best_q = torch.where(take, q, best_q)
            best_err = torch.where(take, err, best_err)
    return best_q.view_as(x).to(x.dtype)


def _e4m3_round(s) -> torch.Tensor:
    """Round a positive float to the nearest **E4M3FN** value.

    E4M3FN = 1 sign, 4 exponent (bias 7), 3 mantissa, NO infinity.
      max finite 448, min normal 2^-6, subnormals down to 2^-9.
    The previous version truncated the mantissa instead of carrying, ignored subnormals
    (2^-9 came back as 2^-6, eight times too large) and did not saturate at 448 --
    Codex caught all three.  Rounding is round-half-to-even, which is what torch.round does.
    """
    x = s.float().clamp(min=0.0)
    # exponent of the value; clamp to the normal range, which also lands subnormals on the
    # correct 2^-9 grid because step = 2^(e-3) = 2^-9 when e = -6.
    e = torch.floor(torch.log2(x.clamp(min=2.0 ** -9))).clamp(-6.0, 8.0)
    step = torch.pow(2.0, e - 3.0)
    q = torch.round(x / step) * step
    return q.clamp(max=448.0)


def q_e2m1_e4m3(x, G=32):
    xf = x.float().view(*x.shape[:-1], x.shape[-1] // G, G)
    sc = _e4m3_round((xf.abs().amax(-1, keepdim=True) / 6).clamp(min=1e-8)).clamp(min=1e-30)
    return (_e2m1_round(xf / sc) * sc).view_as(x).to(x.dtype)


QUANTISERS = {
    "affine_int4": q_affine_int4,
    "sym_int4": q_sym_int4,
    "e2m1_free": q_e2m1_free,
    "e2m1_e8m0": q_e2m1_e8m0,
    "e2m1_e8m0_best": q_e2m1_e8m0_best,
    "e2m1_e4m3": q_e2m1_e4m3,
}


# ------------------------------------------------------------------ capture
class Capture:
    """Collect post-RoPE K/V from the cache, and post-RoPE Q by re-running the projection."""

    def __init__(self):
        self.kv = {}          # layer -> (k, v) as handed to the cache, [T, H, D] float32
        self.recomputed = {}  # layer -> k recomputed from the module's own projections
        self.q = {}

    def reset(self):
        self.kv.clear()
        self.recomputed.clear()
        self.q.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--out", default="/work/results/replay.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3 import modeling_qwen3 as M

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    cap = Capture()
    state = {"hidden": None}

    # capture post-RoPE q by recomputing from the module's own projections, and verify the
    # recomputed k against the k the model actually stores.
    orig_attn_forward = M.Qwen3Attention.forward

    def patched(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kw):
        b, t, _ = hidden_states.shape
        hs = self.q_proj(hidden_states).view(b, t, -1, self.head_dim).transpose(1, 2)
        q = self.q_norm(hs)
        kk = self.k_proj(hidden_states).view(b, t, -1, self.head_dim).transpose(1, 2)
        k = self.k_norm(kk)
        cos, sin = position_embeddings
        q, k = M.apply_rotary_pos_emb(q, k, cos, sin)
        cap.q[self.layer_idx] = q[0].detach().float()              # [Hq, T, D]
        state["k_recomputed"] = k[0].detach().float().transpose(0, 1)   # [T, Hkv, D]
        state["layer"] = self.layer_idx
        return orig_attn_forward(self, hidden_states, position_embeddings,
                                 attention_mask=attention_mask, past_key_values=past_key_values, **kw)

    M.Qwen3Attention.forward = patched

    # capture the K/V that actually reach the cache, so the recomputation can be checked
    from transformers.cache_utils import DynamicCache

    class RecCache(DynamicCache):
        """Records the REAL post-RoPE K/V that the model hands to the cache."""

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            cap.kv[layer_idx] = (key_states[0].detach().float().transpose(0, 1),
                                 value_states[0].detach().float().transpose(0, 1))
            cap.recomputed[layer_idx] = state.get("k_recomputed")
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

    prompts = []
    path = "/work/data/math500/test.jsonl"
    if os.path.exists(path):
        rows = [json.loads(l) for l in open(path)][200:200 + args.prompts]
        prompts = ["Solve step by step, then give the final answer in \\boxed{}.\n\n"
                   f"Problem: {r['problem']}\n\nSolution:" for r in rows]
    if not prompts:
        prompts = ["Explain why a paged KV cache needs a per-token scale."] * args.prompts

    results = {name: {"kl": [], "mse": [], "dz_abs": []} for name in QUANTISERS}
    n_layer_checked = 0

    for pi, text in enumerate(prompts):
        ids = tok(text, return_tensors="pt", truncation=True, max_length=args.max_len).to("cuda")
        cap.reset()
        with torch.no_grad():
            model(**ids, past_key_values=RecCache(), use_cache=True)
        layers = sorted(cap.kv)
        k = cap.kv[0][0]                      # [T, Hkv, D] as stored
        q = cap.q[0]                          # [Hq, T, D]
        n_layer_checked += 1

        # MEANINGFUL self-check: the K we recomputed from the module's own projections must
        # equal the K the model actually wrote into the cache. If this fails, `q` (which is
        # produced by the same code path) is not trustworthy either.
        if args.selftest:
            bad = [l for l in layers
                   if cap.recomputed.get(l) is None
                   or not torch.allclose(cap.recomputed[l], cap.kv[l][0], atol=1e-2, rtol=1e-2)]
            finite = all(torch.isfinite(cap.kv[l][0]).all().item() for l in layers)
            print(f"[selftest] prompt {pi}: layers={len(layers)} T={k.shape[0]} "
                  f"Hq={q.shape[0]} Hkv={k.shape[1]} d={k.shape[-1]} "
                  f"k_match={not bad} finite={finite}")
            if bad:
                worst = bad[0]
                diff = (cap.recomputed[worst] - cap.kv[worst][0]).abs().max().item()
                raise SystemExit(f"[selftest] FAILED: recomputed K != cached K on layer {worst} "
                                 f"(max abs diff {diff:.4g}). Fix the capture before trusting any number.")
            if pi >= 1:
                break
            continue

        # A first/middle/last sample is enough to see whether the ordering is layer-dependent.
        probe = sorted({layers[0], layers[len(layers) // 2], layers[-1]})
        for lid in probe:
            k, q = cap.kv[lid][0], cap.q[lid]
            Hq, Hkv = q.shape[0], k.shape[1]
            gqa = Hq // Hkv
            d = k.shape[-1]

            for name, fn in QUANTISERS.items():
                khat = fn(k)                                   # [T, Hkv, D]
                # score error per query head, averaged over the GQA group
                dz_all = []
                for h in range(Hkv):
                    qg = q[h * gqa:(h + 1) * gqa]              # [gqa, T, D]
                    kh = khat[:, h, :]                          # [T, D]
                    z = torch.einsum("gtd,td->gt", qg.float(), k[:, h, :].float()) / math.sqrt(d)
                    zhat = torch.einsum("gtd,td->gt", qg.float(), kh.float()) / math.sqrt(d)
                    p = torch.softmax(z, dim=-1)
                    logp = torch.log_softmax(z, dim=-1)
                    logph = torch.log_softmax(zhat, dim=-1)
                    kl = (p * (logp - logph)).sum(-1).mean().item()
                    dz = (zhat - z)
                    dz_c = dz - (p * dz).sum(-1, keepdim=True)   # attention-weighted de-mean
                    results[name]["kl"].append(kl)
                    results[name]["dz_abs"].append(dz_c.abs().mean().item())
                    dz_all.append(kh)
                results[name]["mse"].append(
                    ((khat.float() - k.float()).pow(2).mean()
                     / k.float().pow(2).mean()).sqrt().item())

    if args.selftest:
        print("\n[selftest] capture verified; no quantiser numbers computed.")
        return

    print(f"\nlayers checked: {n_layer_checked}")
    print(f"{'quantiser':18s} {'KL':>10s} {'score-err':>12s} {'rel MSE':>10s}")
    for name in QUANTISERS:
        r = results[name]
        print(f"{name:18s} {sum(r['kl'])/len(r['kl']):10.5f} "
              f"{sum(r['dz_abs'])/len(r['dz_abs']):12.5f} "
              f"{sum(r['mse'])/len(r['mse']):10.4f}")

    if not args.selftest:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
