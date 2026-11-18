"""Flash-decoding (split-K) wrapper around the fused INT4 decode kernel.

The kernel maps one CTA to (request, kv_head): 8 CTAs per request, so at batch 4 it puts 32
CTAs onto 82 SMs and reaches ~43% of peak bandwidth. Decode latency then scales with context
and the byte saving never shows up (measured: batch 4 x 13k, packed 22.9 ms/step vs bf16 15.4).

Split-K fixes that without touching the kernel: cut each request's KV range into `nsplit`
chunks, launch the chunks as one ragged batch (the kernel already takes an indptr), and merge
the partial softmax states with base-2 weights. The kernel writes lse in log2 units (verified
against a float64 reference: lse_kernel = log2(sum exp)), which is what makes the merge a
plain 2^(l - max) weighting.

The split count must be a function of the batch size alone, never of the kv length: sglang
captures one cuda graph per batch size and replays it without Python, so anything read at
runtime is frozen at its capture-time value. A kv-length-dependent rule silently collapsed to
"no split" on replay.

All scratch is allocated by prepare(), before capture; see the note in int4_fused.py.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:                                             # pragma: no cover
    HAVE_TRITON = False

from sglang.srt.mem_cache import int4_fused as _f


def _pow2(n):
    """tl.arange needs a power-of-2 extent, so the split count must be one too."""
    n = max(1, int(n))
    return 1 << (n.bit_length() - 1)


def auto_nsplit_for_batch(batch, num_kv_heads, target_ctas=512, max_split=16):
    """Split count from the batch size alone, so it is identical at graph capture and replay."""
    want = -(-target_ctas // max(1, batch * num_kv_heads))
    return _pow2(max(1, min(want, max_split)))


def prepare(device, max_rows=1024, max_hq=32, head_dim=128,
            max_tokens=1 << 20, dtype=torch.bfloat16):
    _f.prepare(device, max_idx=max_tokens, max_batch=512, max_hq=max_hq,
               head_dim=head_dim, dtype=dtype, max_rows=max_rows)
    # Compile every split-count specialisation now. A Triton kernel compiled for the first
    # time *inside* a capture region makes the capture invalid, which is how split-K first
    # produced 0/20 accuracy under cuda graphs while working fine eagerly.
    # Warm every specialisation the serving path can need. `HQ` is a constexpr, so warming
    # with one head count does NOT precompile the kernel for another: the first real call then
    # compiles inside the capture region and invalidates it. That is exactly how split-K ended
    # up at 0/20 accuracy under cuda graphs while working eagerly. Head counts seen in practice
    # for TP1 models are 8/16/32/64, so warm all four.
    dev = torch.device(device if isinstance(device, str) else device)
    indptr = torch.tensor([0, 8, 16], dtype=torch.int32, device=dev)
    warmed = []
    for hq in sorted({8, 16, 32, max_hq}):
        for S in (2, 4, 8, 16):
            plan = torch.empty(2 * S + 1, dtype=torch.int32, device=dev)
            _plan_kernel[(2,)](indptr, plan, S=S)
            o = torch.zeros(2 * S, hq, head_dim, dtype=dtype, device=dev)
            lo = torch.zeros(2 * S, hq, dtype=torch.float32, device=dev)
            out = torch.zeros(2, hq, head_dim, dtype=dtype, device=dev)
            _combine_kernel[(2, hq)](o, lo, out, S=S, HQ=hq, D=head_dim, BLOCK_D=head_dim)
            warmed.append((hq, S))
    torch.cuda.synchronize()
    print(f"[INT4] splitk kernels warmed: {len(warmed)} (hq,S) specialisations", flush=True)


if HAVE_TRITON:
    @triton.jit
    def _plan_kernel(INDPTR, OUT, S: tl.constexpr):
        """OUT[b*S + s] = indptr[b] + s*ceil(len_b/S), clamped to indptr[b+1]."""
        b = tl.program_id(0)
        lo = tl.load(INDPTR + b)
        hi = tl.load(INDPTR + b + 1)
        chunk = (hi - lo + S - 1) // S
        s = tl.arange(0, S)
        tl.store(OUT + b * S + s, tl.minimum(lo + s * chunk, hi))

    @triton.jit
    def _combine_kernel(O, L, OUT, S: tl.constexpr, HQ: tl.constexpr, D: tl.constexpr,
                        BLOCK_D: tl.constexpr):
        b = tl.program_id(0)
        h = tl.program_id(1)
        offs_s = tl.arange(0, S)
        l = tl.load(L + (b * S + offs_s) * HQ + h)
        w = tl.exp2(l - tl.max(l, axis=0))          # lse is log2-based
        nrm = tl.sum(w, axis=0)
        offs_d = tl.arange(0, BLOCK_D)
        os_ = tl.load(O + ((b * S + offs_s)[:, None] * HQ + h) * D + offs_d[None, :])
        acc = tl.sum(os_.to(tl.float32) * w[:, None], axis=0) / nrm
        tl.store(OUT + (b * HQ + h) * D + offs_d, acc.to(OUT.dtype.element_ty))


def build_plan(kv_indptr, nsplit):
    """kv_indptr: [B+1] int32 on device -> [B*nsplit+1] int32."""
    b = kv_indptr.numel() - 1
    out = _f._scratch("plan", b * nsplit + 1, (b * nsplit + 1,), torch.int32, kv_indptr.device)
    _plan_kernel[(b,)](kv_indptr, out, S=nsplit)
    out[b * nsplit: b * nsplit + 1].copy_(kv_indptr[b: b + 1])
    return out


def decode_split(q, kd, ks, kz, vd, vs, vz, kv_indices, kv_indptr, indptr2, o,
                 sm_scale, nsplit):
    """q: [B, HQ, D] bf16 contiguous; o from int4_fused.get_out()."""
    B, HQ, D = q.shape
    rows = B * nsplit
    qp = _f._scratch("qp", rows * HQ * D, (rows, HQ, D), q.dtype, q.device)
    qp.view(B, nsplit, HQ, D).copy_(q[:, None])
    op = _f._scratch("op", rows * HQ * D, (rows, HQ, D), q.dtype, q.device)
    lp = _f._scratch("lp", rows * HQ, (rows, HQ), torch.float32, q.device)
    idx, _ = _f.stage_indices(kv_indices, kv_indptr)
    _f._load().int4_decode_fused_mma(qp, kd, ks, kz, vd, vs, vz, idx, indptr2, op, lp,
                                     HQ, kd.shape[1], float(sm_scale),
                                     torch.cuda.current_stream().cuda_stream)
    _combine_kernel[(B, HQ)](op, lp, o, S=nsplit, HQ=HQ, D=D, BLOCK_D=D)
    return o
