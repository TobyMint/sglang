"""Fused VarN quantiser: the whole 8-iteration normalise+quantise chain in ONE Triton kernel.

Why: the eager path (`quantise_tiles`) issues ~40 small kernels per call. Batching tiles in
`drain_flushes` already removed 6.9x of that, but prefill still spends 0.45-0.53 ms/token
(BF16: 0.31), and the profile puts ~74% of the remaining prefill time in this chain.

Reformulation that makes it kernel-friendly (proven equivalent before writing this file):
  * the eager loop scales `x` in place each iteration -- a kernel would have to keep 8 scaled
    copies or re-read them from global memory;
  * instead keep the ORIGINAL x resident, track cumulative `A[d]` (per channel) and `B[t]`
    (per token), and recompute `r = x / (A*B)` each iteration.
  * measured on CPU fp32, this gives identical codes (max code diff = 0) and residual/A/B
    relative differences of ~1e-7 (pure fp32 rounding).

One program handles one (tile, kv-head): a [T, D] = [128, 128] fp32 block, num_warps=8.
"""
import os
import torch
import triton
import triton.language as tl


@triton.jit
def _varn_quant_kernel(
    K, MASKS,
    CODES, SC, LO, AA, BB,
    ITERS: tl.constexpr,
    T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    QMAX: tl.constexpr, IS_K: tl.constexpr,
):
    pid = tl.program_id(0)          # tile index
    h = tl.program_id(1)            # kv head

    tt = tl.arange(0, T)
    dd = tl.arange(0, D)
    # x layout is [N, T, H, D] -> this tile+head slice is strided by H*D along tokens
    xoff = pid * T * H * D + h * D
    offs = tt[:, None] * (H * D) + dd[None, :]
    x = tl.load(K + xoff + offs)                              # [T, D] fp32

    mraw = tl.load(MASKS + pid * T + tt)                      # [T] (0/1)
    mf = mraw.to(tl.float32)
    denom = tl.maximum(tl.sum(mf), 1.0)

    A = tl.full((D,), 1.0, tl.float32)
    B = tl.full((T,), 1.0, tl.float32)
    for _ in range(ITERS):
        r = x / (A[None, :] * B[:, None])
        sc = tl.sqrt(tl.maximum(tl.sum(r * r * mf[:, None], axis=0) / denom, 1e-16))
        A = A * sc
        r = x / (A[None, :] * B[:, None])
        sr = tl.sqrt(tl.maximum(tl.sum(r * r, axis=1) / D, 1e-16))
        B = B * sr

    r = x / (A[None, :] * B[:, None])
    big = 3.4e38
    # The two sides use DIFFERENT affine grids in the reference implementation, and matching it
    # exactly is the whole point here:
    #   K: per (head, dim) over the tile's tokens, computed on the masked rows only
    #   V: per (token, head) over the 128 dims, mask irrelevant (each row is independent)
    # `libdevice.rint` == torch.round's round-half-to-even, so the codes stay bit-identical.
    if IS_K:
        keep = mf[:, None] > 0
        lo_c = tl.min(tl.where(keep, r, big), axis=0)          # [D]
        hi_c = tl.max(tl.where(keep, r, -big), axis=0)         # [D]
        sc_c = tl.maximum((hi_c - lo_c) / QMAX, 1e-8)
        q = tl.minimum(tl.maximum(
            tl.extra.cuda.libdevice.rint((r - lo_c[None, :]) / sc_c[None, :]), 0.0), QMAX)
        sc_out, lo_out = sc_c, lo_c
    else:
        lo_t = tl.min(r, axis=1)                               # [T]
        hi_t = tl.max(r, axis=1)                               # [T]
        sc_t = tl.maximum((hi_t - lo_t) / QMAX, 1e-8)
        q = tl.minimum(tl.maximum(
            tl.extra.cuda.libdevice.rint((r - lo_t[:, None]) / sc_t[:, None]), 0.0), QMAX)
        sc_out, lo_out = sc_t, lo_t
    qi = q.to(tl.uint8)

    # pack four codes per byte along the D axis: byte = q0 | q1<<2 | q2<<4 | q3<<6
    q3 = tl.reshape(qi, (T, D // 4, 4)).to(tl.int32)
    # four codes per byte, lowest d in the lowest bits -- same layout as `pack2` in the eager path.
    # Triton forbids integer indexing of a block (`q3[:, :, i]`), so shift each of the four slots by
    # its bit offset and sum: the fields do not overlap, so sum == or.
    sh = (tl.arange(0, 4) * 2)[None, None, :]
    packed = tl.sum(q3 << sh, axis=2).to(tl.uint8)
    cc = tl.arange(0, D // 4)
    cout = pid * T * H * (D // 4) + h * (D // 4) + tt[:, None] * (H * D // 4) + cc[None, :]
    tl.store(CODES + cout, packed)

    # K stores sc/lo per channel [H,D]; V stores them per token [T,H]. A is per channel, B per token.
    # Row-major offsets: [H,D] planes index h*D + d, [T,H] planes index t*H + h.
    # (Writing h*T + t here instead transposes the parameter planes; the codes stay correct, which
    # is exactly the kind of bug that would have shown up as a small unexplained accuracy loss.)
    if IS_K:
        tl.store(SC + pid * H * D + h * D + dd, sc_out)
        tl.store(LO + pid * H * D + h * D + dd, lo_out)
    else:
        tl.store(SC + pid * T * H + tt * H + h, sc_out)
        tl.store(LO + pid * T * H + tt * H + h, lo_out)
    tl.store(AA + pid * H * D + h * D + dd, A)
    tl.store(BB + pid * T * H + tt * H + h, B)


def quantise_tiles_fused(ks, vs, masks, iters=8, bits=2):
    """Drop-in replacement for `quantise_tiles` (same dict, same shapes).

    ks, vs: [N,T,H,D] fp32 (already rotated); masks: [N,T] bool.
    """
    assert ks.is_cuda and vs.is_cuda
    N, T, H, D = ks.shape
    qmax = float((1 << bits) - 1)
    dev = ks.device
    mk = torch.empty((N, T), dtype=torch.int32, device=dev)
    mv = torch.empty((N, T), dtype=torch.int32, device=dev)
    mk.copy_(masks.to(torch.int32))
    mv.copy_(masks.to(torch.int32))
    out = {}
    for tag, src in (("k", ks), ("v", vs)):
        is_k = tag == "k"
        codes = torch.empty((N, T, H, D // 4), dtype=torch.uint8, device=dev)
        sh_sc = (N, H, D) if is_k else (N, T, H)
        sc = torch.empty(sh_sc, dtype=torch.float16, device=dev)
        lo = torch.empty(sh_sc, dtype=torch.float16, device=dev)
        aa = torch.empty((N, H, D), dtype=torch.float16, device=dev)
        bb = torch.empty((N, T, H), dtype=torch.float16, device=dev)
        src32 = src.float().contiguous()
        _varn_quant_kernel[(N, H)](
            src32, mk if is_k else mv,
            codes, sc, lo, aa, bb,
            ITERS=iters, T=T, H=H, D=D, QMAX=qmax, IS_K=is_k,
            num_warps=int(os.environ.get("SGLANG_TILE2_QFUSED_WARPS", "8")),
        )
        out["codes_" + tag] = codes
        out[tag + "_sc"] = sc
        out[tag + "_lo"] = lo
        out[tag + "_A"] = aa
        out[tag + "_B"] = bb
    return out


if __name__ == "__main__":
    # Self-test against the eager reference. Run when a GPU is free:
    #   python3 code/tile2_quant_fused.py
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import tile2_store as ts

    torch.manual_seed(0)
    for (N, T, H, D) in ((2, 128, 8, 128), (3, 128, 8, 128)):
        ks = torch.randn(N, T, H, D, device="cuda")
        vs = torch.randn(N, T, H, D, device="cuda")
        masks = torch.ones(N, T, dtype=torch.bool, device="cuda")
        masks[0, 100:] = False
        a = ts.quantise_tiles(ks, vs, masks, bits=2, iters=8)
        b = quantise_tiles_fused(ks, vs, masks, iters=8, bits=2)
        for name in sorted(a):
            same = bool(torch.equal(a[name], b[name])) if a[name].dtype == torch.uint8 else None
            af, bf = a[name].float(), b[name].float()
            md = (af - bf).abs().max().item()
            rr = ((af - bf).norm() / bf.norm().clamp(min=1e-12)).item()
            print("%-9s shape %-16s maxdiff=%.3e relRMS=%.2e%s"
                  % (name, tuple(a[name].shape), md, rr, "  IDENTICAL" if same else ""))
    print("fused quantiser self-test done")
