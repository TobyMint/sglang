"""Packed INT4 KV storage kernels (Triton).

Mirrors the MXFP4 path in `sglang/srt/layers/jit_kernels/mxfp4_kv.py`, but stores
**asymmetric INT4** instead of E2M1+E8M0:

    data  : [S, H, D/2]  uint8   two 4-bit codes per byte (low nibble = even d)
    scale : [S, H, D/G]  fp16    per (token, head, channel-group)
    zero  : [S, H, D/G]  uint8   INTEGER zero-point, dequant is (code - zp) * scale

The integer zero-point matters for speed: fp16 scale + fp16 zero is 4 B of metadata per
32 channels, which is 20% of the traffic the fused kernel moves (16 B of payload). An 8-bit
zero-point brings that to 3 B and makes dequantisation a single subtract-multiply.

Why asymmetric INT4 rather than FP4: our measurements showed the E2M1+E8M0 recipe costs
several points against affine INT4 at the same nominal 4 bits, and the axis does not matter.
So the storage format we ship is the one that actually holds accuracy.

Stage 1 of the plan is *capacity*: these kernels let the pool hold 2x-4x more tokens than
bf16 and still serve correct attention by dequantising on read.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:                                  # pragma: no cover
    HAVE_TRITON = False

HEAD_DIM = 128
GROUP = 32
QMAX = 15


if HAVE_TRITON:
    @triton.jit
    def _int4_pack_kernel(
        SRC, LOC, DATA, SCALE, ZERO,
        T, H: tl.constexpr, D: tl.constexpr, G: tl.constexpr, BLOCK_T: tl.constexpr,
    ):
        """Quantise bf16 rows [T, H, D] and scatter into pool slots `LOC`.

        One program handles one (token, head) pair: D=128 elements, D/G=4 groups.
        """
        pid = tl.program_id(0)                      # token index
        h = tl.program_id(1)                        # head index
        slot = tl.load(LOC + pid)

        offs_d = tl.arange(0, D)
        x = tl.load(SRC + (pid * H + h) * D + offs_d).to(tl.float32)   # [D]

        # per-group min/max -> affine asymmetric INT4
        xg = tl.reshape(x, [D // G, G])
        lo = tl.min(xg, axis=1)                                        # [D/G]
        hi = tl.max(xg, axis=1)
        sc = tl.maximum((hi - lo) / 15.0, 1e-8)

        # Integer zero-point (standard affine quantisation): zp = round(-lo/scale), clamped
        # to [0,15]. Dequantisation is (code - zp) * scale, so the offset is recovered from
        # the code instead of being stored as a float.
        zp = tl.minimum(tl.maximum(tl.floor(-tl.reshape(lo, [D // G]) / sc + 0.5), 0.0), 15.0)
        q = tl.reshape(x, [D // G, G]) / tl.reshape(sc, [D // G, 1]) + tl.reshape(zp, [D // G, 1])
        codes = tl.minimum(tl.maximum(tl.floor(q + 0.5), 0.0), 15.0).to(tl.uint8)   # [D/G, G]

        # pack pairs: low nibble = d even, high nibble = d odd
        codes2 = tl.reshape(tl.reshape(codes, [D]), [D // 2, 2]).to(tl.uint8)
        lo_n, hi_n = tl.split(codes2)                    # each [D/2]
        byte = lo_n | (hi_n << 4)

        tl.store(DATA + (slot * H + h) * (D // 2) + tl.arange(0, D // 2), byte)
        tl.store(SCALE + (slot * H + h) * (D // G) + tl.arange(0, D // G), sc.to(tl.float16))
        tl.store(ZERO + (slot * H + h) * (D // G) + tl.arange(0, D // G), zp.to(tl.uint8))


    @triton.jit
    def _int4_gather_dequant_kernel(
        DATA, SCALE, ZERO, INDICES, OUT,
        N, H: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
    ):
        """Dequantise the slots listed in INDICES into a dense bf16 workspace.

        `N` rows are produced; row i of OUT corresponds to INDICES[i]. This is what the
        paged-attention backend needs, since flashinfer cannot read packed data directly.
        """
        i = tl.program_id(0)
        h = tl.program_id(1)
        if i < N:
            slot = tl.load(INDICES + i)
            offs_d = tl.arange(0, D)
            c = tl.load(DATA + (slot * H + h) * (D // 2) + (offs_d // 2)).to(tl.int32)
            code = tl.where((offs_d % 2) == 0, c & 0x0F, c >> 4).to(tl.float32)
            sc = tl.load(SCALE + (slot * H + h) * (D // G) + (offs_d // G)).to(tl.float32)
            ze = tl.load(ZERO + (slot * H + h) * (D // G) + (offs_d // G)).to(tl.float32)
            tl.store(OUT + (i * H + h) * D + offs_d, ((code - ze) * sc).to(tl.bfloat16))


# ------------------------------------------------------------------ host wrappers
_WS: dict = {}


def _ws(key, shape, dtype, device):
    """Staging buffers: torch.empty() inside a custom-kernel path can land in a pool that
    the kernel cannot touch safely on this driver (same reason the mxfp4 path stages)."""
    k = (key, tuple(shape), dtype)
    buf = _WS.get(k)
    if buf is None:
        buf = torch.empty(shape, dtype=dtype, device=device)
        _WS[k] = buf
    return buf


def quantize_store(src: torch.Tensor, loc: torch.Tensor,
                   data: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> None:
    """src: [T, H, 128] bf16 -> scatter-quantise into the pool."""
    t, h, d = src.shape
    assert d == HEAD_DIM, f"INT4 KV kernel expects head_dim={HEAD_DIM}, got {d}"
    grid = (t, h)
    _int4_pack_kernel[grid](src.contiguous(), loc.contiguous(), data, scale, zero,
                            t, H=h, D=d, G=GROUP, BLOCK_T=1)


def quantize_store_kv(k, v, loc, k_data, k_scale, k_zero, v_data, v_scale, v_zero) -> None:
    quantize_store(k, loc, k_data, k_scale, k_zero)
    quantize_store(v, loc, v_data, v_scale, v_zero)


def gather_dequant(data, scale, zero, indices, out) -> None:
    """indices: [N] int32 slots -> out: [N, H, D] bf16."""
    n = indices.numel()
    h = data.shape[1]
    grid = (n, h)
    _int4_gather_dequant_kernel[grid](data, scale, zero, indices.contiguous(), out,
                                      n, H=h, D=HEAD_DIM, G=GROUP)


# ------------------------------------------------------------------ reference (torch)
def ref_quantize(x: torch.Tensor, G: int = GROUP):
    T, H, D = x.shape
    xf = x.float().view(T, H, D // G, G)
    lo = xf.amin(-1, keepdim=True)
    hi = xf.amax(-1, keepdim=True)
    sc = ((hi - lo) / 15).clamp(min=1e-8)
    ze = torch.round(-lo / sc).clamp_(0, 15)  # integer zero-point
    q = torch.round(xf / sc + ze).clamp_(0, 15).to(torch.uint8).view(T, H, D)
    packed = (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous()
    return packed, sc.squeeze(-1).to(torch.float16), ze.squeeze(-1).to(torch.uint8)


def ref_dequant(packed, scale, zero, G: int = GROUP):
    T, H, Dh = packed.shape
    D = Dh * 2
    lo = (packed & 0x0F).float()
    hi = (packed >> 4).float()
    q = torch.stack([lo, hi], dim=-1).reshape(T, H, D)
    sc = scale.float().repeat_interleave(G, dim=-1)
    ze = zero.float().repeat_interleave(G, dim=-1)
    return ((q - ze) * sc).to(torch.bfloat16)


# ------------------------------------------------------------------ paged decode
if HAVE_TRITON:
    @triton.jit
    def _int4_paged_decode_kernel(
        Q, KD, KS, KZ, VD, VS, VZ, IND, INDPTR, OUT,
        gqa, scale,
        HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """One program per (request, query head): reads packed KV directly.

        This replaces the dequant-to-workspace path: no bf16 copy of the cache is ever
        materialised, which is where the 1.8x slowdown came from.
        """
        pid = tl.program_id(0)
        b = pid // HQ
        h = pid % HQ
        kvh = h // gqa

        lo = tl.load(INDPTR + b).to(tl.int32)
        hi = tl.load(INDPTR + b + 1).to(tl.int32)

        offs_d = tl.arange(0, D)
        q = tl.load(Q + (b * HQ + h) * D + offs_d).to(tl.float32)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([D], dtype=tl.float32)

        nblk = tl.cdiv(tl.maximum(hi - lo, 0), BLOCK_T)
        for k in range(nblk):
            pos = lo + k * BLOCK_T + tl.arange(0, BLOCK_T)
            mask = pos < hi
            slot = tl.load(IND + pos, mask=mask, other=0).to(tl.int64)

            k_base = (slot[:, None] * HKV + kvh) * (D // 2) + (offs_d[None, :] // 2)
            pk = tl.load(KD + k_base, mask=mask[:, None], other=0).to(tl.int32)
            nib = tl.where((offs_d[None, :] % 2) == 0, pk & 0x0F, pk >> 4)
            s_base = (slot[:, None] * HKV + kvh) * (D // G) + (offs_d[None, :] // G)
            sc = tl.load(KS + s_base, mask=mask[:, None], other=0).to(tl.float32)
            ze = tl.load(KZ + s_base, mask=mask[:, None], other=0).to(tl.float32)
            kk = (nib.to(tl.float32) - ze) * sc

            z = tl.sum(kk * q[None, :], axis=1) * scale
            z = tl.where(mask, z, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(z, axis=0))
            p = tl.exp(z - m_new)
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)

            pv = tl.load(VD + k_base, mask=mask[:, None], other=0).to(tl.int32)
            vnib = tl.where((offs_d[None, :] % 2) == 0, pv & 0x0F, pv >> 4)
            vsc = tl.load(VS + s_base, mask=mask[:, None], other=0).to(tl.float32)
            vze = tl.load(VZ + s_base, mask=mask[:, None], other=0).to(tl.float32)
            vv = (vnib.to(tl.float32) - vze) * vsc

            acc = acc * alpha + tl.sum(p[:, None] * vv, axis=0)
            m_i = m_new

        acc = acc / tl.maximum(l_i, 1e-20)
        tl.store(OUT + (b * HQ + h) * D + offs_d, acc.to(tl.bfloat16))


def paged_decode(q, kd, ks, kz, vd, vs, vz, indices, indptr, out, gqa,
                 hkv, hq, D=HEAD_DIM, G=GROUP):
    """q: [B, HQ, D] bf16; indices: [N] int32 slots (request-major); indptr: [B+1]."""
    B = q.shape[0]
    _int4_paged_decode_kernel[(B * hq,)](
        q.reshape(-1), kd, ks, kz, vd, vs, vz, indices.to(torch.int32),
        indptr.to(torch.int32), out.reshape(-1),
        gqa, 1.0 / (D ** 0.5),
        HQ=hq, HKV=hkv, D=D, G=G, BLOCK_T=64,
    )


# ===========================================================================
# INT2 variant (same layout, four 2-bit codes per byte) for the windowless
# 2-bit method: the ablation in the paper is 2-bit, so the system has to be able
# to *serve* 2-bit, not just 4-bit, or the method is never actually run.
# ===========================================================================
INT2_QMAX = 3


if HAVE_TRITON:
    @triton.jit
    def _int2_pack_kernel(
        SRC, LOC, DATA, SCALE, ZERO,
        T, H: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
    ):
        pid = tl.program_id(0)
        h = tl.program_id(1)
        slot = tl.load(LOC + pid)
        offs_d = tl.arange(0, D)
        x = tl.load(SRC + (pid * H + h) * D + offs_d).to(tl.float32)

        xg = tl.reshape(x, [D // G, G])
        lo = tl.min(xg, axis=1)
        hi = tl.max(xg, axis=1)
        sc = tl.maximum((hi - lo) / 3.0, 1e-8)
        zp = tl.minimum(tl.maximum(tl.floor(-lo / sc + 0.5), 0.0), 3.0)
        q = tl.reshape(x, [D // G, G]) / tl.reshape(sc, [D // G, 1]) + tl.reshape(zp, [D // G, 1])
        codes = tl.minimum(tl.maximum(tl.floor(q + 0.5), 0.0), 3.0).to(tl.uint8)

        # pack four codes per byte, little-endian within the byte.
        # NOTE: split() takes the LAST dim, so reshaping [N,4] -> [N,2,2] and splitting twice
        # yields (c0,c2) and (c1,c3) -- not (c0,c1) and (c2,c3). Getting this order wrong
        # permutes the nibbles and silently corrupts the payload.
        q4 = tl.reshape(tl.reshape(codes, [D]), [D // 4, 2, 2])
        ev, od = tl.split(q4)            # ev = (c0, c2), od = (c1, c3)
        c0, c2 = tl.split(ev)
        c1, c3 = tl.split(od)
        byte = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)
        tl.store(DATA + (slot * H + h) * (D // 4) + tl.arange(0, D // 4), byte)
        tl.store(SCALE + (slot * H + h) * (D // G) + tl.arange(0, D // G), sc.to(tl.float16))
        tl.store(ZERO + (slot * H + h) * (D // G) + tl.arange(0, D // G), zp.to(tl.uint8))


    @triton.jit
    def _int2_gather_kernel(
        DATA, SCALE, ZERO, INDICES, OUT,
        N, H: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
    ):
        i = tl.program_id(0)
        h = tl.program_id(1)
        if i < N:
            slot = tl.load(INDICES + i)
            offs_d = tl.arange(0, D)
            c = tl.load(DATA + (slot * H + h) * (D // 4) + (offs_d // 4)).to(tl.int32)
            code = (c >> ((offs_d % 4) * 2)) & 0x3
            sc = tl.load(SCALE + (slot * H + h) * (D // G) + (offs_d // G)).to(tl.float32)
            ze = tl.load(ZERO + (slot * H + h) * (D // G) + (offs_d // G)).to(tl.float32)
            tl.store(OUT + (i * H + h) * D + offs_d,
                     ((code.to(tl.float32) - ze) * sc).to(tl.bfloat16))


def quantize_store_i2(src, loc, data, scale, zero):
    t, h, d = src.shape
    _int2_pack_kernel[(t, h)](src.contiguous(), loc.contiguous(), data, scale, zero,
                              t, H=h, D=d, G=GROUP)


def gather_dequant_i2(data, scale, zero, indices, out):
    n = indices.numel()
    h = data.shape[1]
    _int2_gather_kernel[(n, h)](data, scale, zero, indices.contiguous(), out,
                                n, H=h, D=HEAD_DIM, G=GROUP)


def ref_quantize_i2(x, G=GROUP):
    T, H, D = x.shape
    xf = x.float().view(T, H, D // G, G)
    lo = xf.amin(-1, keepdim=True)
    hi = xf.amax(-1, keepdim=True)
    sc = ((hi - lo) / 3).clamp(min=1e-8)
    ze = torch.round(-lo / sc).clamp_(0, 3)
    q = torch.round(xf / sc + ze).clamp_(0, 3).to(torch.uint8).view(T, H, D)
    packed = (q[..., 0::4] | (q[..., 1::4] << 2) | (q[..., 2::4] << 4) | (q[..., 3::4] << 6)).contiguous()
    return packed, sc.squeeze(-1).to(torch.float16), ze.squeeze(-1).to(torch.uint8)


def ref_dequant_i2(packed, scale, zero, G=GROUP):
    T, H, Dq = packed.shape
    D = Dq * 4
    codes = torch.stack([(packed >> (2 * i)) & 0x3 for i in range(4)], dim=-1).reshape(T, H, D).float()
    sc = scale.float().repeat_interleave(G, dim=-1)
    ze = zero.float().repeat_interleave(G, dim=-1)
    return ((codes - ze) * sc).to(torch.bfloat16)
