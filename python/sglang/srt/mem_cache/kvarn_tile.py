"""KVarN's recipe as a pure 128-token-tile quantiser.

Frozen definition (agreed before the run; do not tune after seeing scores):

  tile size      T = 128 tokens, one tile at a time (the closure engine decides which tiles
                 are complete; this module has no window logic of its own)
  rotation       per-token Hadamard along the channel axis, normalised by 1/sqrt(D), applied in
                 the tensor's own dtype, then all arithmetic in float32
  VarN           8 iterations; each iteration alternates
                   channel scale: RMS over the T axis  -> one scale per (head, channel)
                   token   scale: RMS over the D axis  -> one scale per (token, head)
                 both scale products are accumulated and divided out afterwards
  codebook       asymmetric RTN, qmax = 2**bits - 1
                   K: scale/zero shared over the tile's tokens, i.e. one pair per channel
                   V: scale/zero shared over the channels, i.e. one pair per token
  inverse        de-normalise, rotate back, cast to the pool dtype

This is the same numeric path as `release/kv_qdq.py::_kvarn2` (bf16 rotation, float32 VarN),
restricted to one tile so it can be driven by the closure lifecycle instead of the stateless
write hook.
"""
from __future__ import annotations

import math

import torch


def hadamard(d: int, device, dtype=torch.float32) -> torch.Tensor:
    h = torch.ones(1, 1, device=device, dtype=torch.float32)
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    if h.shape[0] != d:
        raise RuntimeError(f"head_dim {d} is not a power of two")
    return (h / math.sqrt(d)).to(dtype)


def rtn_axis(x: torch.Tensor, bits: int, axis: int) -> torch.Tensor:
    """Asymmetric RTN with scale+zero on every slice along `axis`."""
    qmax = (1 << bits) - 1
    lo = x.amin(dim=axis, keepdim=True)
    hi = x.amax(dim=axis, keepdim=True)
    sc = ((hi - lo) / qmax).clamp(min=1e-8)
    q = torch.round((x - lo) / sc).clamp_(0, qmax)
    return q.mul_(sc).add_(lo)


def varn(x: torch.Tensor, iters: int = 8):
    """Alternating channel/token RMS normalisation. x: [1, 128, H, D] float32."""
    acc = torch.ones_like(x)
    for _ in range(iters):
        sc = x.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_(min=1e-8)   # over tokens
        x = x / sc
        acc = acc * sc
        sr = x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_(min=1e-8)  # over channels
        x = x / sr
        acc = acc * sr
    return x, acc


def kvarn_tile(k: torch.Tensor, v: torch.Tensor, bits: int = 2, iters: int = 8,
               return_codes: bool = False):
    """Quantise one 128-token tile of K and V in place-semantics (returns new tensors).

    k, v: [T, H, D] with T == 128. Returns (k_hat, v_hat) in the same dtype.
    """
    T, H, D = k.shape
    if T != 128:
        raise ValueError(f"kvarn_tile expects 128 tokens, got {T}")
    hk = hadamard(D, k.device, k.dtype)

    kk = (k @ hk).float().unsqueeze(0)          # [1, T, H, D]
    vv = (v @ hk).float().unsqueeze(0)

    kx, kacc = varn(kk, iters)
    vx, vacc = varn(vv, iters)

    kq = rtn_axis(kx, bits, axis=1) * kacc      # axis=1 (tokens) -> per channel over the tile
    vq = rtn_axis(vx, bits, axis=-1) * vacc     # axis=-1 (channels) -> per token

    # inverse rotation in float32 (the quantised tile is float32; the forward rotation used the
    # pool dtype, exactly as the original implementation did)
    hk32 = hk.float()
    k_hat = (kq.squeeze(0) @ hk32).to(k.dtype)
    v_hat = (vq.squeeze(0) @ hk32).to(v.dtype)
    if return_codes:
        return k_hat, v_hat, (kx, vx)
    return k_hat, v_hat


def make_kvarn_quantizer(bits: int = 2, iters: int = 8):
    """Adapter for ClosureDriver: takes K and V rows of one tile, returns both quantised."""
    def quant(k_rows: torch.Tensor, v_rows: torch.Tensor):
        return kvarn_tile(k_rows, v_rows, bits=bits, iters=iters)
    return quant
