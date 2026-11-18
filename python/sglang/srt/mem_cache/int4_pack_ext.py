"""Packed-INT4 KV store kernel (CUDA).

Why this exists: the Triton store kernel costs two Python launches per layer, so 72 launches
per decode step, and each Triton launch is ~200 us of host time. Measured on Qwen3-4B at batch
32 that alone was 15 ms of a 52 ms step -- more than the attention kernel itself. One CUDA
launch per layer (K and V together, 128 threads per (token, head)) removes it.

Layout is exactly the one int4_kv_kernels.py writes:
    data  [S, H, D/2]  uint8    low nibble = even d, high nibble = odd d
    scale [S, H, D/32] fp16
    zero  [S, H, D/32] uint8    integer zero-point, dequant = (code - zp) * scale
"""
from __future__ import annotations

import torch
from torch.utils import cpp_extension

_CPP = r"""
#include <torch/extension.h>
void int4_pack_kv_l(torch::Tensor k, torch::Tensor v, torch::Tensor loc,
                    torch::Tensor kd, torch::Tensor ks, torch::Tensor kz,
                    torch::Tensor vd, torch::Tensor vs, torch::Tensor vz,
                    int64_t H, int64_t stream);
void int4_pack_kv_i(torch::Tensor k, torch::Tensor v, torch::Tensor loc,
                    torch::Tensor kd, torch::Tensor ks, torch::Tensor kz,
                    torch::Tensor vd, torch::Tensor vs, torch::Tensor vz,
                    int64_t H, int64_t stream);
"""

_CU = r"""
#include <torch/extension.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

__device__ __forceinline__ unsigned q_one(float x, float sc, int zp) {
  float q = rintf(x / sc + (float)zp);
  q = fminf(fmaxf(q, 0.f), 15.f);
  return (unsigned)q;
}

// One block per (token, head): 128 threads, one per channel. A warp is exactly one scale
// group (32 channels), so the reduction is a 32-lane butterfly and needs no shared memory.
template <typename IT>
__global__ void __launch_bounds__(128) pack_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const IT* __restrict__ loc,
    uint8_t* __restrict__ kd, __half* __restrict__ ks, uint8_t* __restrict__ kz,
    uint8_t* __restrict__ vd, __half* __restrict__ vs, uint8_t* __restrict__ vz,
    int H, int num_slots) {
  const int t = blockIdx.x;
  const int h = blockIdx.y;
  const int d = threadIdx.x;                 // 0..127
  const int g = d >> 5;                      // group == warp
  const int slot = (int)loc[t];
  // A padded batch can hand us slot ids that are stale or sentinel values. Writing them
  // would be an out-of-bounds store (observed as "illegal memory access" under cuda graphs
  // at batch 6 padded to 8) or quietly corrupt a live row, so refuse them outright.
  if (slot < 0 || slot >= num_slots) return;
  const long src = (long)t * H + h;          // source row: token order
  const long row = (long)slot * H + h;       // destination row: slot order

  const float x = __bfloat162float(k[src * 128 + d]);
  const float y = __bfloat162float(v[src * 128 + d]);

  float xlo = x, xhi = x, ylo = y, yhi = y;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    xlo = fminf(xlo, __shfl_xor_sync(0xffffffffu, xlo, off));
    xhi = fmaxf(xhi, __shfl_xor_sync(0xffffffffu, xhi, off));
    ylo = fminf(ylo, __shfl_xor_sync(0xffffffffu, ylo, off));
    yhi = fmaxf(yhi, __shfl_xor_sync(0xffffffffu, yhi, off));
  }
  const float xsc = fmaxf((xhi - xlo) * (1.f / 15.f), 1e-8f);
  const int   xzp = max(0, min(15, (int)rintf(-xlo / xsc)));
  const float ysc = fmaxf((yhi - ylo) * (1.f / 15.f), 1e-8f);
  const int   yzp = max(0, min(15, (int)rintf(-ylo / ysc)));

  const unsigned xc = q_one(x, xsc, xzp);
  const unsigned yc = q_one(y, ysc, yzp);
  const unsigned xo = __shfl_down_sync(0xffffffffu, xc, 1);
  const unsigned yo = __shfl_down_sync(0xffffffffu, yc, 1);

  if ((d & 1) == 0) {
    const long nb = row * 64 + (d >> 1);
    kd[nb] = (uint8_t)(xc | (xo << 4));
    vd[nb] = (uint8_t)(yc | (yo << 4));
  }
  if ((d & 31) == 0) {
    const long mb = row * 4 + g;
    ks[mb] = __float2half_rn(xsc);
    kz[mb] = (uint8_t)xzp;
    vs[mb] = __float2half_rn(ysc);
    vz[mb] = (uint8_t)yzp;
  }
}

}  // namespace

#define LAUNCH(ITYPE)                                                            \
  const int T = (int)loc.numel();                                                \
  if (T == 0) return;                                                            \
  dim3 grid(T, (unsigned)H);                                                     \
  pack_kernel<ITYPE><<<grid, 128, 0, (cudaStream_t)stream>>>(                     \
      (const __nv_bfloat16*)k.data_ptr(), (const __nv_bfloat16*)v.data_ptr(),     \
      (const ITYPE*)loc.data_ptr(),                                              \
      (uint8_t*)kd.data_ptr(), (__half*)ks.data_ptr(), (uint8_t*)kz.data_ptr(),   \
      (uint8_t*)vd.data_ptr(), (__half*)vs.data_ptr(), (uint8_t*)vz.data_ptr(),   \
      (int)H, (int)kd.size(0));

void int4_pack_kv_l(torch::Tensor k, torch::Tensor v, torch::Tensor loc,
                    torch::Tensor kd, torch::Tensor ks, torch::Tensor kz,
                    torch::Tensor vd, torch::Tensor vs, torch::Tensor vz,
                    int64_t H, int64_t stream) { LAUNCH(int64_t) }

void int4_pack_kv_i(torch::Tensor k, torch::Tensor v, torch::Tensor loc,
                    torch::Tensor kd, torch::Tensor ks, torch::Tensor kz,
                    torch::Tensor vd, torch::Tensor vs, torch::Tensor vz,
                    int64_t H, int64_t stream) { LAUNCH(int32_t) }
"""

_ext = None


def load():
    global _ext
    if _ext is not None:
        return _ext
    _ext = cpp_extension.load_inline(
        name="int4_pack_ext",
        cpp_sources=_CPP,
        cuda_sources=[_CU],
        functions=["int4_pack_kv_l", "int4_pack_kv_i"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_86,code=sm_86", "-std=c++17"],
        verbose=False,
    )
    return _ext


def quantize_store_kv(k, v, loc, kd, ks, kz, vd, vs, vz):
    """k, v: [T, H, 128] bf16 (contiguous). loc: [T] int64/int32 slots."""
    ext = load()
    fn = ext.int4_pack_kv_l if loc.dtype == torch.int64 else ext.int4_pack_kv_i
    fn(k, v, loc, kd, ks, kz, vd, vs, vz, k.shape[1],
       torch.cuda.current_stream().cuda_stream)
