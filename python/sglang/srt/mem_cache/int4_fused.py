"""Affine-INT4 fused decode kernel (CUDA, tensor cores).

Derived from this fork's `mxfp4_kv.decode_fused_mma`. The MMA pipeline, tiling, paging and
merge logic are reused verbatim; only the dequantisation changes:

    MXFP4 :  val = e2m1_table[nibble & 7] * 2^(scale_byte - 127)      (sign in bit 3)
    INT4  :  val = (nibble - zero) * scale_fp16                        (affine, G=32)

Everything this path needs is allocated up front by `prepare()`. That is not tidiness: torch
allocations performed *during* cuda-graph capture land in the graph's private memory pool and
are not valid across replays, which silently produced garbage attention (0/20 accuracy) the
first time this path was captured. `prepare()` runs before capture, and `_scratch` refuses to
allocate while capturing so a too-small cap fails loudly instead of corrupting results.

Self-test:  python3 int4_fused.py --check
"""
from __future__ import annotations

import pathlib
import sys

import torch
from torch.utils import cpp_extension

_HERE = pathlib.Path(__file__).resolve().parent
_CUDA = pathlib.Path("/sgl-workspace/sglang/python/sglang/srt/layers/jit_kernels/cuda_kernels")
_SRC = _CUDA / "int4_decode_fused.cu"

_ext = None

_CPP_SRC = r"""
#include <torch/extension.h>
void int4_decode_fused_mma(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                           torch::Tensor, torch::Tensor, torch::Tensor,
                           int64_t, int64_t, double, int64_t);
"""

_WRAPPER_SRC = r"""
#include <torch/extension.h>
#include <cstdint>
#include <cuda_fp16.h>

static cudaStream_t cur_stream(int64_t s) { return (cudaStream_t)s; }

void int4_decode_fused_mma(torch::Tensor q,
                           torch::Tensor k_data, torch::Tensor k_scale, torch::Tensor k_zero,
                           torch::Tensor v_data, torch::Tensor v_scale, torch::Tensor v_zero,
                           torch::Tensor kv_indices, torch::Tensor kv_indptr,
                           torch::Tensor o, torch::Tensor lse,
                           int64_t num_qo_heads, int64_t num_kv_heads,
                           double sm_scale, int64_t stream) {
  flashinfer::Int4DecodeParams params;
  params.q = (const __nv_bfloat16*)q.data_ptr();
  params.k_data = (const uint8_t*)k_data.data_ptr();
  params.k_scale = (const __half*)k_scale.data_ptr();
  params.k_zero = (const uint8_t*)k_zero.data_ptr();
  params.v_data = (const uint8_t*)v_data.data_ptr();
  params.v_scale = (const __half*)v_scale.data_ptr();
  params.v_zero = (const uint8_t*)v_zero.data_ptr();
  params.kv_indices = (const int*)kv_indices.data_ptr();
  params.kv_indptr = (const int*)kv_indptr.data_ptr();
  params.o = (__nv_bfloat16*)o.data_ptr();
  params.lse = lse.numel() ? (float*)lse.data_ptr() : nullptr;
  params.n = (int)kv_indices.numel();
  params.num_slots = (int)k_data.size(0);
  params.num_qo_heads = (int)num_qo_heads;
  params.num_kv_heads = (int)num_kv_heads;
  params.sm_scale = (float)sm_scale;
  const int batch = (int)kv_indptr.numel() - 1;
  dim3 grid(batch, num_kv_heads);
  dim3 block(32, 4);
  const size_t smem = 16 * flashinfer::kHeadDim * sizeof(__half) +
                      4 * 2 * 16 * flashinfer::kHeadDim * sizeof(__half) +
                      4 * 4 * flashinfer::kHeadDim * sizeof(float) +
                      4 * 4 * 2 * sizeof(float);
  int4_decode_fused_mma_kernel<<<grid, block, smem, cur_stream(stream)>>>(
      *reinterpret_cast<flashinfer::Int4DecodeParams*>(&params));
}
"""


def _load():
    global _ext
    if _ext is not None:
        return _ext
    _ext = cpp_extension.load_inline(
        name="int4_decode_fused",
        cpp_sources=_CPP_SRC,
        cuda_sources=[_SRC.read_text(), _WRAPPER_SRC],
        functions=["int4_decode_fused_mma"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_86,code=sm_86", "-std=c++17"],
        extra_include_paths=[str(_CUDA)],
        verbose=False,
    )
    return _ext


_WS: dict = {}


def prepare(device, max_idx=1 << 20, max_batch=512, max_hq=32, head_dim=128,
            dtype=torch.bfloat16, max_rows=1024):
    """Allocate every scratch buffer before any cuda-graph capture starts."""
    _load()
    _WS["idx"] = torch.empty(max_idx, dtype=torch.int32, device=device)
    _WS["ind"] = torch.empty(max_batch + 1, dtype=torch.int32, device=device)
    _WS["plan"] = torch.empty(max_rows + 1, dtype=torch.int32, device=device)
    _WS["lse"] = torch.empty(max_batch * max_hq, dtype=torch.float32, device=device)
    _WS["lp"] = torch.empty(max_rows * max_hq, dtype=torch.float32, device=device)
    _WS["o"] = torch.empty(max_batch * max_hq * head_dim, dtype=dtype, device=device)
    _WS["qp"] = torch.empty(max_rows * max_hq * head_dim, dtype=dtype, device=device)
    _WS["op"] = torch.empty(max_rows * max_hq * head_dim, dtype=dtype, device=device)
    print(f"[INT4] decode scratch: idx={max_idx} batch={max_batch} rows={max_rows} "
          f"hq={max_hq} (allocated before graph capture)", flush=True)


def _scratch(name, need, shape, dtype, device):
    b = _WS.get(name)
    if b is None or b.numel() < need:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"[INT4] scratch '{name}' needs {need} elements, only "
                f"{0 if b is None else b.numel()} preallocated - raise the cap in "
                f"int4_fused.prepare(); allocating during capture would corrupt results")
        b = torch.empty(need, dtype=dtype, device=device)
        _WS[name] = b
    return b[:need].view(shape)


def get_out(batch, qh, head_dim, dtype=torch.bfloat16, device="cuda"):
    """One output buffer shared by all layers: each layer's attention output is consumed by
    the immediately following o_proj, and the graph preserves that order."""
    return _scratch("o", batch * qh * head_dim, (batch, qh, head_dim), dtype, device)


def stage_indices(kv_indices, kv_indptr):
    """Return (indices, indptr) as int32/contiguous, copying only when necessary.

    No copy in the normal case: the buffers we are handed are the ones sglang's cuda-graph
    plan step refreshes in place (`cuda_graph_kv_indices[i]` / `kv_indptr[i]`), so the captured
    kernel reads fresh contents on every replay. Copying them was also 4 MB per layer at
    batch 64 - 144 MB of pointless traffic per step."""
    idx = kv_indices if (kv_indices.dtype == torch.int32 and kv_indices.is_contiguous()) else None
    if idx is None:
        n = kv_indices.numel()
        idx = _scratch("idx", n, (n,), torch.int32, kv_indices.device)
        idx.copy_(kv_indices)
    ind = kv_indptr if (kv_indptr.dtype == torch.int32 and kv_indptr.is_contiguous()) else None
    if ind is None:
        nb = kv_indptr.numel()
        ind = _scratch("ind", nb, (nb,), torch.int32, kv_indptr.device)
        ind.copy_(kv_indptr)
    return idx, ind


def decode_int4_fused(q, k_data, k_scale, k_zero, v_data, v_scale, v_zero,
                      kv_indices, kv_indptr, o, sm_scale):
    """q: [batch, qo_heads, 128] bf16 contiguous; o same shape, from get_out()."""
    ext = _load()
    batch = kv_indptr.numel() - 1
    qh = q.shape[1]
    idx, ind = stage_indices(kv_indices, kv_indptr)
    lse = _scratch("lse", max(batch * qh, 1), (batch, qh), torch.float32, q.device)
    ext.int4_decode_fused_mma(q, k_data, k_scale, k_zero, v_data, v_scale, v_zero,
                              idx, ind, o, lse, qh, k_data.shape[1], float(sm_scale),
                              torch.cuda.current_stream().cuda_stream)
    return o
