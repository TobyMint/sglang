"""MXFP4 codec (quantize / dequant) correctness.

Verifies:
  1. Triton kernel roundtrip quality (E2M1 is lossy, cos ≈ 0.99)
  2. Triton matches the PyTorch reference (E8M0 rounding may differ slightly)
  3. Empty batches, out-of-range indices, boundary cases
  4. Pre-allocated output buffer reuse
"""

from __future__ import annotations

import torch

from sglang.srt.layers.attention.dsv4.mxfp4_k_cache import (
    MXFP4_BYTES_PER_TOKEN,
    MXFP4_TOTAL_DIM,
    dequantize_dsv4_mxfp4_k_cache_paged,
    quantize_dsv4_mxfp4_k_cache_into,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# Tolerance: E2M1 (4-bit) roundtrip typically gives cos ∈ [0.98, 0.998]
# depending on data distribution.
_COS_MIN = 0.97
_ERR_MAX = 5.0


def _pool(page_size: int, num_pages: int = 4) -> torch.Tensor:
    return torch.zeros(
        num_pages,
        page_size * MXFP4_BYTES_PER_TOKEN,
        dtype=torch.uint8,
        device="cuda",
    )


def test_roundtrip_quality():
    """Triton quantize → dequant roundtrip preserves signal."""
    torch.manual_seed(1)
    page_size, num_tokens = 128, 32
    dev = torch.device("cuda")

    k = torch.randn(num_tokens, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    pool = _pool(page_size)
    loc = torch.arange(num_tokens, dtype=torch.int32, device=dev)

    quantize_dsv4_mxfp4_k_cache_into(k, pool, loc, page_size)
    out = dequantize_dsv4_mxfp4_k_cache_paged(pool, loc, page_size)

    cos = torch.nn.functional.cosine_similarity(
        k.float().flatten(), out[:, 0, :].float().flatten(), dim=0
    ).item()
    max_err = (k.float() - out[:, 0, :].float()).abs().max().item()

    assert cos > _COS_MIN, f"cos {cos} below threshold {_COS_MIN}"
    assert max_err < _ERR_MAX, f"max_err {max_err} above threshold {_ERR_MAX}"


def test_triton_vs_reference():
    """Triton output matches PyTorch reference (CPU fallback)."""
    torch.manual_seed(2)
    page_size, num_tokens = 64, 8
    dev = torch.device("cuda")

    k = torch.randn(num_tokens, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)

    # Triton path
    pool_triton = _pool(page_size)
    loc = torch.arange(num_tokens, dtype=torch.int32, device=dev)
    quantize_dsv4_mxfp4_k_cache_into(k, pool_triton, loc, page_size)
    out_triton = dequantize_dsv4_mxfp4_k_cache_paged(pool_triton, loc, page_size)

    # CPU reference path (move data to CPU, quantize, move back)
    k_cpu = k.cpu()
    pool_cpu = torch.zeros(
        1,
        page_size * MXFP4_BYTES_PER_TOKEN,
        dtype=torch.uint8,
    )
    loc_cpu = torch.arange(num_tokens, dtype=torch.int32)
    quantize_dsv4_mxfp4_k_cache_into(k_cpu, pool_cpu, loc_cpu, page_size)
    out_ref = dequantize_dsv4_mxfp4_k_cache_paged(pool_cpu, loc_cpu, page_size)

    out_cpu = out_ref[:, 0, :].to(dev).float()
    out_gpu = out_triton[:, 0, :].float()

    diff = (out_cpu - out_gpu).abs()
    max_diff = diff.max().item()
    cos = torch.nn.functional.cosine_similarity(
        out_cpu.flatten(), out_gpu.flatten(), dim=0
    ).item()

    # Quantization decisions (E8M0 rounding) may differ slightly between
    # PyTorch and Triton due to FP32 order-of-operations.
    assert cos > 0.999, f"cos {cos} too low"
    assert max_diff < 1e-3, f"max_diff {max_diff} too high"


def test_empty_batch():
    """Zero-token batch produces no-op."""
    page_size = 128
    k = torch.empty(0, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device="cuda")
    pool = _pool(page_size)
    loc = torch.empty(0, dtype=torch.int32, device="cuda")

    quantize_dsv4_mxfp4_k_cache_into(k, pool, loc, page_size)
    out = dequantize_dsv4_mxfp4_k_cache_paged(pool, loc, page_size)

    assert out.shape == (0, 1, MXFP4_TOTAL_DIM)


def test_oob_indices():
    """Out-of-bounds indices yield zero rows (padded graph batch).

    The output is pre-filled with non-zero values so stale memory (or a
    reused pre-allocated buffer) cannot mask a missing zero write.
    """
    torch.manual_seed(3)
    page_size, num_tokens = 32, 16
    dev = torch.device("cuda")
    num_rows = 4 * page_size

    k = torch.randn(num_tokens, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    pool = _pool(page_size)
    loc = torch.arange(num_tokens, dtype=torch.int32, device=dev) - 4  # [-4 .. 11]
    loc[0] = num_rows + 17  # above capacity
    loc[1] = -100  # negative
    assert (loc < 0).any() and (loc >= num_rows).any()

    quantize_dsv4_mxfp4_k_cache_into(k, pool, loc, page_size)

    out = torch.full(
        (num_tokens, 1, MXFP4_TOTAL_DIM), 7.5, dtype=torch.bfloat16, device=dev
    )
    dequantize_dsv4_mxfp4_k_cache_paged(pool, loc, page_size, out=out)

    oob = (loc < 0) | (loc >= num_rows)
    assert (
        out[oob] == 0
    ).all(), f"OOB rows should be zero, got {(out[oob] != 0).sum().item()} non-zero"
    # In-range rows still dequantize (nonzero signal).
    assert (out[~oob] != 0).any()


def test_pre_allocated_output():
    """Using a pre-allocated output tensor (no allocation in dequant)."""
    torch.manual_seed(4)
    page_size, num_tokens = 64, 8
    dev = torch.device("cuda")

    k = torch.randn(num_tokens, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    pool = _pool(page_size)
    loc = torch.arange(num_tokens, dtype=torch.int32, device=dev)

    quantize_dsv4_mxfp4_k_cache_into(k, pool, loc, page_size)

    out1 = dequantize_dsv4_mxfp4_k_cache_paged(pool, loc, page_size)
    out2 = torch.empty(num_tokens, 1, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    out3 = dequantize_dsv4_mxfp4_k_cache_paged(pool, loc, page_size, out=out2)

    assert out3 is out2, "Should return the passed-in tensor"
    assert (out1 == out2).all(), "Pre-allocated output mismatches default-allocated"


def _quantized_scale_rows(k, loc, page_size, group_size):
    pool = _pool(page_size)
    quantize_dsv4_mxfp4_k_cache_into(k, pool, loc, page_size, group_size=group_size)
    rows = pool.view(-1, MXFP4_BYTES_PER_TOKEN)[loc.long()]
    return rows[:, 224:238]  # 14 E8M0 scale slots


def test_group_size_scale_replication():
    """Coarser groups replicate their E8M0 byte across covered 32-dim slots."""
    torch.manual_seed(5)
    page_size, num_tokens = 64, 16
    dev = torch.device("cuda")
    k = torch.randn(num_tokens, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    loc = torch.arange(num_tokens, dtype=torch.int32, device=dev)

    s64 = _quantized_scale_rows(k, loc, page_size, 64)
    assert torch.equal(s64[:, 0::2], s64[:, 1::2]), "g64 slot pairs differ"

    s128 = _quantized_scale_rows(k, loc, page_size, 128)
    for g in range(3):
        quad = s128[:, 4 * g : 4 * g + 4]
        assert torch.equal(quad, quad[:, :1].expand_as(quad)), f"g128 quad {g} differs"
    # 448 = 3x128 + 64: the tail group covers only the last 64 dims.
    assert torch.equal(s128[:, 12], s128[:, 13]), "g128 tail pair differs"
    # Adjacent groups hold (at least some) distinct scales on random data.
    assert bool((s128[:, 0] != s128[:, 4]).any() or (s128[:, 4] != s128[:, 8]).any())


def test_group_size_triton_vs_reference():
    """GPU Triton and CPU reference agree for 64/128 quantization groups."""
    torch.manual_seed(6)
    page_size, num_tokens = 64, 12
    dev = torch.device("cuda")
    k = torch.randn(num_tokens, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    loc = torch.arange(num_tokens, dtype=torch.int32, device=dev)

    for group_size in (64, 128):
        pool_gpu = _pool(page_size)
        quantize_dsv4_mxfp4_k_cache_into(
            k, pool_gpu, loc, page_size, group_size=group_size
        )
        out_gpu = dequantize_dsv4_mxfp4_k_cache_paged(pool_gpu, loc, page_size)

        pool_cpu = torch.zeros(1, page_size * MXFP4_BYTES_PER_TOKEN, dtype=torch.uint8)
        quantize_dsv4_mxfp4_k_cache_into(
            k.cpu(), pool_cpu, loc.cpu(), page_size, group_size=group_size
        )
        out_ref = dequantize_dsv4_mxfp4_k_cache_paged(pool_cpu, loc.cpu(), page_size)

        max_diff = (
            (out_ref[:, 0, :].to(dev).float() - out_gpu[:, 0, :].float())
            .abs()
            .max()
            .item()
        )
        assert max_diff < 1e-3, f"group_size={group_size} max_diff {max_diff}"


def test_group_size_roundtrip_quality():
    """A 128-dim group roundtrip stays within the standard quality envelope."""
    torch.manual_seed(7)
    page_size, num_tokens = 128, 32
    dev = torch.device("cuda")
    k = torch.randn(num_tokens, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    pool = _pool(page_size)
    loc = torch.arange(num_tokens, dtype=torch.int32, device=dev)

    quantize_dsv4_mxfp4_k_cache_into(k, pool, loc, page_size, group_size=128)
    out = dequantize_dsv4_mxfp4_k_cache_paged(pool, loc, page_size)

    cos = torch.nn.functional.cosine_similarity(
        k.float().flatten(), out[:, 0, :].float().flatten(), dim=0
    ).item()
    assert cos > _COS_MIN, f"g128 cos {cos} below threshold {_COS_MIN}"


def test_group_size_invalid_rejected():
    """Only 32/64/128 are accepted quantization group sizes."""
    k = torch.randn(2, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device="cuda")
    pool = _pool(64)
    loc = torch.arange(2, dtype=torch.int32, device="cuda")
    try:
        quantize_dsv4_mxfp4_k_cache_into(k, pool, loc, 64, group_size=96)
    except ValueError:
        return
    raise AssertionError("group_size=96 should be rejected")
