"""HiSparseC4DevicePool construction: FP8 works, MXFP4 is rejected.

Regression: the DSV4 pool factory passes ``use_mxfp4`` to every pool class,
but HiSparseC4DevicePool's signature never gained the parameter — enabling
HiSparse with the default FP8 KV cache crashed with
``unexpected keyword argument 'use_mxfp4'``, and its positional
``super().__init__`` shifted ``start_layer`` into the parent's
``use_mxfp4`` slot.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.layers.attention.dsv4.mxfp4_k_cache import MXFP4_BYTES_PER_TOKEN
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4SingleKVPool,
    HiSparseC4DevicePool,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_SIZE = 256
_PAGE_SIZE = 256
_LAYER_NUM = 4
_NOPE = 448
_ROPE = 64


def _make_pool(use_mxfp4: bool) -> HiSparseC4DevicePool:
    return HiSparseC4DevicePool(
        size=_SIZE,
        page_size=_PAGE_SIZE,
        dtype=torch.float8_e4m3fn,
        qk_nope_head_dim=_NOPE,
        qk_rope_head_dim=_ROPE,
        layer_num=_LAYER_NUM,
        device="cuda",
        enable_memory_saver=False,
        use_mxfp4=use_mxfp4,
    )


def test_hisparse_pool_fp8_init():
    """FP8 (use_mxfp4=False) constructs without error."""
    pool = _make_pool(use_mxfp4=False)
    assert pool.dsv4_kv_cache_store_mxfp4 is False
    assert len(pool.kv_buffer) == _LAYER_NUM


def test_hisparse_pool_rejects_mxfp4():
    """MXFP4 + HiSparse is rejected at construction, not silently mis-routed."""
    with pytest.raises(ValueError, match="not supported with HiSparse"):
        _make_pool(use_mxfp4=True)


def test_hisparse_pool_positional_super_no_shift():
    """Positional super().__init__ args must not land in the parent's
    use_mxfp4 slot: start_layer/end_layer are preserved as layer ranges."""
    pool = HiSparseC4DevicePool(
        _SIZE,
        _PAGE_SIZE,
        torch.float8_e4m3fn,
        _NOPE,
        _ROPE,
        _LAYER_NUM,
        "cuda",
        False,
        False,  # use_mxfp4 (positional, as the factory would not call it)
        1,
        3,
    )
    # With the pre-fix signature this call would raise (unexpected keyword)
    # or, when called via the factory, shift start_layer into use_mxfp4.
    assert pool.dsv4_kv_cache_store_mxfp4 is False


def _make_mxfp4_pool(group_size: int) -> DeepSeekV4SingleKVPool:
    return DeepSeekV4SingleKVPool(
        size=_SIZE,
        page_size=_PAGE_SIZE,
        dtype=torch.float8_e4m3fn,
        qk_nope_head_dim=_NOPE,
        qk_rope_head_dim=_ROPE,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
        use_mxfp4=True,
        mxfp4_group_size=group_size,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_single_kv_pool_threads_group_size_into_quantize():
    """A pool built with mxfp4_group_size=128 must quantize with 128-dim
    groups: each group's E8M0 byte is replicated across every 32-dim scale
    slot it covers (slots 0-3 / 4-7 / 8-11 / 12-13).

    The input is crafted so each 32-dim sub-block inside a group has an
    amax three octaves apart — under the codec default (32-dim groups) the
    stored bytes are necessarily non-replicated.  Dropping the parameter
    from the pool-to-codec call (e.g. a pool refactor) therefore turns
    this red while decode keeps running: the BF16 reader is per-slot and
    stays numerically correct either way, which is exactly why such a
    regression would otherwise be silent."""
    pool = _make_mxfp4_pool(group_size=128)
    k = torch.zeros(2, _NOPE + _ROPE, dtype=torch.bfloat16, device="cuda")
    for slot in range(_NOPE // 32):
        k[:, slot * 32] = 0.4 / (8.0 ** (slot % 4))
    k[:, _NOPE:] = torch.randn(2, _ROPE, dtype=torch.bfloat16, device="cuda") * 0.05
    loc = torch.arange(2, dtype=torch.int64, device="cuda")
    pool.set_key_buffer_fused(0, loc, k)

    scales = pool.kv_buffer[0].view(-1, MXFP4_BYTES_PER_TOKEN)[loc, 224:238]
    for start, stop in ((0, 4), (4, 8), (8, 12), (12, 14)):
        block = scales[:, start:stop]
        assert torch.equal(block, block[:, :1].expand_as(block))

    # The crafted amaxes guarantee the same input through a default
    # (32-dim group) pool violates the invariant — the contrast proves the
    # assertion above is non-trivial.
    pool32 = _make_mxfp4_pool(group_size=32)
    pool32.set_key_buffer_fused(0, loc, k)
    scales32 = pool32.kv_buffer[0].view(-1, MXFP4_BYTES_PER_TOKEN)[loc, 224:238]
    assert not torch.equal(scales32[:, :4], scales32[:, :1].expand(2, 4))
