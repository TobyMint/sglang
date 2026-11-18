"""Single entry point for the packed-INT4 KV write.

The Triton store kernel costs two Python launches per layer (72 per decode step, ~200 us of
host time each). Measured on Qwen3-4B batch 32 that was 15 ms of a 52 ms step. The CUDA
kernel does K and V in one launch per layer. Triton stays as the fallback so a build failure
degrades instead of breaking serving.
"""
from __future__ import annotations

import os

_impl = None


def _triton(k, v, loc, kd, ks, kz, vd, vs, vz):
    from sglang.srt.mem_cache.int4_kv_kernels import quantize_store
    quantize_store(k, loc, kd, ks, kz)
    quantize_store(v, loc, vd, vs, vz)


_KP = {"n": 0, "p": None}
_KPROF_ON = os.environ.get("SGLANG_INT4_KPROF") == "1"


def _kprof_tick():
    """Profile exactly one decode step (36 store calls) when SGLANG_INT4_KPROF=1."""
    if not _KPROF_ON:
        return
    import torch
    n = _KP["n"]
    _KP["n"] = n + 1
    if n == 36 * 10:
        _KP["p"] = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA])
        _KP["p"].__enter__()
    elif n == 36 * 11 and _KP["p"] is not None:
        p = _KP["p"]
        p.__exit__(None, None, None)
        _KP["p"] = None
        print("[KPROF] one decode step, by kernel:", flush=True)
        print(p.key_averages().table(sort_by="cuda_time_total", row_limit=18), flush=True)


def _pick():
    global _impl
    if _impl is not None:
        return _impl
    if os.environ.get("SGLANG_INT4_PACK_TRITON") == "1":
        _impl = _triton
        print("[INT4] pack: triton (forced)", flush=True)
        return _impl
    try:
        from sglang.srt.mem_cache import int4_pack_ext as _m
        _m.load()
        _impl = _m.quantize_store_kv
        print("[INT4] pack: cuda (1 launch/layer)", flush=True)
    except Exception as e:                                   # pragma: no cover
        print(f"[INT4] pack: cuda unavailable ({type(e).__name__}: {e}); triton fallback",
              flush=True)
        _impl = _triton
    return _impl


def store(k, v, loc, kd, ks, kz, vd, vs, vz):
    _kprof_tick()
    # The CUDA kernel indexes as [T, H, D]; the Triton one made the same assumption but called
    # .contiguous() itself. Not every caller hands over a contiguous slice, so do it here -
    # it is a no-op when already contiguous.
    return _pick()(k.contiguous(), v.contiguous(), loc.contiguous(),
                   kd, ks, kz, vd, vs, vz)
