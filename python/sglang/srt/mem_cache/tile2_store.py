"""`packed_tile2`: the tile recipe (rotation + VarN) living in a real 2-bit packed pool.

Why this module exists
----------------------
The fork already has both halves of this, but never together:

  * `packed_int2`  -- a real packed pool (uint8 payload, D/4 bytes per head) with *no* recipe:
                      each token is quantised on its own, so there is no rotation and no VarN.
                      2.75 bit/element, 5.82x denser than bf16 (460,326 tokens at 11 GB).
  * `qdq_kv2*`     -- the KVarN recipe (Hadamard + 8 sweeps of VarN + RTN) driven by the tile
                      closure lifecycle, but on top of a **bf16 pool**, so it saves nothing.

This module puts the recipe inside the packed pool. The storage format is exactly the one the
offline reference measures (`tile_store_min.py`, format = 2.500 bit/element = 6.40x denser):

    per layer   k_codes, v_codes : uint8 [S, H, D/4]        the 2-bit payload
                staging_k/v      : bf16  [R, T, H, D]       one open tile per active request
    per tile    k_sc, k_lo, k_A  : fp16  [H, D]  (per channel, across the tile)
                k_B              : fp16  [T, H]  (per token)
                v_sc, v_lo, v_B  : fp16  [T, H]  (per token)
                v_A              : fp16  [H, D]  (per channel)
    shared      slot_row         : int32 [S]     payload position of a slot, see below

Three decisions worth stating, because each one is a place where the naive version is wrong:

1. Attention runs in the Hadamard domain. R is orthonormal and symmetric, so (qR)(kR)^T = qk^T
   and (sum_t p_t vR) R^T = sum_t p_t v. K and V are therefore never un-rotated per token; the
   query is rotated once and the attention output un-rotated once, both O(b*h*D^2) per layer.
   V's per-channel factor v_A is folded into the output after the sum (it does not depend on t);
   everything else (sc, lo, A_k, B) is applied inside the gather-dequant.

2. `slot_row` is one int32 per slot, shared by all layers, because the tile grid is anchored at
   logical position 0 and therefore identical for every layer. 4 B per token against 512 B of
   payload is 0.8%, and it is what lets attention discover, per slot, whether the row is packed
   (row >= 0: tile = row // T, offset = row % T) or still in the staging window
   (row < 0: staging row = (-row - 1)).

3. `bytes()` counts only what is allocated. The offline `logical` figure 16(1-c)+c*format is a
   mixing rule, not an allocation: this pool pre-allocates the packed region *and* the staging
   window and frees neither when c drops. Reporting one as the other was the mistake the
   reviewer caught in the offline accounting, so the two are kept apart here.
"""
from __future__ import annotations

import math
import os
import time

import torch

T_TILE = 128
_HCACHE: dict = {}
#: SGLANG_TILE2_NOFLUSH=1 keeps every row in the staging window (no packed tile is ever
#: written). That isolates the rotation/attention plumbing from the quantiser: if the
#: output is still wrong with every row exact, the store is not the problem.
_NOFLUSH = os.environ.get("SGLANG_TILE2_NOFLUSH") == "1"
#: SGLANG_TILE2_CHECK=1 prints one round-trip error per process, measured on the rows
#: of a real engine write, so the store is audited in situ and not only in a unit test.
_CHECK = os.environ.get("SGLANG_TILE2_CHECK") == "1"
_AUDIT = os.environ.get("SGLANG_TILE2_AUDIT") == "1"
_SKIPWRITE = os.environ.get("SGLANG_TILE2_SKIPWRITE") == "1"
_NO_EE = os.environ.get("SGLANG_TILE2_NO_EE") == "1"
#: How many requests can be resident at once (the position table is indexed by a slot in this
#: range). 512 is far above normal serving concurrency and keeps the table at ~34 MB.
REQ_SLOTS = int(os.environ.get("SGLANG_TILE2_REQ_SLOTS", "512"))

#: Per-path timing, deliberately living in this file so the attribution does not depend on
#: guessing which profiler entry belongs to the model and which to this store.
_PROF = ({"write": 0.0, "read": 0.0, "fused": 0.0, "rot": 0.0, "kvrot": 0.0, "stgact": 0.0, "layers": 0, "step": 0.0, "t0": None}
         if os.environ.get("SGLANG_TILE2_PROF") == "1" else None)


def _prof_add(name: str, dt: float) -> None:
    if _PROF is None:
        return
    _PROF[name] += dt
    if name == "write":
        _PROF["layers"] += 1
        if _PROF["layers"] % (36 * 40) == 0:
            _PROF["rows"] = 0
            _st = _PROF["step"] * 1e3 / 40 if _PROF["step"] else 0.0
            print(f"[T2ATT] over 40 steps: STEP={_st:.1f} ms/step  "
                  f"tile2_write={_PROF['write'] * 1e3 / 40:.2f}  "
                  f"fused_launch={_PROF['fused'] * 1e3 / 40:.2f}  "
                  f"rotation={_PROF['rot'] * 1e3 / 40:.2f}  "
                  f"staging={_PROF['stgact'] * 1e3 / 40:.2f}  "
                  f"other={_st - (_PROF['write'] + _PROF['fused'] + _PROF['read'] + _PROF['rot'] + _PROF['kvrot']) * 1e3 / 40:.1f}",
                  flush=True)
            _PROF["write"] = _PROF["read"] = _PROF["fused"] = 0.0
            _PROF["rot"] = _PROF["kvrot"] = _PROF["stgact"] = _PROF["step"] = 0.0
#: SGLANG_TILE2_ITERS exists to prove VarN is actually applied: 0 makes A and B identically 1,
#: which must be measurably worse than 8 if the normalisation is doing anything.
_ITERS = int(os.environ.get("SGLANG_TILE2_ITERS", "8"))
#: SGLANG_TILE2_TRACE=3,5,7 prints every write and flush event for those request rows. The defect
#: this exists for only shows up with several requests in flight, so the trace is per row.
_TRACE = {int(x) for x in os.environ.get("SGLANG_TILE2_TRACE", "").split(",") if x.strip()}
#: NOTE: trace keys are the stable per-request id (the position-0 slot), not the req_to_token row.


def hadamard(d: int, device, dtype=torch.float32) -> torch.Tensor:
    key = (d, str(device), dtype)
    h = _HCACHE.get(key)
    if h is None:
        x = torch.ones(1, 1, device=device, dtype=torch.float32)
        while x.shape[0] < d:
            x = torch.cat([torch.cat([x, x], 1), torch.cat([x, -x], 1)], 0)
        if x.shape[0] != d:
            raise RuntimeError(f"head_dim {d} is not a power of two")
        h = (x / math.sqrt(d)).to(dtype)
        _HCACHE[key] = h
    return h


#: SGLANG_TILE2_NOROT=1 skips the Hadamard rotation entirely. Numerically WRONG on purpose --
#: it exists to measure what the rotation costs, the same way SGLANG_TILE2_ITERS=0 does for VarN.
#: `rotate_hadamard_mm` materialises an fp32 copy of its input (`x.float()` below), which during
#: prefill is a 268 MB temporary for q at 16K; this flag tells us whether that is what makes
#: prefill 1.8x slower than BF16.
_NOROT = os.environ.get("SGLANG_TILE2_NOROT") == "1"
#: SGLANG_TILE2_ONEPASS=1 (default) writes the slot/position maps once per chunk rather than once
#: per tile; 0 restores the per-tile writes for an A/B.
_ONEPASS = os.environ.get("SGLANG_TILE2_ONEPASS", "1") != "0"

#: SGLANG_TILE2_NOSTAGE=1 skips the bf16 landing-window writes. DIAGNOSTIC ONLY -- the flush then
#: reads stale rows and every result is wrong. It prices the write half of the staging round trip:
#: every token is written to the ring as bf16 and read back for quantisation, which for prefill is
#: 4 KiB written + 4 KiB read per token per layer (8.6 KiB/token/layer including the packed codes).
_NOSTAGE = os.environ.get("SGLANG_TILE2_NOSTAGE") == "1"

#: SGLANG_TILE2_FUSE=1 (default on) feeds the quantiser straight from the chunk being written
#: whenever a whole tile lies inside it, instead of routing those rows through the bf16 landing
#: window and copying them back out of it. The copy is the expensive part: queue_flush has to
#: clone each closing tile's 512 KiB out of the window because the window is reused before the
#: layer ends, and a chunk-fed slice stays alive until drain_flushes returns. Set to 0 to restore
#: the window path for an A/B.
_FUSE = os.environ.get("SGLANG_TILE2_FUSE", "1") != "0"


def rotate_hadamard_mm(x: torch.Tensor) -> torch.Tensor:
    """x @ H with the Hadamard in x's own dtype, so no casts are needed.

    The fp32 version cost two full casts per call on top of the GEMM, and there are 2*36 calls a
    step; the rotation feeds nothing that needs more than bf16 precision (the stored values are
    bf16 anyway), and cuBLAS accumulates bf16 GEMMs in fp32.
    """
    """x @ H as a single cuBLAS call.

    Only for tensors whose exact rounding order does not matter -- the query, which never enters
    the cache. The stored K/V must keep using the FWHT: cuBLAS changes its accumulation order with
    the batch shape, and a two-ULP change on an outlier moves the per-channel min/max and hence the
    codes, which is what made chunked and single-call writes disagree.
    """
    if _NOROT:
        return x
    d = x.shape[-1]
    # fp32 measured faster than a bf16 Hadamard here (118.0 vs 106.5 tok/s), so the casts stay.
    h = hadamard(d, x.device, torch.float32)
    lead = x.shape[:-1]
    return (x.reshape(-1, d).float() @ h).reshape(*lead, d)


def rotate_hadamard(x: torch.Tensor) -> torch.Tensor:
    """x @ H for the normalised Sylvester Hadamard matrix, computed as a fast Walsh-Hadamard
    transform and returned in float32.

    Not a matmul on purpose. `x @ H` via cuBLAS changes its accumulation order with the batch
    shape, so the same row rotated inside a 100-row write and inside a 430-row write differed by
    two bf16 ULPs -- and a two-ULP change on an outlier moves the per-channel min/max, which
    changes the scale, which changes the codes, which moved the reconstruction by 3.19 in our
    chunking test. The FWHT is additions only, per row, in a fixed order, so the staged bf16
    values are identical no matter how the writes are chunked.
    """
    if _NOROT:
        return x.float()
    n = x.shape[-1]
    lead = x.shape[:-1]
    y = x.float()
    h = 1
    while h < n:
        y = y.view(*lead, n // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack([a + b, a - b], dim=-2).flatten(-2)
        h *= 2
    # the last stage leaves a trailing block axis of size 1; fold it back into the row
    return y.reshape(*lead, n) / math.sqrt(n)


def pack2(q: torch.Tensor) -> torch.Tensor:
    """[..., D] integer codes -> [..., D/4] uint8, four codes per byte (low bits = lowest d)."""
    q = q.to(torch.uint8)
    return (q[..., 0::4] | (q[..., 1::4] << 2) | (q[..., 2::4] << 4) | (q[..., 3::4] << 6))


def unpack2(b: torch.Tensor, d: int) -> torch.Tensor:
    sh = torch.arange(4, device=b.device, dtype=torch.int32) * 2
    o = (b.to(torch.int32).unsqueeze(-1) >> sh) & 3
    return o.reshape(*b.shape[:-1], d).to(torch.float32)


def _varn_masked(x: torch.Tensor, m: torch.Tensor, iters: int):
    """VarN on [1,T,H,D] with reductions weighted by the row mask m: [1,T,1,1]."""
    A = torch.ones_like(x[:, :1, :, :])
    B = torch.ones_like(x[:, :, :, :1])
    # Per-tile denominator so a batch of tiles computes exactly what they would one by one.
    denom = m.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
    for _ in range(iters):
        sc = ((x.pow(2) * m).sum(dim=1, keepdim=True) / denom).sqrt().clamp_(min=1e-8)
        x = x / sc
        A = A * sc
        sr = x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_(min=1e-8)
        x = x / sr
        B = B * sr
    return x, A, B


def quantise_tiles(ks: torch.Tensor, vs: torch.Tensor, masks: torch.Tensor,
                   bits: int = 2, iters: int = 8):
    """Quantise N tiles at once of ALREADY ROTATED values. ks, vs: [N,T,H,D]; masks: [N,T] bool.

    Batching is the whole point: one tile measured 4.4 ms and eight together measured 5.1 ms,
    because the cost is the ~40-strong serial chain of small eager kernels (131k elements each),
    not arithmetic. A prefill step closes ~8 tiles per layer, so per-tile quantisation alone was
    1179 ms per ~1k-token prefill -- over 70% of the whole prefill.
    """
    N, Tn, H, D = ks.shape
    qmax = (1 << bits) - 1
    m = masks.view(N, Tn, 1, 1).to(torch.float32)

    x = ks.float()
    y = vs.float()
    x, A_k, B_k = _varn_masked(x, m, iters)
    y, A_v, B_v = _varn_masked(y, m, iters)

    big = torch.finfo(torch.float32).max
    fill = (~masks).view(N, Tn, 1, 1)
    lo_k = torch.where(fill, torch.full_like(x, big), x).amin(dim=1, keepdim=True)
    hi_k = torch.where(fill, torch.full_like(x, -big), x).amax(dim=1, keepdim=True)
    sc_k = ((hi_k - lo_k) / qmax).clamp(min=1e-8)
    q_k = torch.round((x - lo_k) / sc_k).clamp_(0, qmax)

    lo_v = y.amin(dim=-1, keepdim=True)
    hi_v = y.amax(dim=-1, keepdim=True)
    sc_v = ((hi_v - lo_v) / qmax).clamp(min=1e-8)
    q_v = torch.round((y - lo_v) / sc_v).clamp_(0, qmax)

    h = torch.float16
    return dict(
        codes_k=pack2(q_k),
        codes_v=pack2(q_v),
        k_sc=sc_k.squeeze(1).to(h), k_lo=lo_k.squeeze(1).to(h),
        k_A=A_k.squeeze(1).to(h), k_B=B_k.squeeze(-1).to(h),
        v_sc=sc_v.squeeze(-1).to(h), v_lo=lo_v.squeeze(-1).to(h),
        v_A=A_v.squeeze(1).to(h), v_B=B_v.squeeze(-1).to(h),
    )


def quantise_tile(k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, bits: int = 2,
                  iters: int = 8):
    """One tile of ALREADY ROTATED values. k, v: [T,H,D]; mask: [T] bool. Thin wrapper -- the
    batched form above is what the hot path uses."""
    out = quantise_tiles(k.unsqueeze(0), v.unsqueeze(0), mask.unsqueeze(0), bits=bits,
                         iters=iters)
    return {n: t[0] for n, t in out.items()}


class Tile2Buffers:
    """All GPU-side state of the packed tile store. Owned by the KV pool, one per model."""

    PARAM_SHAPES = {                       # (shape-suffix, dtype) of the per-tile parameters
        "k_sc": (("H", "D"), torch.float16), "k_lo": (("H", "D"), torch.float16),
        "k_A": (("H", "D"), torch.float16), "k_B": (("T", "H"), torch.float16),
        "v_sc": (("T", "H"), torch.float16), "v_lo": (("T", "H"), torch.float16),
        "v_A": (("H", "D"), torch.float16), "v_B": (("T", "H"), torch.float16),
    }

    def __init__(self, size: int, page_size: int, head_num: int, head_dim: int,
                 layer_num: int, device, max_staging: int = 8, bits: int = 2):
        self.size = size
        self.slots = size + page_size
        self.H, self.D, self.L = head_num, head_dim, layer_num
        self.device = device
        self.bits = bits
        #: SGLANG_TILE2_T exists only to make the plumbing testable: with a large T nothing
        #: ever closes, so every row stays in the staging window and the attention path can be
        #: checked without the quantiser in the loop.
        self.T = int(os.environ.get("SGLANG_TILE2_T", T_TILE))
        self.max_staging = max_staging
        #: SGLANG_TILE2_PAGE_BOUND=1 (implied by HiCache) makes a tile BE a device token page: with
        #: `--page-size 128` the allocator hands out page-aligned runs of 128 contiguous slots, so
        #: `tile = slot // T` is well defined and no request-level tile block is needed. The tile
        #: then shares its lifetime with the allocator page, which is the ownership model HiCache
        #: needs (one refcount, the allocator's).
        self.page_bound = os.environ.get("SGLANG_TILE2_PAGE_BOUND") == "1"
        self.n_tiles = ((self.slots // self.T) if self.page_bound
                        else (self.slots + self.T - 1) // self.T)
        assert head_dim % 4 == 0

        pk = head_dim // 4
        # Per layer, not shared. Every layer has its own KV, and the first version of this
        # allocated a single set of payload buffers, which made the pool charge 36x the memory
        # it actually used -- the same class of error as the packed_int4 v_data_buffer bug.
        self.k_codes = torch.zeros((self.L, self.slots, self.H, pk), dtype=torch.uint8,
                                   device=device)
        self.v_codes = torch.zeros((self.L, self.slots, self.H, pk), dtype=torch.uint8,
                                   device=device)
        # -1 means "this slot has never been written"; >= 0 is (tile*T + offset); < -1 is
        # staging (-row-1), see the module docstring.
        #: Sentinel for "never written". It used to be -1, which is EXACTLY the staged encoding of
        #: (staging slot 0, offset 0), so a never-written slot silently returned another request's
        #: row 0 -- coherent-looking data with no error anywhere. The staged encoding now starts at
        #: -2, so -1 stays free to mean "nothing here" and the read path can zero it.
        self.ROW_UNWRITTEN = -1
        self.slot_row = torch.full((self.slots,), self.ROW_UNWRITTEN,
                                   dtype=torch.int32, device=device)
        n, L = self.n_tiles, self.L
        self.k_sc = torch.zeros((L, n, self.H, self.D), dtype=torch.float16, device=device)
        self.k_lo = torch.zeros((L, n, self.H, self.D), dtype=torch.float16, device=device)
        self.k_A = torch.zeros((L, n, self.H, self.D), dtype=torch.float16, device=device)
        self.k_B = torch.zeros((L, n, self.T, self.H), dtype=torch.float16, device=device)
        self.v_sc = torch.zeros((L, n, self.T, self.H), dtype=torch.float16, device=device)
        self.v_lo = torch.zeros((L, n, self.T, self.H), dtype=torch.float16, device=device)
        self.v_A = torch.zeros((L, n, self.H, self.D), dtype=torch.float16, device=device)
        self.v_B = torch.zeros((L, n, self.T, self.H), dtype=torch.float16, device=device)
        #: +1 row is the padding dustbin, and it must live OUTSIDE the staging slots: the first
        #: attempt reused the last row of slot `max_staging-1`, so padded writes clobbered a live
        #: request's row 127 and padding-on accuracy came out 0.565 instead of the 0.710 reference.
        self.staging_k = torch.zeros((self.L, max_staging * self.T + 1, self.H, self.D),
                                     dtype=torch.bfloat16, device=device)
        self.staging_v = torch.zeros_like(self.staging_k)
        self._Hk = None
        self.audit = []
        #: SGLANG_TILE2_TORCH_GATHER=1 forces the reference path (used by the tests)
        self.use_triton = os.environ.get("SGLANG_TILE2_TORCH_GATHER") != "1"
        #: SGLANG_TILE2_SLOTWS=1 keeps the dequantised rows in a SLOT-indexed bf16 buffer and
        #: lets flashinfer read it with the real slot ids, i.e. exactly how the in-place QDQ arm
        #: reads its pool. The compact workspace + arange substitution is then out of the picture,
        #: which is the last link that the evidence had not cleared.
        self.use_slot_ws = os.environ.get("SGLANG_TILE2_SLOTWS") == "1"
        #: CUDA-graph mode. A captured graph may not read device memory on the host, so every
        #: per-request decision has to come from a device tensor. The tables below are filled by the
        #: PREFILL path (which is never captured) and only read during decode.
        self.graphsafe = os.environ.get("SGLANG_TILE2_GRAPH") == "1"
        #: Whether prefixes can be shared, i.e. whether a tile may outlive the request that wrote
        #: it. Set by model_runner from `--disable-radix-cache` (as an env var, because the pool is
        #: built later than server_args is parsed). Defaults to True: the reference-aware reclaim is
        #: the safe choice, and only radix-off runs may skip it.
        self.radix_on = os.environ.get("SGLANG_TILE2_RADIX_ON", "1") != "0"
        self.stg_tab = None
        self.base_tab = None
        self.pos_slot = None  # [req_rows, max_ctx] device position->slot table
        self.closed = None
        self.closed_mask = None
        #: [L, req_rows] int32, how many tiles of this request are FULLY packed for this layer.
        #: The fused decode kernel needs this instead of `seq_len // T`: with a deferred flush a
        #: request can have seq_len = 128 while nothing has been packed for that layer yet, and
        #: then all 128 rows still live in the landing window.
        self.packed_ntiles = None
        #: row -> stable request id (the position-0 slot). tile counts must be keyed by the
        #: request, not the row: with chunked prefill a request changes row between chunks, so a
        #: row-keyed count reads 0 for the new row and the fused kernel would treat a fully packed
        #: history as if it were all still in the landing window.
        self.key_tab = None
        #: row -> compact per-request index used to address `pos_slot`
        self.ridx_tab = None
        #: req_to_token, handed over by install(); the fused kernel reads slot ids from it because
        #: it is the only table that stays correct when a request changes row between chunks.
        self.req_to_token = None
        self.slot_ws = None
        #: SGLANG_TILE2_ROTDOMAIN=1 goes back to computing attention in the Hadamard domain
        # Rotated domain by DEFAULT. Attention is computed on the Hadamard-domain q/k/v, so the read
        # path is a single gather: no un-rotation GEMM, no packed/staged blend, and no second bf16
        # rounding. The earlier "rotated domain costs 12 pp" measurement was taken while the tile
        # block allocator was corrupting every multi-request run, so it was an artefact; re-measured
        # after the fixes it is 82.0% (n=50) versus 80.0% for the original domain.
        self.rot_domain = os.environ.get("SGLANG_TILE2_ROTDOMAIN") != "0"
        self.unrotate = not self.rot_domain
        #: Staging holds EXACTLY what the model wrote. Rotating on the way in would round the
        #: rotated value to bf16 before the quantiser sees it -- one extra rounding that the
        #: in-place QDQ arm does not pay, and it measured ~8% on the reconstruction error at depth.
        #: In rotated-domain mode the wrapper already rotated, so the incoming tensor is the
        #: rotated one and this flag is True.
        self.staging_rotated = self.rot_domain
        # tile ids are handed out once per closure event and shared by all layers, so the
        # allocator itself lives here; see Tile2Driver.tile_id_for.
        self._tiles_free = list(range(self.n_tiles - 1, -1, -1))
        #: Per-page production state. `slot_row >= 0` is NOT proof that a page is complete (it is a
        #: shared mapping that goes positive after layer 0 flushes); offload requires every local
        #: layer's ten fields, so track a bitmask + generation per page.
        self.page_ready_layers = torch.zeros(self.n_tiles, dtype=torch.int32, device=self.device)
        self.page_generation = torch.zeros(self.n_tiles, dtype=torch.int32, device=self.device)
        self._layers_mask = (1 << self.L) - 1
        #: SGLANG_TILE2_TILES_PER_REQ is how many tiles one request may use; the default covers a
        #: 16k context at T=128 and caps concurrency at n_tiles/128 requests, which is the same
        #: limit the slot pool already imposes (500k slots / 16k = 30).
        # Block size decides how many concurrent requests the pool can host: `n_tiles / block`.
        # The historical 128 (= a 16k context) capped a 486k-token pool at 27 requests. Shrinking
        # the block raises that ceiling and measured accuracy-neutral (n=200: 71.0% at 40 tiles,
        # 71.0% at 128). But N blocks with zero slack is fatal -- the first version sized the block
        # as `n_tiles // N` and died immediately with "no free block of 480 tiles (1 blocks left)",
        # because a slot/block is only returned by the LRU reclaimer, never on completion. So keep
        # 4 blocks of slack, and never shrink below what the pool can afford.
        _conc = os.environ.get("SGLANG_TILE2_CONCURRENCY")
        _want = int(os.environ.get("SGLANG_TILE2_TILES_PER_REQ",
                                   str(max(2, 16384 // T_TILE))))
        if "SGLANG_TILE2_TILES_PER_REQ" not in os.environ and _conc is not None:
            _slack = 4
            _want = min(_want, max(2, self.n_tiles // (max(1, int(_conc)) + _slack)))
        # Never reserve so much that a single request cannot be served; on a toy pool (the unit
        # tests) the floor matters, on a real pool _want is the binding term.
        self.tiles_per_req = max(1, min(_want, max(1, self.n_tiles // 2)))
        self._blocks = [(0, self.n_tiles)]
        #: Blocks whose request is gone but which are still referenced by SOME slot: with the radix
        #: cache a shared prefix keeps pointing at another request's tiles after that request
        #: finishes, so returning the block to the free list lets the next request overwrite
        #: parameters a live request is decoding with. Measured at n=50: never recycling tiles is
        #: 0.660, recycling them at n=200 collapses to 0.085.
        self._deferred_blocks = []
        self._live = set()
        self.start_layer = 0
        self._param_owner = {}
        self._cur_row = None
        self._staging_free = list(range(max_staging - 1, -1, -1))
        self._stg_live = set()

    def Hk(self) -> torch.Tensor:
        if self._Hk is None:
            self._Hk = hadamard(self.D, self.device, torch.float32)
        return self._Hk

    def alloc_tile_block(self, k: int) -> int:
        """Carve `k` CONSECUTIVE tile ids for one request.

        Consecutive on purpose: it makes `tile_id = base + position // T` pure arithmetic, which
        is what both remaining problems need. It removes the host-side counter that a CUDA graph
        cannot freeze, and it makes the tile a request owns independent of the req_to_token row,
        which changes between chunks of the same request.
        """
        for i, (start, size) in enumerate(self._blocks):
            if size >= k:
                # Replace the free entry with the REMAINDER only. An earlier version put the
                # allocated range back into the free list as well, so every request was handed
                # start=0 and all of them shared tile ids: codes stayed per slot but the per-tile
                # parameters were overwritten by whichever request flushed last, i.e. every read
                # paired one request's codes with another request's scale factors. With a single
                # request in flight that is invisible, which is exactly the shape of the bug
                # (76% at width 1, 29.5% at width 8).
                self._blocks[i:i + 1] = ([(start + k, size - k)] if size > k else [])
                if _AUDIT:
                    self._assert_blocks_disjoint()
                return start
        self.retry_deferred_blocks()
        for i, (start, size) in enumerate(self._blocks):
            if size >= k:
                self._blocks[i:i + 1] = ([(start + k, size - k)] if size > k else [])
                if _AUDIT:
                    self._assert_blocks_disjoint()
                return start
        raise RuntimeError(
            f"[TILE2] no free block of {k} tiles ({len(self._blocks)} blocks left, "
            f"{len(self._deferred_blocks)} deferred); with "
            f"{self.n_tiles} tiles and {k} per request the pool supports "
            f"{self.n_tiles // k} concurrent requests")

    def _assert_blocks_disjoint(self) -> None:
        """The free list must be a partition of the unallocated tiles: sorted, disjoint.

        Without this the block allocator can hand the same range to two requests and every
        downstream check still passes -- the parameters are simply overwritten.
        """
        prev_end = -1
        for start, size in self._blocks:
            if size <= 0 or start < prev_end:
                print(f"[TILE2-ARGH] free list not a partition: {self._blocks[:8]}", flush=True)
                return
            prev_end = start + size

    def _block_referenced(self, start: int, k: int) -> bool:
        """Is any slot still routing into tiles [start, start+k)?

        `slot_row` holds `tile*T + offset` for every packed slot, so the tile range maps to the
        value range [start*T, (start+k)*T). One vectorised compare over the table (a few MB) is far
        cheaper than the corruption it prevents, and it runs only when a request ends.
        """
        lo, hi = start * self.T, (start + k) * self.T
        sr = self.slot_row
        return bool(((sr >= lo) & (sr < hi)).any())

    def retry_deferred_blocks(self) -> None:
        if not self._deferred_blocks:
            return
        still = []
        for start, k in self._deferred_blocks:
            # Only the radix cache can keep a tile alive past the request that produced it; with it
            # off the reference check is wrong (see free_tile_block).
            if getattr(self, "radix_on", True) and self._block_referenced(start, k):
                still.append((start, k))
            else:
                self._blocks.append((start, k))
        self._deferred_blocks = still
        if self._blocks or not self._deferred_blocks:
            self._blocks.sort()

    def free_tile_block(self, start: int, k: int) -> None:
        # Drop the per-tile ownership records for this range. The old version left them in place,
        # so "this tile's parameters were written twice" fired on every legitimate recycle and the
        # probe that should have caught a genuine theft was drowned in 14,819 false positives.
        lo, hi = start, start + k
        for key in [t for t in self._param_owner if lo <= t[1] < hi]:
            del self._param_owner[key]
        # Only the radix cache can keep a tile alive past the request that produced it. With it
        # off there is no sharing, so a block is free the moment its request ends -- and the
        # reference check is then WRONG, because a freed slot keeps its packed `slot_row` entry
        # until some future request happens to reuse it. Measured: an n=200 radix-off run died at
        # 169/200 with "no free block (3278 deferred)".
        if getattr(self, "radix_on", True) and self._block_referenced(start, k):
            # Someone still reads through these tiles (a shared prefix). Keep the block out of the
            # free list until the last reference is gone; `alloc_tile_block` retries the queue.
            self._deferred_blocks.append((start, k))
        else:
            self._blocks.append((start, k))
            self._blocks.sort()
            # Merge neighbours. Without this the free list fragments after a few hundred requests
            # and a request needing k consecutive tiles fails even though k are free in total.
            merged = []
            for bs, bk in self._blocks:
                if merged and merged[-1][0] + merged[-1][1] == bs:
                    merged[-1] = (merged[-1][0], merged[-1][1] + bk)
                else:
                    merged.append((bs, bk))
            self._blocks = merged

    def alloc_tile(self) -> int:
        if not self._tiles_free:
            raise RuntimeError("[TILE2] tile table exhausted")
        t = self._tiles_free.pop()
        if _AUDIT:
            if t in self._live:
                print(f"[TILE2-ARGH] tile {t} handed out twice", flush=True)
            self._live.add(t)
            self.n_alloc = getattr(self, "n_alloc", 0) + 1
        return t

    def free_tile(self, t: int) -> None:
        if _AUDIT:
            if t not in self._live:
                print(f"[TILE2-ARGH] tile {t} freed but not live", flush=True)
            self._live.discard(t)
            # A recycled tile id is legitimate; its previous owner's parameter rows are gone with
            # it. Without this the detector reports every reuse as a double write.
            for k in [k for k in self._param_owner if k[1] == t]:
                del self._param_owner[k]
        self._tiles_free.append(t)

    def alloc_staging(self) -> int:
        if _AUDIT and self._staging_free:
            pass
        if not self._staging_free:
            raise RuntimeError(
                f"[TILE2] all {self.max_staging} staging slots are in use; raise "
                f"SGLANG_TILE2_STAGING if the batch really is this wide")
        s = self._staging_free.pop()
        if _AUDIT:
            if s in self._stg_live:
                print(f"[TILE2-ARGH] staging {s} handed out twice", flush=True)
            self._stg_live.add(s)
        return s

    def free_staging(self, s: int) -> None:
        if _AUDIT:
            if s not in self._stg_live:
                print(f"[TILE2-ARGH] staging {s} freed but not live", flush=True)
            self._stg_live.discard(s)
        self._staging_free.append(s)

    # -- accounting ---------------------------------------------------------
    def bytes(self) -> dict:
        def nbytes(t):
            return t.numel() * t.element_size()
        payload = nbytes(self.k_codes) + nbytes(self.v_codes)
        params = sum(nbytes(getattr(self, k)) for k in self.PARAM_SHAPES)
        staging = nbytes(self.staging_k) + nbytes(self.staging_v)
        return dict(payload=payload, params=params, staging=staging,
                    slot_row=nbytes(self.slot_row),
                    total=payload + params + staging + nbytes(self.slot_row))

    def format_bits(self) -> float:
        """Format cost of a full pool: payload per token plus metadata amortised over a tile.

        This is the number the design quotes (2.500). It is exact because both terms are
        per-token constants; `format_bits_realized` is what the ceiling of `size/T` tiles makes
        of it on a finite pool.
        """
        payload_pt = 2 * self.H * (self.D // 4)                      # K and V, per token+layer
        params_pt = sum(getattr(self, k).numel() * getattr(self, k).element_size()
                        for k in self.PARAM_SHAPES) / (self.L * self.n_tiles * self.T)
        return (payload_pt + params_pt) * 8 / (self.H * self.D * 2)

    def format_bits_realized(self) -> float:
        b = self.bytes()
        return (b["payload"] + b["params"]) * 8 / (self.L * self.slots * self.H * self.D * 2)

    def report(self) -> str:
        b = self.bytes()
        fmt = self.format_bits()
        lines = [f"[TILE2] pool {self.size} slots, {self.n_tiles} tiles, T={self.T}, "
                 f"staging rows {self.max_staging}",
                 f"[TILE2] payload {b['payload']/2**20:.1f} MiB  params {b['params']/2**20:.1f} MiB  "
                 f"staging {b['staging']/2**20:.1f} MiB  slot_row {b['slot_row']/2**20:.2f} MiB  "
                 f"total {b['total']/2**20:.1f} MiB",
                 f"[TILE2] format {fmt:.4f} bit/element -> {16.0/fmt:.2f}x denser than bf16 "
                 f"(realised {self.format_bits_realized():.4f} on this pool)"]
        return "\n".join(lines)


_RAMP: dict = {}


def _step_cache(buf, name: str, key, build):
    """Build a small device tensor at most once per distinct `key`.

    The KV-write bookkeeping runs once per layer, but `stg_of` / `pre_lens` / the request rows are
    identical across all 36 layers of a step. Rebuilding those tensors per layer cost 576
    host->device copies a step (measured: 75,864 cudaMemcpyAsync against BF16's 475 over the same
    trace), which is pure launch overhead -- the values are the same every time.
    """
    cache = getattr(buf, "_step_tensor_cache", None)
    if cache is None:
        cache = buf._step_tensor_cache = {}
    hit = cache.get(name)
    if hit is not None and hit[0] == key:
        return hit[1]
    val = build()
    cache[name] = (key, val)
    return val


def _ramp(buf, n: int) -> torch.Tensor:
    """A cached 0..n-1 int32 device ramp."""
    key = (buf.device, n)
    r = _RAMP.get(key)
    if r is None:
        r = torch.arange(n, dtype=torch.int32, device=buf.device)
        _RAMP[key] = r
    return r


def own_rows_mask(buf: Tile2Buffers, key: int, ti: int, slots: torch.Tensor) -> torch.Tensor:
    """Which of this tile's rows does THIS request own?

    With the radix cache a request inherits packed slots for a shared prefix and never wrote those
    positions, so its landing-window rows there hold another request's bytes. Packing them would
    (a) compute the tile's per-channel statistics from garbage and (b) repoint the sharer's
    `slot_row` at this tile, so the sharer would decode its own prefix with these parameters.

    A row is ours iff its slot is not packed yet (`slot_row < 0` covers both "never written" and
    the staged encoding). The answer is CACHED per (request, tile index) because `slot_row` is
    shared by all layers and layer 0 flips it to "packed": recomputing for layers 1..35 would
    report "nothing is ours" and silently drop the whole tile.
    """
    cache = buf.__dict__.setdefault("_own_cache", {})
    ck = (key, int(ti))
    om = cache.get(ck)
    if om is None:
        om = buf.slot_row.index_select(0, slots.to(torch.long)) < 0
        cache[ck] = om
    return om


def _flush_prepare(buf: Tile2Buffers, li: int, stg: int, n_valid: int, tile: int,
                   slots: torch.Tensor, update_map: bool, count_tokens: bool = False,
                   req_row: int = -1, key: int = -1, own_mask=None, direct=None):
    """Gather one closing tile out of the landing window and build its row mask.

    Split out of `flush_tile` so that all the tiles closing in one layer can be quantised in ONE
    batched call: a single `quantise_tile` measured 4375 us for a 128x8x128 tile while eight
    together measured 5104 us, i.e. 6.9x cheaper per tile. The cost there is the ~40-strong serial
    chain of small eager kernels, not arithmetic, so batching is what removes it.
    """

    base = stg * buf.T
    slots = slots.to(torch.long)
    if direct is not None:
        # The whole tile came out of the chunk that is being written right now, so take it straight
        # from there instead of bouncing through the landing window. `direct` stays alive until
        # `drain_flushes` returns (the caller still holds it), which also makes the defensive
        # `.clone()` in queue_flush unnecessary for this path.
        k, v = direct
    else:
        k = buf.staging_k[li, base:base + n_valid]
        v = buf.staging_v[li, base:base + n_valid]
    mask = torch.zeros(buf.T, dtype=torch.bool, device=buf.device)
    mask[:n_valid] = True
    if own_mask is not None:
        mask[:own_mask.numel()] &= own_mask
        if not bool(mask.any()):
            return None
    if n_valid < buf.T:                       # pad with zeros; the mask keeps them out of the stats
        pad = buf.T - n_valid
        k = torch.cat([k, torch.zeros(pad, buf.H, buf.D, dtype=k.dtype, device=k.device)], 0)
        v = torch.cat([v, torch.zeros(pad, buf.H, buf.D, dtype=v.dtype, device=v.device)], 0)

    if os.environ.get("SGLANG_TILE2_AUDIT") == "1" and li == 0 and len(buf.audit) < 64:
        buf.audit.append((slots.clone(), k.clone(), v.clone()))
    if getattr(buf, "staging_rotated", False):
        kq_, vq_ = k, v
    else:
        kq_, vq_ = rotate_hadamard(k.float()), rotate_hadamard(v.float())
    if _AUDIT and _STATE.get("last_w") is not None:
        w_loc, w_k, w_v = _STATE["last_w"]
        pos = {int(s_): i for i, s_ in enumerate(w_loc.tolist())}
        hit = [j for j in range(buf.T) if int(slots[j]) in pos]
        if hit:
            ref = torch.stack([w_k[pos[int(slots[j])]] for j in hit]).float()
            got = buf.staging_k[li, base:base + buf.T][hit].float()
            bad = ((got - ref).pow(2).mean(dim=(1, 2)).sqrt()
                   / ref.pow(2).mean(dim=(1, 2)).sqrt()).clamp_max(1e9)
            nbad = int((bad > 0.05).sum())
            if nbad:
                print(f"[TILE2-ARGH] staging mismatch at flush: {nbad}/{len(hit)} rows "
                      f"worse than 5% (worst {float(bad.max()):.3f}) layer={li}", flush=True)
    return dict(k=kq_, v=vq_, mask=mask, slots=slots, tile=int(tile), direct=direct is not None,
                update_map=update_map, count_tokens=count_tokens, key=key,
                n_valid=int(n_valid))


def tile_for_slots(buf: Tile2Buffers, tile_of, key, ti, slots):
    """Global tile id for one group of slots.

    Page-bound mode derives it from the device page the slots belong to, which is what makes a
    tile's lifetime identical to the allocator page's. All T slots must be in ONE page, otherwise
    the codes and their parameters would straddle two host pages.
    """
    if getattr(buf, "page_bound", False):
        t = torch.unique(slots.to(torch.long) // buf.T)
        if int(t.numel()) != 1:
            raise RuntimeError(
                "[TILE2] page-bound tile spans %d device pages (slots %s..%s); --page-size must "
                "equal the tile size (%d)" % (int(t.numel()), int(slots.min()), int(slots.max()),
                                              buf.T))
        return int(t[0])
    return tile_of(key, ti)


def _flush_commit(buf: Tile2Buffers, li: int, ctx: dict, parts: dict) -> None:
    """Write one tile's codes, its eight parameter planes and its routing entry."""
    slots, mask = ctx["slots"], ctx["mask"]
    tile, update_map = ctx["tile"], ctx["update_map"]
    count_tokens, key = ctx["count_tokens"], ctx["key"]
    kq_, n_valid = ctx["k"], ctx["n_valid"]
    if os.environ.get("SGLANG_TILE2_QERR") == "1" and li == 0 and (int(tile) % 20 == 0):
        # Homogeneity of the rows that went into this tile's per-channel statistics. If a tile
        # mixes rows from different requests (or from different staging leases), the token-axis
        # statistics are computed over unrelated data, the quantiser loses far more than it should,
        # and the model degrades without any addressing error. `cv` = coefficient of variation of
        # the per-row norms; a clean tile is one chunk of one request.
        _n = kq_.float().norm(dim=(1, 2))[:int(n_valid)]
        _cv = float(_n.std() / (_n.mean() + 1e-9))
        print(f"[T2Q] tile={int(tile)} rows={int(n_valid)} row_norm_cv={_cv:.4f} "
              f"k_sc_absmean={float(parts['k_sc'].abs().mean()):.5f}", flush=True)
    if _AUDIT and li == 0 and len(buf.audit) == 1:
        print(f"[TILE2-VARN] iters={_ITERS} "
              f"k_A rng=[{parts['k_A'].min().item():.3g},{parts['k_A'].max().item():.3g}] "
              f"k_B rng=[{parts['k_B'].min().item():.3g},{parts['k_B'].max().item():.3g}] "
              f"v_A rng=[{parts['v_A'].min().item():.3g},{parts['v_A'].max().item():.3g}] "
              f"v_B rng=[{parts['v_B'].min().item():.3g},{parts['v_B'].max().item():.3g}] "
              f"sc_k rng=[{parts['k_sc'].min().item():.3g},{parts['k_sc'].max().item():.3g}]",
              flush=True)

    if _AUDIT:
        key = (li, tile)
        owner = buf._param_owner.get(key)
        if owner is not None:
            print(f"[TILE2-ARGH] params (layer {li}, tile {tile}) written twice; "
                  f"first by row {owner}, now by row {buf._cur_row}. "
                  f"slots[0]={int(slots[0])} slots[-1]={int(slots[-1])}", flush=True)
        else:
            buf._param_owner[key] = buf._cur_row
    if bool(mask.all()):
        buf.k_codes[li].index_copy_(0, slots, parts["codes_k"])
        buf.v_codes[li].index_copy_(0, slots, parts["codes_v"])
    else:
        buf.k_codes[li].index_copy_(0, slots[mask], parts["codes_k"][mask])
        buf.v_codes[li].index_copy_(0, slots[mask], parts["codes_v"][mask])
    for name in buf.PARAM_SHAPES:
        getattr(buf, name)[li, tile] = parts[name]
    if update_map:
        off = torch.arange(buf.T, dtype=torch.int32, device=buf.device)
        if bool(mask.all()):
            buf.slot_row.index_copy_(0, slots, (tile * buf.T + off))
        else:
            buf.slot_row.index_copy_(0, slots[mask], (tile * buf.T + off)[mask])
    if buf.packed_ntiles is not None and key is not None:
        # Only now, with the codes and all eight planes written, may the fused kernel treat this
        # tile as readable from codes. Incrementing before the writes would let a decode step read
        # a half-written tile.
        buf.packed_ntiles[li, key] += 1
    if getattr(buf, "page_bound", False):
        # This page has its ten fields for layer `li` once the writes above are enqueued. The page is
        # usable for offload only when every local layer is present; a host sync per page (not per
        # layer) is acceptable because it happens once per 128 tokens.
        buf.page_ready_layers[int(tile)] |= int(1 << li)
        if int(buf.page_ready_layers[int(tile)]) == int(buf._layers_mask):
            buf.page_generation[int(tile)] += 1
    if count_tokens:
        # The map write happens on every layer (idempotent, and it removes the last structural
        # difference between layer 0 and the layers that read it). Counting must not: it is a
        # per-tile quantity and would otherwise be multiplied by the number of layers.
        buf.n_packed_tokens = getattr(buf, "n_packed_tokens", 0) + buf.T


#: SGLANG_TILE2_QFUSED=1 routes the quantiser through `tile2_quant_fused.py`, which does the whole
#: 8-sweep VarN + affine + packing chain in ONE Triton kernel instead of ~40 eager calls.
#: Measured on 8 tiles: 3.100 ms -> 0.203 ms (15.3x), with bit-identical codes and parameter planes
#: differing only by fp32 rounding (relRMS 1e-5). Off by default so the verified path stays default.
_QFUSED = os.environ.get("SGLANG_TILE2_QFUSED") == "1"
#: SGLANG_TILE2_LIFEDBG=1 prints every request retirement (rid -> key + step). This is how the
#: "one slot per resident request" claim is verified rather than assumed.
_LIFEDBG = os.environ.get("SGLANG_TILE2_LIFEDBG") == "1"


def _flush_q(buf: Tile2Buffers) -> dict:
    """Per-buffer flush queue, keyed by layer.

    Not a module-level dict keyed by `id(buf)`: a freed buffer's id is reused by the next one, and
    a queue left behind (a path that stages a tile but never drains) would then be written into an
    unrelated pool. Owned by the buffer, that cannot happen.
    """
    q = buf.__dict__.get("_flush_q")
    if q is None:
        q = buf.__dict__["_flush_q"] = {}
    return q


def queue_flush(buf: Tile2Buffers, li: int, stg: int, n_valid: int, tile: int,
                slots: torch.Tensor, update_map: bool, count_tokens: bool = False,
                req_row: int = -1, key: int = -1, own_mask=None, direct=None) -> None:
    """Stage a closing tile; `drain_flushes` quantises the whole layer's batch in one go."""
    ctx = _flush_prepare(buf, li, stg, n_valid, tile, slots, update_map, count_tokens,
                         req_row, key, own_mask, direct=direct)
    if ctx is not None and os.environ.get("SGLANG_TILE2_FLUSH_NOW") == "1":
        # Diagnostic: quantise immediately (pre-batching behaviour) so an A/B can tell whether
        # deferring the quantisation to the end of the layer changed anything numerically.
        _flush_commit(buf, li, ctx, quantise_tile(ctx["k"], ctx["v"], ctx["mask"],
                                                  bits=buf.bits, iters=_ITERS))
        return
    if ctx is not None:
        # COPY before queueing. `ctx["k"]` is a VIEW into the landing window, and as soon as this
        # tile closes the window restarts at fill = 0 for the request's next tile -- so by the time
        # the layer ends and the batch is quantised, that memory holds a different tile's rows.
        # Batching without this copy quantised the wrong data (P2 caught it: V relRMS 0.997).
        if not ctx.get("direct"):
            ctx["k"] = ctx["k"].clone()
            ctx["v"] = ctx["v"].clone()
        ctx["slots"] = ctx["slots"].clone()
        _flush_q(buf).setdefault(li, []).append(ctx)


def drain_flushes(buf: Tile2Buffers, li: int) -> None:
    q = _flush_q(buf).pop(li, None)
    if not q:
        return
    ks = torch.stack([c["k"] for c in q])
    vs = torch.stack([c["v"] for c in q])
    ms = torch.stack([c["mask"] for c in q])
    if _QFUSED:
        from sglang.srt.mem_cache.tile2_quant_fused import quantise_tiles_fused
        allp = quantise_tiles_fused(ks, vs, ms, iters=_ITERS, bits=buf.bits)
    else:
        allp = quantise_tiles(ks, vs, ms, bits=buf.bits, iters=_ITERS)
    if len(q) == 1:
        _flush_commit(buf, li, q[0], {n: t[0] for n, t in allp.items()})
        return
    for i, ctx in enumerate(q):
        _flush_commit(buf, li, ctx, {n: t[i] for n, t in allp.items()})


def flush_tile(buf: Tile2Buffers, li: int, stg: int, n_valid: int, tile: int,
               slots: torch.Tensor, update_map: bool, count_tokens: bool = False,
               req_row: int = -1, key: int = -1, own_mask=None) -> None:
    """Immediate one-tile path (tests and diagnostics); the hot paths queue + drain."""
    ctx = _flush_prepare(buf, li, stg, n_valid, tile, slots, update_map, count_tokens,
                         req_row, key, own_mask)
    if ctx is None:
        return
    _flush_commit(buf, li, ctx, quantise_tile(ctx["k"], ctx["v"], ctx["mask"],
                                              bits=buf.bits, iters=_ITERS))


def write_and_flush(buf: Tile2Buffers, li: int, stg_of, pre_lens, n_writes, rows, keys, loc,
                    k_write: torch.Tensor, v_write: torch.Tensor, r2t: torch.Tensor,
                    tile_of, update_map: bool = True, rotate: bool = True, pending=None,
                    track_positions: bool = True, ridx_of=None, src_dev=None):
    """Stage one write call's rows and flush every tile that closed.

    All requests in the batch are handled together because `loc`, `k_write` and `v_write` arrive
    concatenated in batch order. A single call can span several tiles (chunked prefill writes up
    to 2048 rows), so the walk below is segmented by the position grid -- assuming "one call is
    at most one tile" is the mistake that makes the first tile lose its data.

    stg_of   : staging slot id per batch row (host list)
    pre_lens : positions already written before this call, per batch row (host list)
    n_writes : rows written by this call, per batch row (host list)
    rows     : req_to_token row per batch row (host list). Kept for diagnostics only: the slots of
               a tile are ACCUMULATED as they are written, not looked up in req_to_token. A tile
               that straddles a chunked-prefill boundary has half its positions in the previous
               chunk's req_to_token row, and that row is not the one this call is holding, so a
               lookup silently returns uninitialised slots -- which is exactly the defect that
               made three of eight concurrent requests come out badly quantised.
    pending  : {key: [slot tensors written since the last flush]}
    keys     : stable per-request identity per batch row (host list). NOT the req_to_token row:
               with chunked prefill the same request is extended in several calls and sglang hands
               it a different req_pool_idx in each, so a row-keyed store loses the partial tile at
               every chunk boundary. The position-0 slot is stable for the request's lifetime.
    tile_of  : callable (req_row, tile_index) -> global tile id, allocated once per closure
    """
    if rotate:
        k_write, v_write = rotate_hadamard(k_write), rotate_hadamard(v_write)
    kr = k_write.to(torch.bfloat16)
    vr = v_write.to(torch.bfloat16)

    if _NOFLUSH and rows:
        # keep everything staged; overflow the window loudly rather than silently flushing
        pass

    if _AUDIT:
        tot = int(sum(int(x) for x in n_writes))
        if tot != k_write.shape[0]:
            print(f"[TILE2-ARGH] segmentation mismatch sum(nwrites)={tot} "
                  f"loc={k_write.shape[0]} pre={list(pre_lens)} nw={list(n_writes)}",
                  flush=True)
    if _TRACE:
        for i, r in enumerate(keys):
            if r in _TRACE:
                print(f"[T2T] write row={r} pre={pre_lens[i]} n={n_writes[i]} li={li} "
                      f"loc0={None if loc.numel() == 0 else int(loc[0])} "
                      f"nloc={loc.numel()}", flush=True)
    ridx_list = ([ridx_of(int(k)) for k in keys] if ridx_of is not None
                 else [int(rows[i]) for i in range(len(rows))])
    # ---- decode fast path: stage every request's single row in one shot -------------------
    # The per-request loop below is 8 iterations x 36 layers = 288 Python/torch calls a step, and it
    # measured 26 ms/step for what is two index_copies. When every request writes exactly one row
    # the staging half of the work is fully vectorisable; the map writes and the (rare) tile close
    # stay in the general path.
    staged_vec = False
    if (not _NOFLUSH) and len(rows) > 0 and all(int(x) == 1 for x in n_writes):
        if src_dev is not None:
            src = src_dev
        else:
            pre_t = torch.tensor([int(x) for x in pre_lens], dtype=torch.long, device=buf.device)
            stg_t = torch.tensor([int(x) for x in stg_of], dtype=torch.long, device=buf.device)
            src = stg_t * buf.T + (pre_t % buf.T)
        if _PROF is not None:
            _ts0 = time.perf_counter()
        if not _NOSTAGE:
            buf.staging_k[li].index_copy_(0, src, kr)
            buf.staging_v[li].index_copy_(0, src, vr)
        if _PROF is not None:
            _prof_add("stgact", time.perf_counter() - _ts0)
        staged_vec = True
        # Decode closes a tile only once every T steps per request. When none closes, the staging
        # writes above are the whole job and the per-request loop below (8 iterations x 36 layers
        # = 288 a step) can be skipped.
        if (_NO_EE and False) or (not _NO_EE and not any((int(x) + 1) % buf.T == 0
                                                        for x in pre_lens)):
            if update_map:
                _loc = loc.to(torch.long)
                _sr_val = _step_cache(buf, "srow_val", (tuple(stg_of), tuple(pre_lens)),
                                      lambda: -(2 + (
                                          torch.tensor([int(x) for x in stg_of], dtype=torch.int32,
                                                       device=buf.device) * buf.T
                                          + torch.tensor([int(x) % buf.T for x in pre_lens],
                                                         dtype=torch.int32, device=buf.device))))
                buf.slot_row.index_copy_(0, _loc, _sr_val)
                if buf.pos_slot is not None:
                    # The position table is indexed by the request's COMPACT index, not by the
                    # req_to_token row -- the row changes between prefill chunks.
                    if ridx_of is not None:
                        _ri = _step_cache(buf, "posrow", tuple(keys),
                                          lambda: torch.tensor(
                                              [int(ridx_of(int(k))) for k in keys],
                                              dtype=torch.long, device=buf.device))
                    else:
                        _ri = _step_cache(buf, "posrow", tuple(rows),
                                          lambda: torch.tensor(
                                              [int(x) for x in rows],
                                              dtype=torch.long, device=buf.device))
                    _pt = torch.tensor([int(x) for x in pre_lens], dtype=torch.long, device=buf.device)
                    buf.pos_slot[(_ri, _pt)] = _loc.to(torch.int32)
            return
    start = 0
    for i in range(len(rows)):
        n = int(n_writes[i])
        if n == 0:
            continue
        pre = int(pre_lens[i])
        row = int(rows[i])
        key = int(keys[i])
        stg = int(stg_of[i])
        ridx = ridx_list
        j, end = start, start + n
        p = pre
        fill = pre % buf.T
        _tile_j0 = -1                      # where the tile currently filling began, in this chunk
        if pending is not None and pre == 0 and li == buf.start_layer:
            # pre == 0 means a request is starting. The identity we key on (the position-0 slot) is
            # recycled once the old request is gone, so anything left under it is stale.
            #
            # ONLY on the first layer. This reset used to run once per layer, so layer 0 filled the
            # position->slot map and layer 1 immediately wiped it again; every tile that straddled
            # a chunked-prefill boundary then fell back to req_to_token, where the first half of
            # the tile lives in the PREVIOUS chunk's row and reads as uninitialised slots.
            pending.pop(key, None)
        # One map write for the whole chunk instead of one per tile. `slot_row` is a pure function
        # of (staging slot, position) and `pos_slot` of (compact request index, position); neither
        # depends on where the tile boundaries fall, so writing them per tile only multiplied the
        # Python loop count by `n / T`. Measured motivation: prefill leaves the GPU idle ~42 ms per
        # layer-chunk, and 8 requests x 64 tiles x 36 layers = 18,432 iterations carry 4 device ops
        # each. Set SGLANG_TILE2_ONEPASS=0 to restore the per-tile writes for an A/B.
        # `slot_row` and `pos_slot` are shared by all layers -- the tile grid is anchored at
        # position 0, so layer L and layer 0 must write the same values. Writing them on every
        # layer is therefore 36x redundant. Layer 0 runs first in every forward, so the maps are
        # already correct by the time layers 1..35 read them.
        if update_map and _ONEPASS and n > 0 and li == buf.start_layer:
            _pos_all = torch.arange(p, p + n, dtype=torch.long, device=buf.device)
            if buf.pos_slot is not None and not track_positions:
                buf.pos_slot[(ridx[i], _pos_all)] = loc[j:j + n].to(torch.int32)
            buf.slot_row.index_copy_(
                0, loc[j:j + n].to(torch.long),
                -(2 + stg * buf.T + (_pos_all % buf.T).to(torch.int32)))
        while j < end:
            take = min(end - j, buf.T - fill)
            if fill == 0:
                _tile_j0 = j
            base = stg * buf.T + fill
            if not staged_vec:
                if not _NOSTAGE:
                    buf.staging_k[li, base:base + take] = kr[j:j + take]
                    buf.staging_v[li, base:base + take] = vr[j:j + take]
            if update_map and not (_ONEPASS and n > 0):
                # Reuse one ramp buffer. `torch.arange` per request per layer is 8 x 36 small
                # allocations a step, and the write path measured 21-47 ms/step for what is
                # otherwise two index_copies.
                off = _ramp(buf, buf.T)[fill:fill + take]
                buf.slot_row.index_copy_(0, loc[j:j + take].to(torch.long),
                                         -(2 + stg * buf.T + off))
            if (buf.pos_slot is not None and not track_positions and update_map
                    and not (_ONEPASS and n > 0)):
                # Device-side position table. The dict version below needs `loc[...].tolist()`,
                # which is a device->host sync, and write_and_flush runs once per layer -- 36 syncs
                # a step, each one draining the pipeline. That alone made packed_tile2 4.2x slower
                # than bf16 while packed_int4 (whose write is a pure kernel) was only 1.34x slower.
                # p .. p+take are the logical positions; the row index into pos_slot is the
                # request's COMPACT index, which survives a chunked-prefill row change.
                pos_t = torch.arange(p, p + take, dtype=torch.long, device=buf.device)
                buf.pos_slot[(ridx[i], pos_t)] = loc[j:j + take].to(torch.int32)
            elif pending is not None:
                # position -> slot, idempotent by construction: write_and_flush runs once per layer
                # (36 times a step) and a tile can straddle a chunked-prefill boundary, so anything
                # that counts appends has to be exactly right 36 times over. A map only has to be
                # right once.
                m = pending.setdefault(key, {})
                for off_i, sl_ in zip(range(p, p + take), loc[j:j + take].tolist()):
                    m[off_i] = sl_
            fill += take
            j += take
            p += take
            if _NOFLUSH and fill >= buf.T:
                raise RuntimeError("[TILE2] staging window full and NOFLUSH is set")
            if fill == buf.T and not _NOFLUSH:
                # Prefer the slots we accumulated while writing: a tile that straddles a chunked
                # prefill boundary has half its positions in an earlier req_to_token row, so the
                # lookup below can only be trusted when the accumulation is complete.
                slots = None
                nacc = -1
                if buf.pos_slot is not None:
                    # Device position table: one host read of T entries per FLUSH (rare), instead of
                    # a `.tolist()` per layer. It is also the only source that is correct when a
                    # tile straddles a chunked-prefill boundary, because it is written by the same
                    # call that stages the rows.
                    # `pos_slot` is indexed by the request's COMPACT index (ridx), never by the
                    # req_to_token row: the table only has REQ_SLOTS(+1) rows, while row ids run
                    # into the thousands, so using `row` here read another request's positions or
                    # went out of bounds. Measured consequence: an n=500 run died at item 239 with
                    # "IndexError: index 513 is out of bounds for dimension 0 with size 513".
                    _ri_tile = ridx[i] if ridx is not None else (
                        ridx_of(int(key)) if ridx_of is not None else int(row))
                    cand = buf.pos_slot[_ri_tile, p - buf.T:p]
                    if int(cand.numel()) == buf.T and int(cand.min()) >= 0:
                        slots = cand.to(torch.long)
                        nacc = buf.T
                if slots is None and pending is not None:
                    m = pending.get(key, {})
                    want_pos = list(range(p - buf.T, p))
                    if all(q in m for q in want_pos):
                        # Do NOT remove the entries. The map is layer-independent and read once
                        # per layer (36 times a step); popping here let the first layer consume
                        # the positions and forced the other 35 onto the req_to_token fallback,
                        # which is exactly where the straddling tiles were being corrupted.
                        slots = torch.tensor([m[q] for q in want_pos],
                                             dtype=torch.long, device=buf.device)
                        nacc = buf.T
                if slots is None:
                    slots = r2t[row, p - buf.T:p]
                    if int(slots.numel()) != buf.T:
                        raise RuntimeError(f"[TILE2] tile {p // buf.T - 1}: short position map")
                    if _AUDIT:
                        # NOTE: report the count taken BEFORE the pop. Reading it back out of
                        # `pending` here is always zero and cost an hour of chasing a diagnostic
                        # artefact instead of the defect.
                        print(f"[TILE2-ARGH] tile {p // buf.T - 1} key={key} li={li} "
                              f"have={nacc} want={buf.T} fill={fill} "
                              f"map={0 if pending is None else len(pending.get(key, {}))}",
                              flush=True)
                _ti = p // buf.T - 1
                if _TRACE and key in _TRACE:
                    print(f"[T2T] flush key={key} ti={_ti} stg={stg} li={li} "
                          f"slots0={int(slots[0])} slotT={int(slots[-1])}", flush=True)
                if _AUDIT and li == buf.start_layer:
                    # Only the first layer of a step sees the map in its pre-flush state: every
                    # layer now writes the map, so from layer 1 on the entries are legitimately
                    # "packed" already.
                    # The 128 slots about to be packed must be tagged in the position map as
                    # "staged, at exactly these offsets, in this order". If the slot list were
                    # offset by even one tile, the map would say "packed" (previous tile) or carry
                    # different offsets, and this fires.
                    cur = buf.slot_row.index_select(0, slots.to(torch.long)).to(torch.long)
                    want = -(2 + stg * buf.T
                             + torch.arange(buf.T, device=buf.device, dtype=torch.long))
                    if not torch.equal(cur, want):
                        nd = int((cur != want).sum())
                        print(f"[TILE2-ARGH] tile {_ti} key={key} li={li}: {nd}/{buf.T} slots "
                              f"in the map do not match the staged encoding "
                              f"(map[0]={int(cur[0])} want[0]={int(want[0])})", flush=True)
                # ONLY with the radix cache can a row of this tile belong to someone else. With
                # it off, a slot whose `slot_row` still looks packed is simply a slot that a
                # PREVIOUS request used and sglang has since freed -- `own_rows_mask` would then
                # skip packing this request's own row, and that position would read another
                # request's bytes forever. Slot reuse only starts after enough requests have run,
                # which is exactly the shape of the observed cliff (fine for 160 requests, then
                # 0.05). Measured: 0.595 -> 0.71 by gating this on the radix flag.
                _own = own_rows_mask(buf, key, _ti, slots) if getattr(buf, "radix_on", True) else None
                if _own is not None and not bool(_own.any()):
                    fill = 0          # fully inherited tile: nothing of ours to pack
                    continue
                # A tile that began at fill == 0 inside this chunk is entirely in `kr`, so hand
                # the quantiser the chunk slice directly (no landing-window round trip, no clone).
                _direct = None
                if _FUSE and _tile_j0 >= 0 and _tile_j0 + buf.T <= end:
                    _direct = (kr[_tile_j0:_tile_j0 + buf.T], vr[_tile_j0:_tile_j0 + buf.T])
                queue_flush(buf, li, stg, buf.T, tile_for_slots(buf, tile_of, key, _ti, slots), slots, update_map,
                            count_tokens=(li == buf.start_layer), req_row=row, key=key,
                            own_mask=_own, direct=_direct)
                fill = 0
        start = end
    drain_flushes(buf, li)


def gather_dequant_torch(buf: Tile2Buffers, li: int, idx: torch.Tensor,
                         out_k: torch.Tensor, out_v: torch.Tensor) -> None:
    """Dequantise the slots `idx` into the dense bf16 workspace, staying in the rotated domain.

    Both sources are handled in one pass: packed rows read their codes from the pool and their
    parameters from the tile table; staging rows are already rotated bf16 and are copied as is.
    """
    n = idx.numel()
    if n == 0:
        return
    idx = idx.to(torch.long)
    rows = buf.slot_row.index_select(0, idx).to(torch.long)
    packed = rows >= 0

    def per_token(tab, tile, off):
        """tab: [L, n_tiles, T, H] -> [np, H] at (li, tile, off)."""
        return tab[li].index_select(0, tile).gather(
            1, off.view(-1, 1, 1).expand(-1, 1, buf.H)).squeeze(1)

    if bool(packed.any()):
        rr = rows[packed]
        sl = idx[packed]
        tile, off = rr // buf.T, rr % buf.T

        ck = unpack2(buf.k_codes[li].index_select(0, sl), buf.D)
        kd = ck * buf.k_sc[li].index_select(0, tile) + buf.k_lo[li].index_select(0, tile)
        kd = kd * buf.k_A[li].index_select(0, tile) * per_token(buf.k_B, tile, off).unsqueeze(-1)
        out_k[packed] = kd.to(out_k.dtype)

        cv = unpack2(buf.v_codes[li].index_select(0, sl), buf.D)
        sv = per_token(buf.v_sc, tile, off) * per_token(buf.v_B, tile, off)
        lv = per_token(buf.v_lo, tile, off) * per_token(buf.v_B, tile, off)
        vd = cv * sv.unsqueeze(-1) + lv.unsqueeze(-1)
        vd = vd * buf.v_A[li].index_select(0, tile)
        out_v[packed] = vd.to(out_v.dtype)

    staged = (~packed) & (rows != buf.ROW_UNWRITTEN)
    if bool(staged.any()):
        sv = (-rows[staged] - 2)
        out_k[staged] = buf.staging_k[li].index_select(0, sv).to(out_k.dtype)
        out_v[staged] = buf.staging_v[li].index_select(0, sv).to(out_v.dtype)
    unwritten = rows == buf.ROW_UNWRITTEN
    if bool(unwritten.any()):
        out_k[unwritten] = 0
        out_v[unwritten] = 0

    if os.environ.get("SGLANG_TILE2_AUDIT") == "1" and li == 0 and buf.audit:
        # running worst-case over every gather, against the exact rows that were flushed
        idx_l = idx.to(torch.long)
        pos_of = {int(sv_): i for i, sv_ in enumerate(idx_l.tolist())}
        wk = wv = 0.0
        for sl, kraw, vraw in buf.audit:
            sel = [j for j, sv_ in enumerate(sl.tolist()) if sv_ in pos_of]
            if not sel:
                continue
            rows_out = torch.tensor([pos_of[sl[j].item()] for j in sel], device=out_k.device)
            ref_k = kraw[sel].float()
            ref_v = vraw[sel].float()
            wk = max(wk, ((out_k[rows_out].float() - ref_k).pow(2).mean().sqrt()
                          / ref_k.pow(2).mean().sqrt()).item())
            wv = max(wv, ((out_v[rows_out].float() - ref_v).pow(2).mean().sqrt()
                          / ref_v.pow(2).mean().sqrt()).item())
        st = _STATE.setdefault("au", {"n": 0, "wk": 0.0, "wv": 0.0})
        st["n"] += 1
        st["wk"] = max(st["wk"], wk)
        st["wv"] = max(st["wv"], wv)
        if st["n"] == 1:
            for ti, (sl, kraw, vraw) in enumerate(buf.audit):
                sel = [j for j, sv_ in enumerate(sl.tolist()) if sv_ in pos_of]
                if len(sel) != sl.numel():
                    print(f"[TILE2-AUDIT] tile#{ti} only {len(sel)}/{sl.numel()} slots visible",
                          flush=True)
                    continue
                ro = torch.tensor([pos_of[sl[j].item()] for j in sel], device=out_k.device)
                rk = kraw[sel].float()
                rv = vraw[sel].float()
                ek = ((out_k[ro].float() - rk).pow(2).mean().sqrt()
                      / rk.pow(2).mean().sqrt()).item()
                ev = ((out_v[ro].float() - rv).pow(2).mean().sqrt()
                      / rv.pow(2).mean().sqrt()).item()
                pr = buf.slot_row.index_select(0, sl.to(torch.long))
                print(f"[TILE2-AUDIT] tile#{ti} row0={int(pr[0])} tiles="
                      f"{sorted(set((pr // buf.T).tolist()))[:3]} K={ek:.4f} V={ev:.4f} "
                      f"|A_k|max={buf.k_A[0].abs().max().item():.3g}", flush=True)
        if st["n"] % 200 == 0:
            print(f"[TILE2-AUDIT] gathers={st['n']} tiles_saved={len(buf.audit)} "
                  f"worst K={st['wk']:.4f} V={st['wv']:.4f} (this call K={wk:.4f} V={wv:.4f})",
                  flush=True)


#: Actually freeing a request's landing-window slot and tile block looks obviously right, and it
#: is -- except that `flush_deferred` records liveness under `_row_key.get(row, row)`, so a row
#: whose mapping is absent poisons the liveness map with a ROW number. Once frees really happen,
#: live requests then look stale and their slots are handed to someone else, and because the read
#: path serves the *newest* tokens out of the landing window, that corrupts exactly the tokens
#: that matter most: n=200 accuracy fell 71.5% -> 38.0% with truncation 31.5% -> 63.0%.
#: So the translating (effectively no-op) path stays the default until the liveness map is keyed
#: consistently; set SGLANG_TILE2_FREE_RELEASE=1 to exercise the correct behaviour.
def _free_release() -> bool:
    # Default ON now that liveness is recorded in the request-key namespace (see flush_deferred).
    # SGLANG_TILE2_FREE_RELEASE=0 restores the old, effectively no-op behaviour.
    return os.environ.get("SGLANG_TILE2_FREE_RELEASE", "1") != "0"


def _write_ranges(fb, nreq):
    """(positions already written, rows written now) per batch row.

    Decode leaves `extend_prefix_lens/extend_seq_lens` unset, so the decode case is derived from
    `seq_lens` (which already counts the token being written): every request writes exactly one
    row, at position seq_len - 1. The `_cpu` variants are preferred when present because they
    avoid a device sync, which is what the CUDA-graph path will need.
    """
    nw = getattr(fb, "extend_seq_lens_cpu", None)
    pre = getattr(fb, "extend_prefix_lens_cpu", None)
    if nw is None:
        t = getattr(fb, "extend_seq_lens", None)
        nw = [1] * nreq if t is None else [int(x) for x in t.tolist()]
    if pre is None:
        t = getattr(fb, "extend_prefix_lens", None)
        pre = ([int(x) for x in t.tolist()] if t is not None
               else [int(x) - n for x, n in zip(fb.seq_lens.tolist(), nw)])
    return list(pre), list(nw)


class Tile2Driver:
    """Per-step glue between a ForwardBatch and the store.

    Two pieces of state, both keyed by the request-pool row and released from
    `ReqToTokenPool.free`, so a recycled row cannot inherit either:

      staging slot  -- where the request's open tile accumulates (bf16, rotated)
      tile ids      -- (row, tile index) -> global tile id, allocated ONCE per closure event

    The second one is why `tile_of` is a callback rather than a counter: all 36 layers close the
    same tiles at the same step, and a counter advanced per layer would hand out 36 different ids
    for the same tile, silently corrupting every layer but the one that won.
    """

    def __init__(self, buf: Tile2Buffers, start_layer: int = 0):
        self.buf = buf
        self.start_layer = start_layer
        self._stg: dict = {}
        self._tile: dict = {}
        self._blk: dict = {}
        self._len: dict = {}
        #: stable request key -> compact index used to address the per-request position table
        self._ridx: dict = {}
        self._ridx_next = 0
        self.packed_tokens = 0
        self.steps = 0
        self._ctx = None
        self._src_dev = None
        self._lru = []
        # --- request lifecycle -------------------------------------------------------------
        # `retire_request(rid)` is called from the SCHEDULER thread when a request is finished or
        # aborted. It only enqueues: the physical reclaim happens on the WORKER thread inside
        # `drain_retires()` (called from flush_deferred), because returning a staging buffer while
        # a forward that still reads it is in flight is a GPU use-after-free.
        self._retire_q = []
        self._retire_tries = {}
        self._retired = {}          # key -> step at which it was retired (diagnostics only)
        #: free list for compact indices; `ridx_of` used to hand out `n % REQ_SLOTS`, which can
        #: collide with a long-lived request once the counter wraps.
        self._ridx_free = []
        self._last_seen = {}
        self._released = {}
        self._row_key = {}
        self._active = set()
        self._pending = {}
        #: request-pool row -> STABLE per-request id (see note_request). The identity used to be
        #: the slot of position 0: stable across the row change that chunked prefill causes, but
        #: NOT unique once the radix cache shares a prefix -- every request reusing that prefix
        #: got the same key, so `_stg`/`_tile`/`pos_slot`/`packed_ntiles` aliased across live
        #: requests and decode read a neighbour's KV (measured: n=20 accuracy 0.00, all truncated,
        #: vs 0.70 with the cache off; a trace showed three concurrent requests with the same key
        #: AND the same staging slot). A row is unique but not stable. `Request.rid` is both.
        self._row_rid: dict = {}
        self._rid_by_req: dict = {}
        self._rid_next = 0

    # -- request identity ---------------------------------------------------
    def note_request(self, row: int, rid) -> None:
        """Bind a request-pool row to a stable per-request id. Called from the scheduler hook.

        Only `ScheduleBatch` sees `Request.rid` next to `req_pool_indices`, and it does so on the
        host before the batch is handed to the model, so this is both the cheapest and the only
        place where the two can be joined.
        """
        known = self._rid_by_req.get(rid)
        first = known is None
        if first:
            known = self._rid_next
            self._rid_next += 1
            self._rid_by_req[rid] = known
        self._row_rid[row] = known
        return first

    def key_of_row(self, row: int, fallback=None) -> int:
        """Stable per-request key for this row, else `fallback` (or the union of both worlds)."""
        k = self._row_rid.get(row)
        if k is not None:
            return k
        if os.environ.get("SGLANG_TILE2_KEYDBG") == "1":
            print(f"[T2KEY-MISS] row={row} has no rid (the scheduler hook never described it); "
                  f"falling back to slot-0={fallback}", flush=True)
        if fallback is not None:
            return fallback
        # Rows the scheduler hook has not described are capture-time synthetic batches; they never
        # flush, so a per-row private namespace is enough to keep them from aliasing a real key.
        return -(row + 1)

    # -- allocation ---------------------------------------------------------
    def tile_of(self, key: int, ti: int) -> int:
        """Global tile id for (request, tile index), taking blocks ON DEMAND.

        `tiles_per_req` is the BLOCK SIZE (allocation granularity), not a per-request reservation.
        Reserving it up front wasted most of it: a 2.6k-token request uses ~21 tiles but reserved
        24-128, and under the radix cache every finished request keeps its block pinned in the
        tree, so a 3583-tile pool ran out at 108-145 of 200 problems ("no free block") while the
        token pool was only 60-80% full. Blocks stay contiguous internally (all `ti % K` needs);
        the read path never assumes tiles are contiguous across blocks -- it derives the tile id
        from `slot_row`.
        """
        K = max(1, int(self.buf.tiles_per_req))
        bi = int(ti) // K
        blocks = self._blk.get(key)
        if blocks is None:
            blocks = []
            self._blk[key] = blocks
        while len(blocks) <= bi:
            blocks.append(self.buf.alloc_tile_block(K))
        return blocks[bi] + (int(ti) % K)

    def retire_request(self, rid) -> None:
        """SCHEDULER thread: this request is finished/aborted, so its slot may be returned.

        Deliberately does not touch any buffer: it only records the rid. `drain_retires()` does the
        work on the worker thread. An unknown rid is ignored rather than creating an identity --
        inventing one here is how a stranger's slot gets freed.
        """
        if _LIFEDBG:
            print(f"[T2LIFE] retire_request(rid={rid!r}) called; known="
                  f"{str(rid) in self._rid_by_req}", flush=True)
        self._retire_q.append(str(rid))

    def drain_retires(self) -> None:
        """WORKER thread (from flush_deferred): physically return slots of retired requests.

        A retired request may still appear in `_active` (the batch whose result the scheduler is
        processing), so it is retried next step. Nothing is freed while still scheduled.
        """
        if not self._retire_q:
            return
        q, self._retire_q = self._retire_q, []
        active = set(getattr(self, "_active", ()) or ())
        for rid in q:
            key = self._rid_by_req.get(rid)
            if key is None:
                if _LIFEDBG:
                    print(f"[T2LIFE] retire rid={rid!r} has NO key binding "
                          f"(known rids sample={list(self._rid_by_req)[:3]})", flush=True)
                continue
            if key in active and self._retire_tries.get(rid, 0) < 8:
                self._retire_tries[rid] = self._retire_tries.get(rid, 0) + 1
                self._retire_q.append(rid)
                continue
            if key in active:
                # Refusing loudly is the point. Freeing a slot that a live request is still writing
                # is silent corruption (measured once before: n=200 accuracy 0.710 -> 0.555), and a
                # crash here is far cheaper than a wrong number in a paper table.
                raise RuntimeError(
                    f"[TILE2] refusing to retire {rid} (key={key}): still in the active batch "
                    f"after 8 steps. This means the lifecycle hook and the write path disagree "
                    f"about ownership; do NOT free it.")
            self._retire_tries.pop(rid, None)
            if _AUDIT and key in active:
                print(f"[TILE2-ARGH] freeing ACTIVE key {key} on retire of {rid}", flush=True)
            self._retired[key] = self.steps
            if _LIFEDBG:
                print(f"[T2LIFE] retire rid={rid} key={key} step={self.steps} "
                      f"free_before={len(getattr(self.buf, '_staging_free', []))}", flush=True)
            self.release(key)

    def staging_of(self, key: int) -> int:
        """Staging slot for a request, with LRU recycling. `key` = stable request id.

        Not tied to ReqToTokenPool.free alone: warmup requests never go through the scheduler's
        normal teardown, so with a free-list-only policy eight warmup rows took every slot and the
        first real request died with "all staging slots are in use". A slot is only interesting
        while its request is in the current batch, so the least recently used row is a safe
        victim -- and if every slot belongs to the current batch, this is a real capacity error
        and it says so.
        """
        self._last_seen[key] = self.steps
        s = self._stg.get(key)
        if s is not None:
            if key in self._lru:
                self._lru.remove(key)
            self._lru.append(key)
            return s
        if not self.buf._staging_free:
            # CORRECT release: requests that actually ENDED, reported by the scheduler hook.
            self.drain_retires()
        if not self.buf._staging_free:
            # Leak guard ONLY. A key that never saw a retire and has been untouched for a very long
            # time is almost certainly a warmup/synthetic identity (they never reach the scheduler's
            # teardown). 512 steps is a leak detector, NOT a correctness mechanism: the previous
            # version used 32 steps as one and recycled LIVE requests -- measured n=200 0.710 ->
            # 0.555, and short requests (MMLU) exhausted 24 slots on the first item.
            current = set(getattr(self, "_active", ()) or ())
            victim = next((r for r in self._lru
                           if r not in current and r not in self._retired
                           and self.steps - self._last_seen.get(r, 0) > 512), None)
            if victim is not None:
                print(f"[TILE2] leak guard: recycling key {victim}, unseen for "
                      f"{self.steps - self._last_seen.get(victim, 0)} steps, no retire seen",
                      flush=True)
                self.release(victim)
        if not self.buf._staging_free:
            _ages = sorted(self.steps - self._last_seen.get(k, -10**9) for k in self._lru)
            raise RuntimeError(
                f"[TILE2] all {self.buf.max_staging} staging slots are held by live requests; "
                f"with the retire hook this means concurrency > SGLANG_TILE2_STAGING. "
                f"steps={self.steps} n_active={len(set(getattr(self, '_active', ()) or ()))} "
                f"n_slots={len(self._lru)} youngest_ages={_ages[:6]}")
        s = self.buf.alloc_staging()
        self._stg[key] = s
        # Tiles are NOT reserved here any more: `tile_of` takes a block when a tile actually
        # closes. Reserving one per request at staging time was the source of the waste that made
        # an n=200 run die of "no free block" while the token pool was 60-80% empty.
        self._lru.append(key)
        return s

    def release(self, key: int) -> None:
        """Free one request's staging slot and tile block.

        `key` is the STABLE request id (the slot of its position 0), which is what `_lru` holds
        and what `staging_of` is called with -- every map in this class is keyed by it.

        This used to translate the argument through `_row_key` first, i.e. it treated a key as a
        req_to_token ROW. The two namespaces overlap numerically (rows are 0..~4096, keys are slot
        ids starting at 0), so whenever a key happened to equal a live row the call popped a
        stranger's entry: either nothing was freed (the pool then died with "all N staging slots
        are in use") or another LIVE request's slot was handed out from under it. Both were
        silent; the second one is corruption. `_row_key` is now only read for row -> key lookups.
        """
        # `key` is ALWAYS the stable request key, never a req_to_token row. The old row-translating
        # branch (which an env var could re-enable) silently freed a stranger's entries because the
        # two namespaces overlap numerically. One switch for one meaning only.
        if key in self._retired:
            self._retired.pop(key)
        if _AUDIT:
            self._released[key] = self.steps
        if key in self._lru:
            self._lru.remove(key)
        blocks = self._blk.pop(key, None)
        if os.environ.get("SGLANG_TILE2_KEEP_TILES") != "1":
            for base in (blocks or []):
                self.buf.free_tile_block(base, max(1, int(self.buf.tiles_per_req)))
        if False:
            # With the radix cache this request's tiles may still be referenced by another request
            # that shares its prefix, so returning them to the free list lets the next request
            # overwrite parameters the sharer is decoding with. Correct lifetime is "when the last
            # slot pointing at the tile is evicted"; until that is wired, KEEP_TILES=1 isolates the
            # effect by never recycling tiles.
            self.buf.free_tile_block(base, self.buf.tiles_per_req)
        self._pending.pop(key, None)
        s = self._stg.pop(key, None)
        if _LIFEDBG:
            print(f"[T2LIFE] release(key={key}) staging={'FREED' if s is not None else 'MISSING'}"
                  f" lru={len(self._lru)} free={len(getattr(self.buf, '_staging_free', []))}",
                  flush=True)
        if s is not None:
            assert self._stg.get(key) is None, "release() left a staging binding behind"
            self.buf.free_staging(s)
        elif _AUDIT:
            print(f"[TILE2-ARGH] release(key={key}) found no staging slot", flush=True)
        self._len.pop(key, None)
        self._last_seen.pop(key, None)
        if True:
            # Hand the compact index back so it can be reused (see `ridx_of`).
            _ri = self._ridx.pop(key, None)
            if _ri is not None:
                self._ridx_free.append(_ri)
            for r, kk in list(self._row_key.items()):
                if kk == key:
                    self._row_key.pop(r, None)

    def _ensure_pos_slot(self, fb) -> None:
        b = self.buf
        r2t = fb.req_to_token_pool.req_to_token
        if b.pos_slot is None:
            # Indexed by a COMPACT per-request index, not by the req_to_token row. A request that
            # is prefilled in two chunks lives on two rows, and the second row's req_to_token only
            # holds that chunk's positions -- so a row-keyed table (or req_to_token itself) has
            # holes for everything written earlier. That is exactly the 2048-token cliff: correct
            # at n <= 2048, wrong above it.
            # Sized by concurrent requests, not by the request-pool row count: the table only has
            # to hold the requests that are live at once, and 4096 rows cost 268 MB of the KV
            # budget (that showed up as capacity 499,718 -> 486,613 tokens).
            b.pos_slot = torch.full((REQ_SLOTS + 1, r2t.shape[1]), -1,
                                    dtype=torch.int32, device=b.device)
            # CUDA-graph padding support. `replay_prepare` writes `pad_sentinel` into the padded
            # rows' `out_cache_loc`, and padded tokens are REDIRECTED to these dustbin rows rather
            # than filtered out -- filtering needs `nonzero`, whose data-dependent shape capture
            # rejects. The extra pos_slot row and the last staging row are never read by anyone.
            # Slot 0 is the spare one, NOT `size`. The page-size-one allocator hands out
            # `arange(1, size + 1)`, i.e. 1..size inclusive, so `size` is a REAL slot a live
            # request can own. Using it as the sentinel meant a token that happened to be
            # allocated slot `size` had its K/V and position-map write redirected to the dustbin,
            # leaving attention to read stale data. Slot 0 is never handed out (the allocator's
            # own comment calls it "the padded slot 0"), so it is the safe spare.
            b.pad_sentinel = 0
            b.dustbin_pos_row = int(REQ_SLOTS)
            b.dustbin_stg_row = int(b.max_staging * b.T)   # the extra row, outside every slot
        if b.ridx_tab is None:
            b.ridx_tab = torch.zeros((r2t.shape[0],), dtype=torch.int32, device=b.device)
        if b.packed_ntiles is None:
            # Indexed by the stable request id, which is a SLOT id, so it needs `slots` columns
            # (36 x 493k x 4 B = 71 MB, about 1% of the pool). Needed by the fused decode kernel in
            # every mode, not just for CUDA graphs.
            b.packed_ntiles = torch.zeros((b.L, b.slots), dtype=torch.int32, device=b.device)
        if b.base_tab is None:
            b.base_tab = torch.zeros((r2t.shape[0],), dtype=torch.int64, device=b.device)
            b.stg_tab = torch.zeros((r2t.shape[0],), dtype=torch.int64, device=b.device)
            b.key_tab = torch.zeros((r2t.shape[0],), dtype=torch.int32, device=b.device)

    def ridx_of(self, key: int) -> int:
        """A compact index for this request, stable for its lifetime.

        Allocation is a real free list. Wrapping `_ridx_next % REQ_SLOTS` (the previous version)
        can hand an index to a new request while a long-lived request still owns it, which makes two
        live requests share a `pos_slot` row -- silent cross-talk, not a crash.
        """
        r = self._ridx.get(key)
        if r is None:
            if self._ridx_free:
                r = self._ridx_free.pop()
            else:
                r = self._ridx_next % REQ_SLOTS
                self._ridx_next = (self._ridx_next + 1) % REQ_SLOTS
            self._ridx[key] = r
        return r

    def _publish_tables(self, rows, keys) -> None:
        """Copy the host-side per-request routing into device tables.

        Called only from the prefill path (never captured), so a host round trip is fine here.
        Decode then reads `stg_tab`/`base_tab` by `req_pool_indices` on the device.
        """
        b = self.buf
        dev = b.device
        if b.stg_tab is None:
            n_rows = 4096
            b.stg_tab = torch.full((n_rows,), -1, dtype=torch.int64, device=dev)
            b.base_tab = torch.full((n_rows,), -1, dtype=torch.int64, device=dev)
            b.packed_ntiles = torch.zeros((b.L, b.slots), dtype=torch.int32, device=dev)
        for r, kk in zip(rows, keys):
            b.stg_tab[int(r)] = int(self._stg[kk])
            # `base_tab` used to carry the request's tile base. With on-demand blocks there is no
            # single base; every consumer derives the tile id from `slot_row` or `tile_of`.
            b.base_tab[int(r)] = -1
            b.key_tab[int(r)] = int(kk)
            b.ridx_tab[int(r)] = int(self.ridx_of(kk))

    # -- decode write, capture-safe -----------------------------------------
    def _step_decode_g(self, li: int, fb, k, v, loc) -> None:
        """One layer of a decode step with no device->host reads at all.

        The tile that closes is only *marked* here; `flush_deferred` does the quantisation after
        the forward returns, i.e. outside the captured region. Deferring is safe because a token
        written in this step is not read by any attention until the next step.
        """
        b = self.buf
        # Capture runs before any prefill publishes anything, so create every table here.
        self._ensure_pos_slot(fb)
        r2t = fb.req_to_token_pool.req_to_token
        if b.pos_slot is None:
            # Indexed by a COMPACT per-request index, not by the req_to_token row. A request that
            # is prefilled in two chunks lives on two rows, and the second row's req_to_token only
            # holds that chunk's positions -- so a row-keyed table (or req_to_token itself) has
            # holes for everything written earlier. That is exactly the 2048-token cliff: correct
            # at n <= 2048, wrong above it.
            # Sized by concurrent requests, not by the request-pool row count: the table only has
            # to hold the requests that are live at once, and 4096 rows cost 268 MB of the KV
            # budget (that showed up as capacity 499,718 -> 486,613 tokens).
            b.pos_slot = torch.full((REQ_SLOTS + 1, r2t.shape[1]), -1,
                                    dtype=torch.int32, device=b.device)
            # CUDA-graph padding support. `replay_prepare` writes `pad_sentinel` into the padded
            # rows' `out_cache_loc`, and padded tokens are REDIRECTED to these dustbin rows rather
            # than filtered out -- filtering needs `nonzero`, whose data-dependent shape capture
            # rejects. The extra pos_slot row and the last staging row are never read by anyone.
            b.pad_sentinel = 0    # slot 0 is the spare one; `size` is a real slot (see above)
            b.dustbin_pos_row = int(REQ_SLOTS)
            b.dustbin_stg_row = int(b.max_staging * b.T)   # the extra row, outside every slot
        if b.ridx_tab is None:
            b.ridx_tab = torch.zeros((r2t.shape[0],), dtype=torch.int32, device=b.device)
        if b.stg_tab is None:
            # Capture runs a synthetic batch before any prefill has published the routing, so the
            # tables must exist (and be in range) from the first captured call. Their values during
            # capture are irrelevant; only the absence of host reads matters.
            b.stg_tab = torch.zeros((r2t.shape[0],), dtype=torch.int64, device=b.device)
            b.base_tab = torch.zeros((r2t.shape[0],), dtype=torch.int64, device=b.device)
        rows = fb.req_pool_indices.to(torch.long)
        pre = fb.seq_lens.to(torch.long) - 1
        T = b.T
        fill = pre % T
        # CUDA-graph batch padding. When the true batch size is not an exactly captured one,
        # `cuda_graph_runner.replay_prepare` pads up to the next captured size, fills the padded
        # rows' `seq_lens` with the backend's fill value and zeroes their `out_cache_loc`; the
        # padded rows' `req_pool_indices` are left STALE. Everything below indexes per-row tables,
        # so a padded row would steer a write into whatever live request happens to own that stale
        # row: its landing-window rows get dummy KV, its position table gets a bogus entry, and
        # `closed_mask` flags it for a flush. Measured: padding on 40.5% vs off 71.0% on MATH500
        # n=200, with no error anywhere.
        #
        # The marker is a dedicated sentinel that `replay_prepare` writes into the padded
        # `out_cache_loc` entries (see install()): a spare KV slot no request can ever own. Padded
        # tokens are then REDIRECTED to dustbin rows instead of being filtered out, because
        # filtering would need `nonzero` -- a data-dependent shape, which capture rejects.
        sent = getattr(b, "pad_sentinel", None)
        if sent is not None:
            valid = loc != sent
        else:
            valid = torch.ones_like(loc, dtype=torch.bool)
        stg = b.stg_tab.index_select(0, rows)
        if (not torch.cuda.is_current_stream_capturing()) and bool(((stg < 0) & valid).any()):
            raise RuntimeError("[TILE2] decode reached a row with no staging slot; the routing "
                               "tables are only filled by the prefill path")
        src = torch.where(valid, stg * T + fill, torch.full_like(stg * T + fill,
                                                                b.dustbin_stg_row))
        # k/v arrive ALREADY rotated: the wrapper rotates them when rot_domain is on, and the
        # host write path is told `rotate=False` for exactly that reason. Rotating again here was
        # a double rotation and made the whole decode store wrong (0/8 with CUDA graphs).
        kw = k.to(torch.bfloat16)
        vw = v.to(torch.bfloat16)
        b.staging_k[li].index_copy_(0, src, kw)
        b.staging_v[li].index_copy_(0, src, vw)
        if li == self.start_layer:
            if b.pos_slot is None:
                raise RuntimeError("[TILE2] position table missing in graph-safe mode")
            # ridx (per-request compact index), NOT the row: the two diverge after a
            # chunked-prefill row change, and using the row silently loses the whole history.
            _ri = b.ridx_tab.index_select(0, rows)
            if sent is not None:
                _ri = torch.where(valid, _ri, torch.full_like(_ri, b.dustbin_pos_row))
            b.pos_slot.index_put_((_ri, pre), loc.to(torch.int32))
            # `slot_row` is layer-shared and records where each slot currently lives. The host
            # write path refreshes it, but this graph-safe writer used to update only the ring and
            # `pos_slot`. A slot recycled from an earlier tile therefore kept that tile's POSITIVE
            # (packed) tag, and with USE_SROW the kernel classified the new, still-staging tile as
            # packed and loaded the previous tile's parameters. Codex audit; only reachable with a
            # radix cache, where slot reuse across requests is the norm.
            b.slot_row.index_copy_(0, loc.to(torch.long),
                                   (-(2 + stg * T + fill)).to(torch.int32))
        # mark rows whose tile just closed, for the post-forward flush
        if b.closed is None or b.closed.shape[1] < rows.numel():
            b.closed = torch.full((b.L, max(4096, int(rows.numel()))), -1,
                                  dtype=torch.int64, device=b.device)
        # Fixed-shape only: `torch.nonzero` is rejected during capture because its output size
        # depends on the data. Write the whole batch plus a per-row flag and let the deferred
        # flush (which runs on the host, outside the graph) do the filtering.
        bs = int(rows.numel())
        c = b.closed[li]
        c.fill_(-1)
        c[:bs].copy_(rows)
        if b.closed_mask is None or b.closed_mask.shape[1] < bs:
            b.closed_mask = torch.zeros((b.L, max(4096, bs)), dtype=torch.bool, device=b.device)
        m = b.closed_mask[li]
        m.fill_(False)
        # A padded row must never be reported as a closed tile: `flush_deferred` would quantise a
        # tile for whatever request owns that stale row.
        m[:bs].copy_((fill == (T - 1)) & valid)

    def flush_deferred(self, fb) -> None:
        """Quantise every tile that a decode step closed. Runs OUTSIDE the captured graph."""
        b = self.buf
        # Ownership masks are per step: `slot_row` flips to "packed" as layer 0 writes it.
        b.__dict__.pop("_own_cache", None)
        if b.closed is None:
            return
        seq = [int(x) for x in fb.seq_lens.tolist()]
        rows_now = [int(x) for x in fb.req_pool_indices.tolist()]
        pre_of = {r: seq[i] - 1 for i, r in enumerate(rows_now)}
        # Refresh liveness here, because the decode write path runs inside the graph and cannot
        # touch host state. Without this a request that is happily decoding looks "unseen" to the
        # reclaimer after a few dozen steps and its staging slot is recycled underneath it.
        self.steps += 1
        # `_active` must be the set of keys the WRITE path is actually touching, not what some other
        # hook recorded. Measured failure: `n_active=1` while all 8 slots had age 0, so the reclaimer
        # believed 7 slots were free while 8 requests were mid-decode. Rows are the source of truth
        # here because flush_deferred sees exactly the batch's req_pool_indices.
        try:
            _act = set()
            for _r in rows_now:
                _rid = self._row_rid.get(int(_r))
                if _rid is None:
                    continue
                _k = self._rid_by_req.get(_rid)
                if _k is not None:
                    _act.add(_k)
            if _act:
                self._active = _act
        except Exception:
            pass
        # Retired requests are reclaimed here: this is the worker thread, after the previous
        # batch's kernels were enqueued, so no in-flight read can still reference these buffers.
        self.drain_retires()
        # The graph-safe decode path cannot maintain the row -> stable-request-id map (it may not
        # read device memory), so do it here, outside the captured region. Without this the tile
        # count is never published and the fused kernel treats the whole history as unflushed.
        if b.key_tab is not None:
            for r in rows_now:
                self._row_key[r] = int(b.key_tab[r])
        # Liveness must live in the SAME namespace the allocator uses (the stable request key).
        # The old fallback `_row_key.get(r, r)` wrote a ROW number into the key namespace whenever
        # a row had no mapping, and rows/keys overlap numerically (rows 0..~4096, keys are slot
        # ids from 0 up). Those phantom entries never match a real key, so live requests looked
        # stale and their landing-window slots -- which hold the NEWEST tokens -- were handed to
        # the next request. A row with no published key is not a request we can name, so skip it.
        self._active = set()
        for r in rows_now:
            kk = self._row_key.get(r)
            if kk is None:
                continue
            self._active.add(kk)
            self._last_seen[kk] = self.steps
            if kk in self._lru:
                self._lru.remove(kk)
            self._lru.append(kk)
        for li in range(b.L):
            col = b.closed[li]
            msk = b.closed_mask[li] if b.closed_mask is not None else None
            sel = col >= 0
            if msk is not None:
                sel = sel & msk
            live = col[sel]
            if live.numel() == 0:
                continue
            for r in live.tolist():
                pre = pre_of.get(int(r), -1)
                if pre < 0:
                    continue
                T = b.T
                tail = pre - T + 1
                if tail < 0:
                    continue
                _ri = int(b.ridx_tab[int(r)])
                slots = b.pos_slot[_ri, tail:pre + 1].to(torch.long)
                if int(slots.min()) < 0:
                    continue
                _key0 = int(self._row_key.get(int(r), -1))
                if _key0 < 0:
                    continue
                stg = int(b.stg_tab[r])
                _key = int(self._row_key.get(int(r), -1))
                # See the write path: ownership only means something when prefixes are shared.
                _own = (own_rows_mask(b, _key, pre // T, slots)
                        if getattr(b, "radix_on", True) else None)
                if _own is not None and not bool(_own.any()):
                    continue
                queue_flush(b, li, stg, T, tile_for_slots(b, self.tile_of, _key0, pre // T, slots), slots, True,
                            count_tokens=(li == b.start_layer), req_row=int(r),
                            key=_key, own_mask=_own)
            drain_flushes(b, li)
            col.fill_(-1)
            if b.closed_mask is not None:
                b.closed_mask[li].fill_(False)

    # -- one layer of one step ----------------------------------------------
    def step(self, li: int, fb, k, v, loc) -> None:
        if _SKIPWRITE:
            return
        if self.buf.graphsafe and not fb.forward_mode.is_extend():
            return self._step_decode_g(li, fb, k, v, loc)
        # The batch metadata is identical for all 36 layers of a step, and reading it costs a
        # device->host sync (`tolist()`). Doing it per layer cost 36 syncs per step and pinned
        # decode at 8 tokens/s; layer `start_layer` is the first call of a step, so it refreshes
        # the cache and the rest reuse it.
        if li == self.start_layer or self._ctx is None:
            # Gather the whole batch in ONE vectorised read. The old form did one scalar device
            # read per request (`int(r2t[r, 0])`), i.e. bs separate device->host syncs every step,
            # and each of those drains the whole pipeline -- that, not the dequantisation, was what
            # made packed_tile2 3.6x slower than bf16 with the read path switched off.
            rows_t = fb.req_pool_indices.to(torch.long)
            r2t = fb.req_to_token_pool.req_to_token
            rows = rows_t.tolist()
            # NOT `req_to_token[row, 0]` any more: with a shared prefix that slot is common to
            # every request that reuses the prefix. See the `_row_rid` note above.
            _slot0 = r2t.index_select(0, rows_t)[:, 0].tolist()
            keys = [self.key_of_row(r, fallback=int(s0)) for r, s0 in zip(rows, _slot0)]
            for r_, k_ in zip(rows, keys):
                self._row_key[r_] = k_
            self._ensure_pos_slot(fb)
            # Inherited prefix: when the radix cache hands a request slots it never wrote, this
            # request's write path never sees those positions, so its per-request position table
            # would stay -1 there and the fused kernel would MASK THE WHOLE PREFIX OUT of
            # attention (it reads slots from `pos_slot`, not from req_to_token, because a chunked
            # prefill changes req_to_token rows). Copy the inherited range in once, from the row
            # that still holds it, on the request's first step.
            # `extend_prefix_lens_cpu` is Optional and is NOT populated in every code path; when it
            # is None this entire block used to be skipped, so an inherited prefix never reached
            # the position table and the fused kernel masked it out of attention (the warm request
            # then attends only to its own tail -- coherent-looking garbage). Fall back to the
            # device tensor and pay one sync here: this is the extend path, never captured.
            _pl = None if os.environ.get("SGLANG_TILE2_PFXCOPY") == "0" else getattr(fb, "extend_prefix_lens_cpu", None)
            if _pl is None:
                _dev = getattr(fb, "extend_prefix_lens", None)
                if _dev is not None:
                    _pl = [int(x) for x in _dev.tolist()]
            if _pl is not None:
                _pre_lens = _pl
                for _i, (r_, k_) in enumerate(zip(rows, keys)):
                    _np = int(_pre_lens[_i]) if _i < len(_pre_lens) else 0
                    if _np <= 0:
                        continue
                    _ri = self.ridx_of(int(k_))
                    _src = r2t[int(r_), :_np].to(torch.int32)
                    if int(_src.numel()) < _np:
                        continue
                    buf_pos = self.buf.pos_slot
                    buf_pos[_ri, :_np] = _src
                    if os.environ.get("SGLANG_TILE2_PFXDBG") == "1":
                        _pk = (self.buf.slot_row.index_select(0, _src.to(torch.long)) >= 0)
                        print(f"[T2PFX] ridx={_ri} prefix={_np} packed_frac=%.3f slot0=%d"
                              % (float(_pk.float().mean()), int(_src[0])), flush=True)
                    if os.environ.get("SGLANG_TILE2_PFXDBG") == "1":
                        print(f"[T2PFX] key={int(k_)} ridx={_ri} prefix={_np} "
                              f"copied_slots0={int(_src[0])} copied_slots_last={int(_src[-1])}",
                              flush=True)
            pre, nw = _write_ranges(fb, len(rows))
            # Publish the batch BEFORE allocating slots. `staging_of` refuses to evict a request
            # that is in the current batch, and the check used to read the PREVIOUS step's batch:
            # while building this list, a slot handed out for one request could be recycled to
            # serve the next request of the same step.
            self._active = set(keys)
            _stg = [self.staging_of(k) for k in keys]
            # One host->device copy per step instead of two per layer (72 a step).
            _src = (torch.tensor(_stg, dtype=torch.long, device=self.buf.device) * self.buf.T
                    + (torch.tensor(pre, dtype=torch.long, device=self.buf.device)
                       % self.buf.T))
            self._src_dev = _src
            self._ctx = (rows, keys, pre, nw, _stg)
            # Publish in every mode: the fused decode kernel reads these tables, and they must
            # exist whether or not CUDA graphs are on.
            self._publish_tables(rows, keys)
            if os.environ.get("SGLANG_TILE2_TRACE_ROWS") == "1" and self.steps < 60:
                print(f"[T2R] step={self.steps} rows={rows} keys={keys} stg={self._ctx[4]} "
                      f"pre={pre} nw={nw}", flush=True)
        if _TRACE and li != self.start_layer:
            # Is the cached batch metadata really the same for every layer of a step?
            live_rows = [int(x) for x in fb.req_pool_indices.tolist()]
            if live_rows != self._ctx[0]:
                print(f"[T2T-STALE] li={li} cached rows={self._ctx[0]} live rows={live_rows}",
                      flush=True)
        rows, keys, pre, nw, stg = self._ctx
        if _AUDIT:
            if len(set(rows)) != len(rows):
                print(f"[TILE2-ARGH] duplicate req rows in one batch: {rows}", flush=True)
            if len(set(stg)) != len(stg):
                print(f"[TILE2-ARGH] duplicate staging slots in one batch: {stg}", flush=True)
            self.buf._cur_row = tuple(rows)
            # The write must continue exactly where this row stopped last time. Any violation
            # means the position map is being fed from the wrong batch metadata, which silently
            # writes one request's tokens over another's positions.
            for i, r in enumerate(keys):
                prev = self._len.get(r, 0)
                # pre == 0 is a legitimate restart (a new request on a recycled row, which is
                # exactly what warmup does). Anything else must continue the previous write.
                if pre[i] != 0 and pre[i] != prev:
                    self._nbad = getattr(self, "_nbad", 0) + 1
                    if self._nbad <= 8:
                        print(f"[TILE2-ARGH] row {r}: write starts at {pre[i]} but last write "
                              f"ended at {prev} (rows={rows} pre={pre} nw={nw})", flush=True)
                if nw[i] > 0 and pre[i] + nw[i] != int(fb.seq_lens[i]):
                    print(f"[TILE2-ARGH] row {r}: pre+ nw = {pre[i] + nw[i]} but seq_len = "
                          f"{int(fb.seq_lens[i])}", flush=True)
            for r in keys:
                if r in self._released:
                    print(f"[TILE2-ARGH] row {r} reappears at step {self.steps} after being "
                          f"released at step {self._released[r]} (premature free)", flush=True)
                    del self._released[r]
        _tp0 = time.perf_counter() if _PROF is not None else 0.0
        write_and_flush(self.buf, li, stg, pre, nw, rows, keys, loc, k, v,
                        fb.req_to_token_pool.req_to_token, self.tile_of,
                        # Layer-independent and idempotent, so write it once per step. Doing it from
                        # all 36 layers cost 36 scatter ops a step; it was switched on while hunting
                        # the allocator bug and never switched back.
                        update_map=(li == self.start_layer),
                        rotate=not getattr(self, "rotated", False),
                        pending=self._pending,
                        # The device position table replaces the Python dict, whose update needs
                        # `loc[...].tolist()`, i.e. a device->host sync once per layer.
                        track_positions=False, ridx_of=self.ridx_of,
                        src_dev=self._src_dev)
        if _PROF is not None:
            _prof_add("write", time.perf_counter() - _tp0)
            if li == self.start_layer:
                _now = time.perf_counter()
                _prev = _PROF.get("t0")
                _PROF["t0"] = _now
                if _prev is not None:
                    _PROF["step"] = _PROF.get("step", 0.0) + (_now - _prev)
        if li == self.start_layer:
            # Invariant: a step is exactly `L` layer writes, the map is refreshed exactly once,
            # and it is refreshed by the FIRST layer of the step. If any of this is false the
            # per-slot position map can be updated with stale metadata, which is silent.
            if _AUDIT:
                if getattr(self, "_open_step", False):
                    print(f"[TILE2-ARGH] step {self.steps} saw {self._calls} layer writes, "
                          f"expected {self.buf.L}", flush=True)
                self._open_step = True
                self._calls = 0
            for r, p, n in zip(keys, pre, nw):
                self._len[r] = int(p) + int(n)
            self.steps += 1
        if _AUDIT or _CHECK:
            self._calls = getattr(self, "_calls", 0) + 1
            if self._calls == self.buf.L:
                self._open_step = False

        if os.environ.get("SGLANG_TILE2_COV") == "1" and self.steps % 200 == 0:
            # c is the number the whole project is about, so it gets its own switch: it must be
            # printable without turning on any of the audit machinery.
            c = self.coverage()
            print(f"[TILE2-COV] step={self.steps} reqs={c['reqs']} packed={c['packed']} "
                  f"staged={c['staged']} c={c['c']:.4f}", flush=True)
        if _AUDIT or _CHECK:
            if (_CHECK and not getattr(self, "_checked", False)
                    and getattr(self.buf, "n_packed_tokens", 0)):
                self._checked = True
                b = self.buf
                idx = loc.to(torch.long)
                ok = torch.empty((idx.numel(), b.H, b.D), dtype=torch.bfloat16, device=b.device)
                ov = torch.empty_like(ok)
                gather_dequant(b, li, idx, ok, ov)
                # k/v arrive already rotated (see install), and gather stays in that domain,
                # so the comparison is direct -- no inverse transform on either side.
                ref, refv = k.float(), v.float()
                rel = ((ok.float() - ref).pow(2).mean().sqrt()
                       / ref.pow(2).mean().sqrt()).item()
                relv = ((ov.float() - refv).pow(2).mean().sqrt()
                        / refv.pow(2).mean().sqrt()).item()
                print(f"[TILE2-CHECK] rows={idx.numel()} "
                      f"packed_tokens={getattr(self.buf, 'n_packed_tokens', 0)} "
                      f"K relRMS={rel:.4f} V relRMS={relv:.4f}", flush=True)

    # -- coverage, the thing that has to be reported alongside any bit figure --
    def coverage(self) -> dict:
        """c = packed / (packed + open), computed from live requests, not from a formula."""
        staged = sum(int(L) % self.buf.T for L in self._len.values())
        packed = int(getattr(self.buf, "n_packed_tokens", 0))
        live = packed + staged
        return dict(packed=packed, staged=staged, live=live,
                    c=(packed / live) if live else 0.0, reqs=len(self._len))


# ---------------------------------------------------------------- engine wiring
_STATE: dict = {}
_INSTALLED = False


def state() -> dict:
    return _STATE


def install(pool) -> None:
    """Wire the store into the sglang backend. Called once, from the KV pool constructor.

    Three hooks, each for a reason that cannot be expressed as a small local edit:

      forward_decode / forward_extend : the query has to be rotated AFTER RoPE and after QK-norm
          (the backend receives it that way) and the attention output has to be un-rotated before
          it reaches o_proj. Both are one line, but they have to bracket the call, so they are
          wrapped rather than sprinkled through branches. The wrapper also stashes the current
          ForwardBatch, because `set_kv_buffer` does not receive one and the write path needs the
          position map.

      ReqToTokenPool.free : staging slots and tile ids are owned per request-pool row. A row is
          recycled, so without a release hook the next request inherits a half-filled tile.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

    # `ScheduleBatch` is the only place where `Request.rid` sits next to `req_pool_indices`, and it
    # runs on the host before every forward (never captured), so it is where the row -> request
    # binding is stamped. Without it the store has no identity that is both unique and stable.
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    _orig_gmwb = getattr(ScheduleBatch, "get_model_worker_batch", None)
    if _orig_gmwb is not None and not getattr(_orig_gmwb, "_tile2_bound", False):
        def _gmwb(self, *a, **kw):
            drv = pool.tile2_drv
            idx = self.req_pool_indices
            _dbg = os.environ.get("SGLANG_TILE2_KEYDBG") == "1"
            if _dbg:
                print("[T2HOOK-CALL] reqs=%d" % len(self.reqs), flush=True)
            for i, req in enumerate(self.reqs):
                _row = int(idx[i])
                _new = drv.note_request(_row, req.rid)
                if _dbg:
                    print(f"[T2KEY] row={_row} rid={req.rid} key={drv._row_rid[_row]} "
                          f"first_seen={_new}", flush=True)
            return _orig_gmwb(self, *a, **kw)
        _gmwb._tile2_bound = True
        ScheduleBatch.get_model_worker_batch = _gmwb
        if os.environ.get("SGLANG_TILE2_KEYDBG") == "1":
            print("[T2HOOK] ScheduleBatch.get_model_worker_batch patched", flush=True)

    def bracket(fn):
        def inner(self, q, k, v, layer, forward_batch, save_kv_cache=True):
            _STATE["fb"] = forward_batch
            _is_ext = not forward_batch.forward_mode.is_decode()
            try:
                hq, hd = layer.tp_q_head_num, layer.head_dim
                rot = getattr(pool, "tile2_buf").rot_domain
                # q only: a single matmul. k/v keep the FWHT because their rounded bits are what
                # the store quantises, and the FWHT is the variant that is stable across chunking.
                _rf = getattr(pool.tile2_buf, "rot_fused", False) and not _is_ext
                if rot and _PROF is not None:
                    _tr0 = time.perf_counter()
                qr = (rotate_hadamard_mm(q.view(-1, hq, hd)).to(q.dtype).view(q.shape)
                      if (rot and not _rf) else q)
                if rot and _PROF is not None:
                    _prof_add("rot", time.perf_counter() - _tr0)
                # k/v are [N, kv_heads, head_dim]; rotate along the last axis, no reshape needed.
                # They are rotated HERE and not in the store, because the ragged prefill attends
                # on these tensors directly (use_ragged is True for non-multimodal models), so
                # rotating only the query would put q and k in different spaces.
                if rot and k is not None and _PROF is not None:
                    _tk0 = time.perf_counter()
                # k/v also use the single-matmul rotation. This was kept on the FWHT for
                # bit-identical codes across differently chunked writes, but that property is
                # stricter than the engine needs: every position is rotated exactly once, at the
                # moment it is written, so a per-call rounding difference cannot make two writes
                # of the SAME position disagree. The FWHT cost 7 fp32 passes x 2 tensors x 36
                # layers ~= 1500 tiny kernels a step and measured 51.7 ms/step, i.e. half the step.
                _kvf = os.environ.get("SGLANG_TILE2_KV_FWHT") == "1"
                kr = ((rotate_hadamard(k) if _kvf else rotate_hadamard_mm(k)).to(k.dtype)
                      if (rot and k is not None) else k)
                vr = ((rotate_hadamard(v) if _kvf else rotate_hadamard_mm(v)).to(v.dtype)
                      if (rot and v is not None) else v)
                if rot and k is not None and _PROF is not None:
                    _prof_add("kvrot", time.perf_counter() - _tk0)
                if (pool.tile2_buf.rot_domain and kr is None
                        and save_kv_cache):
                    pass
                if kr is not None and save_kv_cache:
                    loc_ = forward_batch.out_cache_loc
                    rotated_in = pool.tile2_buf.rot_domain
                    _STATE["last_w"] = (loc_, kr if rotated_in
                                        else rotate_hadamard(kr).to(kr.dtype),
                                        vr if rotated_in
                                        else rotate_hadamard(vr).to(vr.dtype))
                o = fn(self, qr, kr, vr, layer, forward_batch, save_kv_cache)
                # `o` comes back already flattened to [N, hq*head_dim]. Un-rotating that without
                # reshaping runs a 4096-point transform over the concatenated heads, which is
                # silently wrong rather than an error.
                if not rot or _rf:
                    # the fused kernel already produced the output in the model's basis
                    return o
                # The output un-rotation has the same status as the query rotation: it never feeds
                # the cache, so one matmul is fine and seven fp32 passes are not affordable here.
                if _PROF is not None:
                    _tu0 = time.perf_counter()
                _res = rotate_hadamard_mm(o.view(-1, hq, hd)).to(o.dtype).view(o.shape)
                if _PROF is not None:
                    _prof_add("rot", time.perf_counter() - _tu0)
                return _res
            finally:
                _STATE["fb"] = None
        return inner

    FlashInferAttnBackend.forward_decode = bracket(FlashInferAttnBackend.forward_decode)
    FlashInferAttnBackend.forward_extend = bracket(FlashInferAttnBackend.forward_extend)

    drv = pool.tile2_drv
    # The store never rotates on the write path in either mode: it stages exactly what the model
    # produced. In rotated-domain mode the wrapper already rotated q/k/v, so nothing rotates
    # again; in original-domain mode the rotation happens inside flush_tile, right before the
    # quantiser needs it, and the read path un-rotates only the packed rows.
    # The store never rotates on the way in, in either mode: the wrapper rotates q/k/v in
    # rotated-domain mode, and in the default (original-domain) mode flush_tile rotates in fp32 so
    # the quantiser sees the same thing the in-place QDQ arm sees.
    drv.rotated = True
    #: rotate q / un-rotate the output INSIDE the fused decode kernel instead of with two cuBLAS
    #: calls per layer (72 a step, measured 8.7 ms/step).
    pool.tile2_buf.rot_fused = os.environ.get("SGLANG_TILE2_ROT_FUSED") == "1"
    try:
        pool.tile2_buf.req_to_token = pool.req_to_token_pool.req_to_token
    except Exception:
        pass
    # Deferred flush. The decode write path only marks the tiles it closed; the quantisation runs
    # here, after the model forward (and therefore outside the captured graph, where a device->host
    # read is legal). A token written in a step is not read by attention until the next step, so
    # deferring by one forward is not observable.
    if getattr(pool.tile2_buf, "graphsafe", False):
        from sglang.srt.model_executor.model_runner import ModelRunner as _MR
        _orig_mr_forward = _MR.forward

        def _mr_forward(self, *args, **kwargs):
            out = _orig_mr_forward(self, *args, **kwargs)
            try:
                # `forward_batch` is an argument in this sglang version, not an attribute.
                fb = args[0] if args else kwargs.get("forward_batch")
                if fb is not None and fb.forward_mode.is_decode():
                    drv.flush_deferred(fb)
            except Exception as exc:  # never break the forward pass over a bookkeeping issue
                print(f"[TILE2] deferred flush failed: {exc}", flush=True)
            return out

        _MR.forward = _mr_forward

    # Torch-profiler hook. Every previous attempt to attribute the decode step by bisection gave a
    # different answer, because switching a path off also changes what the other path does; this
    # asks the profiler instead.
    if os.environ.get("SGLANG_TILE2_TORCHPROF") == "1":
        from sglang.srt.model_executor.model_runner import ModelRunner as _MR2
        _orig_fwd2 = _MR2.forward
        _st2 = {"n": 0, "prof": None}

        def _prof_forward(self, *args, **kwargs):
            import torch as _t
            fb = args[0] if args else kwargs.get("forward_batch")
            dec = fb is not None and fb.forward_mode.is_decode()
            st = _st2
            if dec and st["prof"] != "done":
                st["n"] += 1
            if dec and st["n"] >= 5 and st["prof"] is None:
                st["prof"] = _t.profiler.profile(
                    activities=[_t.profiler.ProfilerActivity.CUDA],
                    record_shapes=False, with_stack=False)
                st["prof"].__enter__()
            _out = _orig_fwd2(self, *args, **kwargs)
            if st["prof"] is not None and st["n"] >= 25:
                st["prof"].__exit__(None, None, None)
                try:
                    ka = st["prof"].key_averages()
                    rows = sorted(ka, key=lambda e: getattr(e, "self_device_time_total", 0.0),
                                  reverse=True)[:18]
                    tot = sum(getattr(e, "self_device_time_total", 0.0) for e in ka)
                    print(f"[T2PROF] 20 decode steps, total device {tot/1e3:.1f} ms, "
                          f"per step {tot/20/1e3:.2f} ms", flush=True)
                    for e in rows:
                        us = getattr(e, "self_device_time_total", 0.0)
                        print(f"[T2PROF] {us/20/1e3:9.3f} ms/step  {e.key[:84]}", flush=True)
                except Exception as exc:
                    print(f"[T2PROF] dump failed: {exc}", flush=True)
                st["prof"] = "done"
            return _out

        _MR2.forward = _prof_forward

    # No request-lifecycle hooks at all. Two attempts were made and both were wrong for the same
    # reason: with chunked prefill a request is extended in several calls, each with a DIFFERENT
    # req_to_token row, so alloc/free per row does not correspond to per-request start/finish.
    #   * releasing on alloc freed the tiles of requests that were still live;
    #   * releasing on free ran between two chunks of the same request and threw away the partial
    #     tile plus the slots accumulated for it -- exactly 36 rows of the straddling tile went
    #     missing, which is what `have=92 want=128` was telling us.
    # Staging lifetime is therefore decided by activity alone (see staging_of): a row that has not
    # appeared in a batch for a while is gone, whichever hook sglang did or did not call.
    # ---- lifecycle hook: request END -> return the slot immediately -------------------------
    # Installed on the INSTANCE the scheduler actually picks (RadixCache / ChunkCache / SWA / the
    # C++ cache all implement cache_finished_req separately, and BasePrefixCache's version is
    # abstract, so patching a class would silently miss cases). NOT on ReqToTokenPool.free: chunked
    # prefill (scheduler.py:get_next_batch_to_run) and retract_decode also free rows for requests
    # that are still alive, so a row free is not a death certificate.
    try:
        from sglang.srt.managers.scheduler import Scheduler as _Sched
        _orig_init_cache = _Sched.init_memory_pool_and_cache

        def _init_memory_pool_and_cache(self, *a, **kw):
            out = _orig_init_cache(self, *a, **kw)
            try:
                cache = self.tree_cache
                if not getattr(cache, "_tile2_hooked", False):
                    _orig_fin = cache.cache_finished_req

                    def cache_finished_req(req, *aa, **kk):
                        rid = getattr(req, "rid", None)
                        res = _orig_fin(req, *aa, **kk)      # only notify AFTER the cleanup ran
                        if rid is not None:
                            try:
                                pool.tile2_drv.retire_request(rid)
                            except Exception as e:           # never break teardown
                                print(f"[TILE2] retire notify failed for {rid}: {e}", flush=True)
                        return res

                    cache.cache_finished_req = cache_finished_req
                    cache._tile2_hooked = True
                    print("[T2HOOK] cache_finished_req patched on the live cache instance",
                          flush=True)
            except Exception as e:
                print(f"[T2HOOK] FAILED to patch cache_finished_req: {e}", flush=True)
            return out

        _Sched.init_memory_pool_and_cache = _init_memory_pool_and_cache
        print("[T2HOOK] Scheduler.init_memory_pool_and_cache patched (lifecycle hook)", flush=True)
    except Exception as e:
        print(f"[T2HOOK] lifecycle hook NOT installed: {e}", flush=True)

    _INSTALLED = True
    print("[TILE2] installed: q rotation, output un-rotation, write divert, request release",
          flush=True)
    # The run configuration is otherwise invisible in the log, and "which flags produced this
    # number" was unknowable often enough to matter. Print it once, from the one place that is
    # definitely reached by every server.
    _flags = " ".join("%s=%s" % (k[len("SGLANG_TILE2_"):], os.environ[k])
                      for k in sorted(os.environ) if k.startswith("SGLANG_TILE2_"))
    # Export SGLANG_TILE2_CONCURRENCY at the source. The slot count used to be 2N+2 because
    # nothing returned a slot when a request ended: the only recycler was "this slot has not been
    # touched for 32 steps", which recycled LIVE requests (n=200 accuracy 0.710 -> 0.555) and still
    # ran out under short requests (MMLU needed 48 slots for 8 concurrent requests). With the
    # lifecycle hook the rule is simply: one slot per resident request.
    print("[TILE2] concurrency=%s (staging slots=%s; with SGLANG_TILE2_STAGING unset this equals "
          "the concurrency, because slots are now returned when a request finishes)"
          % (os.environ.get("SGLANG_TILE2_CONCURRENCY", "?"), pool.tile2_buf.max_staging), flush=True)
    _buf = getattr(pool, "tile2_buf", None)
    if _buf is not None:
        print("[TILE2] tile block = %d tiles per request (pool has %d tiles -> %d concurrent)"
              % (_buf.tiles_per_req, _buf.n_tiles, _buf.n_tiles // max(1, _buf.tiles_per_req)),
              flush=True)
    print("[TILE2] flags: %s" % (_flags or "(none)"), flush=True)


# ---------------------------------------------------------------- Triton read path
try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:                                   # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:
    @triton.jit
    def _tile2_gather_kernel(
        KC, KS, KL, KA, KB,
        VC, VS, VL, VA, VB,
        SROW, IND, OUTK, OUTV, STGK, STGV,
        N, H: tl.constexpr, D: tl.constexpr, T: tl.constexpr,
    ):
        """One program per (gathered row, head): decode a 2-bit row into the bf16 workspace.

        The torch version of this is O(D) elementwise fp32 over every live token on every layer
        of every step, which measured out at 3.8 tokens/s. Same arithmetic, one kernel.

        Both sources are handled here rather than pre-split: index 0 of `SROW` says whether the
        slot is packed (>=0) or still in the staging window (<0). Doing the split on the host
        would make the launch shape data-dependent, which is exactly what CUDA graphs forbid.
        """
        i = tl.program_id(0)
        h = tl.program_id(1)
        if i < N:
            slot = tl.load(IND + i)
            row = tl.load(SROW + slot)
            offs = tl.arange(0, D)
            outk = OUTK + (i * H + h) * D + offs
            outv = OUTV + (i * H + h) * D + offs
            if row >= 0:
                tile = row // T
                off = row % T
                sh = (offs % 4) * 2
                cb = tl.load(KC + (slot * H + h) * (D // 4) + offs // 4).to(tl.float32)
                code = tl.floor(cb / tl.exp2(sh.to(tl.float32))) % 4.0
                base = (tile * H + h) * D + offs
                kv = (code * tl.load(KS + base).to(tl.float32)
                      + tl.load(KL + base).to(tl.float32))
                kv = kv * tl.load(KA + base).to(tl.float32)
                kv = kv * tl.load(KB + (tile * T + off) * H + h).to(tl.float32)
                tl.store(outk, kv)

                vb_ = tl.load(VC + (slot * H + h) * (D // 4) + offs // 4).to(tl.float32)
                codev = tl.floor(vb_ / tl.exp2(sh.to(tl.float32))) % 4.0
                bv = tl.load(VB + (tile * T + off) * H + h).to(tl.float32)
                sv = (tl.load(VS + (tile * T + off) * H + h).to(tl.float32) * bv)
                lv = (tl.load(VL + (tile * T + off) * H + h).to(tl.float32) * bv)
                vv = (codev * sv + lv) * tl.load(VA + base).to(tl.float32)
                tl.store(outv, vv)
            elif row != -1:
                srow = -row - 2
                src = (srow * H + h) * D + offs
                tl.store(outk, tl.load(STGK + src))
                tl.store(outv, tl.load(STGV + src))
            else:
                tl.store(outk, 0.0)
                tl.store(outv, 0.0)


    @triton.jit
    def _tile2_gather_kernel_hm(
        KC, KS, KL, KA, KB,
        VC, VS, VL, VA, VB,
        SROW, IND, OUTK, OUTV, STGK, STGV,
        N, H: tl.constexpr, D: tl.constexpr, T: tl.constexpr,
    ):
        """Same arithmetic as `_tile2_gather_kernel`, one program per row and all heads inside.

        The per-(row, head) grid launched 166k tiny programs for 20.8k live rows and measured
        scheduling-bound, so this exists only to divide the program count by H. The body is the
        original one verbatim with `h` becoming a compile-time loop variable; a bit-exactness test
        against the original is what makes that claim safe.
        """
        i = tl.program_id(0)
        if i < N:
            slot = tl.load(IND + i)
            row = tl.load(SROW + slot)
            offs = tl.arange(0, D)
            for h in tl.static_range(0, H):
                outk = OUTK + (i * H + h) * D + offs
                outv = OUTV + (i * H + h) * D + offs
                if row >= 0:
                    tile = row // T
                    off = row % T
                    sh = (offs % 4) * 2
                    cb = tl.load(KC + (slot * H + h) * (D // 4) + offs // 4).to(tl.float32)
                    code = tl.floor(cb / tl.exp2(sh.to(tl.float32))) % 4.0
                    base = (tile * H + h) * D + offs
                    kv = (code * tl.load(KS + base).to(tl.float32)
                          + tl.load(KL + base).to(tl.float32))
                    kv = kv * tl.load(KA + base).to(tl.float32)
                    kv = kv * tl.load(KB + (tile * T + off) * H + h).to(tl.float32)
                    tl.store(outk, kv)

                    vb_ = tl.load(VC + (slot * H + h) * (D // 4) + offs // 4).to(tl.float32)
                    codev = tl.floor(vb_ / tl.exp2(sh.to(tl.float32))) % 4.0
                    bv = tl.load(VB + (tile * T + off) * H + h).to(tl.float32)
                    sv = (tl.load(VS + (tile * T + off) * H + h).to(tl.float32) * bv)
                    lv = (tl.load(VL + (tile * T + off) * H + h).to(tl.float32) * bv)
                    vv = (codev * sv + lv) * tl.load(VA + base).to(tl.float32)
                    tl.store(outv, vv)
                elif row != -1:
                    srow = -row - 2
                    src = (srow * H + h) * D + offs
                    tl.store(outk, tl.load(STGK + src))
                    tl.store(outv, tl.load(STGV + src))
                else:
                    tl.store(outk, 0.0)
                    tl.store(outv, 0.0)


def gather_dequant_triton(buf: Tile2Buffers, li: int, idx: torch.Tensor,
                          out_k: torch.Tensor, out_v: torch.Tensor) -> bool:
    """Returns False when Triton is unavailable or the shape is unsupported."""
    if not HAVE_TRITON or buf.D % 4 or buf.D != 128:
        return False
    n = idx.numel()
    if n == 0:
        return True
    if _PROF is not None:
        import time as _tq
        _tq0 = _tq.perf_counter()
    _hm = os.environ.get("SGLANG_TILE2_GATHER_HM") == "1"
    grid = (n,) if _hm else (n, buf.H)
    _kern = _tile2_gather_kernel_hm if _hm else _tile2_gather_kernel
    _kern[grid](
        buf.k_codes[li], buf.k_sc[li], buf.k_lo[li], buf.k_A[li], buf.k_B[li],
        buf.v_codes[li], buf.v_sc[li], buf.v_lo[li], buf.v_A[li], buf.v_B[li],
        buf.slot_row, idx, out_k, out_v,
        buf.staging_k[li].view(-1, buf.H, buf.D),
        buf.staging_v[li].view(-1, buf.H, buf.D),
        n, H=buf.H, D=buf.D, T=buf.T,
    )
    return True


def gather_dequant(buf: Tile2Buffers, li: int, idx: torch.Tensor,
                   out_k: torch.Tensor, out_v: torch.Tensor) -> None:
    """Dequantise the visible rows into the bf16 workspace, in the rotated domain.

    Dispatches to the Triton kernel when it applies. The torch version is kept because it is the
    reference the kernel is checked against (`test_tile2_store.py` P2/P3/P4), and because a
    missing Triton install must degrade rather than fail.
    """
    if os.environ.get("SGLANG_TILE2_DIAG") == "1" and li == 0:
        st = _STATE.setdefault("dg", {"n": 0})
        if st["n"] < 6:
            st["n"] += 1
            r = buf.slot_row.index_select(0, idx.to(torch.long))
            neg = int((r < 0).sum())
            uninit = int((r == -1).sum())
            hi = int(r.max())
            bad = idx.to(torch.long)[r < 0]
            print(f"[TILE2-DIAG] n={idx.numel()} staged={neg} uninit={uninit} "
                  f"bad_slots={bad[:8].tolist()} row_max={hi} "
                  f"idx_min={int(idx.min())} idx_max={int(idx.max())} "
                  f"in_pool_bound={buf.slots}", flush=True)
    if buf.use_triton and gather_dequant_triton(buf, li, idx, out_k, out_v):
        pass
    else:
        gather_dequant_torch(buf, li, idx, out_k, out_v)
    if buf.unrotate:
        # Only the PACKED rows are in the Hadamard domain; staging holds the model's own values.
        # The blend is done with torch.where on a device-side mask: indexing by a host boolean
        # mask costs a device sync per layer.
        hk = buf.Hk()
        rot_k = out_k.float() @ hk
        rot_v = out_v.float() @ hk
        m = (buf.slot_row.index_select(0, idx) >= 0).view(-1, 1, 1)
        out_k.copy_(torch.where(m, rot_k, out_k.float()).to(out_k.dtype))
        out_v.copy_(torch.where(m, rot_v, out_v.float()).to(out_v.dtype))


# ---------------------------------------------------------------- KV dump (cross-arm)
def install_kv_dump(pool) -> None:
    """Dump the K/V a layer actually attends on, for ANY cache recipe, once per process.

    This exists because the remaining question was "does packed_tile2 hand the attention the same
    rows the in-place QDQ arm does?" -- and unit tests can only answer it inside one store. The
    hook sits in the same place for both arms (right before the backend call, after the write and
    after any dequantisation), so whatever difference it reports is a difference the model sees.
    """
    import os as _os
    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

    path = _os.environ["SGLANG_KVDUMP"]
    layer_want = int(_os.environ.get("SGLANG_KVDUMP_LAYER", "0"))
    want = {int(x) for x in _os.environ.get("SGLANG_KVDUMP_AT", "0").split(",")}
    state = {"step": 0, "done": set()}

    base_path = path
    all_layers = _os.environ.get("SGLANG_KVDUMP_ALL_LAYERS") == "1"
    #: sparse layer set, so a chunk-by-chunk scan does not write 36 files per chunk per arm
    layer_set = {int(x) for x in _os.environ.get("SGLANG_KVDUMP_LAYERS", "").split(",") if x.strip()}
    #: also dump right after each EXTEND call. With 8 requests the prefill is split into several
    #: chunks; the layer-2+ corruption only appears at width 8, and this is what pins down which
    #: chunk introduces it.
    do_extend = _os.environ.get("SGLANG_KVDUMP_EXTEND") == "1"
    minrows = int(_os.environ.get("SGLANG_KVDUMP_MINROWS", "200"))
    counters: dict = {}

    def bracket(fn):
        def inner(self, q, k, v, layer, forward_batch, save_kv_cache=True):
            o = fn(self, q, k, v, layer, forward_batch, save_kv_cache)
            # is_extend() is not the complement of decode in this sglang version, and getting it
            # wrong made the decode dump silently produce nothing while the extend dump worked.
            is_ext = not forward_batch.forward_mode.is_decode()
            n = getattr(self, "cur_kv_len", 0)
            if is_ext and not do_extend:
                return o
            if (not is_ext) and do_extend:
                return o
            li = layer.layer_id
            if _os.environ.get("SGLANG_KVDUMP_DBG") == "1":
                _dbg = state.setdefault("dbg", 0)
                if _dbg < 24:
                    state["dbg"] = _dbg + 1
                    print(f"[KVD-DBG] li={li} is_ext={is_ext} n={n} "
                          f"step={counters.get(li, 0)} want={sorted(want)}", flush=True)
            if layer_set:
                if li not in layer_set:
                    return o
            elif li != layer_want and not all_layers:
                return o
            # Count only calls that carry a real batch. Warmup runs tiny decodes (n=7) and would
            # otherwise consume the step numbers we are asking for, so the dump would never fire.
            if n < minrows:
                return o
            # count decode steps per layer, so a dump can name the same step on every layer
            counters[li] = counters.get(li, 0) + 1
            step = counters[li] - 1
            if step not in want or (li, step) in state["done"]:
                return o
            state["done"].add((li, step))
            if is_ext:
                # `cur_kv_len` is 0 during prefill (the prefill updater only counts the cached
                # prefix), so build the slot list from req_to_token and read the store directly.
                pool_ = forward_batch.token_to_kv_pool
                _r2t = forward_batch.req_to_token_pool.req_to_token
                _rows = [int(x) for x in forward_batch.req_pool_indices.tolist()]
                _lens = [int(x) for x in forward_batch.seq_lens.tolist()]
                _sl = torch.cat([_r2t[_rows[i], : _lens[i]] for i in range(len(_rows))]).long()
                if getattr(pool_, "is_packed_tile2", False):
                    _ok = torch.empty((_sl.numel(), pool_.tile2_buf.H, pool_.tile2_buf.D),
                                      dtype=torch.bfloat16, device=_sl.device)
                    _ov = torch.empty_like(_ok)
                    gather_dequant(pool_.tile2_buf, li, _sl, _ok, _ov)
                else:
                    # bf16-storage arm: read the pool rows the same way, so the comparison is
                    # row-for-row against what the model actually attends on.
                    _kb, _ = pool_.get_kv_buffer(layer.layer_id)
                    _ok = _kb.index_select(0, _sl).to(torch.bfloat16)
                torch.save({"slots": _sl.cpu(),
                            "seq_lens": forward_batch.seq_lens.cpu(),
                            "k": _ok.to(torch.bfloat16).cpu()},
                           f"{base_path}.E{step}.L{li}.pt")
                print(f"[KVDUMP] {base_path}.E{step}.L{li}.pt rows={_sl.numel()}", flush=True)
                return o
            if n < minrows:
                return o
            idx = self.cur_kv_indices[: self.cur_kv_len].to(torch.long)
            kk, vv = self._packed_kv_for_attn(layer, forward_batch)
            if not getattr(forward_batch.token_to_kv_pool, "is_packed_tile2", False):
                kk = kk.index_select(0, idx)
                vv = vv.index_select(0, idx)
            torch.save({"slots": idx.cpu(),
                        "seq_lens": forward_batch.seq_lens.cpu(),
                        "req_rows": forward_batch.req_pool_indices.cpu(),
                        # K only, bf16: 36 layers x 8 requests is already ~0.6 GB an arm
                        "k": kk.to(torch.bfloat16).cpu()},
                       f"{base_path}.{'E' if is_ext else 'step'}{step}.L{li}.pt")
            print(f"[KVDUMP] {base_path}.{'E' if is_ext else 'step'}{step}.L{li}.pt "
                  f"rows={idx.numel()}", flush=True)
            return o
        return inner

    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend as _F
    # BOTH, or an extend-chunk scan silently produces nothing at all: the hook only ever saw
    # decode calls, and "no files" looks exactly like "no problem".
    _F.forward_decode = bracket(_F.forward_decode)
    _F.forward_extend = bracket(_F.forward_extend)
    print(f"[KVDUMP] armed -> {path}", flush=True)
