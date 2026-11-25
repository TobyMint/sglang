"""Fused decode attention over the packed_tile2 store.

Why this file exists
--------------------
The first implementation read the cache by gathering every visible row into a dense
bf16 workspace and then handing that workspace to flashinfer. That "materialise"
path moves, per element of KV, the 2-bit code plus a full bf16 write plus a full
bf16 read -- roughly 2.15x the traffic of a plain bf16 cache, which is why decode
ran at 30.5 tok/s against bf16's 134.5. The reference implementation this is
modelled on (KVarN, Apache-2.0, `references/kvarn_triton_fused_decode.py.txt`)
states the same thing from the other side: a fused kernel moves only the packed
code, so decode becomes ~0.2x of bf16's KV traffic and can actually be *faster*
than bf16 instead of slower.

Structure (ported from the reference, adapted to this store):

  * one program per (request, kv_head) -- every query head that shares this KV
    head is served by the same dequantisation, so a tile is decoded once;
  * a device-side loop over the request's tiles, bounded by its sequence length,
    so the launch shape is constant and the kernel is CUDA-graph safe;
  * two sources, exactly like the reference's "int4 cache + fp16 pool":
    tiles that have been packed (``packed_ntiles``) are decoded from the 2-bit
    codes, the single unflushed tile is read from the bf16 landing window;
  * online softmax with fp32 accumulation and bf16 MMA;
  * the output stays in the Hadamard domain; the caller rotates q once per step
    and un-rotates the output once.

Addressing note that cost me a wrong turn before: the *parameters* are indexed by
a contiguous per-request tile id (``base_tab[row] + ti``), but the *codes* are
indexed by physical slot. `tile_id` being contiguous does not make the codes
contiguous, so the packed branch must look each slot up in `pos_slot` -- and it
must NOT use `req_to_token`, which does not hold the full history after a
chunked-prefill row change.
"""

from __future__ import annotations

#: SGLANG_TILE2_RECENT=R reads the most recent R positions from the bf16 landing window instead
#: of the packed codes (R <= 128 = one ring). Default 0 reproduces the tile-aligned behaviour.
_RECENT = int(__import__("os").environ.get("SGLANG_TILE2_RECENT", "0"))
#: SGLANG_TILE2_REGS=1 reports the compiled kernel's register count, local-memory spills and
#: shared-memory use after the first launch. Register pressure is what caps this kernel at 3
#: blocks/SM (168 regs/thread -> 19.7% occupancy -> 13% of peak bandwidth), so every change aimed
#: at occupancy has to be judged on this number before a benchmark is worth running.
_DBG_REGS = __import__("os").environ.get("SGLANG_TILE2_REGS") == "1"


def _report_regs(jitfn, tag: str) -> None:
    try:
        caches = getattr(jitfn, "device_caches", None) or getattr(jitfn, "cache", None)
        if not caches:
            print(f"[TILE2-REGS] {tag}: 没有编译缓存", flush=True)
            return
        for _dev, entry in caches.items():
            kc = entry[0] if isinstance(entry, tuple) else entry
            items = kc.items() if hasattr(kc, "items") else []
            for _key, kern in items:
                md = getattr(kern, "metadata", None)
                print(f"[TILE2-REGS] {tag} regs={getattr(kern, 'n_regs', '?')} "
                      f"spills={getattr(kern, 'n_spills', '?')} "
                      f"smem={getattr(md, 'shared', '?')} "
                      f"warps={getattr(md, 'num_warps', '?')} "
                      f"stages={getattr(md, 'num_stages', '?')}", flush=True)
                return
        print(f"[TILE2-REGS] {tag}: 缓存为空", flush=True)
    except Exception as exc:                      # Triton internals move between versions
        print(f"[TILE2-REGS] {tag}: 读取失败 {type(exc).__name__}: {exc}", flush=True)

#: SGLANG_TILE2_NODQ=1 skips the per-element affine in the K dequantisation. DIAGNOSTIC ONLY --
#: the outputs are wrong. It measures the cost of the `ck * sA + lA` that D3's third step would
#: remove (fold sA into q, lA into a constant), which is blocked today because staged and packed
#: rows share one tl.dot and therefore one affine semantics.
_NODQ = __import__("os").environ.get("SGLANG_TILE2_NODQ") == "1"
if _RECENT:
    print(f"[TILE2] decode read path: the last {_RECENT} positions come from the bf16 landing "
          f"window; the rest from the packed codes", flush=True)

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

    class _Stub:
        @staticmethod
        def jit(*a, **k):
            def deco(fn):
                return fn

            return deco

        @staticmethod
        def constexpr(x):
            return x

    triton = _Stub()
    tl = None


#: Q rows per program. Ampere MMA wants M >= 16, and Qwen3-4B's GQA group is 4,
#: so the query tile is padded and the padding is masked out. This is a
#: correctness-first choice; a per-head-group kernel with the real G is the
#: obvious next optimisation once the path is proven.
BQ = 16


@triton.jit
def _tile2_fused_decode_kernel(
    Q, ROWS, LENS, BASE, STG, POS, RIDX, KEY, SROW, CLOSED, OUT,
    KC, KS, KL, KA, KB,
    VC, VS, VL, VA, VB,
    STGK, STGV,
    SCALE,
    RSTR: tl.constexpr,          # row stride of req_to_token
    H: tl.constexpr,             # kv heads
    D: tl.constexpr,             # head dim
    T: tl.constexpr,             # tokens per tile
    G: tl.constexpr,             # query heads per kv head
    HQ: tl.constexpr,            # query heads
    BQ_: tl.constexpr,           # padded query tile
    BN: tl.constexpr,            # tokens per k/v chunk
    USE_SROW: tl.constexpr = True,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    n = tl.load(LENS + b)                      # includes this step's token
    req = tl.load(ROWS + b).to(tl.int64)
    base = tl.load(BASE + req, mask=n > 0, other=0).to(tl.int64)
    stg = tl.load(STG + req, mask=n > 0, other=0).to(tl.int64)
    key = tl.load(KEY + req, mask=n > 0, other=0).to(tl.int64)
    # Counted per REQUEST, not per row: a chunked-prefill request changes row between chunks.
    nc = tl.load(CLOSED + key, mask=n > 0, other=0)   # tiles already packed

    d = tl.arange(0, D)
    g = tl.arange(0, BQ_)
    hq = h * G + g
    qo = (b * HQ + hq[:, None]) * D + d[None, :]

    q = tl.load(Q + qo, mask=(g[:, None] < G) & (n > 0), other=0.0)
    q = q.to(tl.bfloat16)

    m = tl.full((BQ_,), -float("inf"), tl.float32)
    z = tl.zeros((BQ_,), tl.float32)
    acc = tl.zeros((BQ_, D), tl.float32)

    tl.static_assert(D % 4 == 0)
    db = tl.arange(0, D // 4)
    sh4 = 2 * tl.arange(0, 4)

    ntiles = (n + T - 1) // T
    for ti in range(0, ntiles):
        # A launch-time choice between two rules.
        #  USE_SROW (radix cache on): `slot_row` is the only entry that knows a tile belongs to
        #    ANOTHER request, so a shared prefix decodes from codes instead of a landing window
        #    that no longer holds it.
        #  not USE_SROW (radix off): keep the layer-local count. `slot_row` is shared by all 36
        #    layers and is published by layer 0's flush, while a layer's codes are written only
        #    when THAT layer flushes -- so `slot_row` makes the upper layers decode a just-closed
        #    tile out of a different tile's codes. Measured (MATH500 n=200, radix off): 0.710 with
        #    the count, 0.555 with slot_row, loss concentrated on the longest generations.
        if USE_SROW:
            _rid0 = tl.load(RIDX + req, mask=n > 0, other=0).to(tl.int64)
            _pos0 = ti * T
            _s0 = tl.load(POS + _rid0 * RSTR + _pos0, mask=_pos0 < n, other=-1).to(tl.int64)
            _sr0 = tl.load(SROW + _s0, mask=_s0 >= 0, other=-1)
            packed = _sr0 >= 0
        else:
            packed = ti < nc

        for c0 in range(0, T, BN):
            c = c0 + tl.arange(0, BN)
            posn = ti * T + c
            valid = posn < n

            if packed:
                # Slots come from req_to_token, NOT from the per-row position table: a request
                # changes req_to_token row between prefill chunks, so a row-keyed table has holes
                # for everything written under the earlier rows -- which silently zeroed most of
                # the history. sglang's req_to_token is maintained across the row change.
                # Slots come from the per-request position table, NOT from req_to_token: above
                # chunked_prefill_size a request is prefilled in two pieces on two different rows
                # and the second row's req_to_token does not hold the first piece's positions.
                ridx = tl.load(RIDX + req, mask=n > 0, other=0).to(tl.int64)
                slots = tl.load(POS + ridx * RSTR + posn, mask=valid, other=-1).to(tl.int64)
                ok = valid & (slots >= 0)
                # K codes are packed along the CHANNEL axis: four consecutive d share a byte.
                kaddr = (slots[:, None] * H + h) * (D // 4) + db[None, :]
                u = tl.load(KC + kaddr, mask=ok[:, None], other=0)
                ck = tl.reshape((u[:, :, None] >> sh4[None, None, :]) & 3, (BN, D)).to(tl.float32)
                vaddr = (slots[:, None] * H + h) * (D // 4) + db[None, :]
                w = tl.load(VC + vaddr, mask=ok[:, None], other=0)
                cv = tl.reshape((w[:, :, None] >> sh4[None, None, :]) & 3, (BN, D)).to(tl.float32)

                # Tile id and offset come from slot_row -- the value the store itself recorded --
                # not from base + position // T. Recomputing them assumes the request's block base
                # is still the one its earliest tiles were packed with; if it ever is not, the
                # whole history is decoded with another tile's parameters. Reading them back makes
                # that class of error impossible.
                srow = tl.load(SROW + slots, mask=ok, other=-1).to(tl.int64)
                # A slot whose entry is still the staged encoding would make tidr negative and the
                # parameter gathers run off the end of the tensor (illegal access, not a wrong
                # answer). Guard it and clamp the indices; masking the row out is correct for a
                # consistent store, and the count of such rows is exactly what an audit wants.
                okp = ok & (srow >= 0)
                safe_srow = tl.where(okp, srow, 0)
                tidr = safe_srow // T
                offr = safe_srow % T
                pcr = (tidr[:, None] * H + h) * D + d[None, :]
                ksr = tl.load(KS + pcr, mask=okp[:, None], other=0.0).to(tl.float32)
                klr = tl.load(KL + pcr, mask=okp[:, None], other=0.0).to(tl.float32)
                kar = tl.load(KA + pcr, mask=okp[:, None], other=0.0).to(tl.float32)
                var = tl.load(VA + pcr, mask=okp[:, None], other=0.0).to(tl.float32)
                ptr = (tidr * T + offr) * H + h
                kb = tl.load(KB + ptr, mask=okp, other=0.0).to(tl.float32)
                vs = tl.load(VS + ptr, mask=okp, other=0.0).to(tl.float32)
                vl = tl.load(VL + ptr, mask=okp, other=0.0).to(tl.float32)
                vb = tl.load(VB + ptr, mask=okp, other=0.0).to(tl.float32)

                k = ((ck * ksr + klr) * kar) * kb[:, None]
                v = (cv * (vs * vb)[:, None] + (vl * vb)[:, None]) * var
                # Per-row source choice, exactly like gather_dequant: a row whose slot_row entry is
                # still the staged encoding must be read from the landing window, not decoded from
                # codes with zeroed parameters. Without this those rows contribute score=0/v=0 and
                # silently pollute the softmax -- and the unit test cannot see it, because there the
                # two sources always agree.
                fallback = valid & ~okp
                sa = ((stg * T + c[:, None]) * H + h) * D + d[None, :]
                kf = tl.load(STGK + sa, mask=fallback[:, None], other=0.0).to(tl.float32)
                vf = tl.load(STGV + sa, mask=fallback[:, None], other=0.0).to(tl.float32)
                k = tl.where(fallback[:, None], kf, k)
                v = tl.where(fallback[:, None], vf, v)
                k = k.to(tl.bfloat16)
                v = v.to(tl.bfloat16)
            else:
                # The one unflushed tile. It may hold every one of the T rows.
                a = ((stg * T + c[:, None]) * H + h) * D + d[None, :]
                k = tl.load(STGK + a, mask=valid[:, None], other=0.0)
                v = tl.load(STGV + a, mask=valid[:, None], other=0.0)

            s = tl.dot(q, tl.trans(k)) * SCALE
            s = tl.where(valid[None, :], s, -float("inf"))

            mn = tl.maximum(m, tl.max(s, 1))
            dead = mn == -float("inf")
            safe = tl.where(dead, 0.0, mn)
            p = tl.exp(s - safe[:, None])
            alpha = tl.where(dead, 0.0, tl.exp(m - safe))
            z = z * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m = mn

    ans = acc / tl.maximum(z, 1.0e-20)[:, None]
    tl.store(OUT + qo, ans.to(tl.bfloat16), mask=(g[:, None] < G) & (n > 0))


def audit_sources(buf, li: int, req: int, n: int, nc: int) -> dict:
    """How the two sources actually line up for one request, per the STORE.

    The fused kernel decides packed-vs-staged from `packed_ntiles` and then reads the tile id out
    of `slot_row`. If the store disagrees with `packed_ntiles` for some rows inside the packed
    region, those rows are silently dropped from the attention -- an error of exactly quantisation
    magnitude. This reports the counts so the disagreement is visible instead of inferred.
    """
    import torch as _t

    # slots come from the per-request position table (see fused_decode): req_to_token's row only
    # holds the current chunk's positions once the prefill has been split.
    ridx = int(buf.ridx_tab[req]) if buf.ridx_tab is not None else req
    slots = buf.pos_slot[ridx, :n].to(_t.long)
    srow = buf.slot_row.index_select(0, slots).to(_t.long)
    pstart = nc * buf.T
    head_p = srow[:pstart]
    return {
        "n": int(n),
        "nc": int(nc),
        "packed_region": int(pstart),
        "packed_region_negative": int((head_p < 0).sum()),
        "packed_region_unwritten": int((head_p == -1).sum()),
        "staged_region_negative": int((srow[pstart:] < 0).sum()),
        "staged_region_rows": int(n - pstart),
        "max_tile": int((srow[srow >= 0] // buf.T).max()) if bool((srow >= 0).any()) else -1,
    }


@triton.jit
def _tile2_split_decode_kernel(
    Q, ROWS, LENS, BASE, STG, POS, RIDX, KEY, SROW, CLOSED, HMAT,
    KC, KS, KL, KA, KB, VC, VS, VL, VA, VB, STGK, STGV,
    MIDO, MIDLSE, SCALE,
    RSTR: tl.constexpr, H: tl.constexpr, D: tl.constexpr, T: tl.constexpr,
    G: tl.constexpr, HQ: tl.constexpr, BQ_: tl.constexpr, BN: tl.constexpr,
    SPLITS: tl.constexpr, ROT: tl.constexpr = False, USE_SROW: tl.constexpr = True,
    RECENT: tl.constexpr = 0,
    NODQ: tl.constexpr = False,
):
    """Stage 1 of split-K flash decoding.

    The single-program version put one program per (request, kv_head) -- 64 programs on a card with
    82 SMs, each walking the whole context serially, which measured out at ~2 % of peak: the GPU was
    idle, not slow. Splitting the tile range across `SPLITS` programs fills it, and the split axis
    only affects occupancy -- stage 2 combines with log-sum-exp, so the result is unchanged.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    sp = tl.program_id(2)

    n = tl.load(LENS + b)
    if n <= 0:
        return
    req = tl.load(ROWS + b).to(tl.int64)
    stg = tl.load(STG + req).to(tl.int64)
    key = tl.load(KEY + req).to(tl.int64)
    nc = tl.load(CLOSED + key)
    ridx = tl.load(RIDX + req).to(tl.int64)

    d = tl.arange(0, D)
    g = tl.arange(0, BQ_)
    hq = h * G + g
    qo = (b * HQ + hq[:, None]) * D + d[None, :]
    q = tl.load(Q + qo, mask=g[:, None] < G, other=0.0).to(tl.bfloat16)
    # The query arrives already rotated (see _tile2_rot_q_kernel) and the output leaves in the
    # rotated frame (stage 2 un-rotates once per request, not once per split).

    m = tl.full((BQ_,), -float("inf"), tl.float32)
    z = tl.zeros((BQ_,), tl.float32)
    acc = tl.zeros((BQ_, D), tl.float32)

    db = tl.arange(0, D // 4)
    sh4 = 2 * tl.arange(0, 4)

    ntiles = (n + T - 1) // T
    per = (ntiles + SPLITS - 1) // SPLITS
    lo = sp * per
    hi = tl.minimum(lo + per, ntiles)
    for ti in range(lo, hi):
        # See the non-split kernel for why this is a launch-time choice.
        if USE_SROW:
            _rid0 = tl.load(RIDX + req, mask=n > 0, other=0).to(tl.int64)
            _pos0 = ti * T
            _s0 = tl.load(POS + _rid0 * RSTR + _pos0, mask=_pos0 < n, other=-1).to(tl.int64)
            _sr0 = tl.load(SROW + _s0, mask=_s0 >= 0, other=-1)
            packed = _sr0 >= 0
        else:
            packed = ti < nc
        # Tile-uniform parameter planes, loaded ONCE per tile (a chunk never straddles tiles).
        # With BN=32 and T=128 the old placement read them 4x per tile; the profile said L1/TEX.
        if packed:
            _rid_h = tl.load(RIDX + req, mask=n > 0, other=0).to(tl.int64)
            _pos_h = ti * T
            _s0_h = tl.load(POS + _rid_h * RSTR + _pos_h, mask=_pos_h < n, other=-1).to(tl.int64)
            _sr0_h = tl.load(SROW + _s0_h, mask=_s0_h >= 0, other=-1)
            _ok_h = _sr0_h >= 0
            _pc_h = ((_sr0_h // T) * H + h) * D + d
            sA_t = (tl.load(KS + _pc_h, mask=_ok_h, other=0.0).to(tl.float32)
                    * tl.load(KA + _pc_h, mask=_ok_h, other=0.0).to(tl.float32))
            lA_t = (tl.load(KL + _pc_h, mask=_ok_h, other=0.0).to(tl.float32)
                    * tl.load(KA + _pc_h, mask=_ok_h, other=0.0).to(tl.float32))
            vA_t = tl.load(VA + _pc_h, mask=_ok_h, other=0.0).to(tl.float32)
        else:
            sA_t = tl.zeros((D,), tl.float32)
            lA_t = tl.zeros((D,), tl.float32)
            vA_t = tl.full((D,), 1.0, tl.float32)

        # UNROLLED (T/BN = 4 chunks at T=128, BN=32). The compiler emits the four chunk bodies
        # together, so the four independent `slots`->`codes` gathers can be issued before the
        # first one is consumed. With a runtime `range` each iteration waited for its own
        # `slots` load before the code load could even be addressed, so no memory-level
        # parallelism existed across chunks and `num_stages` had nothing to work with
        # (the kernel is SHARED:0: there is no pipeline buffer to stage into).
        for c0 in tl.static_range(0, T, BN):
            c = c0 + tl.arange(0, BN)
            posn = ti * T + c
            valid = posn < n
            if packed:
                slots = tl.load(POS + ridx * RSTR + posn, mask=valid, other=-1).to(tl.int64)
                ok = valid & (slots >= 0)
                kaddr = (slots[:, None] * H + h) * (D // 4) + db[None, :]
                u = tl.load(KC + kaddr, mask=ok[:, None], other=0)
                ck = tl.reshape((u[:, :, None] >> sh4[None, None, :]) & 3, (BN, D)).to(tl.float32)
                srow = tl.load(SROW + slots, mask=ok, other=-1).to(tl.int64)
                okp = ok & (srow >= 0)
                safe_srow = tl.where(okp, srow, 0)
                tidr = safe_srow // T
                offr = safe_srow % T
                # ---- dequantisation, rearranged so the per-token factors leave the D axis ----
                #   K: (c*s + l) * A * B  ==  (c*(s*A) + (l*A)) * B   -> B becomes a per-token scalar
                #      applied to the finished score row, not a [BN, D] plane inside the loop.
                #   V: (c*(vs*vB) + vl*vB) * vA  ->  vA comes out of the accumulation entirely, and
                #      (vs*vB), (vl*vB) are per-token scalars multiplying p.
                # Both are exact rearrangements; what they buy is that the inner loop no longer reads
                # four D-wide parameter planes per token, which is what made the kernel L1/TEX bound.
                # A chunk never straddles tiles, so the per-channel planes are tile-uniform: take
                # the tile id from the rows themselves (max of slot_row // T) rather than from a base.
                sA = sA_t
                lA = lA_t
                vA_d = vA_t
                if RECENT > 0:
                    # A row redirected to the ring carries its EXACT bf16 value, but the chunk
                    # output is multiplied by this tile's per-channel V scale `vA_d` further down.
                    # Pre-dividing the staged value cancels that multiply. Without this a recent
                    # row contributes V*vA instead of V (Codex audit, tile2_attn.py:432).
                    inv_vA = tl.where(vA_d != 0.0, 1.0 / vA_d, 0.0)
                ptr = (tidr * T + offr) * H + h
                k_b = tl.load(KB + ptr, mask=okp, other=0.0).to(tl.float32)
                a_v = (tl.load(VS + ptr, mask=okp, other=0.0).to(tl.float32)
                       * tl.load(VB + ptr, mask=okp, other=0.0).to(tl.float32))
                b_v = (tl.load(VL + ptr, mask=okp, other=0.0).to(tl.float32)
                       * tl.load(VB + ptr, mask=okp, other=0.0).to(tl.float32))
                if NODQ:
                    # DIAGNOSTIC ONLY: drops the per-element affine. Results are wrong; this
                    # exists to measure what the dequantisation ALU costs, the same way
                    # SGLANG_TILE2_ITERS=0 does for VarN and NOROT does for the rotation.
                    khat = ck.to(tl.bfloat16)
                else:
                    khat = (ck * sA[None, :] + lA[None, :]).to(tl.bfloat16)             # [BN, D]

                fb = valid & ~okp
                if RECENT > 0:
                    # The landing window is a 128-deep ring over positions mod T, so the most
                    # recent RECENT positions are still bit-exact there even after their tile
                    # closed. Reading them from the ring instead of the 2-bit codes is exactly
                    # the "recent R tokens in high precision" that KIVI/KVarN implement with an
                    # explicit FP16 window -- here it costs no extra memory, because the ring is
                    # allocated anyway. RECENT <= T only (one ring's worth).
                    fb = fb | (valid & (posn >= n - RECENT))
                sa = ((stg * T + c[:, None]) * H + h) * D + d[None, :]
                kf = tl.load(STGK + sa, mask=fb[:, None], other=0.0)
                vf = tl.load(STGV + sa, mask=fb[:, None], other=0.0)
                if RECENT > 0:
                    vf = (vf.to(tl.float32) * inv_vA[None, :]).to(vf.dtype)
                # Staged rows are already exact: fold them in as k_b = v_b = 1, a_v = 1, b_v = 0.
                use_st = fb | ~okp
                k_b = tl.where(use_st, 1.0, k_b)
                a_v = tl.where(use_st, 1.0, a_v)
                b_v = tl.where(use_st, 0.0, b_v)
                khat = tl.where(fb[:, None], kf, khat).to(tl.bfloat16)
                # bf16 for the dot; raw codes are 0..3 so they are exact there, and staged
                # rows carry their true value through the identity factors.
                # V is loaded HERE, not next to the K load above: nothing between the two needs it,
                # and holding a [BN, D] tile across the whole score path costs registers that this
                # kernel cannot spare (168/thread puts only 3 blocks on an SM).
                w = tl.load(VC + kaddr, mask=ok[:, None], other=0)
                cv = tl.reshape((w[:, :, None] >> sh4[None, None, :]) & 3, (BN, D)).to(tl.bfloat16)
                vcode = tl.where(fb[:, None], vf, cv)
                sA_eff = sA
                m_extra = None
            else:
                a_st = ((stg * T + c[:, None]) * H + h) * D + d[None, :]
                khat = tl.load(STGK + a_st, mask=valid[:, None], other=0.0)
                vcode = tl.load(STGV + a_st, mask=valid[:, None], other=0.0)
                # Staged rows are exact, so every factor below is the identity.
                k_b = tl.full((BN,), 1.0, tl.float32)
                a_v = tl.full((BN,), 1.0, tl.float32)
                b_v = tl.zeros((BN,), tl.float32)
                vA_d = tl.full((D,), 1.0, tl.float32)

            # score = q . khat, then the per-token K factor scales the finished row
            sc = tl.dot(q, tl.trans(khat)) * SCALE * k_b[None, :]
            sc = tl.where(valid[None, :], sc, -float("inf"))
            mn = tl.maximum(m, tl.max(sc, 1))
            dead = mn == -float("inf")
            safe = tl.where(dead, 0.0, mn)
            p = tl.exp(sc - safe[:, None])
            alpha = tl.where(dead, 0.0, tl.exp(m - safe))
            z = z * alpha + tl.sum(p, 1)
            # acc = sum_j p_j * (code_j * a_j + b_j) * vA  -- two cheap accumulations, no vA plane
            # in the inner loop: the code term is a dot, the offset term is a scalar sum.
            pa = (p * a_v[None, :]).to(tl.bfloat16)
            # vA multiplies the WHOLE bracket, not just the offset term.
            acc = (acc * alpha[:, None]
                   + (tl.dot(pa, vcode) + tl.sum(p * b_v[None, :], 1)[:, None])
                   * vA_d[None, :])
            m = mn

    nonempty = z > 0
    o = acc / tl.where(nonempty, z, 1.0)[:, None]
    lse = tl.where(nonempty, m + tl.log(tl.where(nonempty, z, 1.0)), -float("inf"))
    mid = (b * HQ + hq[:, None]) * SPLITS * D + sp * D + d[None, :]
    tl.store(MIDO + mid, o, mask=g[:, None] < G)
    tl.store(MIDLSE + (b * HQ + hq) * SPLITS + sp, lse, mask=g < G)


@triton.jit
def _tile2_rot_q_kernel(Q, OUT, HMAT, NQ: tl.constexpr, D: tl.constexpr):
    """Rotate one request's query tile [HQ, D] in a single dot.

    Stage 1 runs `SPLITS` times per (request, kv_head), so rotating the query there did the same
    [16,128]x[128,128] product eight times over. This kernel does it once per request, and the
    splits then just read the result.
    """
    b = tl.program_id(0)
    r = tl.arange(0, NQ)
    d = tl.arange(0, D)
    q = tl.load(Q + (b * NQ + r[:, None]) * D + d[None, :])
    hm = tl.load(HMAT + d[:, None] * D + d[None, :]).to(tl.bfloat16)
    o = tl.dot(q.to(tl.bfloat16), hm)
    tl.store(OUT + (b * NQ + r[:, None]) * D + d[None, :], o.to(tl.bfloat16))


@triton.jit
def _tile2_split_combine_kernel(MIDO, MIDLSE, OUT, HMAT,
                                D: tl.constexpr, SPLITS: tl.constexpr,
                                HQ: tl.constexpr, ROT: tl.constexpr = False):
    """Stage 2: log-sum-exp combine, one program per request, and the output un-rotation.

    Doing the un-rotation here instead of in stage 1 removes it from the `SPLITS`-fold redundant
    path: the combine is linear, so un-rotating after it is exact.
    """
    b = tl.program_id(0)
    r = tl.arange(0, HQ)
    d = tl.arange(0, D)
    row = b * HQ + r                                   # [HQ]
    m = tl.full((HQ,), -float("inf"), tl.float32)
    for s in range(0, SPLITS):
        m = tl.maximum(m, tl.load(MIDLSE + row * SPLITS + s))
    z = tl.zeros((HQ,), tl.float32)
    acc = tl.zeros((HQ, D), tl.float32)
    for s in range(0, SPLITS):
        v = tl.load(MIDLSE + row * SPLITS + s)
        w = tl.exp(v - m)
        z += w
        acc += (tl.load(MIDO + row[:, None] * SPLITS * D + s * D + d[None, :])
                * w[:, None])
    o = acc / z[:, None]
    if ROT:
        hm = tl.load(HMAT + d[:, None] * D + d[None, :]).to(tl.bfloat16)
        o = tl.dot(o.to(tl.bfloat16), hm)
    tl.store(OUT + row[:, None] * D + d[None, :], o.to(tl.bfloat16))


def fused_decode_split(buf, li: int, q_rot: torch.Tensor, rows: torch.Tensor,
                       lens: torch.Tensor, packed_ntiles: torch.Tensor,
                       out: torch.Tensor, scale: float, scratch,
                       splits: int = 8, bn: int = 32, rot_fused: bool = False,
                       qrot=None) -> bool:
    if not HAVE_TRITON or buf.D != 128 or buf.T % bn or splits < 1:
        return False
    hq = q_rot.shape[1]
    if hq % buf.H:
        return False
    bcap = q_rot.shape[0]
    if bcap == 0:
        return True
    mido, midlse = scratch
    need = bcap * hq * splits
    if mido.numel() < need * buf.D or midlse.numel() < need:
        return False
    hmat = buf.Hk().to(torch.bfloat16)
    q_in = q_rot
    if rot_fused:
        if qrot.numel() < bcap * hq * buf.D:
            return False
        q_in = qrot[: bcap * hq * buf.D].view(bcap, hq, buf.D)
        _tile2_rot_q_kernel[(bcap,)](q_rot, q_in, hmat, NQ=hq, D=buf.D)
    _tile2_split_decode_kernel[(bcap, buf.H, splits)](
        q_in, rows, lens, buf.base_tab, buf.stg_tab, buf.pos_slot, buf.ridx_tab,
        buf.key_tab, buf.slot_row, packed_ntiles, hmat,
        buf.k_codes[li], buf.k_sc[li], buf.k_lo[li], buf.k_A[li], buf.k_B[li],
        buf.v_codes[li], buf.v_sc[li], buf.v_lo[li], buf.v_A[li], buf.v_B[li],
        buf.staging_k[li].view(-1, buf.H, buf.D), buf.staging_v[li].view(-1, buf.H, buf.D),
        mido, midlse, scale,
        RSTR=buf.pos_slot.stride(0), H=buf.H, D=buf.D, T=buf.T,
        G=hq // buf.H, HQ=hq, BQ_=max(16, hq // buf.H), BN=bn, SPLITS=splits,
        ROT=rot_fused,
        USE_SROW=getattr(buf, "radix_on", True),
        RECENT=_RECENT,
        NODQ=_NODQ,
        num_warps=int(__import__("os").environ.get("SGLANG_TILE2_WARPS", "4")),
        num_stages=int(__import__("os").environ.get("SGLANG_TILE2_STAGES", "2")),
    )
    _tile2_split_combine_kernel[(bcap,)](
        mido, midlse, out.view(-1, buf.D), hmat,
        D=buf.D, SPLITS=splits, HQ=hq, ROT=rot_fused,
    )
    if _DBG_REGS:
        _report_regs(_tile2_split_decode_kernel, "split_decode")
    return True


def fused_decode(buf, li: int, q_rot: torch.Tensor, rows: torch.Tensor,
                 lens: torch.Tensor, packed_ntiles: torch.Tensor,
                 out: torch.Tensor, scale: float, bn: int = 32) -> bool:
    """Decode attention straight off the packed store.

    q_rot  : [B_CAP, Hq, D] bf16, already in the Hadamard domain
    rows   : [B_CAP] int32 req_to_token rows (0 for padded slots)
    lens   : [B_CAP] int32 sequence lengths INCLUDING the token written this step
    packed_ntiles : [req_rows] int32 number of fully packed tiles for this layer
    out    : [B_CAP, Hq, D] bf16, written in the Hadamard domain
    Returns False when Triton or the shape is unsupported.
    """
    if not HAVE_TRITON or buf.D != 128 or buf.T % bn:
        return False
    if q_rot.shape[-1] != buf.D or buf.H == 0:
        return False
    hq = q_rot.shape[1]
    if hq % buf.H:
        return False
    bcap = q_rot.shape[0]
    if bcap == 0:
        return True
    import time as _tf
    _tf0 = _tf.perf_counter()
    _tile2_fused_decode_kernel[(bcap, buf.H)](
        q_rot, rows, lens,
        buf.base_tab, buf.stg_tab, buf.pos_slot, buf.ridx_tab, buf.key_tab,
        buf.slot_row, packed_ntiles, out,
        buf.k_codes[li], buf.k_sc[li], buf.k_lo[li], buf.k_A[li], buf.k_B[li],
        buf.v_codes[li], buf.v_sc[li], buf.v_lo[li], buf.v_A[li], buf.v_B[li],
        buf.staging_k[li].view(-1, buf.H, buf.D),
        buf.staging_v[li].view(-1, buf.H, buf.D),
        scale,
        RSTR=buf.pos_slot.stride(0),
        H=buf.H, D=buf.D, T=buf.T,
        # tl.dot needs M >= 16 on this hardware, so the tile is padded from G to 16. Using
        # G here is rejected at compile time; the padding is the price of the MMA path.
        G=hq // buf.H, HQ=hq, BQ_=max(16, hq // buf.H), BN=bn,
        USE_SROW=getattr(buf, "radix_on", True),
    )
    if _DBG_REGS:
        _report_regs(_tile2_fused_decode_kernel, "fused_decode")
    try:
        from sglang.srt.mem_cache.tile2_store import _prof_add as _pa
        _pa("fused", _tf.perf_counter() - _tf0)
    except Exception:
        pass
    return True


def reference_decode(buf, li: int, q_rot: torch.Tensor, rows, lens,
                     packed_ntiles, scale: float) -> torch.Tensor:
    """Slow, obviously-correct version of the same computation, for the unit test.

    Reuses `gather_dequant` so the K/V values are produced by code that is already
    validated, then does full attention with torch. The fused kernel must agree
    with this to within reduction-order noise.
    """
    from tile2_store import gather_dequant

    bcap, hq, D = q_rot.shape
    out = torch.zeros((bcap, hq, D), dtype=torch.bfloat16, device=q_rot.device)
    for b in range(bcap):
        n = int(lens[b])
        if n <= 0:
            continue
        req = int(rows[b])
        slots = buf.pos_slot[req, :n].to(torch.long)
        assert int(slots.min()) >= 0, "position table has a hole"
        k = torch.empty((n, buf.H, buf.D), dtype=torch.bfloat16, device=q_rot.device)
        v = torch.empty_like(k)
        gather_dequant(buf, li, slots.to(torch.int32), k, v)
        # gather_dequant un-rotates when buf.unrotate; the fused kernel does not, so
        # re-rotate to compare in the same frame.
        if buf.unrotate:
            from tile2_store import rotate_hadamard

            k = rotate_hadamard(k.float()).to(torch.bfloat16)
            v = rotate_hadamard(v.float()).to(torch.bfloat16)
        g = hq // buf.H
        for h in range(buf.H):
            kk = k[:, h, :].float()                 # [n, D]
            vv = v[:, h, :].float()
            qq = q_rot[b, h * g:(h + 1) * g, :].float()   # [g, D]
            s = (qq @ kk.T) * scale
            p = torch.softmax(s, dim=-1)
            out[b, h * g:(h + 1) * g, :] = (p @ vv).to(torch.bfloat16)
    return out
