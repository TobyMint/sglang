"""Tile-closure lifecycle for in-place KV quantisation.

Why this exists: the previous reproduction hook had no cross-call state, so a token written by
a decode step (T=1) never formed a 128-token tile and stayed at BF16 for the whole generation.
Every accuracy number measured through that hook therefore describes "partial prefill
quantisation with the generation KV left at BF16", not a fully quantised cache.

Semantics (positions are logical positions inside one request, NOT pool slots):

  tile grid      tile_k = [k*T, (k+1)*T), anchored at position 0, half-open
  protected      P = {0..S-1} union {L-R .. L-1}, a set union (never double counted)
  closable       k >= ceil(S/T)   and   (k+1)*T <= L - R
  sink overlap   a tile intersecting the sink region is kept WHOLE, so when S is not a multiple
                 of T there is an extra sub-tile non-sink prefix that stays exact and must be
                 counted as unquantised
  completion     a tile counts as done only when BOTH K and V have been applied, because they
                 are separate buffers; K-only is not "done"
  invariants     `closed` is monotone; a tile enters QDQ at most once per tensor; validation
                 happens BEFORE any gather/QDQ/scatter, so an illegal call mutates nothing;
                 attention must read after the scatter is complete (stream ordering, not the
                 Python return)

Review-driven fixes in this revision (each was a real defect):
  * pending() is now (closed tiles) - (done tiles). The old version assumed marks arrive in
    order and reported the wrong tile when they did not.
  * mark()/validate() reject tiles that are not closed yet, tiles protected by sink/recent, and
    out-of-range tiles. Previously any tile index was accepted.
  * validation runs before the quantisation loop, so a duplicate call raises before the second
    quantisation has already been written into the pool.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

KV = ("K", "V")


@dataclass
class _State:
    L: int = 0
    closed: int = 0                                     # next tile index to consider
    done: Dict[str, set] = field(default_factory=lambda: {kv: set() for kv in KV})


class TileClosure:
    """Tracks, per (request, layer), which 128-token tiles have closed and been applied."""

    def __init__(self, T: int = 128, sink: int = 0, recent: int = 0,
                 per_write: bool = False) -> None:
        # per_write=True is the POLICY-INTERVENTION arm: a tile is quantised only if it is fully
        # contained in the current write and nothing carries over to later calls, so decode writes
        # never close a tile. It is NOT a faithful reproduction of the old stateless hook (that one
        # sliced from each write's local start and could tile across requests); it exists to
        # measure the effect of "no cross-step closure" as a policy.
        self.per_write = per_write
        if T <= 0:
            raise ValueError("T must be positive")
        if sink < 0 or recent < 0:
            raise ValueError("sink and recent must be non-negative")
        self.T = T
        self.sink = sink
        self.recent = recent
        self._st: Dict[Tuple[int, int], _State] = {}

    # -- helpers ---------------------------------------------------------
    @property
    def first_tile(self) -> int:
        """First tile that does not intersect the sink region: ceil(S / T)."""
        return -(-self.sink // self.T)

    def state(self, req: int, layer: int) -> _State:
        key = (req, layer)
        st = self._st.get(key)
        if st is None:
            st = _State(closed=self.first_tile)
            self._st[key] = st
        return st

    def closed_tiles(self, req: int, layer: int) -> set:
        st = self.state(req, layer)
        return set(range(self.first_tile, st.closed))

    # -- lifecycle -------------------------------------------------------
    def start(self, req: int, layer: Optional[int] = None) -> None:
        """Reset state for a new request, or for a reused request id."""
        for key in [k for k in self._st if k[0] == req and (layer is None or k[1] == layer)]:
            del self._st[key]

    def write(self, req: int, layer: int, n_new: int) -> List[int]:
        """Record n_new freshly written positions; return tile indices that just closed."""
        if n_new < 0:
            raise ValueError("n_new must be non-negative")
        st = self.state(req, layer)
        prev = st.L
        st.L += n_new
        out: List[int] = []
        if self.per_write:
            # Only tiles FULLY CONTAINED in this write [prev, prev+n_new) may be quantised, i.e.
            #   k >= ceil(prev / T)  and  (k+1)*T <= prev + n_new
            # Using k0 = prev // T would claim a tile that starts before the write (e.g. a 100-
            # then-128 write would return tile 0 = [0,128), which is not inside [100,228)) - that
            # is a real semantic gap an external reviewer caught.
            lo_k = -(-prev // self.T)
            hi_k = (prev + n_new) // self.T
            out = list(range(lo_k, hi_k))
            if out:
                st.closed = max(st.closed, out[-1] + 1)
            return out
        k = max(st.closed, self.first_tile)
        while (k + 1) * self.T <= st.L - self.recent:
            out.append(k)
            k += 1
        st.closed = k
        return out

    def end(self, req: int, layer: Optional[int] = None) -> Dict[str, int]:
        """Finish a request: the unclosed tail is released, never force-quantised."""
        L = q = pend = 0
        done_tiles = 0
        for (r, l_), st in list(self._st.items()):
            if r != req or (layer is not None and l_ != layer):
                continue
            L += st.L
            fully = st.done["K"] & st.done["V"]
            done_tiles += len(fully)
            q += len(fully) * self.T
            pend += len(self.pending(r, l_))
        self.start(req, layer)
        return {"released_positions": L, "quantised_positions": q,
                "unquantised_tail": L - q, "closed_but_unapplied": pend,
                "fully_applied_tiles": done_tiles}

    # -- validation and bookkeeping --------------------------------------
    def validate(self, req: int, layer: int, tiles: Iterable[int], kv: str = "K") -> None:
        """Raise unless every tile is closed, unprotected, in range and not already done.

        Called BEFORE touching the pool: the previous version validated after the scatter, so a
        duplicate request had already quantised the tile a second time by the time it raised.
        """
        if kv not in KV:
            raise ValueError(f"kv must be one of {KV}")
        st = self.state(req, layer)
        closed = self.closed_tiles(req, layer)
        tiles = list(tiles)
        seen: set = set()
        for k in tiles:
            if k in seen:
                raise RuntimeError(f"tile {k} listed twice in one call")
            seen.add(k)
            if not isinstance(k, int):
                raise TypeError(f"tile index must be int, got {type(k)}")
            if k < self.first_tile:
                raise RuntimeError(
                    f"tile {k} is protected by the sink region (first closable is "
                    f"{self.first_tile})")
            if k >= st.closed:
                raise RuntimeError(f"tile {k} has not closed yet (closed cursor {st.closed})")
            if k not in closed:
                raise RuntimeError(f"tile {k} is not in the closed set")
            if k in st.done[kv]:
                raise RuntimeError(f"tile {k} of request {req} layer {layer} {kv} "
                                   f"quantised twice")

    def mark(self, req: int, layer: int, tiles: Iterable[int], kv: str = "K") -> None:
        tiles = list(tiles)
        self.validate(req, layer, tiles, kv)
        st = self.state(req, layer)
        for k in tiles:
            st.done[kv].add(k)

    def pending(self, req: int, layer: int) -> List[int]:
        """Closed tiles missing K or V. Empty means every closed tile was fully applied."""
        st = self.state(req, layer)
        return sorted(k for k in self.closed_tiles(req, layer)
                      if not (k in st.done["K"] and k in st.done["V"]))

    def coverage(self, req: int, layer: int) -> Dict[str, object]:
        """Numerical coverage from explicit state - never inferred from value changes.

        A tile counts only when both K and V were applied. `c` is over the effective length L,
        so an unclosed tail and any sink/recent region reduce it.
        """
        st = self.state(req, layer)
        fully = st.done["K"] & st.done["V"]
        q = len(fully) * self.T
        if q > st.L:
            raise RuntimeError("quantised count exceeds effective length")
        return {"L": st.L, "quantised": q, "bf16_positions": st.L - q,
                "closed": len(self.closed_tiles(req, layer)),
                "pending": len(self.pending(req, layer)),
                "c": (q / st.L) if st.L else 0.0}


# --------------------------------------------------------------------------- pool side
def quantize_tiles(engine: TileClosure,
                   req: int, layer: int, tiles: Sequence[int],
                   pool: torch.Tensor,
                   pos_to_slot: Sequence[int],
                   quantizer,
                   kv: str = "K",
                   ) -> List[int]:
    """Gather each closing tile by logical position, QDQ it, scatter back in place.

    pool: [num_slots, ...] BF16 tensor. pos_to_slot maps a logical position to a slot; it is
    supplied explicitly because pool slots are an allocation order, NOT a sequence order.

    Validation happens first, so an illegal call (duplicate, unclosed, protected) raises without
    writing anything. The scatter is a stream-ordered op: a caller that then reads the pool on
    the same stream sees quantised values. A Python return alone is not that guarantee - the
    read must be enqueued on the same stream (or after an event wait).
    """
    engine.validate(req, layer, tiles, kv)
    for k in tiles:
        lo, hi = k * engine.T, (k + 1) * engine.T
        slots = torch.as_tensor([pos_to_slot[p] for p in range(lo, hi)],
                                dtype=torch.long, device=pool.device)
        rows = pool.index_select(0, slots)
        q = quantizer(rows, kv)
        pool.index_copy_(0, slots, q.to(pool.dtype))
    engine.mark(req, layer, tiles, kv)
    return list(tiles)


# --------------------------------------------------------------------------- reference quantiser
def asym_rtn(x: torch.Tensor, bits: int, group: int = 32) -> torch.Tensor:
    """Asymmetric round-to-nearest, per `group` along the last dim. For tests and controls."""
    levels = (1 << bits) - 1
    xf = x.float()
    shape = xf.shape
    g = xf.reshape(*shape[:-1], shape[-1] // group, group)
    lo = g.amin(-1, keepdim=True)
    hi = g.amax(-1, keepdim=True)
    sc = ((hi - lo) / levels).clamp(min=1e-8)
    zp = torch.round(-lo / sc).clamp(0, levels)
    qc = torch.round(g / sc + zp).clamp(0, levels)
    out = (qc - zp) * sc
    return out.reshape(*shape).to(x.dtype)


def make_quantizer(bits: int):
    """Keyed by 'K'/'V' so the same callable serves both buffers (see completion semantics)."""
    def q(rows, kv="K"):
        return asym_rtn(rows, bits)
    return q
