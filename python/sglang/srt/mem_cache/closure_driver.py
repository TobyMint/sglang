"""Driver for the tile-closure lifecycle on a live BF16 pool.

`code/tile_closure.py` decides *which* tiles have closed. This module is the piece that turns
that decision into work on the real cache: for each closing tile it gathers the K and V rows of
that tile from the pool via the request's position->slot map, runs the QDQ, and scatters the
result back in place.

Design constraints (from report/实验方案.md and the review):

  * positions come from `req_to_token[req, position]`, never from slot contiguity;
  * the gather/QDQ/scatter happens after the KV write for the current step and before the
    attention read, on the same stream, so the read sees quantised values;
  * a tile is done only when both K and V have been applied;
  * validation happens before any pool mutation (a duplicate call must not write).

The driver is deliberately written against plain tensors (`req_to_token`, `seq_lens`) so it can be
unit-tested without a server. The sglang wiring only has to supply those tensors per layer.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

try:                                              # inside the sglang package
    from .tile_closure import TileClosure, asym_rtn
except ImportError:                               # standalone, from /work/code
    from tile_closure import TileClosure, asym_rtn


class ClosureDriver:
    """Applies tile closure to one layer's K/V buffers for a whole batch of requests."""

    def __init__(self, T: int = 128, sink: int = 0, recent: int = 0, bits: int = 2,
                 group: int = 32, tile_quant=None, per_write: bool = False,
                 fault: Optional[str] = None) -> None:
        self.engine = TileClosure(T=T, sink=sink, recent=recent, per_write=per_write)
        self.bits = bits
        self.group = group
        self.events: List[Dict[str, object]] = []      #: (req, layer, tile, kv) applied, in order
        self._H: dict = {}                             #: cached Hadamard per head_dim
        #: TEST ONLY: inject a known defect so the auditor can be shown to catch it.
        #:   "skip"     - record the tile as applied but leave the pool untouched
        #:   "double"   - apply the quantiser twice to the same rows
        #:   "misalign" - apply to the tile shifted by 7 positions
        self.fault = fault
        #: optional joint quantiser quant(k_rows, v_rows) -> (k_hat, v_hat) for one whole tile.
        #: KVarN needs this because its K and V axes share the same VarN normalisation but use
        #: different scale-sharing axes, so a per-buffer callable is not expressive enough.
        self.tile_quant = tile_quant
        self._last_len: Dict[tuple, int] = {}
        self._slot0: Dict[tuple, int] = {}      #: position-0 slot (fallback witness only)
        self._gen: Dict[tuple, int] = {}        #: explicit generation id per (request, layer)

    # -- quantiser -------------------------------------------------------
    def _hadamard(self, d: int, device) -> torch.Tensor:
        """Sylvester Hadamard, normalised by 1/sqrt(d) - same construction as kv_qdq._hadamard.

        The frozen config says all four arms use "per-head Hadamard + asymmetric RTN". The first
        deployed driver quantised without the rotation, which is a recipe error (all arms shared
        it, so the contrasts were still internally consistent, but it was not the frozen recipe).
        """
        key = (d, str(device))
        h = self._H.get(key)
        if h is None:
            h = torch.ones(1, 1, device=device, dtype=torch.float32)
            while h.shape[0] < d:
                h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
            if h.shape[0] != d:
                raise RuntimeError(f"head_dim {d} is not a power of two")
            h = (h / (d ** 0.5)).to(device)
            self._H[key] = h
        return h

    def _q(self, rows: torch.Tensor) -> torch.Tensor:
        """Rotate -> asymmetric RTN -> rotate back (the frozen recipe, not plain RTN).

        I4LIFE_NO_ROT=1 disables the rotation. That switch exists only for the recipe-pairing
        control (rotation on/off at the same nominal width) and is what the "no-rotation" condition
        in report/四臂结果_v1.md uses. Without it both conditions would be identical.
        """
        import os as _os
        if _os.environ.get("I4LIFE_NO_ROT") == "1":
            return asym_rtn(rows, self.bits, self.group)
        h = self._hadamard(rows.shape[-1], rows.device)
        x = rows.float() @ h
        return (asym_rtn(x, self.bits, self.group) @ h).to(rows.dtype)

    # -- batch step ------------------------------------------------------
    def step(self,
             layer: int,
             req_ids: Sequence[int],
             seq_lens: Sequence[int],
             req_to_token: torch.Tensor,
             k_pool: torch.Tensor,
             v_pool: torch.Tensor,
             req_rows: Optional[Sequence[int]] = None,
             gens: Optional[Sequence[int]] = None,
             ) -> Dict[int, List[int]]:
        """Process one layer of one forward pass.

        layer        : layer index (the engine keeps state per (request, layer))
        req_ids      : logical request id per batch row (stable across steps)
        seq_lens     : effective cache length per batch row AFTER this step's write
        req_to_token : [num_req_rows, max_len] position -> pool slot
        req_rows     : row of req_to_token to use per batch row; defaults to req_ids
        k_pool/v_pool: [num_slots, heads, head_dim] BF16 buffers for this layer

        Returns {request_id: [tile indices applied in this step]}.
        """
        out: Dict[int, List[int]] = {}
        rows = list(req_rows) if req_rows is not None else list(req_ids)
        for rid, L, rrow in zip(req_ids, seq_lens, rows):
            L = int(L)
            if gens is not None:
                # Authoritative generation id from ReqToTokenPool.alloc/free. When it is supplied
                # it is the ONLY reset criterion - the length/slot fallbacks are skipped so the
                # two mechanisms cannot disagree.
                g = int(gens[list(req_ids).index(rid)])
                gprev = self._gen.get((rid, layer))
                self._gen[(rid, layer)] = g
                if gprev is not None and g != gprev:
                    self.engine.start(rid, layer)
                    self._last_len[(rid, layer)] = 0
                prev = self._last_len.get((rid, layer), 0)
            else:
                # Fallback when no id is available (offline tests). Length alone is insufficient
                # (old request ends at 256, new one starts at 257) and so is the position-0 slot
                # (it can be reallocated to the same value), hence both are checked.
                prev = self._last_len.get((rid, layer), 0)
                slot0 = int(req_to_token[rrow, 0])
                prev_slot0 = self._slot0.get((rid, layer))
                self._slot0[(rid, layer)] = slot0
                if prev > 0 and (L <= prev or (prev_slot0 is not None and slot0 != prev_slot0)):
                    self.engine.start(rid, layer)
                    prev = 0
            n_new = L - prev
            # instrument: a write where the loose (prev//T) and strict (ceil(prev/T)) per-write
            # definitions would differ. Zero here means the semantic fix cannot change results.
            if getattr(self, "count_unaligned", False) and (prev % self.engine.T) and n_new >= self.engine.T:
                self.unaligned_writes = getattr(self, "unaligned_writes", 0) + 1
            self._last_len[(rid, layer)] = L
            closed = self.engine.write(rid, layer, n_new)
            applied: List[int] = []
            for tile in closed:
                lo, hi = tile * self.engine.T, (tile + 1) * self.engine.T
                if hi > L:
                    raise RuntimeError(f"tile {tile} closes beyond L={L}")
                slots = req_to_token[rrow, lo:hi].to(torch.long)
                if slots.numel() != self.engine.T:
                    raise RuntimeError(f"short position map for tile {tile}")
                # validate both tensors before touching either buffer
                self.engine.validate(rid, layer, [tile], kv="K")
                self.engine.validate(rid, layer, [tile], kv="V")
                if self.fault == "nosched" and tile == 1:
                    # silently drop the tile: no write, no bookkeeping. This is the case the
                    # first version of the audit test could NOT catch (it deleted bookkeeping
                    # after a correct write, which the pool-level audit rightly passes).
                    applied.append(tile)
                    continue
                apply_slots = slots
                if self.fault == "misalign":
                    shifted = lo + 7
                    apply_slots = req_to_token[rrow, shifted:shifted + self.engine.T].to(torch.long)
                k_src = k_pool.index_select(0, apply_slots)
                v_src = v_pool.index_select(0, apply_slots)
                if self.tile_quant is not None:
                    k_hat, v_hat = self.tile_quant(k_src, v_src)
                else:
                    k_hat, v_hat = self._q(k_src), self._q(v_src)
                if self.fault == "skip":
                    self.engine.mark(rid, layer, [tile], kv="K")
                    self.engine.mark(rid, layer, [tile], kv="V")
                    self.events.append({"req": rid, "layer": layer, "tile": tile,
                                        "slots": slots.tolist(), "kv": "SKIPPED"})
                    applied.append(tile)
                    continue
                if self.fault == "double":
                    k_hat, v_hat = (self.tile_quant(k_hat, v_hat) if self.tile_quant is not None
                                    else (self._q(k_hat), self._q(v_hat)))
                k_pool.index_copy_(0, apply_slots, k_hat.to(k_pool.dtype))
                v_pool.index_copy_(0, apply_slots, v_hat.to(v_pool.dtype))
                for kv, sl in (("K", slots), ("V", slots)):
                    self.engine.mark(rid, layer, [tile], kv=kv)
                    self.events.append({"req": rid, "layer": layer, "tile": tile,
                                        "slots": sl.tolist(), "kv": kv})
                applied.append(tile)
            out[rid] = applied
        return out

    # -- reporting -------------------------------------------------------
    def finish(self, req_id: int) -> Dict[str, int]:
        for key in [k for k in self._last_len if k[0] == req_id]:
            del self._last_len[key]
        for key in [k for k in self._slot0 if k[0] == req_id]:
            del self._slot0[key]
        for key in [k for k in self._gen if k[0] == req_id]:
            del self._gen[key]
        return self.engine.end(req_id)

    def coverage(self, req_id: int, layer: int) -> Dict[str, object]:
        return self.engine.coverage(req_id, layer)

    def pending(self, req_id: int, layer: int) -> List[int]:
        return self.engine.pending(req_id, layer)
