"""P7: the storage contract the recent-window (R) read path depends on.

`SGLANG_TILE2_RECENT=R` makes decode read the last R positions straight out of the bf16
landing window instead of the 2-bit codes. That shortcut is only sound if two things hold:

  (a) after a tile closes and is packed, its rows are STILL bit-exact in the ring, and
  (b) a row is only reused once its position has fallen out of the window.

It is worth pinning down because the one bug R did have (`tile2_attn.py` multiplied the whole
chunk output by the packed tile's per-channel V scale, so a ring row came out as V*vA instead
of V; MATH500 84/100 vs 87/100 at R=0) was invisible to every store-level test.

NOTE on the harness: the store rotates K/V itself unless it is told the input is already in the
rotation domain. `test_tile2_store.drive()` builds its driver without setting `rotated`, so it
rotates a SECOND time -- values then come out as H(H(x)) and nothing matches. This file sets
`driver.rotated = True` and feeds pre-rotated rows, which is what the real decode path does
(the wrapper rotates, then calls the store with rotate=False).

Run inside the container:  python3 code/test_recent_window.py
"""
import os
import sys
sys.path.insert(0, "/work/code")
os.environ.setdefault("SGLANG_TILE2_TILES_PER_REQ", "4")

import torch

import test_tile2_store as T


def _write(drv, buf, kw, vw, loc, r2t, pre, n_req=1):
    fb = type("F", (), {})()
    fb.req_to_token_pool = type("R", (), {"req_to_token": r2t})()
    fb.extend_prefix_lens_cpu = [pre] * n_req
    fb.extend_seq_lens_cpu = [len(loc) // n_req] * n_req
    fb.req_pool_indices = torch.tensor(list(range(n_req)), dtype=torch.int64, device=T.DEV)
    for li in range(buf.L):
        drv.step(li, fb, kw, vw, loc)


def main():
    torch.manual_seed(0)
    d = torch.load(T.DUMP, map_location="cpu", weights_only=False)
    k_all = d["prefill"][0][0].float()
    v_all = d["prefill"][0][1].float()
    T.H, T.D = k_all.shape[1], k_all.shape[2]
    DEV, TT = T.DEV, T.T_TILE

    def staged(x):
        return T.rotate_hadamard(x.to(DEV)).to(torch.bfloat16)

    N = TT + 1
    ks, vs = staged(k_all[:N]), staged(v_all[:N])

    buf = T.make_store(size=1024, layers=1, staging=2)
    slots = list(range(1, N + 1))
    r2t = torch.zeros((2, 4096), dtype=torch.int64, device=DEV)
    r2t[0, :N] = torch.tensor(slots, dtype=torch.int64, device=DEV)
    drv = T.Tile2Driver(buf, start_layer=0)
    drv.rotated = True                      # see the NOTE above

    # ---- phase 1: exactly one tile, which closes and is packed ------------------------------
    _write(drv, buf, torch.stack([ks[j] for j in range(TT)]),
           torch.stack([vs[j] for j in range(TT)]),
           torch.tensor(slots[:TT], dtype=torch.int64, device=DEV), r2t, 0)

    ring = buf.staging_k[0]
    exact0 = [p for p in range(TT) if torch.equal(ring[p], ks[p])]
    print(f"P7a closed tile still bit-exact in ring : {len(exact0)}/{TT}")
    assert len(exact0) == TT, f"positions {[p for p in range(TT) if p not in exact0][:8]} not exact"

    packed = int((buf.slot_row[torch.tensor(slots[:TT], device=DEV)].to(torch.int64) >= 0).sum())
    print(f"P7b slot_row marks the tile packed      : {packed}/{TT}")
    assert packed == TT, "tile did not close/pack; the rest of the test would be vacuous"

    # ---- phase 2: one more token; only the row it lands on may change -----------------------
    _write(drv, buf, ks[TT].unsqueeze(0), vs[TT].unsqueeze(0),
           torch.tensor([slots[TT]], dtype=torch.int64, device=DEV), r2t, TT)

    assert torch.equal(ring[0], ks[TT]), "the new position did not land on ring row 0"
    bad = [p for p in range(1, TT) if not torch.equal(ring[p], ks[p])]
    print(f"P7c window intact after one wrap        : {'OK' if not bad else f'{len(bad)} clobbered'}")
    assert not bad, f"positions {bad[:8]} were clobbered before leaving the window"

    print("\nRESULT: PASS  (ring is bit-exact and only reuses rows that aged out)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
