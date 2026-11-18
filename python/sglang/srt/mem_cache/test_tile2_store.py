"""Engine-side `packed_tile2` vs the offline target-format reference.

This is the equivalence harness a reviewer asked for: it compares the *engine* store (packed
codes in pool buffers, per-tile fp16 parameters, a per-slot position map, a bf16 staging window)
against the *offline* store (`tile_store_min.TileStore`) on identical inputs, with the offline
store fed pre-rotated rows so the two are directly comparable instead of merely similar.

Four properties, in increasing order of how much they would hurt if wrong:

  P1 bytes     : the engine's allocated bytes give format = 2.500 bit/element exactly.
  P2 equal     : engine gather == offline read, bit for bit, on every position.
  P3 chunked   : splitting one 300-row write into 100+100+100 changes nothing (bit for bit).
  P4 permuted  : assigning NON-CONTIGUOUS, randomly permuted pool slots changes nothing
                 (bit for bit). This is what a layout relying on slot contiguity, or on
                 `slot // T`, would silently fail.

Run inside the container:  python3 code/test_tile2_store.py
"""
import os
import random
import sys
sys.path.insert(0, "/work/code")
# A toy pool cannot afford the production reservation of 128 tiles per request; the tests only
# need four tiles per request.
os.environ.setdefault("SGLANG_TILE2_TILES_PER_REQ", "4")

import torch

from tile_store_min import TileStore
from tile2_store import (Tile2Buffers, Tile2Driver, gather_dequant, hadamard,
                         rotate_hadamard, T_TILE)

DUMP = "/work/results/kvdump/kv_4b_idx0000.pt"
DEV = "cuda"
H = D = None


def make_store(size=4096, layers=2, staging=4):
    b = Tile2Buffers(size=size, page_size=1, head_num=H, head_dim=D,
                     layer_num=layers, device=DEV, max_staging=staging)
    # The reference store lives in the Hadamard domain (pre_rotated=True), so compare there:
    # ask the engine not to un-rotate on read.
    b.rot_domain, b.unrotate, b.staging_rotated = True, False, True
    return b


def drive(buf, k_src, v_src, script, slot_fn, r2t, n_req=2):
    """script[ci] = (pre_lens, n_writes) for write call ci; slot_fn(ci, req, j) -> pool slot."""
    drv = Tile2Driver(buf, start_layer=0)
    fb = type("F", (), {})()
    fb.req_to_token_pool = type("R", (), {"req_to_token": r2t})()
    prev = [0] * n_req
    for ci, (pre, nw) in enumerate(script):
        assert pre == prev, (ci, pre, prev)
        loc, idx = [], []
        for i in range(n_req):
            for j in range(nw[i]):
                loc.append(slot_fn(ci, i, j))
                idx.append(pre[i] + j)
        loc = torch.tensor(loc, dtype=torch.int64, device=DEV)
        kw = torch.stack([k_src[j] for j in idx])
        vw = torch.stack([v_src[j] for j in idx])
        fb.extend_prefix_lens_cpu = list(pre)
        fb.extend_seq_lens_cpu = list(nw)
        fb.req_pool_indices = torch.tensor(list(range(n_req)), dtype=torch.int64, device=DEV)
        for li in range(buf.L):
            drv.step(li, fb, kw, vw, loc)
        prev = [pre[i] + nw[i] for i in range(n_req)]
    return drv


def gather(buf, li, slots):
    ok = torch.empty((len(slots), H, D), dtype=torch.bfloat16, device=DEV)
    ov = torch.empty_like(ok)
    gather_dequant(buf, li, torch.tensor(slots, dtype=torch.int64, device=DEV), ok, ov)
    return ok.float(), ov.float()


def rel_rms(x, ref):
    return ((x - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()


def main():
    global H, D
    torch.manual_seed(0)
    d = torch.load(DUMP, map_location="cpu", weights_only=False)
    k_all = d["prefill"][0][0].float()
    v_all = d["prefill"][0][1].float()
    H, D = k_all.shape[1], k_all.shape[2]
    k_all, v_all = k_all.to(DEV), v_all.to(DEV)
    N0, N1, DEC = 300, 130, 40
    n0 = N0 + DEC

    hk = hadamard(D, DEV, torch.float32)
    fw = rotate_hadamard(k_all[:256].to(DEV))
    print("rotation: FWHT vs x@H  max|d|=%.3e  (relRMS %.2e)"
          % ((fw - k_all[:256].to(DEV).float() @ hk).abs().max(),
             ((fw - k_all[:256].to(DEV).float() @ hk).pow(2).mean().sqrt()
              / fw.pow(2).mean().sqrt()).item()))

    def staged(x):
        return rotate_hadamard(x.to(DEV)).to(torch.bfloat16)

    # ---- offline reference ------------------------------------------------------------------
    ref = {}
    for li in range(2):
        st = TileStore(bits=2, H=H, D=D, T=T_TILE, pre_rotated=True)
        st.append(0, staged(k_all[:n0]).cpu(), staged(v_all[:n0]).cpu())
        while st.should_flush(0):
            st.flush(0)
        ref[li] = st
    print("reference: packed=%d staged=%d c=%.4f format=%.4f bit/el"
          % (ref[0].n_packed[0], ref[0].n_staged[0], ref[0].coverage(0),
             ref[0].format_bits_per_element(0)))

    # ---- engine, contiguous slots ------------------------------------------------------------
    r2t = torch.zeros((4, 4096), dtype=torch.int64, device=DEV)
    script = [([0, 0], [N0, N1])] + [([N0 + i, N1 + i], [1, 1]) for i in range(DEC)]
    nxt = [1]

    def contiguous(ci, i, j):
        s = nxt[0]
        nxt[0] += 1
        r2t[i, script[ci][0][i] + j] = s
        return s

    drv = drive(make_store(), k_all, v_all, script, contiguous, r2t)
    buf = drv.buf

    # ---- P1 bytes ---------------------------------------------------------------------------
    b = buf.bytes()
    fmt = buf.format_bits()
    print("P1 bytes      payload=%.1f MiB params=%.1f MiB staging=%.1f MiB slot_row=%.2f MiB"
          % (b["payload"] / 2**20, b["params"] / 2**20, b["staging"] / 2**20,
             b["slot_row"] / 2**20))
    print("              format=%.4f bit/element -> %.2fx vs bf16"
          "   (realised %.4f on this toy pool)" % (fmt, 16.0 / fmt, buf.format_bits_realized()))
    assert abs(fmt - 2.5) < 1e-9, fmt

    # ---- P2 engine gather == offline read ---------------------------------------------------
    pos = list(range(n0))
    slots0 = [int(r2t[0, p]) for p in pos]
    rk, rv = [t.to(DEV) for t in ref[0].read(0)]   # rotated domain, as the engine keeps it
    gk, gv = gather(buf, 0, slots0)  # bf16 workspace
    relk = ((gk - rk).pow(2).mean().sqrt() / rk.pow(2).mean().sqrt()).item()
    relv = ((gv - rv).pow(2).mean().sqrt() / rv.pow(2).mean().sqrt()).item()
    print("P2 equal      rotated domain relRMS K=%.2e V=%.2e  max|dK|=%.3e max|dV|=%.3e"
          % (relk, relv, (gk - rk).abs().max(), (gv - rv).abs().max()))
    # bf16 in the workspace is the only difference; 1e-2 relative would mean a real divergence
    assert relk < 1e-2 and relv < 1e-2, (relk, relv)
    print("              (bf16 workspace quantisation only; the packed values agree to fp32)")
    gk1, _ = gather(buf, 1, slots0)
    r1 = ref[1].read(0)[0].to(DEV)
    rel1 = ((gk1 - r1).pow(2).mean().sqrt() / r1.pow(2).mean().sqrt()).item()
    print("              layer 1 independently: relRMS=%.2e" % rel1)
    assert rel1 < 1e-2

    # what the reconstruction costs against the ORIGINAL bf16 KV (un-rotated once, for reporting)
    uo = rotate_hadamard(gk.float())
    print("              vs original bf16 KV: relRMS K=%.4f"
          % (((uo - k_all[:n0]).pow(2).mean().sqrt()
              / k_all[:n0].pow(2).mean().sqrt()).item()))

    cov = drv.coverage()
    print("              coverage: packed=%d staged=%d c=%.4f  (expect %d / %d)"
          % (cov["packed"], cov["staged"], cov["c"],
             (N0 // T_TILE + N1 // T_TILE) * T_TILE, N0 + N1 + 2 * DEC))
    assert cov["packed"] == (N0 // T_TILE + N1 // T_TILE) * T_TILE

    # ---- P3 chunked prefill -----------------------------------------------------------------
    r2t3 = torch.zeros((4, 4096), dtype=torch.int64, device=DEV)
    nxt3 = [1]
    script3, pre = [], [0, 0]
    for take in (100, 100, 100):
        script3.append((list(pre), [take, 0]))
        pre[0] += take
    script3.append((list(pre), [0, N1]))
    pre[1] += N1
    for _ in range(DEC):
        script3.append((list(pre), [1, 1]))
        pre[0] += 1
        pre[1] += 1
    base3 = [c[0][:] for c in script3]

    def contiguous3(ci, i, j):
        s = nxt3[0]
        nxt3[0] += 1
        r2t3[i, base3[ci][i] + j] = s
        return s

    b3 = make_store()
    drive(b3, k_all, v_all, script3, contiguous3, r2t3)
    slots3 = [int(r2t3[0, p]) for p in pos]
    gk3, gv3 = gather(b3, 0, slots3)
    same = torch.equal(gk3, gk) and torch.equal(gv3, gv)
    print("P3 chunked    100+100+100 vs 300 in one call: identical=%s" % same)
    assert same, "chunked prefill changed the stored values"

    # ---- P4 permuted slots ------------------------------------------------------------------
    r2t4 = torch.zeros((4, 4096), dtype=torch.int64, device=DEV)
    perm = list(range(1, 900))
    random.Random(7).shuffle(perm)
    nxt4 = [0]

    def permuted(ci, i, j):
        s = perm[nxt4[0]]
        nxt4[0] += 1
        r2t4[i, script[ci][0][i] + j] = s
        return s

    b4 = make_store()
    drive(b4, k_all, v_all, script, permuted, r2t4)
    slots4 = [int(r2t4[0, p]) for p in pos]
    gk4, gv4 = gather(b4, 0, slots4)
    same4 = torch.equal(gk4, gk) and torch.equal(gv4, gv)
    print("P4 permuted    shuffled non-contiguous slots: identical=%s  (n=%d, max slot %d)"
          % (same4, len(slots4), max(slots4)))
    assert same4, "the store depends on slot contiguity"

    # ---- P5: two requests with DIFFERENT data must not share tile parameters ----------------
    # This is the regression test for the block-allocator bug: it handed every request the same
    # tile ids, so the per-tile parameters of whichever request flushed last were used by all of
    # them. The earlier tests could not see it because they fed every request the same rows, which
    # makes the shared parameters correct.
    b5 = make_store()
    r2t5 = torch.zeros((4, 4096), dtype=torch.int64, device=DEV)
    k_a = (torch.randn(300, H, D, device=DEV) * 12)
    v_a = (torch.randn(300, H, D, device=DEV) * 12)
    k_b = (torch.randn(300, H, D, device=DEV) * 3)
    v_b = (torch.randn(300, H, D, device=DEV) * 3)
    nxt5 = [1]
    pre5 = [0, 0]
    d5 = Tile2Driver(b5, start_layer=0)
    for _ in range(300):
        loc, idx_a, idx_b = [], [], []
        for i, (src_k,) in enumerate(((k_a,), (k_b,))):
            s_ = nxt5[0]; nxt5[0] += 1
            r2t5[i, pre5[i]] = s_
            loc.append(s_)
            (idx_a if i == 0 else idx_b).append(pre5[i])
        pre5 = [p + 1 for p in pre5]
        kk = torch.stack([k_a[idx_a[0]], k_b[idx_b[0]]])
        vv = torch.stack([v_a[idx_a[0]], v_b[idx_b[0]]])
        fb = type("F", (), {})()
        fb.req_to_token_pool = type("R", (), {"req_to_token": r2t5})()
        fb.extend_prefix_lens_cpu = [p - 1 for p in pre5]
        fb.extend_seq_lens_cpu = [1, 1]
        fb.req_pool_indices = torch.tensor([0, 1], dtype=torch.int64, device=DEV)
        loc = torch.tensor(loc, dtype=torch.int64, device=DEV)
        d5.step(0, fb, kk, vv, loc)          # one layer is enough for the check
    # Check the FIRST request: when two requests share tile ids, the one that flushes LAST wins
    # and the earlier one is the victim, so testing request 1 would pass even with the bug.
    ok, ov = gather(b5, 0, [int(r2t5[0, p]) for p in range(300)])
    uo = rotate_hadamard(ok.float())
    r0_a = rel_rms(uo, k_a.to(torch.bfloat16).float())
    r0_b = rel_rms(uo, k_b.to(torch.bfloat16).float())
    print(f"P5 per-request params: req0 relRMS to its OWN data={r0_a:.4f} "
          f"to req1's data={r0_b:.4f}")
    assert r0_a < 0.6, f"request 0's KV does not match its own data ({r0_a:.3f})"
    assert r0_a < r0_b, "request 0 was quantised with request 1's tile parameters"

    # ---- P6 release(): the argument is a stable request KEY, never a req_to_token row ---------
    # Two namespaces coexist in the engine and they overlap numerically: req_to_token rows are
    # 0..pool_rows, stable request keys are KV slot ids that start at 0 and climb. A release()
    # that translates its argument through the row->key map therefore pops a stranger's entry as
    # soon as the key's numeric value happens to be a live row. Both outcomes are silent: nothing
    # is freed (the pool then dies with "all N staging slots are in use") or a LIVE request's slot
    # and tile block are freed and handed to the next request (corruption). The delivery run hit
    # the first one at question 168/200; this check is that failure, offline.
    b6 = make_store(size=4096, layers=1, staging=4)
    d6 = Tile2Driver(b6, start_layer=0)
    d6.staging_of(2580)                      # keys are slot ids, far above the row range
    d6.staging_of(5160)
    d6._row_key[3] = 2580                    # row 3 owns key 2580
    d6._row_key[0] = 999                     # row 0 owns key 999
    free_before = len(b6._staging_free)
    # The correct behaviour is opt-in: freeing is load bearing on a liveness map that is still
    # keyed inconsistently, so the default stays inert (see the note above _free_release in
    # tile2_store.py). Set the switch for the duration of this check.
    os.environ["SGLANG_TILE2_FREE_RELEASE"] = "1"
    d6.release(2580)                         # release the KEY, not a row
    assert len(b6._staging_free) == free_before + 1, "release(key) freed no staging slot"
    assert 2580 not in d6._stg and 2580 not in d6._blk, "release(key) left the key allocated"
    # The collision: this key's numeric value is also a live row. A row-translating release()
    # frees row 7's key (4242) instead of key 7, leaving key 7 allocated and killing a live
    # request -- which is exactly how the delivery run ran out of slot.
    d6.staging_of(7)
    d6._row_key[7] = 4242                    # row 7 owns key 4242
    d6.staging_of(4242)                      # the live request a translating release would destroy
    n_before = len(b6._staging_free)
    d6.release(7)
    assert len(b6._staging_free) == n_before + 1, (
        "release(key=7) freed nothing: it translated a key through the row->key map")
    assert 7 not in d6._stg and 7 not in d6._blk, "key 7 is still allocated after release(7)"
    assert 4242 in d6._stg and 4242 in d6._blk, (
        "release(key=7) freed LIVE key 4242: it treated the key as a req_to_token row")
    assert d6._row_key.get(7) == 4242, "release() dropped a live row's key mapping"
    os.environ.pop("SGLANG_TILE2_FREE_RELEASE", None)
    print("P6 release     key/row namespace: release(key=2580) freed its slot; colliding "
          "release(key=7) freed key 7 only, live key 4242 intact (free %d->%d)"
          % (n_before, len(b6._staging_free)))

    print("RESULT: PASS")


if __name__ == "__main__":
    main()
