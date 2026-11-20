"""Acceptance (3): boundaries 127/128/129 and 255/256/257 must never seal a partial page.

A page may only be sealed (hence ever become host-valid) when all 128 of its rows were written AND
every local layer wrote its ten fields. The store flushes only when a tile closes, so a partial
page's rows stay in staging and `page_ready_layers` must stay 0 -- this test proves that rather than
assuming it.
"""
import os, sys
os.environ["SGLANG_TILE2_PAGE_BOUND"] = "1"
os.environ.setdefault("SGLANG_TILE2_TILES_PER_REQ", "4")
sys.path.insert(0, "/sgl-workspace/sglang/python")
sys.path.insert(0, "/work/code")
import torch
from sglang.srt.mem_cache.tile2_store import Tile2Buffers, Tile2Driver, write_and_flush

DEV = "cuda"
T, H, D, L = 128, 8, 128, 2
buf = Tile2Buffers(size=4 * T, page_size=T, head_num=H, head_dim=D, layer_num=L,
                   device=DEV, max_staging=8)
buf.rot_domain, buf.unrotate, buf.staging_rotated = True, False, True
mask_all = buf._layers_mask
print("layers=%d layers_mask=0x%x n_tiles=%d" % (L, mask_all, buf.n_tiles))

def feed(n_tokens, page, seed=0):
    """Write `n_tokens` rows that live in device page `page` via the real write path."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    k = torch.randn(n_tokens, H, D, generator=g).to(DEV, torch.bfloat16)
    v = torch.randn(n_tokens, H, D, generator=g).to(DEV, torch.bfloat16)
    r2t = torch.zeros((1, 8192), dtype=torch.int32, device=DEV)
    stg = buf.alloc_staging()
    stg_of, pre_lens, n_writes, rows, keys, loc = [stg], [0], [n_tokens], [0], [0], []
    for j in range(n_tokens):
        loc.append(page * T + j)
    loc = torch.tensor(loc, dtype=torch.int64, device=DEV)
    r2t[0, :n_tokens] = loc.to(torch.int32)
    base = page * T
    # position grid for this request: rows [0..n) are slots base..base+n
    drv = Tile2Driver(buf, start_layer=0)
    for li in range(L):
        write_and_flush(buf, li, stg_of, pre_lens, n_writes, rows, keys, loc, k, v, r2t,
                        drv.tile_of, update_map=True, rotate=False, pending=None,
                        track_positions=False, ridx_of=None)
    return loc

def sealed(page):
    v = int(buf.page_ready_layers[page])
    return v == mask_all, v

# ---- 127: nothing closed -> page must NOT be sealed
buf.slot_row.zero_()
buf.page_ready_layers.zero_()
try:
    feed(127, page=1)
except Exception as e:
    print("127 write raised:", type(e).__name__, str(e)[:60])
ok, v = sealed(1)
print("127 tokens -> page1 ready_layers=0x%02x sealed=%s  %s" % (v, ok, "PASS" if not ok else "FAIL"))
assert not ok

# ---- 128: exactly one page closes -> sealed
feed(128, page=2, seed=1)
ok2, v2 = sealed(2)
print("128 tokens -> page2 ready_layers=0x%02x sealed=%s  %s" % (v2, ok2, "PASS" if ok2 else "FAIL"))
assert ok2

# ---- 129: the 129th row starts a new page and must NOT seal it
feed(129, page=3, seed=2)
ok3, v3 = sealed(3)
ok3b, v3b = sealed(4)
print("129 tokens -> page3 ready=0x%02x sealed=%s ; page4 ready=0x%02x sealed=%s  %s"
      % (v3, ok3, v3b, ok3b, "PASS" if ok3 and not ok3b else "FAIL"))
assert ok3 and not ok3b
print("边界 127/128/129：未满页绝不被 seal —— PASS")
