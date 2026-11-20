"""Goal acceptance (2): offload -> clobber the device page -> onload to a DIFFERENT page id
must be BIT-EXACT, for every layer.

Run in the container:  SGLANG_TILE2_PAGE_BOUND=1 python3 /tmp/test_pagebound.py
"""
import os, sys
os.environ["SGLANG_TILE2_PAGE_BOUND"] = "1"
os.environ.setdefault("SGLANG_TILE2_TILES_PER_REQ", "4")
sys.path.insert(0, "/sgl-workspace/sglang/python")
sys.path.insert(0, "/work/code")
import torch
from sglang.srt.mem_cache.tile2_store import Tile2Buffers, gather_dequant_torch
from sglang.srt.mem_cache.tile2_page import Tile2PageSpec
from sglang.srt.mem_cache.tile2_pool_host import PackedTile2KVPoolHost

DEV = "cuda"
T, H, D, L = 128, 8, 128, 2
NPAGES_REQ = 4
SIZE = NPAGES_REQ * T
buf = Tile2Buffers(size=SIZE, page_size=T, head_num=H, head_dim=D, layer_num=L,
                   device=DEV, max_staging=2)
NPAGES = buf.n_tiles          # the pool adds one padded page (page_size extra slots)
assert buf.page_bound and NPAGES == NPAGES_REQ + 1, (buf.page_bound, NPAGES)
print("page_bound=%s n_tiles=%d n_slots=%d" % (buf.page_bound, buf.n_tiles, buf.slots))

class FakePool:
    pass
pool = FakePool()
pool.tile2_buf = buf
pool.start_layer, pool.end_layer = 0, L
pool.layer_num = L
pool.size = SIZE
pool.store_dtype = torch.bfloat16

spec = Tile2PageSpec(T, H, D)
host = PackedTile2KVPoolHost(pool, host_to_device_ratio=4, host_size=0, page_size=T,
                             layout="layer_first", pin_memory=False, device="cpu")
print(host.report())

torch.manual_seed(0)
P0, P1 = 0, 2                      # write page 0, later restore into page 2 (different id)
slots0 = torch.arange(P0 * T, (P0 + 1) * T, dtype=torch.long, device=DEV)
slots1 = torch.arange(P1 * T, (P1 + 1) * T, dtype=torch.long, device=DEV)

# ---- fill page 0 with random packed content (this is what `_flush_commit` would write) ----
def fill(page, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        buf.k_codes[:, page * T:(page + 1) * T].copy_(
            torch.randint(0, 256, (L, T, H, D // 4), generator=g, dtype=torch.uint8).to(DEV))
        buf.v_codes[:, page * T:(page + 1) * T].copy_(
            torch.randint(0, 256, (L, T, H, D // 4), generator=g, dtype=torch.uint8).to(DEV))
        for name, attr, per_tile in PackedTile2KVPoolHost.FIELD_SRC:
            if not per_tile:
                continue
            t = getattr(buf, attr)
            shape = t[0, page].shape
            t[:, page].copy_(torch.randn(shape, generator=g).half().expand(L, *shape).to(DEV))
        # identity mapping, as page-bound mode produces
        buf.slot_row[page * T:(page + 1) * T] = torch.arange(page * T, (page + 1) * T, device=DEV)

fill(P0, 0)

def read(page):
    ks = torch.empty(T, H, D, dtype=torch.bfloat16, device=DEV)
    vs = torch.empty_like(ks)
    sl = torch.arange(page * T, (page + 1) * T, dtype=torch.long, device=DEV)
    outs = []
    for li in range(L):
        gather_dequant_torch(buf, li, sl, ks, vs)
        outs.append((ks.clone(), vs.clone()))
    return outs

ref = read(P0)

# ---- offload page 0 to host page 0 ----
host.backup_from_device_all_layer(pool, host_indices=[0], device_indices=[P0 * T])
print("offload done; host page bytes nonzero:", int(host.kv_buffer[:, 0].ne(0).any()))

# ---- clobber page 0 AND page 1, so a stale device copy cannot masquerade as a restore ----
fill(P0, 111)
fill(1, 222)                        # page 1 gets different content
after_clobber = read(P0)
assert any(not torch.equal(a, b) for a, b in zip(ref[0], after_clobber[0])), "clobber had no effect"

# ---- onload host page 0 into device page 2 (DIFFERENT id) ----
for li in range(L):
    n = host.load_to_device_per_layer(pool, host_indices=[0], device_indices=[P1 * T],
                                     layer_id=li + pool.start_layer)
    assert n == 1
buf.slot_row[P1 * T:(P1 + 1) * T] = torch.arange(P1 * T, (P1 + 1) * T, device=DEV)

got = read(P1)
ok = True
for li in range(L):
    for what, a, b in (("K", got[li][0], ref[li][0]), ("V", got[li][1], ref[li][1])):
        same = torch.equal(a, b)
        if not same:
            ok = False
            d = (a.float() - b.float()).abs().max().item()
            print("  layer %d %s MISMATCH max|d|=%.3e" % (li, what, d))
print("验收 (2) 逐位往返（换 page id）:", "PASS" if ok else "FAIL")
assert ok

# ---- host flat page round-trip through set/get must be byte-identical too ----
flat = host.get_flat_data_page(0)
host.set_from_flat_data_page(2 * T, flat)          # copy into host page 2
assert torch.equal(host.kv_buffer[:, 2], host.kv_buffer[:, 0])
print("host flat page 复制逐位一致: PASS")
print("ALL PASS")
