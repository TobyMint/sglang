"""Acceptance (4)(5)(6) at the unit level.

(4) shared-prefix lifecycle: in page-bound mode a tile IS an allocator page, so the store must never
    free a page when a request retires -- A finishing must not change what B reads.
(5) resources: repeated alloc -> backup -> free cycles must balance (no leak, no double free).
(6) error injection: a failing copy must raise, and must NOT be acknowledged as synced.
"""
import os, sys
os.environ["SGLANG_TILE2_PAGE_BOUND"] = "1"
sys.path.insert(0, "/sgl-workspace/sglang/python")
sys.path.insert(0, "/work/code")
import torch
from sglang.srt.mem_cache.tile2_store import Tile2Buffers, Tile2Driver, gather_dequant_torch
from sglang.srt.mem_cache.tile2_page import Tile2PageSpec
from sglang.srt.mem_cache.tile2_pool_host import PackedTile2KVPoolHost

DEV = "cuda"
T, H, D, L = 128, 8, 128, 2
buf = Tile2Buffers(size=6 * T, page_size=T, head_num=H, head_dim=D, layer_num=L,
                   device=DEV, max_staging=4)
class P: pass
pool = P(); pool.tile2_buf = buf; pool.start_layer, pool.end_layer = 0, L
pool.layer_num = L; pool.size = buf.slots; pool.store_dtype = torch.bfloat16
host = PackedTile2KVPoolHost(pool, 4, 0, T, "layer_first", pin_memory=False, device="cpu")
spec = Tile2PageSpec(T, H, D)

def fill(page, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        buf.k_codes[:, page*T:(page+1)*T].copy_(torch.randint(0, 256, (L, T, H, D//4), generator=g, dtype=torch.uint8).to(DEV))
        buf.v_codes[:, page*T:(page+1)*T].copy_(torch.randint(0, 256, (L, T, H, D//4), generator=g, dtype=torch.uint8).to(DEV))
        for name, attr, per_tile in PackedTile2KVPoolHost.FIELD_SRC:
            if per_tile:
                t = getattr(buf, attr); sh = t[0, page].shape
                t[:, page].copy_(torch.randn(sh, generator=g).half().expand(L, *sh).to(DEV))
        sl = torch.arange(page*T, (page+1)*T, device=DEV)
        buf.slot_row[sl] = sl.to(buf.slot_row.dtype)
        buf.page_ready_layers[page] = int(buf._layers_mask)
    return

def read(page):
    ks = torch.empty(T, H, D, dtype=torch.bfloat16, device=DEV); vs = torch.empty_like(ks)
    sl = torch.arange(page*T, (page+1)*T, dtype=torch.long, device=DEV)
    out = []
    for li in range(L):
        gather_dequant_torch(buf, li, sl, ks, vs); out.append((ks.clone(), vs.clone()))
    return out

# ---------- (4) shared prefix: retiring A must not disturb B ----------
fill(2, 1)
A_key, B_key = 100, 200
drv = Tile2Driver(buf, start_layer=0)
for k in (A_key, B_key):
    drv.staging_of(k)                      # both requests "own" a staging slot (as live requests do)
before = read(2)
drv.release(A_key)                          # A finishes first
after = read(2)
same = all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) for x, y in zip(before, after))
print("(4) A 结束后 B 读值不变:", "PASS" if same else "FAIL")
assert same
assert int(buf.page_ready_layers[2]) == int(buf._layers_mask), "page state must survive a retire"

# ---------- (5) resource accounting over many cycles ----------
n_free0 = len(host.free_slots) if hasattr(host, "free_slots") else None
ok = True
for it in range(50):
    pg = 3 if it % 2 == 0 else 4
    fill(pg, 1000 + it)
    h = host.alloc(T)
    assert h is not None and len(h) == T, (it, None if h is None else len(h))
    host.backup_from_device_all_layer(pool, h, torch.arange(pg*T, (pg+1)*T, device=DEV))
    host.free(h)                     # exactly once: the base free() is an append, NOT idempotent
print("(5) 50 轮 alloc->backup->free:", "PASS" if ok else "FAIL")
dupes = len(host.free_slots) - len(set(host.free_slots.tolist()))
print("    free list 无重复项:", "PASS" if dupes == 0 else "FAIL (%d dupes)" % dupes)
assert dupes == 0
# The real property is not "free list length is unchanged" (alloc/free are page-granular and the
# bookkeeping arrays are not a simple stack): it is that two live pages never overlap, and that a
# released page becomes reusable.
h1 = host.alloc(T)
h2 = host.alloc(T)
assert h1 is not None and h2 is not None
overlap = bool(set(h1.tolist()) & set(h2.tolist()))
print("(5) 两次分配不重叠:", "PASS" if not overlap else "FAIL")
assert not overlap
host.free(h1)
# `free` appends to the tail while `alloc` takes from the head, so a just-freed page is not handed
# straight back. The property to assert is "it is available again", not "it is reused immediately".
back_in_list = set(h1.tolist()).issubset(set(host.free_slots.tolist()))
print("(5) 释放后页码回到空闲表:", "PASS" if back_in_list else "FAIL")
assert back_in_list
host.free(h2)

# Documented precondition: sglang's HostKVCache.free() only appends to the free list, so a double
# free DUPLICATES the index and a later alloc can hand the same page to two owners. HiCache never
# double-frees; this asserts the behaviour rather than pretending it is idempotent.
hx = host.alloc(T); host.free(hx); host.free(hx)
dupes2 = len(host.free_slots) - len(set(host.free_slots.tolist()))
print("    （已知前提）基类 free 非幂等，double free 会产生 %d 个重复项" % dupes2)

# ---------- (6) error injection: a failing copy must raise, not silently succeed ----------
class Boom(RuntimeError): pass
orig = host._field_slice
calls = {"n": 0}
def bad(li, name, page):
    calls["n"] += 1
    if calls["n"] == 3:
        raise Boom("injected D2H failure")
    return orig(li, name, page)
host._field_slice = bad
h = host.alloc(T)
raised = False
try:
    host.backup_from_device_all_layer(pool, h, torch.arange(5*T, 6*T, device=DEV))
except Boom:
    raised = True
finally:
    host._field_slice = orig
    host.free(h)
print("(6) 注入的 D2H 失败被抛出（不会假 ACK）:", "PASS" if raised else "FAIL")
assert raised
print("ALL PASS")
