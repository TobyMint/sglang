"""`PackedTile2KVPoolHost`: the HiCache host pool for a packed 2.5-bit KV store.

HiCache's own host pool (`MHATokenToKVPoolHost`) assumes dense bf16 KV: it allocates
`[L, size, H, D]` fp16 buffers and asserts `host_tokens > device_tokens`. Our store keeps NO bf16
KV at all -- a tile is ten tensors (2 code planes + 8 fp16 parameter planes) and 640 B/token/layer
instead of 4096 -- so that pool cannot describe it. Measured before this class existed:
`packed_tile2 + --enable-hierarchical-cache` died in the constructor with
"The host memory should be larger than the device memory with the current protocol".

This class stores the packed bytes verbatim (no dequantise / requantise, so migration is exact) in
one flat byte buffer per page: `uint8[L, n_pages, B]`, where B = 81,920 B for T=H=D=128
(64 KiB codes + 32 KiB parameters) and the field order is `Tile2PageSpec`'s.

Why page-granular and not token-granular: the store runs in PAGE-BOUND mode (tile = slot // T), so
one host page is exactly one device page and the parameters (shared by all T tokens) never have to
be duplicated or reference-counted separately.
"""
import torch

from sglang.srt.mem_cache.memory_pool_host import HostKVCache
from sglang.srt.mem_cache.tile2_page import Tile2PageSpec


class PackedTile2KVPoolHost(HostKVCache):
    """Opaque, page-granular host pool for `packed_tile2`."""

    #: field name -> (device attribute on Tile2Buffers, is it per-tile (True) or per-slot (False))
    FIELD_SRC = [
        ("codes_k", "k_codes", False),
        ("codes_v", "v_codes", False),
        ("k_sc", "k_sc", True),
        ("k_lo", "k_lo", True),
        ("k_A", "k_A", True),
        ("k_B", "k_B", True),
        ("v_sc", "v_sc", True),
        ("v_lo", "v_lo", True),
        ("v_A", "v_A", True),
        ("v_B", "v_B", True),
    ]

    def __init__(self, device_pool, host_to_device_ratio, host_size, page_size, layout,
                 pin_memory: bool = True, device: str = "cpu"):
        buf = getattr(device_pool, "tile2_buf", None)
        if buf is None:
            raise ValueError("PackedTile2KVPoolHost requires a packed_tile2 device pool")
        if not getattr(buf, "page_bound", False):
            raise ValueError(
                "packed_tile2 + HiCache requires page-bound mode (tile = device page). Set "
                "SGLANG_TILE2_PAGE_BOUND=1 and --page-size %d." % buf.T)
        if int(page_size) != int(buf.T):
            raise ValueError(
                "packed_tile2 + HiCache requires --page-size == tile size (%d), got %s: a tile must "
                "be exactly one host page or the parameters would straddle two pages"
                % (buf.T, page_size))
        self.buf = buf
        self.spec = Tile2PageSpec(buf.T, buf.H, buf.D)
        self.layer_num = buf.L
        self.n_local_layers = buf.L
        super().__init__(device_pool, host_to_device_ratio, host_size, page_size, layout,
                         pin_memory, device)
        self.n_pages = self.size // self.page_size
        assert self.kv_buffer.shape == (self.layer_num, self.n_pages, self.spec.page_bytes), (
            self.kv_buffer.shape, (self.layer_num, self.n_pages, self.spec.page_bytes))

    # ---- size / allocation ---------------------------------------------------------------
    def get_size_per_token(self):
        # bytes per token across every local layer = L * B / T (= 23,040 for the 4B model)
        return self.layer_num * self.spec.bytes_per_token()

    def init_kv_buffer(self):
        n_pages = (self.size // self.page_size) if getattr(self, "size", None) else 0
        # `HostKVCache.__init__` sets `self.size` BEFORE calling us, so this is always > 0.
        n_pages = self.size // self.page_size
        return torch.zeros((self.layer_num, n_pages, self.spec.page_bytes), dtype=torch.uint8,
                           pin_memory=self.pin_memory, device=self.device)

    # ---- flat page <-> host buffer ------------------------------------------------------
    def get_flat_data_page(self, index) -> torch.Tensor:
        page = int(index) // self.page_size
        assert 0 <= page < self.n_pages, (index, page, self.n_pages)
        return self.kv_buffer[:, page, :].reshape(-1).clone()

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(self.layer_num * self.spec.page_bytes, dtype=torch.uint8,
                           device=self.device)

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        page = int(index) // self.page_size
        assert 0 <= page < self.n_pages, (index, page, self.n_pages)
        flat = data_page.reshape(-1)
        assert flat.numel() == self.layer_num * self.spec.page_bytes, (
            flat.numel(), self.layer_num * self.spec.page_bytes)
        if flat.dtype != torch.uint8:
            flat = flat.view(torch.uint8)
        self.kv_buffer[:, page, :] = flat.reshape(self.layer_num, self.spec.page_bytes)

    # ---- device <-> host transfers ------------------------------------------------------
    def _field_slice(self, li, name, page):
        """Device tensor slice holding `name` for local layer `li`, page `page`."""
        _, attr, per_tile = next(f for f in self.FIELD_SRC if f[0] == name)
        t = getattr(self.buf, attr)[li]
        if per_tile:
            return t[page]
        return t[page * self.page_size:(page + 1) * self.page_size]

    def backup_from_device_all_layer(self, device_pool, device_indices, host_indices, stream=None):
        """D2H: copy every field of these device pages into host pages. Byte copy, no conversion."""
        host_indices = host_indices.tolist() if torch.is_tensor(host_indices) else host_indices
        device_indices = device_indices.tolist() if torch.is_tensor(device_indices) else device_indices
        with _stream_ctx(stream):
            for d_idx, h_idx in zip(device_indices, host_indices):
                d_page = int(d_idx) // self.page_size
                h_page = int(h_idx) // self.page_size
                for li in range(self.layer_num):
                    for name, _attr, _per_tile in self.FIELD_SRC:
                        off, nbytes, _shape, _dt = self.spec.offsets[name]
                        src = self._field_slice(li, name, d_page).contiguous().view(torch.uint8).reshape(-1)
                        assert src.numel() == nbytes, (name, src.numel(), nbytes)
                        self.kv_buffer[li, h_page, off:off + nbytes].copy_(src, non_blocking=True)

    def load_to_device_per_layer(self, device_pool, device_indices, host_indices, layer_id,
                                 stream=None):
        """H2D for ONE layer. Returns the number of tokens restored (HiCache's completion protocol)."""
        host_indices = host_indices.tolist() if torch.is_tensor(host_indices) else host_indices
        device_indices = device_indices.tolist() if torch.is_tensor(device_indices) else device_indices
        li = int(layer_id) - self.start_layer
        if not (0 <= li < self.layer_num):
            return 0
        with _stream_ctx(stream):
            for d_idx, h_idx in zip(device_indices, host_indices):
                d_page = int(d_idx) // self.page_size
                h_page = int(h_idx) // self.page_size
                for name, _attr, _per_tile in self.FIELD_SRC:
                    off, nbytes, shape, dt = self.spec.offsets[name]
                    dst = self._field_slice(li, name, d_page)
                    dst.copy_(self.kv_buffer[li, h_page, off:off + nbytes].view(dt).reshape(shape),
                              non_blocking=True)
        return len(device_indices)

    # ---- lifecycle ----------------------------------------------------------------------
    def clear(self):
        if not hasattr(self, "kv_buffer") or self.kv_buffer is None:
            return
        self.kv_buffer.zero_()
        super().clear() if hasattr(super(), "clear") else None

    def report(self) -> str:
        return ("[TILE2-HOST] %s pages=%d page_size=%d page_bytes=%d (%.0f KiB) "
                "bytes_per_token=%d (all %d layers) buffer=%.2f GB pinned=%s"
                % (type(self).__name__, self.n_pages, self.page_size, self.spec.page_bytes,
                   self.spec.page_bytes / 1024, self.get_size_per_token(), self.layer_num,
                   self.kv_buffer.nelement() / 1e9, self.pin_memory))


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _stream_ctx(stream):
    """`stream` is sometimes a torch Stream and sometimes just a device string (measured:
    HiCache passed 'cuda', and `torch.cuda.stream("cuda")` raises
    `'str' object has no attribute 'device'`). Only wrap a real Stream."""
    if isinstance(stream, torch.cuda.Stream):
        return torch.cuda.stream(stream)
    return _nullcontext()
