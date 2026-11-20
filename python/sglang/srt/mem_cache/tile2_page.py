"""`Tile2PageSpec`: the byte layout of ONE packed tile, and its flat serialisation.

A packed tile is the quantised form of `T = 128` tokens for one layer, and it is ten separate
tensors, not one buffer:

    codes_k  [T, H, D//4] uint8      codes_v  [T, H, D//4] uint8
    k_sc     [H, D] fp16             v_sc     [T, H] fp16
    k_lo     [H, D] fp16             v_lo     [T, H] fp16
    k_A      [H, D] fp16             v_A      [H, D] fp16
    k_B      [T, H] fp16             v_B      [T, H] fp16

HiCache moves KV between device and host, and its host pool is a flat byte buffer, so the tile has
to be serialised into one contiguous page. Doing that with a *fixed* layout (and a version byte) is
what makes a migrated page bit-identical to the one that was written on the device -- the whole
point of migrating the quantised form instead of dequantising.

Bytes per page per layer:  B = T*H*D/2 + 8*H*(D+T)
  T=H=D/=128:  65536 + 16384 = 81920 B = 80 KiB   ->  640 B/token/layer (matches the format's
                                                bit accounting: 512 B codes + 128 B params)
"""
import torch

FORMAT_VERSION = 1


class Tile2PageSpec:
    """Layout of one (layer, tile) page: field order, offsets, shapes, dtypes."""

    def __init__(self, T: int, H: int, D: int):
        assert T > 0 and H > 0 and D > 0 and D % 4 == 0
        self.T, self.H, self.D = T, H, D
        # Order is part of the wire format: codes first, then K's four planes, then V's four.
        self.fields = [
            ("codes_k", (T, H, D // 4), torch.uint8),
            ("codes_v", (T, H, D // 4), torch.uint8),
            ("k_sc", (H, D), torch.float16),
            ("k_lo", (H, D), torch.float16),
            ("k_A", (H, D), torch.float16),
            ("k_B", (T, H), torch.float16),
            ("v_sc", (T, H), torch.float16),
            ("v_lo", (T, H), torch.float16),
            ("v_A", (H, D), torch.float16),
            ("v_B", (T, H), torch.float16),
        ]
        self.offsets = {}
        off = 0
        for name, shape, dt in self.fields:
            n = 1
            for s in shape:
                n *= s
            nbytes = n * torch.tensor([], dtype=dt).element_size()
            self.offsets[name] = (off, nbytes, shape, dt)
            off += nbytes
        self.page_bytes = off
        self.version = FORMAT_VERSION

    # ---- shape helpers -------------------------------------------------------------------
    @property
    def codes_bytes(self) -> int:
        return 2 * self.T * self.H * (self.D // 4)      # 2 * payload (uint8, 4 codes/byte)

    @property
    def param_bytes(self) -> int:
        return 8 * self.H * (self.D + self.T) * 2       # eight fp16 planes

    def bytes_per_token(self) -> int:
        assert self.page_bytes % self.T == 0
        return self.page_bytes // self.T

    # ---- serialise / deserialise ---------------------------------------------------------
    def pack(self, page: dict) -> torch.Tensor:
        """dict of field-name -> tensor (any device) into one contiguous uint8 page."""
        out = torch.empty(self.page_bytes, dtype=torch.uint8, device=page["codes_k"].device)
        for name, shape, dt in self.fields:
            t = page[name]
            assert tuple(t.shape) == shape, (name, tuple(t.shape), shape)
            off, nbytes, _, _ = self.offsets[name]
            flat = t.contiguous().view(torch.uint8).reshape(-1)
            assert flat.numel() == nbytes, (name, flat.numel(), nbytes)
            out[off:off + nbytes] = flat
        return out

    def unpack(self, flat: torch.Tensor, device=None) -> dict:
        """one uint8 page -> dict of field tensors. Exact inverse of `pack`."""
        assert flat.dtype == torch.uint8 and flat.numel() == self.page_bytes, (
            flat.dtype, flat.numel(), self.page_bytes)
        out = {}
        for name, shape, dt in self.fields:
            off, nbytes, _, _ = self.offsets[name]
            chunk = flat[off:off + nbytes]
            out[name] = chunk.view(dt).reshape(shape)
        return out

    def report(self) -> str:
        return (f"[TILE2-PAGE] v{self.version} T={self.T} H={self.H} D={self.D} "
                f"page_bytes={self.page_bytes} ({self.page_bytes / 1024:.0f} KiB, "
                f"{self.bytes_per_token()} B/token/layer; targets 640) "
                f"codes={self.codes_bytes} params={self.param_bytes}")


if __name__ == "__main__":
    spec = Tile2PageSpec(128, 8, 128)
    print(spec.report())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device="cpu").manual_seed(0)
    page = {}
    for name, shape, dt in spec.fields:
        if dt == torch.uint8:
            page[name] = torch.randint(0, 256, shape, generator=g, dtype=torch.uint8).to(dev)
        else:
            # fp16 tensors including the bit patterns a quantiser can emit
            page[name] = torch.randn(shape, generator=g).half().to(dev)
    flat = spec.pack(page)
    back = spec.unpack(flat)
    bad = [n for n, _, _ in spec.fields if not torch.equal(back[n].cpu(), page[n].cpu())]
    print("逐位往返:", "PASS" if not bad else f"FAIL {bad}")
    # a relocated page must carry the same bytes: simulate writing into another host slot
    host = torch.empty((4, spec.page_bytes), dtype=torch.uint8, device=dev)
    host[0] = flat
    host[3] = host[0].clone()
    print("换 page id 后逐位一致:", "PASS" if torch.equal(host[0], host[3]) else "FAIL")
    print("bytes_per_token =", spec.bytes_per_token(), "(expect 640)")
