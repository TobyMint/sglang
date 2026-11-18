"""OSCAR with the *query*-covariance rotation, implemented at the attention call site.

Why a second implementation exists
---------------------------------
OSCAR (arXiv 2605.17757) builds its rotation from the QUERY covariance,

    R_K = U_Q . H_had . P_br,   C_Q = (1/N) sum_n q_n^T q_n,

and the whole "INT2 is nearly lossless" claim rests on it: our write-time shim
(`kv_qdq.py::_oscar2`) has no query available and had to substitute the K covariance,
which measured *worse* than plain Hadamard (0.371 vs 0.360 relative reconstruction error)
and collapsed GSM8K to 8/100.

`RadixAttention.forward(q, k, v, forward_batch)` is the first place where q, k and v for the
new tokens are all present, so the paper's construction becomes implementable there.
The values we hand to the backend are already quantize-dequantized, so the KV pool stores
exactly what OSCAR would have stored (the pool-level shim must stay disabled for this recipe).

Signature notes
---------------
q : [T, H_q,  D]   (H_q = 32 for Qwen3-4B)
k : [T, H_kv, D]   (H_kv = 8,  GQA: H_q / H_kv = 4 queries per KV head)
v : [T, H_kv, D]
"""
from __future__ import annotations

import atexit
import json
import math
import os
from typing import Dict

import torch

# OSCAR defaults (arXiv 2605.17757): sink S=64, recent W=256 kept in BF16.
N_SINK = 64
RECENT = 256
GROUP = 64          # quantisation group along the channel dimension
RHO_K = 0.96        # pre-quantisation clip percentile for K
RHO_V = 0.92        # ... and for V
MIN_CAL_TOKENS = 256
STRICT = os.environ.get("QDQ_STRICT", "0") == "1"   # raise instead of silently falling back
ROT_SAVE = os.environ.get("QDQ_ROT_SAVE", "")        # dir: dump calibrated rotations here
ROT_LOAD = os.environ.get("QDQ_ROT_LOAD", "")        # dir: load frozen rotations instead of calibrating
STATS_PATH = os.environ.get("QDQ_STATS", "")         # json: per-layer coverage counters


def _hadamard(d: int, device, dtype) -> torch.Tensor:
    h = torch.ones(1, 1, device=device, dtype=torch.float32)
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    assert h.shape[0] == d, f"head_dim {d} is not a power of two"
    return (h / math.sqrt(d)).to(dtype)


def _bitrev(d: int, device) -> torch.Tensor:
    bits = int(math.log2(d))
    idx = torch.arange(d)
    rev = torch.zeros_like(idx)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    return rev.to(device)


def make_br_perm_matrix(eigvals: torch.Tensor) -> torch.Tensor:
    """Verbatim port of OSCAR's rotation/compute_kv_rotation.py::make_br_perm_matrix.

    NOTE: this is NOT a plain bit-reversal permutation of the identity. It sorts the
    eigen-directions by descending eigenvalue first, then applies the bit-reversal *index
    map* to them. Getting this wrong is silent: any composition of orthogonal factors is
    still orthogonal, so only the importance-uniformity identity catches it.
    """
    d = eigvals.shape[-1]
    dev = eigvals.device
    sorted_idx = torch.argsort(eigvals, descending=True)
    br = _bitrev(d, dev)
    perm = torch.zeros(d, dtype=torch.long, device=dev)
    perm[br] = sorted_idx
    return torch.eye(d, dtype=torch.float64, device=dev)[:, perm]


def compose_rotation(rotation: torch.Tensor, eigvals: torch.Tensor,
                     hadamard: torch.Tensor) -> torch.Tensor:
    """OSCAR's default composition `r_h_pbr` = U @ H_had @ P_br (paper :258)."""
    return rotation @ hadamard.float() @ make_br_perm_matrix(eigvals)


def _rtn_group(x: torch.Tensor, bits: int = 2) -> torch.Tensor:
    """Affine asymmetric RTN, scale+zero per token and per group along the last dim."""
    qmax = (1 << bits) - 1
    lo = x.amin(dim=-1, keepdim=True)
    hi = x.amax(dim=-1, keepdim=True)
    sc = ((hi - lo) / qmax).clamp(min=1e-8)
    q = torch.round((x - lo) / sc).clamp_(0, qmax)
    return q.mul_(sc).add_(lo)


def _clip_pct(x: torch.Tensor, rho: float) -> torch.Tensor:
    s, _ = x.abs().sort(dim=-1)
    d = s.shape[-1]
    k = min(d - 1, int(rho * d))          # author's index: int(rho*D), not int(rho*(D-1))
    tau = s[..., k:k + 1]
    return x.clamp(-tau, tau)


class KVQDQAttn:
    """Per-layer query-covariance rotation + clipped affine INT2, OSCAR's recipe."""

    def __init__(self, method: str = "oscar2a", sink: int = N_SINK, recent: int = RECENT,
                 group: int = GROUP):
        self.method = method
        self.n_sink = sink
        self.recent = recent
        env_s, env_r = os.environ.get("QDQ_SINK"), os.environ.get("QDQ_RECENT")
        if env_s is not None:
            self.n_sink = int(env_s)
        if env_r is not None:
            self.recent = int(env_r)
        self.G = group
        self.state: Dict[int, dict] = {}
        self.stats: Dict[str, int] = {}
        atexit.register(self.dump_stats)
        self._H: Dict[tuple, torch.Tensor] = {}

    # -- helpers ---------------------------------------------------------
    def _had(self, d, device, dtype):
        key = (d, str(device))
        if key not in self._H:
            self._H[key] = _hadamard(d, device, dtype)
        return self._H[key]

    @staticmethod
    def _rot_from(cov: torch.Tensor, hk: torch.Tensor) -> torch.Tensor:
        """Author-faithful: U = eigh(cov) eigenvectors (ascending, NOT flipped), then U @ H @ P."""
        d = cov.shape[-1]
        cov = cov + 1e-6 * torch.eye(d, device=cov.device, dtype=cov.dtype)
        eigvals, evec = torch.linalg.eigh(cov.double())
        # compose in float64 (author runs offline in float64), cast once at the end
        pbr = make_br_perm_matrix(eigvals)
        return (evec @ hk.double() @ pbr).float()

    def _window(self, T: int):
        s = min(self.n_sink, T)
        r = min(self.recent, T - s)
        return s, T - r

    def _row_mask(self, fb, n: int):
        """Per-REQUEST sink/recent mask (paper semantics), from the ForwardBatch.

        The old `_window(T)` treated one write as one sequence, so in a 2-request prefill
        the second request's sink and the first request's recent rows were quantised.
        Returns None when the batch does not expose positions (then we fall back).
        """
        if fb is None:
            return None
        try:
            pos = fb.positions.long()
            if pos.numel() != n:
                return None
            if fb.forward_mode.is_decode():
                req = torch.arange(n, device=pos.device)
            else:
                counts = fb.extend_seq_lens.long()
                req = torch.repeat_interleave(
                    torch.arange(counts.numel(), device=pos.device), counts)
            if req.numel() != n:
                return None
            seq_end = fb.seq_lens.long()[req]
            return (pos >= self.n_sink) & (pos < seq_end - self.recent)
        except Exception:
            if STRICT:
                raise
            return None

    def _qdq(self, x, R, rho, mask=None):
        T, H, D = x.shape
        if mask is not None:
            idx = mask.nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                return x
            sel = x.index_select(0, idx)
            G = self.G
            xr = sel.float() @ R
            xr = _clip_pct(xr, rho)
            q = _rtn_group(xr.view(idx.numel(), H, D // G, G)).view(idx.numel(), H, D)
            out = x.clone()
            out.index_copy_(0, idx, (q @ R.T).to(x.dtype))
            return out
        s, e = self._window(T)
        if e - s <= 0:
            return x
        G = self.G
        xr = x[s:e].float() @ R
        xr = _clip_pct(xr, rho)
        q = _rtn_group(xr.view(e - s, H, D // G, G)).view(e - s, H, D)
        return torch.cat([x[:s], (q @ R.T).to(x.dtype), x[e:]], dim=0)

    # -- calibration (author-faithful) -----------------------------------
    @staticmethod
    def _calibrate(q, k, v, hk):
        """Port of OSCAR rotation/compute_kv_rotation.py::compute_qqt + compute_sst.

        C_Q = mean over KV-head groups of (q_g^T q_g / N)
        C_S = mean over KV heads of  (vw^T vw / N),  vw = v * sqrt(w),
              w_t = k_t^T C_Q_h k_t normalised to mean 1  (attention-aware V target)
        """
        T, Hkv, D = k.shape
        Hq = q.shape[1] if q is not None else Hkv
        gqa = max(1, Hq // Hkv)
        dev = k.device
        kf = k.reshape(-1, Hkv, D).double()
        vf = v.reshape(-1, Hkv, D).double()
        qf = q.reshape(-1, Hq, D).double() if q is not None else None
        n = kf.shape[0]
        c_q = torch.zeros(D, D, dtype=torch.float64, device=dev)
        c_v = torch.zeros(D, D, dtype=torch.float64, device=dev)
        for h in range(Hkv):
            if qf is not None:
                qg = qf[:, h * gqa:(h + 1) * gqa, :].reshape(-1, D)
                qtq = qg.T @ qg / max(1, qg.shape[0])
            else:
                qtq = torch.eye(D, dtype=torch.float64, device=dev)
            c_q = c_q + qtq
            kh, vh = kf[:, h, :], vf[:, h, :]
            w = (kh @ qtq * kh).sum(1)
            w = w / w.sum().clamp(min=1e-12) * n
            vw = vh * w.sqrt().unsqueeze(1)
            c_v = c_v + vw.T @ vw / n
        c_q = (c_q / Hkv); c_v = (c_v / Hkv)
        c_q = (c_q + c_q.T) / 2
        c_v = (c_v + c_v.T) / 2
        return (KVQDQAttn._rot_from(c_q, hk).float(),
                KVQDQAttn._rot_from(c_v, hk).float())

    # -- entry point -----------------------------------------------------
    # -- coverage counters (P0) ------------------------------------------
    def _bump(self, key, n=1):
        self.stats[key] = self.stats.get(key, 0) + n

    def dump_stats(self):
        if not STATS_PATH:
            return
        try:
            with open(STATS_PATH, "w") as fh:
                json.dump(self.stats, fh, indent=1)
        except Exception:
            pass

    def _get_rot(self, st, key, layer_id, q, k, v, hk):
        """Frozen rotations win over in-situ calibration (P2)."""
        if key in st:
            return st[key]
        if ROT_LOAD:
            path = os.path.join(ROT_LOAD, f"{key}_layer{layer_id}.pt")
            if os.path.exists(path):
                st[key] = torch.load(path, map_location=k.device).float()
                self._bump(f"loaded_{key}")
                return st[key]
        if k.shape[0] < MIN_CAL_TOKENS:
            return None
        if "rk" not in st or "rv" not in st:
            st["rk"], st["rv"] = self._calibrate(q, k, v, hk)
            self._bump("calibrated_layers")
            if ROT_SAVE:
                try:
                    os.makedirs(ROT_SAVE, exist_ok=True)
                    for kk in ("rk", "rv"):
                        torch.save(st[kk].cpu(),
                                   os.path.join(ROT_SAVE, f"{kk}_layer{layer_id}.pt"))
                except Exception:
                    if STRICT:
                        raise
        return st.get(key)

    def __call__(self, q, k, v, layer_id: int, fb=None):
        T, H, D = k.shape
        st = self.state.setdefault(layer_id, {})
        hk = self._had(D, k.device, k.dtype)
        rk = self._get_rot(st, "rk", layer_id, q, k, v, hk)
        rv = st.get("rv")
        if rk is None or rv is None:
            self._bump("skipped_uncalibrated")
            return k, v
        mask = self._row_mask(fb, T)
        if mask is None:
            self._bump("fallback_batch_window")
        else:
            self._bump("masked_calls")
            self._bump("masked_rows", int(mask.sum().item()))
            self._bump("skipped_rows", int((~mask).sum().item()))
        try:
            k_out = self._qdq(k, rk, RHO_K, mask)
            v_out = self._qdq(v, rv, RHO_V, mask)
        except Exception:
            self._bump("qdq_errors")
            if STRICT:
                raise
            return k, v
        self._bump("qdq_calls")
        return k_out, v_out


def install_attn_hook(method: str = "oscar2a") -> bool:
    """Wrap RadixAttention.forward so (q, k, v) can be intercepted together."""
    from sglang.srt.layers.radix_attention import RadixAttention

    if getattr(RadixAttention, "_qdq_attn", False):
        return False
    mod = KVQDQAttn(method)
    orig = RadixAttention.forward

    def forward(self, q, k, v, forward_batch, save_kv_cache=True, **kwargs):
        # NOTE: at this call site k/v arrive FLAT, [T, H_kv * D] (RadixAttention.forward does the
        # .view(-1, H, D) itself), so they must be reshaped before quantising and flattened back.
        if k is not None and v is not None:
            lid = getattr(self, "layer_id", None)
            if lid is None:
                lid = id(self) % 100003
            try:
                ks, vs, qs = k.shape, v.shape, (q.shape if q is not None else None)
                k3 = k.view(-1, self.tp_k_head_num, self.qk_head_dim) if k.dim() == 2 else k
                v3 = v.view(-1, self.tp_v_head_num, self.v_head_dim) if v.dim() == 2 else v
                q3 = q.view(-1, self.tp_q_head_num, self.qk_head_dim) if (q is not None and q.dim() == 2) else q
                k3, v3 = mod(q3, k3, v3, lid, forward_batch)
                k = k3.reshape(ks) if k.dim() == 2 else k3
                v = v3.reshape(vs) if v.dim() == 2 else v3
            except Exception as e:                      # never break serving because of us
                if STRICT:
                    raise
                if not getattr(RadixAttention, "_qdq_warned", False):
                    RadixAttention._qdq_warned = True
                    print(f"[QDQ-ATTN] disabled after error: {type(e).__name__}: {e}", flush=True)
        return orig(self, q, k, v, forward_batch, save_kv_cache, **kwargs)

    RadixAttention.forward = forward
    RadixAttention._qdq_attn = True
    return True
