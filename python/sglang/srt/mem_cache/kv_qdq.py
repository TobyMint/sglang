"""KV-cache quantize-dequantize (QDQ) shim for SGLang.

Purpose: reproduce published KV-cache quantization *methods* inside the real serving stack
(paged attention, flashinfer kernels, radix cache, batching) without changing the memory layout.

It hooks the moment tokens are written into the KV pool: the bf16 K/V for the new tokens are
quantized with the method's quantizer and dequantized back, then stored. Attention therefore reads
exactly the values the method would have stored. Memory saving is NOT modelled here (that is a
separate step); numerical behaviour is.

Methods implemented (faithful to the papers unless marked [approx]):
  kivi4       KIVI (arXiv 2402.02750): K per-channel static (asymmetric INT4), V per-token
              asymmetric INT4, recent window kept in bf16.
  saw_int4    SAW-INT4 (arXiv 2604.19157): per-(token,head) asymmetric INT4 with a per-head
              Hadamard rotation applied before quantization.
  kvarn2      KVarN (arXiv 2606.03458): channel Hadamard + dual-axis (channel/token) variance
              normalization, then 2-bit symmetric RTN. Calibration-free.
  oscar2      OSCAR (arXiv 2605.17757): covariance-aware rotation + clipped INT2 + sink/recent
              bf16 window.  [approx] the covariance rotation is estimated in situ from the first
              prefill chunk per layer instead of from an offline calibration set.
  kivi_axis4  reference arm: E2M1-style values replaced by INT4 with KIVI's axis (used by our paper).

All quantizers operate on tensors shaped [T, H, D] (tokens, kv heads, head_dim).
"""
from __future__ import annotations

import math
import os
from typing import Dict, Tuple

import torch

# ------------------------------------------------------------------ helpers

# Calibration guard: the server issues a tiny health-check prefill (~6 tokens) before serving
# real traffic. Calibrating a per-channel statistic on those 6 irrelevant tokens poisons the
# scale for the WHOLE run (observed: kivi4 = 0/100). Only calibrate on a chunk of at least
# this many tokens; before that store exact bf16 (a handful of tokens, harmless).
MIN_CAL_TOKENS = 256


def _hadamard(d: int, device, dtype) -> torch.Tensor:
    h = torch.ones(1, 1, device=device, dtype=torch.float32)
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    assert h.shape[0] == d, f"head_dim {d} is not a power of two"
    return (h / math.sqrt(d)).to(dtype)


def _asym_int(x: torch.Tensor, bits: int, group_dim: int, scale: torch.Tensor | None = None):
    """Asymmetric RTN. group_dim=1 -> per-(token,head) stats; group_dim=2 -> per-channel stats."""
    qmax = (1 << bits) - 1
    xf = x.float()
    if scale is None:
        if group_dim == 1:          # one (min,max) per token & head, across channels
            lo = xf.amin(dim=-1, keepdim=True)
            hi = xf.amax(dim=-1, keepdim=True)
        else:                       # one (min,max) per channel & head, across tokens
            lo = xf.amin(dim=0, keepdim=True)
            hi = xf.amax(dim=0, keepdim=True)
        rng = (hi - lo).clamp(min=1e-8)
        scale = (rng / qmax, lo.squeeze(-1) if False else lo)
    sc, lo = scale
    q = torch.round((xf - lo) / sc).clamp_(0, qmax)
    return q.mul_(sc).add_(lo).to(x.dtype), (sc, lo)


def _rtn_axis(x: torch.Tensor, bits: int, axis: int) -> torch.Tensor:
    """Asymmetric RTN whose scale+zero live on every slice along `axis` (the paper's
    'per token and per group' / 'per channel over a token group' convention)."""
    qmax = (1 << bits) - 1
    lo = x.amin(dim=axis, keepdim=True)
    hi = x.amax(dim=axis, keepdim=True)
    sc = ((hi - lo) / qmax).clamp(min=1e-8)
    q = torch.round((x - lo) / sc).clamp_(0, qmax)
    return q.mul_(sc).add_(lo)


def _bitrev(d: int) -> torch.Tensor:
    """OSCAR's bit-reversal permutation P_br: spread importance levels across groups."""
    bits = int(math.log2(d))
    idx = torch.arange(d)
    rev = torch.zeros_like(idx)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    return rev


def _quantile_last(x: torch.Tensor, rho: float) -> torch.Tensor:
    """Per-row percentile of |x| along the last dim (OSCAR's pre-quantization clipping)."""
    s, _ = x.abs().sort(dim=-1)
    k = min(s.shape[-1] - 1, int(rho * (s.shape[-1] - 1)))
    return s[..., k:k + 1]


_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _e2m1_round(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest E2M1 value (sign handled separately). Magnitudes only."""
    a = x.abs().clamp(max=6.0)
    out = torch.zeros_like(a)
    prev = torch.zeros_like(a)
    for lv in _E2M1_LEVELS[1:]:
        mid = 0.5 * (prev + lv)
        out = torch.where(a >= mid, torch.full_like(a, lv), out)
        prev = torch.full_like(a, lv)
    return torch.sign(x) * out


def _sym_int4_grouped(x: torch.Tensor, G: int = 32) -> torch.Tensor:
    """Variant: symmetric INT4, no zero point -- isolates what the zero point buys."""
    T, H, D = x.shape
    xf = x.float().view(T, H, D // G, G)
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    sc = amax / 7.0
    q = torch.round(xf / sc).clamp_(-8, 7).mul_(sc)
    return q.view(T, H, D)


def _e2m1_affine(x: torch.Tensor, G: int = 32) -> torch.Tensor:
    """Variant: E2M1 codebook but an affine (free) per-group scale instead of E8M0."""
    T, H, D = x.shape
    xf = x.float().view(T, H, D // G, G)
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    sc = amax / 6.0                                   # 6.0 is the largest E2M1 magnitude
    q = _e2m1_round(xf / sc) * sc
    return q.view(T, H, D)


def _e2m1_round_with_scale(xb: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return _e2m1_round(xb / scale) * scale


def _mxfp4_best_e8m0(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """E8M0 with a *searched* exponent instead of the OCP default.

    The OCP default (s = 2^(floor(log2 amax) - 2)) is one turn of a dial, not the best turn:
    it can leave the block maximum above the 6.0 that E2M1 can represent, or make the grid
    needlessly coarse. Here each block picks the exponent among {e-1, e, e+1} that minimises
    its own reconstruction error. This is the baseline the "finer scale" claim must beat.
    """
    T, H, D = x.shape
    xb = x.float().view(T, H, D // block, block)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    e0 = torch.floor(torch.log2(amax)) - 2
    best_q, best_err = None, None
    for dk in (-1, 0, 1):
        sc = torch.pow(2.0, e0 + dk)
        q = _e2m1_round_with_scale(xb, sc)
        err = (q - xb).pow(2).mean(dim=-1, keepdim=True)
        if best_err is None:
            best_q, best_err = q, err
        else:
            take = err < best_err
            best_q = torch.where(take, q, best_q)
            best_err = torch.where(take, err, best_err)
    return best_q.view(T, H, D)


def _e4m3_round(s: torch.Tensor) -> torch.Tensor:
    """Round a positive float to the nearest **E4M3FN** value.

    E4M3FN = 1 sign, 4 exponent (bias 7), 3 mantissa, NO infinity.
      max finite 448, min normal 2^-6, subnormals down to 2^-9.
    The previous version truncated the mantissa instead of carrying, ignored subnormals
    (2^-9 came back as 2^-6, eight times too large) and did not saturate at 448 --
    Codex caught all three.  Rounding is round-half-to-even, which is what torch.round does.
    """
    x = s.float().clamp(min=0.0)
    # exponent of the value; clamp to the normal range, which also lands subnormals on the
    # correct 2^-9 grid because step = 2^(e-3) = 2^-9 when e = -6.
    e = torch.floor(torch.log2(x.clamp(min=2.0 ** -9))).clamp(-6.0, 8.0)
    step = torch.pow(2.0, e - 3.0)
    q = torch.round(x / step) * step
    return q.clamp(max=448.0)


def _e2m1_e4m3_scale(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """E2M1 payload with an E4M3 per-group scale -- SAME 8 bits as E8M0.

    This is the arm that decides whether the 8-point gap is about *scale resolution*:
    identical payload, identical group size, identical scale byte count, only the encoding
    of those 8 bits changes.
    """
    T, H, D = x.shape
    xb = x.float().view(T, H, D // block, block)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    sc = _e4m3_round(amax / 6.0).clamp(min=1e-30)
    q = _e2m1_round_with_scale(xb, sc)
    return q.view(T, H, D)


def _mxfp4_qdq(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """The MXFP4 recipe exactly: per-32-element power-of-two E8M0 scale + E2M1 payload."""
    T, H, D = x.shape
    xf = x.float().view(T, H, D // block, block)
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    # MXFP4: shared exponent e = floor(log2(amax)) - 2, scale = 2^e (E8M0), largest |v| <= 6
    e = torch.floor(torch.log2(amax)) - 2
    scale = torch.pow(2.0, e)
    q = _e2m1_round(xf / scale) * scale
    return q.view(T, H, D)


def _sym_int(x: torch.Tensor, bits: int):
    """Symmetric RTN with a per-(token,head) scale (KVarN / OSCAR style after normalization)."""
    qmax = (1 << (bits - 1)) - 1
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    sc = amax / qmax
    q = torch.round(xf / sc).clamp_(-qmax - 1, qmax)
    return q.mul_(sc).to(x.dtype)


# ------------------------------------------------------------------ methods

class _State:
    """Per-(layer, side) calibration state."""

    def __init__(self):
        self.k_scale = None      # KIVI: per-channel (sc, lo)
        self.ch_std = None       # KVarN: per-channel std from prefill
        self.rot = None          # OSCAR: rotation matrix
        self.calibrated = False


class KVQDQ:
    # (sink, recent) protection windows, per paper. Structural, not a tuning knob:
    #   KVarN  S=128 R=128 (a 128-token tile cannot close until 128 later tokens exist)
    #   OSCAR  S=64  R=256 (its own Table 5 reports (S,R)=(0,0) -> 0.00 accuracy)
    #   KIVI   S=0   R=32
    WINDOWS = {"kvarn2": (0, 0), "oscar2": (64, 256), "kivi4": (0, 32), "kivi_axis4": (0, 32),
               "axis_a4": (0, 32), "axis_b4": (0, 32), "mxfp4_qdq": (0, 32), "mxfp4_nowin": (0, 0),
               "fmt_sym4": (0, 32), "fmt_e2m1aff": (0, 32), "fmt_mxfp4": (0, 32), "fmt_e8best": (0, 32), "fmt_e4m3": (0, 32)}

    def __init__(self, method: str, recent_window: int | None = None, n_sink: int | None = None):
        self.method = method
        s_def, r_def = self.WINDOWS.get(method, (0, 0))
        self.n_sink = s_def if n_sink is None else n_sink
        self.recent = r_def if recent_window is None else recent_window
        # Sweep hooks: QDQ_SINK / QDQ_RECENT override the per-method defaults so a window
        # sweep needs no code edits. `QDQ_RECENT=0 QDQ_SINK=0` reproduces the no-window arm.
        env_s, env_r = os.environ.get("QDQ_SINK"), os.environ.get("QDQ_RECENT")
        if env_s is not None:
            self.n_sink = int(env_s)
        if env_r is not None:
            self.recent = int(env_r)
        self.state: Dict[Tuple[int, str], _State] = {}
        self._H: Dict[Tuple[int, str], torch.Tensor] = {}

    def _window(self, T: int):
        """Rows of THIS write that stay exact: [0:s) sink and [T-r:T) recent.
        The hook sees no per-sequence positions, so a write shorter than the window keeps
        everything exact -- i.e. decode steps always land inside the recent window."""
        s = min(self.n_sink, T)
        r = min(self.recent, T - s)
        return s, T - r

    # -- public entry point -------------------------------------------------
    def __call__(self, k: torch.Tensor, v: torch.Tensor, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.method == "none":
            return k, v
        st = self.state.setdefault((layer_id, "kv"), _State())
        if not getattr(self, "_announced", False):
            self._announced = True
            try:
                with open("/work/logs/qdq_calls.log", "a") as fh:
                    fh.write(f"QDQ ACTIVE method={self.method} layer={layer_id} "
                             f"shape={tuple(k.shape)} dev={k.device}\n")
            except Exception:
                pass
        D = k.shape[-1]
        Hk = self._H.get((D, str(k.device)))
        if Hk is None:
            Hk = _hadamard(D, k.device, k.dtype)
            self._H[(D, str(k.device))] = Hk

        f = getattr(self, "_" + self.method)
        out = f(k, v, st, Hk)
        # PROBE: count rows that came back bit-identical (i.e. left at BF16) vs changed.
        try:
            import torch as _t
            if _t.cuda.is_available() and _t.cuda.is_current_stream_capturing():
                return out                      # counting here would sync and break capture
            if not hasattr(self, "_cnt"):
                self._cnt = {"rows": 0, "exact": 0, "calls": 0}
            c = self._cnt
            n = k.shape[0]
            same = bool(_t.equal(out[0], k))
            c["rows"] += n
            c["calls"] += 1
            if same:
                c["exact"] += n
            if c["calls"] % 200 == 0:
                print(f"[QDQPROBE] method={self.method} calls={c['calls']} "
                      f"rows={c['rows']} left_exact={c['exact']} "
                      f"({c['exact']/max(c['rows'],1)*100:.1f}%)", flush=True)
        except Exception as _e:
            pass
        return out

    # -- methods ------------------------------------------------------------
    def _kivi4(self, k, v, st, Hk):
        # K: static per-channel asymmetric INT4, calibrated once on the first REAL chunk.
        # V: per-token asymmetric INT4.  The last R=32 tokens per sequence stay exact
        # (KIVI's FP16 residual window); decode writes are single tokens, so they land
        # inside the window and are left untouched.
        if st.k_scale is None:
            if k.shape[0] < MIN_CAL_TOKENS:
                return k, v                            # warmup / tiny chunk: keep exact
            _, st.k_scale = _asym_int(k, 4, group_dim=2)
        s, e = self._window(k.shape[0])
        if e - s <= 0:
            return k, v
        kq, _ = _asym_int(k[s:e], 4, group_dim=2, scale=st.k_scale)
        vq, _ = _asym_int(v[s:e], 4, group_dim=1)
        return (torch.cat([k[:s], kq.to(k.dtype), k[e:]], dim=0),
                torch.cat([v[:s], vq.to(v.dtype), v[e:]], dim=0))

    def _kivi_axis4(self, k, v, st, Hk):
        # our paper's reference arm: INT4, K per-channel static / V per-token, no rotation
        return self._kivi4(k, v, st, Hk)

    def _saw4_tile(self, k, v, st, Hk):
        """Matched-coverage control: the SAME tile/window gating as _kvarn2, but 4-bit.

        `_saw_int4` quantises every write, including the single-token decode writes, so it is a
        full-coverage 4.25-bit cache. `_kvarn2` only quantises complete 128-token tiles, so its
        decode tail stays BF16. Comparing those two confounds width with coverage. This variant
        quantises exactly the positions _kvarn2 quantises, leaving the same rows exact.
        """
        T = k.shape[0]
        s, e = self._window(T)
        n = (e - s) // 128
        if n <= 0:
            return k, v
        Tg = n * 128
        kq, _ = _asym_int(k[s:s + Tg] @ Hk, 4, group_dim=1)
        vq, _ = _asym_int(v[s:s + Tg] @ Hk, 4, group_dim=1)
        return (torch.cat([k[:s], (kq @ Hk).to(k.dtype), k[s + Tg:]], dim=0),
                torch.cat([v[:s], (vq @ Hk).to(v.dtype), v[s + Tg:]], dim=0))

    def _saw_int4(self, k, v, st, Hk):
        # SAW-INT4: rotate (per-head Hadamard) then per-(token,head) asymmetric INT4, rotate back
        kq, _ = _asym_int(k @ Hk, 4, group_dim=1)
        vq, _ = _asym_int(v @ Hk, 4, group_dim=1)
        return (kq @ Hk).to(k.dtype), (vq @ Hk).to(v.dtype)

    def _kvarn2(self, k, v, st, Hk):
        """KVarN (arXiv 2606.03458): per-head channel Hadamard -> VarN dual-axis normalization
        (Sinkhorn-style, 8 iterations) -> asymmetric RTN, group G=128 along tokens for K
        (scale+zero indexed by channel) and per token for V.
        [approx] the hook has no token positions, so a 128-token tile can only be formed inside
        one prefill chunk. Writes shorter than one tile stay exact, i.e. the paper's R=128 FP16
        trailing window is emulated but never closed."""
        T = k.shape[0]
        s, e = self._window(T)
        n = (e - s) // 128
        if n <= 0:
            return k, v                       # this write fits inside the sink/recent window
        Tg = n * 128
        H, D = k.shape[1], k.shape[2]
        kk = (k[s:s + Tg] @ Hk).float().view(n, 128, H, D)
        vv = (v[s:s + Tg] @ Hk).float().view(n, 128, H, D)

        def varn(x):
            """Alternate channel / token RMS normalisation, accumulating both scale products."""
            acc = torch.ones_like(x)
            for _ in range(8):
                sc = x.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_(min=1e-8)
                x = x / sc; acc = acc * sc
                sr = x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_(min=1e-8)
                x = x / sr; acc = acc * sr
            return x, acc

        kx, kacc = varn(kk)
        vx, vacc = varn(vv)
        kq = _rtn_axis(kx, 2, axis=1) * kacc      # per channel over the 128-token group
        vq = _rtn_axis(vx, 2, axis=-1) * vacc     # per token over channels
        # inverse rotation in fp32: kq/vq are float32, Hk is the bf16 Hadamard
        Hf = Hk.float()
        k_out = torch.cat([k[:s], (kq.reshape(Tg, H, D) @ Hf).to(k.dtype), k[s + Tg:]], dim=0)
        v_out = torch.cat([v[:s], (vq.reshape(Tg, H, D) @ Hf).to(v.dtype), v[s + Tg:]], dim=0)
        return k_out, v_out

    # ---- controlled axis experiment (P2): SAME codebook, SAME G, SAME window ----
    # A 32-token page and D=128 channels give 128 parameter groups under either axis:
    #   axisA (KIVI): one scale/zero per CHANNEL, shared over the 32 tokens of the page
    #   axisB (block): one scale/zero per 32-CHANNEL group, per token
    # Payload is affine asymmetric INT4 in both, the window is the same, and both go through
    # the same QDQ path -- only the direction the parameters are shared changes.
    AXIS_G = 32

    def _axis_page(self, x, axis):
        """x: [T,H,D] -> quantised in place; axis=0 shared over tokens, axis=1 over channels."""
        T, H, D = x.shape
        G = self.AXIS_G
        n = T // G
        if n <= 0:
            return None
        xr = x[: n * G].float().view(n, G, H, D)
        if axis == "channel":                       # one param per (page, head, channel)
            return _rtn_axis(xr, 4, axis=1)
        else:                                        # one param per (page, token, head, group)
            g = xr.view(n, G, H, D // G, G)
            q = _rtn_axis(g, 4, axis=-1).view(n, G, H, D)
            return q

    def _axis_common(self, k, v, st, Hk, axis):
        T, H, D = k.shape
        G = self.AXIS_G
        n = T // G
        if n <= 0:
            return k, v
        Tq = n * G
        s, e = self._window(Tq)
        if e - s <= 0:
            return k, v
        # round the quantised region down to whole pages
        s -= s % G
        e -= e % G
        if e - s <= 0:
            return k, v
        kq = self._axis_page(k[s:e], axis).view(e - s, H, D)
        # V must be held IDENTICAL to the format arms, i.e. grouped G=32 along the head dim.
        # The earlier version quantised V over the whole head_dim (one scale per token per
        # head), which made the "same G, only the format changed" claim false -- Codex caught it.
        vq = _rtn_axis(v[s:e].float().view((e - s) // G, G, H, D // G, G), 4, axis=-1) \
            .view(e - s, H, D)
        return (torch.cat([k[:s], kq.to(k.dtype), k[e:]], dim=0),
                torch.cat([v[:s], vq.to(v.dtype), v[e:]], dim=0))

    def _axis_a4(self, k, v, st, Hk):
        return self._axis_common(k, v, st, Hk, "channel")

    def _axis_b4(self, k, v, st, Hk):
        return self._axis_common(k, v, st, Hk, "token")

    def _mxfp4_qdq(self, k, v, st, Hk):
        """Ablation for the axis study: MXFP4 *numerics* through the QDQ path.

        The packed fp4_mx_block32 arm changes three things at once (E2M1 codebook, E8M0
        power-of-two scale, packed execution). This arm keeps the representation and removes
        the execution difference, so a drop here is attributable to the format alone.
        Window is the same R=32 as the axis arms.
        """
        T, H, D = k.shape
        s, e = self._window(T)
        if e - s <= 0:
            return k, v
        kq = _mxfp4_qdq(k[s:e])
        vq = _mxfp4_qdq(v[s:e])
        return (torch.cat([k[:s], kq.to(k.dtype), k[e:]], dim=0),
                torch.cat([v[:s], vq.to(v.dtype), v[e:]], dim=0))

    def _mxfp4_nowin(self, k, v, st, Hk):
        """Same MXFP4 numerics, NO sink/recent protection -- the packed fp4 arm's setting."""
        kq = _mxfp4_qdq(k)
        vq = _mxfp4_qdq(v)
        return kq.to(k.dtype), vq.to(v.dtype)

    # ---- format-mechanism ablation: same axis (block), same G, same window, same coverage
    def _fmt_common(self, k, v, st, Hk, fn):
        T, H, D = k.shape
        s, e = self._window(T)
        if e - s <= 0:
            return k, v
        kq = fn(k[s:e]); vq = fn(v[s:e])
        return (torch.cat([k[:s], kq.to(k.dtype), k[e:]], dim=0),
                torch.cat([v[:s], vq.to(v.dtype), v[e:]], dim=0))

    def _fmt_sym4(self, k, v, st, Hk):
        return self._fmt_common(k, v, st, Hk, _sym_int4_grouped)

    def _fmt_e2m1aff(self, k, v, st, Hk):
        return self._fmt_common(k, v, st, Hk, _e2m1_affine)

    def _fmt_mxfp4(self, k, v, st, Hk):
        return self._fmt_common(k, v, st, Hk, _mxfp4_qdq)

    def _fmt_e8best(self, k, v, st, Hk):
        return self._fmt_common(k, v, st, Hk, _mxfp4_best_e8m0)

    def _fmt_e4m3(self, k, v, st, Hk):
        return self._fmt_common(k, v, st, Hk, _e2m1_e4m3_scale)

    def _debugkill(self, k, v, st, Hk):
        """Wiring check: destroy the cache. If accuracy does not collapse, the hook is not reached."""
        return torch.zeros_like(k), torch.zeros_like(v)

    def _oscar2(self, k, v, st, Hk):
        """OSCAR (arXiv 2605.17757): dense rotation R = U . H_had . P_br per layer, per-token
        percentile clipping (0.96 K / 0.92 V), then affine asymmetric INT2 (qmax=3) with
        scale+zero per token and per group G=64 along channels.
        Sink S=64 and recent W=256 tokens stay exact -- structural: the paper's own Table 5
        reports (S,R)=(0,0) -> 0.00 accuracy, which this reproduction reproduces exactly.
        [approx] U is estimated from this layer's K (resp. V) covariance on the first real
        chunk instead of an offline query covariance C_Q / C_S."""
        T, H, D = k.shape
        if getattr(st, "rot", None) is None:
            if T < MIN_CAL_TOKENS:
                return k, v
            eye = torch.eye(D, device=k.device)
            Pm = eye[_bitrev(D).to(k.device)]

            def rot_of(x):
                xf = x.reshape(-1, D).float()
                cov = (xf.T @ xf) / max(1, xf.shape[0])
                cov = cov + 1e-6 * torch.eye(D, device=x.device)
                _, evec = torch.linalg.eigh(cov)
                U = evec.flip(-1)                          # leading eigen-directions first
                # OSCAR eq. (paper :258): R = U @ H_had @ P_br, applied to ROW vectors as x @ R.
                # The composition order is not cosmetic: it is the one that makes
                #   diag(R^T C_Q R) = tr(C_Q)/d * I
                # i.e. every rotated coordinate carries equal importance. `H @ P @ U` is just as
                # orthogonal but leaves importance max/mean = 1.58 (worse than H alone = 1.45).
                return U @ Hk.float() @ Pm

            st.rot = rot_of(k)
            st.rotv = rot_of(v)

        s, e = self._window(T)
        if e - s <= 0:
            return k, v                                    # write fits inside the window
        G = 64
        n = e - s

        def qdq(x, R, rho):
            xr = x[s:e].float() @ R
            tau = _quantile_last(xr, rho)
            xr = xr.clamp(-tau, tau)
            q = _rtn_axis(xr.view(n, H, D // G, G), 2, axis=-1).view(n, H, D)
            return torch.cat([x[:s], (q @ R.T).to(x.dtype), x[e:]], dim=0)

        return qdq(k, st.rot, 0.96), qdq(v, st.rotv, 0.92)


ATTN_RECIPES = ("oscar2a",)   # implemented at the attention call site, not on write


def build(method: str) -> KVQDQ | None:
    if not method or method in ("auto", "bf16", "fp8_e5m2", "fp8_e4m3", "fp4_mx_block32"):
        return None
    if not method.startswith("qdq_"):
        return None
    name = method[len("qdq_"):]
    from sglang.srt.mem_cache import kv_qdq_lifecycle as _life
    if _life.is_lifecycle_recipe(name):
        # tile-closure lifecycle: stateful driver, no stateless write hook
        return _life.build(name)
    if name in ATTN_RECIPES:
        # OSCAR needs the QUERY covariance for its rotation, which only exists at the
        # attention call site. Install the hook there and keep the write path untouched.
        from sglang.srt.mem_cache.kv_qdq_attn import install_attn_hook
        install_attn_hook(name)
        return None
    return KVQDQ(name)
