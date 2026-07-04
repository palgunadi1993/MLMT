"""Differentiable forward model and Rao-Blackwellized MT solve.

All waveform operations (taper, zero-phase bandpass, sub-sample time shift)
are implemented in torch and applied IDENTICALLY to observed data and to
G-matrix columns, so the linear WLS solve sees a consistent forward operator.

The moment tensor is never learned: given a nuisance sample
theta = (dn, de, dz, dt0), the MT is obtained analytically by weighted linear
least squares against the interpolated Green's functions, together with its
conditional Gaussian covariance (PLAN §4).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .gf import GFGrid, assemble_G

# ----------------------------------------------------------------------------
# moment tensor utilities (NED convention, m6 = (mnn, mee, mdd, mne, mnd, med))
# ----------------------------------------------------------------------------

#: orthonormal basis of the zero-trace (deviatoric) subspace in m6 coords,
#: with the tensor Frobenius metric (off-diagonals carry weight sqrt(2))
_DEV_BASIS = np.array([
    [1/math.sqrt(2), -1/math.sqrt(2), 0, 0, 0, 0],
    [1/math.sqrt(6), 1/math.sqrt(6), -2/math.sqrt(6), 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
], dtype=np.float64).T                                   # (6, 5)


def m6_to_matrix(m6: np.ndarray) -> np.ndarray:
    mnn, mee, mdd, mne, mnd, med = np.moveaxis(np.asarray(m6), -1, 0)
    row0 = np.stack([mnn, mne, mnd], axis=-1)
    row1 = np.stack([mne, mee, med], axis=-1)
    row2 = np.stack([mnd, med, mdd], axis=-1)
    return np.stack([row0, row1, row2], axis=-2)


def matrix_to_m6(m: np.ndarray) -> np.ndarray:
    return np.stack([m[..., 0, 0], m[..., 1, 1], m[..., 2, 2],
                     m[..., 0, 1], m[..., 0, 2], m[..., 1, 2]], axis=-1)


def scalar_moment(m6: np.ndarray) -> np.ndarray:
    """M0 = |M| / sqrt(2) (Frobenius convention, matches pyrocko)."""
    m = m6_to_matrix(m6)
    return np.sqrt(np.sum(m * m, axis=(-2, -1)) / 2.0)


def mw_to_m0(mw: float | np.ndarray) -> np.ndarray:
    return 10.0 ** (1.5 * (np.asarray(mw) + 6.07))


def m0_to_mw(m0: float | np.ndarray) -> np.ndarray:
    return 2.0 / 3.0 * np.log10(np.asarray(m0)) - 6.07


def project_deviatoric(m6: np.ndarray) -> np.ndarray:
    tr = (m6[..., 0] + m6[..., 1] + m6[..., 2]) / 3.0
    out = np.array(m6, copy=True)
    for i in range(3):
        out[..., i] -= tr
    return out


def project_to_dc(m6: np.ndarray) -> np.ndarray:
    """Best double couple: eigenframe of the deviatoric part with eigenvalues
    (-M0', 0, +M0'), M0' = (e_max - e_min)/2. Post-hoc projection only —
    the DC constraint is intentionally NOT part of the linear solve."""
    m = m6_to_matrix(project_deviatoric(m6))
    w, v = np.linalg.eigh(m)
    m0p = (w[..., -1] - w[..., 0]) / 2.0
    w_dc = np.stack([-m0p, np.zeros_like(m0p), m0p], axis=-1)
    m_dc = np.einsum('...ij,...j,...kj->...ik', v, w_dc, v)
    return matrix_to_m6(m_dc)


def decompose_mt(m6: np.ndarray) -> dict[str, float]:
    """ISO/DC/CLVD percentages (standard deviatoric decomposition)."""
    m = m6_to_matrix(np.asarray(m6, dtype=np.float64))
    w = np.linalg.eigvalsh(m)
    iso = np.trace(m) / 3.0
    dev = np.sort(w - iso)                # e1 <= e2 <= e3
    e1, e2, e3 = dev
    denom = max(abs(e1), abs(e3))
    if denom == 0:
        return {'iso': 100.0, 'dc': 0.0, 'clvd': 0.0}
    f = -e2 / denom                       # in [-0.5, 0.5]
    m0 = abs(iso) + max(abs(e1), abs(e3))
    p_iso = 100.0 * abs(iso) / m0
    p_clvd = 2.0 * abs(f) * (100.0 - p_iso)
    p_dc = 100.0 - p_iso - p_clvd
    return {'iso': float(p_iso), 'dc': float(p_dc), 'clvd': float(p_clvd)}


# ----------------------------------------------------------------------------
# torch waveform operators
# ----------------------------------------------------------------------------

def cosine_taper(n_t: int, fraction: float,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device | str = 'cpu') -> torch.Tensor:
    """Symmetric cosine (Tukey) taper, `fraction` of the window each end."""
    t = torch.arange(n_t, dtype=dtype, device=device)
    n_ramp = max(1, int(round(fraction * n_t)))
    w = torch.ones(n_t, dtype=dtype, device=device)
    ramp = 0.5 * (1 - torch.cos(math.pi * t[:n_ramp] / n_ramp))
    w[:n_ramp] = ramp
    w[-n_ramp:] = ramp.flip(0)
    return w


def bandpass_zerophase(x: torch.Tensor, deltat: float, fmin: float,
                       fmax: float, order: int) -> torch.Tensor:
    """FFT-domain zero-phase Butterworth bandpass: the real transfer function
    |H_lp|^2 * |H_hp|^2 (squared magnitudes = forward-backward filtering).
    Deterministic and identical for data and synthetics; differentiable."""
    n_t = x.shape[-1]
    f = torch.fft.rfftfreq(n_t, d=deltat, dtype=x.dtype, device=x.device)
    hp = torch.zeros_like(f)
    pos = f > 0
    hp[pos] = 1.0 / (1.0 + (fmin / f[pos]) ** (2 * order))
    lp = 1.0 / (1.0 + (f / fmax) ** (2 * order))
    X = torch.fft.rfft(x, dim=-1)
    return torch.fft.irfft(X * (hp * lp), n=n_t, dim=-1)


def time_shift(x: torch.Tensor, deltat: float,
               shift: torch.Tensor) -> torch.Tensor:
    """y(t) = x(t - shift) via FFT phase ramp; differentiable in `shift`.
    `shift` [s] must broadcast against x's leading dims (e.g. (B, n_tr) for
    x of shape (B, n_tr, n_t))."""
    n_t = x.shape[-1]
    f = torch.fft.rfftfreq(
        n_t, d=deltat,
        dtype=torch.promote_types(x.dtype, torch.float32), device=x.device)
    X = torch.fft.rfft(x, dim=-1)
    phase = torch.exp(-2j * math.pi * f * shift.unsqueeze(-1).to(f.dtype))
    return torch.fft.irfft(X * phase, n=n_t, dim=-1)


# ----------------------------------------------------------------------------
# WLS moment tensor solve
# ----------------------------------------------------------------------------

@dataclass
class WLSResult:
    m_hat: torch.Tensor        # (B, 6) [Nm]
    cov: torch.Tensor          # (B, 6, 6) conditional covariance
    misfit: torch.Tensor       # (B,) weighted residual mean square
    vr: torch.Tensor           # (B,) global variance reduction (weighted)
    vr_trace: torch.Tensor     # (B, n_tr) per-trace variance reduction
    synth: torch.Tensor        # (B, n_tr, n_t) best-fit synthetics


def wls_solve(G: torch.Tensor, d: torch.Tensor, weights: torch.Tensor,
              lam_rel: float = 1e-3,
              constraint: str = 'full') -> WLSResult:
    """m_hat = (Gt W G + lam I)^-1 Gt W d, per batch element.

    G: (B, n_tr, n_t, 6); d: (n_tr, n_t) or (B, n_tr, n_t);
    weights: per-trace inverse noise variances, (n_tr,) or (B, n_tr).
    lam_rel: Tikhonov lambda relative to trace(GtWG)/6 (scale-free).
    constraint: 'full' (6) | 'deviatoric' (5, zero-trace projection).
    The DC constraint is post-hoc only (see :func:`project_to_dc`).

    cov = s^2 (Gt W G + lam I)^-1 with s^2 the weighted residual mean square
    — the classical WLS covariance with estimated scale; s^2 ~ 1 when the
    weights are true inverse noise variances.
    """
    if constraint not in ('full', 'deviatoric'):
        raise ValueError(f'unknown constraint {constraint!r}')

    B = G.shape[0]
    if d.dim() == 2:
        d = d.unsqueeze(0).expand(B, -1, -1)
    if weights.dim() == 1:
        weights = weights.unsqueeze(0).expand(B, -1)
    weights = weights.to(G.dtype)
    d = d.to(G.dtype)

    # Exact internal rescaling: G columns are unit-moment (~1e-20 m/Nm) and
    # weights are 1/sigma^2 (~1e12+), so GtWG ~ 1e-26 — products G*G
    # underflow float32 (denormals) and the linalg backward emits NaNs.
    # Solve in per-batch normalized units and map back; algebraically the
    # regularization (lambda relative to trace) and results are unchanged.
    tiny = torch.finfo(G.dtype).tiny
    gs = G.pow(2).mean(dim=(1, 2, 3)).sqrt().clamp_min(tiny)      # (B,)
    ds = d.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(tiny)         # (B,)
    ws = weights.mean(dim=1).clamp_min(tiny)                      # (B,)
    Gn = G / gs.view(-1, 1, 1, 1)
    dn = d / ds.view(-1, 1, 1)
    wn = weights / ws.view(-1, 1)

    A = Gn
    if constraint == 'deviatoric':
        P = torch.as_tensor(_DEV_BASIS, dtype=G.dtype, device=G.device)
        A = torch.einsum('bsnk,kq->bsnq', Gn, P)

    k = A.shape[-1]
    H = torch.einsum('bsnk,bsnl,bs->bkl', A, A, wn)
    b = torch.einsum('bsnk,bsn,bs->bk', A, dn, wn)
    lam = lam_rel * torch.diagonal(H, dim1=-2, dim2=-1).sum(-1) / k
    eye = torch.eye(k, dtype=G.dtype, device=G.device)
    H_reg = H + lam.view(-1, 1, 1) * eye
    q_hat = torch.linalg.solve(H_reg, b)
    H_inv = torch.linalg.inv(H_reg)

    if constraint == 'deviatoric':
        q_hat = torch.einsum('kq,bq->bk', P, q_hat)
        H_inv = torch.einsum('kq,bqr,lr->bkl', P, H_inv, P)

    m_hat = q_hat * (ds / gs).view(-1, 1)

    synth = torch.einsum('bsnk,bk->bsn', G, m_hat)
    resid = d - synth
    n_data = d.shape[1] * d.shape[2]
    dof = max(n_data - k, 1)
    # s2 in normalized units; cov = s2_orig * H_orig^-1 with
    # H_orig = H_norm * gs^2 * ws and s2_orig = s2_norm * ds^2 * ws
    s2n = torch.einsum('bsn,bsn,bs->b', resid / ds.view(-1, 1, 1),
                       resid / ds.view(-1, 1, 1), wn) / dof
    cov = H_inv * (s2n * ds**2 / gs**2).view(-1, 1, 1)

    misfit = torch.einsum('bsn,bsn,bs->b', resid, resid, weights) / n_data
    num_tr = (resid**2).sum(-1)
    den_tr = (d**2).sum(-1).clamp_min(torch.finfo(G.dtype).tiny)
    vr_trace = 1.0 - num_tr / den_tr
    num_g = torch.einsum('bsn,bsn,bs->b', resid, resid, weights)
    den_g = torch.einsum('bsn,bsn,bs->b', d, d, weights)
    vr = 1.0 - num_g / den_g.clamp_min(torch.finfo(G.dtype).tiny)

    return WLSResult(m_hat=m_hat, cov=cov, misfit=misfit, vr=vr,
                     vr_trace=vr_trace, synth=synth)


# ----------------------------------------------------------------------------
# cross-correlation station alignment (per-station time shifts delta t_i)
# ----------------------------------------------------------------------------

def align_station_shifts(synth: torch.Tensor, data: torch.Tensor,
                         deltat: float, max_shift: float,
                         n_components: int = 3) -> torch.Tensor:
    """Per-station time shifts by cross-correlation, sub-sample refined.

    Sums the correlation functions of a station's components so all three
    share one shift. Returns shifts (B, n_sta) [s] such that
    ``time_shift(synth, deltat, shifts_expanded)`` best aligns with data.
    Measurement step (argmax): intentionally not differentiable — the shift
    values are applied through the differentiable :func:`time_shift`.

    Tier B (local, high-frequency) leans on this heavily; it is implemented
    and tested in v1 (PLAN §0).
    """
    B, n_tr, n_t = synth.shape
    n_sta = n_tr // n_components
    n_fft = 2 * n_t
    S = torch.fft.rfft(synth, n=n_fft, dim=-1)
    D = torch.fft.rfft(data.expand_as(synth), n=n_fft, dim=-1)
    xc = torch.fft.irfft(D * S.conj(), n=n_fft, dim=-1)
    # lag axis: circular; lag k in [-n_t, n_t) -> roll so index 0 = -max_lag
    max_lag = min(int(round(max_shift / deltat)), n_t - 1)
    xc = torch.roll(xc, shifts=max_lag, dims=-1)[..., :2 * max_lag + 1]
    xc = xc.view(B, n_sta, n_components, -1).sum(dim=2)   # (B, n_sta, lags)

    idx = xc.argmax(dim=-1)
    # parabolic sub-sample refinement (guard the window edges)
    idx_c = idx.clamp(1, 2 * max_lag - 1)
    ym = torch.gather(xc, -1, (idx_c - 1).unsqueeze(-1)).squeeze(-1)
    y0 = torch.gather(xc, -1, idx_c.unsqueeze(-1)).squeeze(-1)
    yp = torch.gather(xc, -1, (idx_c + 1).unsqueeze(-1)).squeeze(-1)
    denom = (ym - 2 * y0 + yp)
    frac = torch.where(denom.abs() > 0, 0.5 * (ym - yp) / denom,
                       torch.zeros_like(y0))
    frac = frac.clamp(-0.5, 0.5)
    lag = (idx.to(synth.dtype) - max_lag
           + torch.where(idx == idx_c, frac, torch.zeros_like(frac)))
    return (lag * deltat).clamp(-max_shift, max_shift)


# ----------------------------------------------------------------------------
# end-to-end forward model
# ----------------------------------------------------------------------------

class ForwardModel:
    """Bundles a GFGrid with the processing chain (taper -> zero-phase
    bandpass) applied identically to data and synthetics, the WLS solve and
    the optional cross-correlation station alignment."""

    def __init__(self, grid: GFGrid, cfg: dict[str, Any]):
        self.grid = grid
        proc = cfg['processing']
        self.deltat = grid.deltat
        self.band: tuple[float, float] = tuple(proc['band_hz'])
        self.order = int(proc['filter_order'])
        self.taper_fraction = float(proc['taper_fraction'])
        wls = cfg['wls']
        self.lam_rel = float(wls['tikhonov'])
        self.constraint = str(wls['constraint'])
        self.station_shift_max = float(wls['station_shift_max_s'])
        self.station_shift_enable = bool(wls['station_shift_enable'])
        self.n_components = len(grid.components)
        self._taper: torch.Tensor | None = None

    def _get_taper(self, x: torch.Tensor) -> torch.Tensor:
        if (self._taper is None or self._taper.dtype != x.dtype
                or self._taper.device != x.device):
            self._taper = cosine_taper(
                self.grid.n_t, self.taper_fraction, x.dtype, x.device)
        return self._taper

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Taper + zero-phase bandpass; used verbatim on windowed data and on
        every synthetic/G column."""
        x = x * self._get_taper(x)
        return bandpass_zerophase(
            x, self.deltat, self.band[0], self.band[1], self.order)

    def assemble(self, theta: torch.Tensor,
                 station_shifts: torch.Tensor | None = None) -> torch.Tensor:
        """G at theta = (dn[m], de[m], dz[m], dt0[s]), shape (B, 4) ->
        preprocessed G (B, n_tr, n_t, 6)."""
        theta = torch.atleast_2d(theta)
        G = assemble_G(self.grid, theta[:, 0], theta[:, 1], theta[:, 2])
        shift = theta[:, 3].view(-1, 1).expand(-1, G.shape[1])
        if station_shifts is not None:
            shift = shift + torch.repeat_interleave(
                station_shifts, self.n_components, dim=1)
        G = G.permute(0, 1, 3, 2)                        # (B, n_tr, 6, n_t)
        G = time_shift(G, self.deltat, shift.unsqueeze(-1))
        G = self.preprocess(G)
        return G.permute(0, 1, 3, 2)

    def synthetics(self, theta: torch.Tensor, m6: torch.Tensor,
                   station_shifts: torch.Tensor | None = None
                   ) -> torch.Tensor:
        """Waveforms (B, n_tr, n_t) for nuisance theta and moment tensor m6
        ((6,) or (B, 6)) [Nm]."""
        G = self.assemble(theta, station_shifts)
        m6 = torch.atleast_2d(m6).to(G.dtype)
        return torch.einsum('bsnk,bk->bsn', G, m6)

    def solve(self, theta: torch.Tensor, data: torch.Tensor,
              weights: torch.Tensor,
              align_stations: bool | None = None
              ) -> tuple[WLSResult, torch.Tensor | None]:
        """Preprocessed-data WLS solve at each theta sample.

        data: (n_tr, n_t) already preprocessed via :meth:`preprocess`.
        Returns the WLS result and, if station alignment is enabled, the
        measured per-station shifts (B, n_sta).
        """
        if align_stations is None:
            align_stations = self.station_shift_enable
        G = self.assemble(theta)
        res = wls_solve(G, data, weights, self.lam_rel, self.constraint)
        shifts = None
        if align_stations and self.station_shift_max > 0:
            shifts = align_station_shifts(
                res.synth, data if data.dim() == 3 else data.unsqueeze(0),
                self.deltat, self.station_shift_max, self.n_components)
            G = self.assemble(theta, station_shifts=shifts)
            res = wls_solve(G, data, weights, self.lam_rel, self.constraint)
        return res, shifts
