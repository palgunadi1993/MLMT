"""Validation metrics and benchmarks (PLAN §9, Phase 7).

Metric primitives (Kagan angle, credible-interval coverage, SBC ranks,
Gaussian-mixture MT draws) live here so train.py can log validation Kagan
and coverage during training; the orchestration entry points (SBC over
held-out synthetics, ablations, Grond benchmark) are built on top of them.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import torch

from .forward import m6_to_matrix, project_deviatoric

logger = logging.getLogger('sbi_mt.evaluate')


# ----------------------------------------------------------------------------
# Kagan angle
# ----------------------------------------------------------------------------

#: proper symmetry operations of a double couple (identity + 180 deg
#: rotations about each principal axis)
_DC_SYMMETRIES = [np.diag(s).astype(float) for s in
                  ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]


def _principal_frame(m6: np.ndarray) -> np.ndarray:
    """Right-handed principal-axes rotation matrix of the deviatoric part
    (columns = eigenvectors, ascending eigenvalues: P, B, T)."""
    m = m6_to_matrix(project_deviatoric(np.asarray(m6, dtype=np.float64)))
    _, v = np.linalg.eigh(m)
    if np.linalg.det(v) < 0:
        v = v.copy()
        v[:, 0] = -v[:, 0]
    return v


def kagan_angle(m6_a: np.ndarray, m6_b: np.ndarray) -> float:
    """Minimum rotation angle [deg] between the principal-axes frames of two
    moment tensors, minimized over the 4-element DC symmetry group
    (Kagan 1991)."""
    va = _principal_frame(m6_a)
    vb = _principal_frame(m6_b)
    best = 180.0
    for s in _DC_SYMMETRIES:
        r = va @ s @ vb.T
        c = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
        best = min(best, float(np.degrees(np.arccos(c))))
    return best


# ----------------------------------------------------------------------------
# posterior draws and coverage
# ----------------------------------------------------------------------------

def mt_posterior_draws(m_hat: np.ndarray, cov: np.ndarray,
                       rng: np.random.Generator,
                       n_per: int = 1) -> np.ndarray:
    """Draws from the Gaussian-mixture MT posterior: for each theta sample j,
    n_per draws m ~ N(m_hat_j, cov_j). m_hat (n, 6), cov (n, 6, 6) ->
    (n * n_per, 6). Covariances get a relative jitter on the diagonal so a
    rank-deficient solve (e.g. deviatoric constraint) still factorizes."""
    m_hat = np.asarray(m_hat, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    n = m_hat.shape[0]
    jitter = 1e-12 * np.trace(cov, axis1=-2, axis2=-1).reshape(-1, 1, 1)
    chol = np.linalg.cholesky(cov + jitter * np.eye(6))
    z = rng.standard_normal((n, n_per, 6))
    return (m_hat[:, None, :]
            + np.einsum('nkl,npl->npk', chol, z)).reshape(n * n_per, 6)


def central_interval_hits(samples: np.ndarray, truth: np.ndarray,
                          levels: Sequence[float]) -> np.ndarray:
    """Empirical central credible intervals per component: hits (n_levels,
    k) — True where truth falls inside the level-alpha interval of the
    marginal sample distribution. samples (n, k), truth (k,)."""
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    hits = np.zeros((len(levels), truth.size), dtype=bool)
    for il, lev in enumerate(levels):
        lo = np.quantile(samples, (1.0 - lev) / 2.0, axis=0)
        hi = np.quantile(samples, (1.0 + lev) / 2.0, axis=0)
        hits[il] = (truth >= lo) & (truth <= hi)
    return hits


def sbc_ranks(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Simulation-based calibration rank of truth within the marginal
    posterior samples, per dimension: rank (k,) in {0..n}. Uniform ranks
    over many simulations <=> calibrated posterior (Talts et al. 2018)."""
    return (np.asarray(samples) < np.asarray(truth)[None, :]).sum(axis=0)


def rank_uniformity_pvalue(ranks: np.ndarray, n_samples: int,
                           n_bins: int = 20) -> float:
    """Chi-square p-value against uniformity of SBC ranks in {0..n_samples}."""
    from scipy import stats
    edges = np.linspace(0, n_samples + 1, n_bins + 1)
    counts, _ = np.histogram(ranks, bins=edges)
    expected = len(ranks) / n_bins
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    return float(stats.chi2.sf(chi2, df=n_bins - 1))


# ----------------------------------------------------------------------------
# per-event posterior evaluation (shared by train-val logging and Phase 7)
# ----------------------------------------------------------------------------

def wls_over_thetas(fm: Any, thetas: torch.Tensor, data: torch.Tensor,
                    weights: torch.Tensor, chunk: int = 64
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chunked WLS solve at flow samples: (m_hats (n, 6), covs (n, 6, 6),
    vr (n,)). fm: ForwardModel on the reference store's full grid."""
    m_hats, covs, vrs = [], [], []
    with torch.no_grad():
        for i in range(0, thetas.shape[0], chunk):
            res, _ = fm.solve(thetas[i:i + chunk], data, weights)
            m_hats.append(res.m_hat.cpu().numpy())
            covs.append(res.cov.cpu().numpy())
            vrs.append(res.vr.cpu().numpy())
    return (np.concatenate(m_hats), np.concatenate(covs),
            np.concatenate(vrs))


def posterior_metrics(m6_true: np.ndarray, m_hats: np.ndarray,
                      covs: np.ndarray, rng: np.random.Generator,
                      levels: Sequence[float] = (0.68, 0.90),
                      n_per: int = 1) -> dict[str, Any]:
    """Kagan angle of the posterior-mean MT + per-level coverage hits of the
    6 MT components under the Gaussian-mixture posterior."""
    m_mean = m_hats.mean(axis=0)
    draws = mt_posterior_draws(m_hats, covs, rng, n_per)
    hits = central_interval_hits(draws, m6_true, levels)
    return {
        'kagan_deg': kagan_angle(m_mean, m6_true),
        'coverage_hits': hits,                       # (n_levels, 6) bool
        'm6_mean': m_mean,
    }


# ----------------------------------------------------------------------------
# synthetic evaluation suite: SBC + coverage + binned accuracy (PLAN §9.1-4)
# ----------------------------------------------------------------------------

def evaluate_synthetic(model: Any, events: Sequence[dict[str, Any]],
                       cfg: dict[str, Any], fetcher: Any,
                       device: Any, rng: np.random.Generator
                       ) -> dict[str, np.ndarray]:
    """Full posterior evaluation on held-out synthetics: per event, flow
    samples + full-grid WLS -> SBC ranks (theta dims and MT components),
    coverage hits at evaluate.coverage_levels, Kagan angle of the posterior
    mean, and the conditioning covariates (SNR, station count, perturbed).

    Returns arrays keyed for HDF5 storage; n_samples per event =
    evaluate.n_theta_samples.
    """
    from .train import event_forward_model   # local import (cycle)

    ecfg = cfg['evaluate']
    n_ts = int(ecfg['n_theta_samples'])
    chunk = int(cfg['inference']['wls_chunk'])
    levels = [float(x) for x in ecfg['coverage_levels']]
    n_comp = len(cfg['processing']['components'])

    out: dict[str, list] = {k: [] for k in (
        'ranks_theta', 'ranks_m6', 'hits_theta', 'hits_m6', 'kagan_deg',
        'kagan_direct_deg', 'snr_median', 'n_stations', 'perturbed',
        'theta_true', 'theta_mean', 'm6_true', 'm6_mean', 'vr_best')}
    model.eval()
    for i_ev, ex in enumerate(events):
        wf = torch.as_tensor(ex['waveforms'],
                             dtype=torch.float32).unsqueeze(0).to(device)
        md = torch.as_tensor(ex['metadata'],
                             dtype=torch.float32).unsqueeze(0).to(device)
        mask_t = torch.as_tensor(np.asarray(ex['mask'], dtype=bool)
                                 ).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.embed(wf, md, mask_t)
            thetas = model.sample_emb(emb, n_ts)[0]

        fm, data, weights = event_forward_model(
            ex, cfg, fetcher, np.arange(len(ex['station_codes'])), device)
        m_hats, covs, vr = wls_over_thetas(fm, thetas, data, weights,
                                           chunk)
        thetas_np = thetas.cpu().numpy().astype(np.float64)
        theta_true = np.asarray(ex['theta'], dtype=np.float64)
        m6_true = np.asarray(ex['m6'], dtype=np.float64)
        draws = mt_posterior_draws(m_hats, covs, rng, 1)

        out['ranks_theta'].append(sbc_ranks(thetas_np, theta_true))
        out['ranks_m6'].append(sbc_ranks(draws, m6_true))
        out['hits_theta'].append(
            central_interval_hits(thetas_np, theta_true, levels))
        out['hits_m6'].append(
            central_interval_hits(draws, m6_true, levels))
        out['kagan_deg'].append(kagan_angle(m_hats.mean(axis=0), m6_true))

        # ablation (c): direct regression head, if the model has one
        if getattr(model, 'direct_mt', None) is not None:
            with torch.no_grad():
                direction, log_m0 = model.direct_mt(emb)
            m6_dir = (direction[0].cpu().numpy().astype(np.float64)
                      * 10.0 ** float(log_m0[0]))
            out['kagan_direct_deg'].append(kagan_angle(m6_dir, m6_true))
        else:
            out['kagan_direct_deg'].append(np.nan)

        mask = np.asarray(ex['mask'], dtype=bool)
        out['snr_median'].append(
            float(np.median(np.asarray(ex['snr'])[mask])))
        out['n_stations'].append(
            int(mask.reshape(-1, n_comp).any(axis=1).sum()))
        out['perturbed'].append(bool(ex['perturbed']))
        out['theta_true'].append(theta_true)
        out['theta_mean'].append(thetas_np.mean(axis=0))
        out['m6_true'].append(m6_true)
        out['m6_mean'].append(m_hats.mean(axis=0))
        out['vr_best'].append(float(vr.max()))
        if (i_ev + 1) % 50 == 0 or i_ev + 1 == len(events):
            logger.info('evaluated %d / %d events', i_ev + 1, len(events))

    arrays = {k: np.asarray(v) for k, v in out.items()}
    arrays['coverage_levels'] = np.asarray(levels)
    arrays['n_theta_samples'] = np.asarray(n_ts)
    return arrays


def save_evaluation(arrays: dict[str, np.ndarray], path: str) -> None:
    import h5py
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, 'w') as f:
        for k, v in arrays.items():
            f.create_dataset(k, data=v)


def load_evaluation(path: str) -> dict[str, np.ndarray]:
    import h5py
    with h5py.File(path, 'r') as f:
        return {k: f[k][()] for k in f.keys()}


def binned_kagan(arrays: dict[str, np.ndarray], key: str,
                 edges: Sequence[float]) -> dict[str, np.ndarray]:
    """Median Kagan angle binned by a covariate ('snr_median' or
    'n_stations'), split reference vs velocity-perturbed test events."""
    x = np.asarray(arrays[key], dtype=float)
    kag = np.asarray(arrays['kagan_deg'], dtype=float)
    pert = np.asarray(arrays['perturbed'], dtype=bool)
    edges = np.asarray(list(edges), dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = np.full((2, centers.size), np.nan)
    count = np.zeros((2, centers.size), dtype=int)
    for ip, sel_p in enumerate((~pert, pert)):
        for ib in range(centers.size):
            sel = sel_p & (x >= edges[ib]) & (x < edges[ib + 1])
            count[ip, ib] = int(sel.sum())
            if sel.any():
                med[ip, ib] = float(np.median(kag[sel]))
    return {'edges': edges, 'centers': centers, 'median_kagan': med,
            'count': count, 'rows': np.array(['reference', 'perturbed'])}


def empirical_coverage(arrays: dict[str, np.ndarray],
                       which: str = 'hits_m6') -> np.ndarray:
    """Empirical coverage per (level, dim): fraction of events whose truth
    fell inside the central interval."""
    return np.asarray(arrays[which], dtype=float).mean(axis=0)


# ----------------------------------------------------------------------------
# ablations (PLAN §9.4) — training-config variants, one flag each
# ----------------------------------------------------------------------------

ABLATIONS = ('gaussian_noise', 'no_velocity_perturbation',
             'direct_regression')


def apply_ablation(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    """Deep-copied config with ONE ablation enabled: (a) Gaussian noise
    instead of the empirical library, (b) reference-velocity-only training,
    (c) direct MT regression head (no WLS layer). Train each variant with
    scripts/04 on its config; evaluate on the SAME test set."""
    import copy
    if name not in ABLATIONS:
        raise ValueError(f'unknown ablation {name!r}; choose {ABLATIONS}')
    out = copy.deepcopy(cfg)
    for k in ABLATIONS:
        out['evaluate']['ablations'][k] = False
    out['evaluate']['ablations'][name] = True
    if name == 'gaussian_noise':
        out['synth']['noise'] = 'gaussian'
    elif name == 'no_velocity_perturbation':
        out['synth']['velocity_perturbed_fraction'] = 0.0
    return out


# ----------------------------------------------------------------------------
# real-event benchmark (PLAN §9.5-6)
# ----------------------------------------------------------------------------

def benchmark_event(ev: Any, result: dict[str, Any],
                    grond: dict[str, Any] | None = None) -> dict[str, Any]:
    """SBI posterior vs the catalog reference MT for one event, optionally
    vs a Grond posterior summary.

    ev: data.CatalogEvent with m6_ref (NED [Nm]); result: payload from
    inference.run_event/load_results. grond (user-provided YAML in
    paths.grond_runs/<event>.yaml) keys: m6 (NED [Nm], 6 floats), mw,
    depth_km, wall_clock_s, and optional ci68 (6 floats, credible-interval
    widths of the MT components normalized by M0).
    """
    from .forward import m0_to_mw, scalar_moment
    s = result['summary']
    primary = 'mean_is' if s.get('importance_primary') else 'mean'
    m6_sbi = np.asarray(s[primary]['m6'], dtype=np.float64)
    theta_mean = np.asarray(s['theta_mean'], dtype=np.float64)
    out: dict[str, Any] = {
        'event': str(ev.name),
        'mw_sbi': float(s[primary]['mw']),
        'vr_best': float(s['best_vr']),
        'depth_sbi_km': (float(ev.depth) + theta_mean[2]) / 1000.0,
        'time_sbi_s': float(s['timing_s']['total']),
        'ess': float(s['ess']),
    }
    if getattr(ev, 'm6_ref', None) is not None:
        m6_ref = np.asarray(ev.m6_ref, dtype=np.float64)
        out.update(
            kagan_ref_deg=kagan_angle(m6_sbi, m6_ref),
            dmw_ref=out['mw_sbi'] - float(m0_to_mw(scalar_moment(m6_ref))),
            ref_source=getattr(ev, 'ref_source', ''),
        )
        if ev.depth is not None:
            out['ddepth_ref_km'] = out['depth_sbi_km'] - ev.depth / 1000.0
    # SBI credible-interval widths (68%), normalized by M0
    draws = np.asarray(result['m6_samples'], dtype=np.float64)
    m0 = scalar_moment(m6_sbi)
    ci = (np.quantile(draws, 0.84, axis=0)
          - np.quantile(draws, 0.16, axis=0)) / m0
    out['ci68_sbi'] = [float(x) for x in ci]
    if grond is not None:
        m6_g = np.asarray(grond['m6'], dtype=np.float64)
        out.update(
            kagan_grond_deg=kagan_angle(m6_sbi, m6_g),
            kagan_grond_ref_deg=(
                kagan_angle(m6_g, np.asarray(ev.m6_ref, dtype=np.float64))
                if getattr(ev, 'm6_ref', None) is not None else np.nan),
            mw_grond=float(grond.get('mw', np.nan)),
            time_grond_s=float(grond.get('wall_clock_s', np.nan)),
            ci68_grond=[float(x) for x in grond['ci68']]
            if 'ci68' in grond else None,
        )
    return out
