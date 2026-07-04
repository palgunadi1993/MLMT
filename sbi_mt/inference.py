"""Amortized inference (PLAN §8, Phase 6).

Per event: one network forward pass gives q(theta | data); N theta samples
are pushed in chunks through the differentiable forward operator on the
FULL reference GF grid, each with a closed-form WLS moment tensor solve.
The MT posterior is the resulting mixture of analytic Gaussians
m ~ N(m_hat_j, Sigma_j).

Optional importance reweighting (config inference.importance_reweight):
log w_j = log p(d | theta_j) + log p(theta_j) - log q(theta_j | d), with the
Rao-Blackwellized marginal likelihood over m (flat MT prior):
log p(d|theta) = -0.5 chi2_min(theta) - 0.5 log|GtWG + lam I| + const.
Both unweighted and reweighted summaries are always stored.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import h5py
import numpy as np
import torch

from .evaluate import mt_posterior_draws
from .forward import decompose_mt, m0_to_mw, scalar_moment
from .model import PosteriorModel, prior_bounds_si
from .train import event_forward_model

logger = logging.getLogger('sbi_mt.inference')

Fetcher = Any  # GFCube | StoreFetcher


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (x * w[:, None]).sum(axis=0) / w.sum()


def run_event(model: PosteriorModel, ex: dict[str, Any],
              cfg: dict[str, Any], fetcher: Fetcher,
              device: torch.device | str,
              rng: np.random.Generator,
              name: str | None = None) -> dict[str, Any]:
    """Amortized posterior for one preprocessed event (example dict from
    data.prepare_event_data or synth). Returns the full posterior payload;
    write it with :func:`save_results`."""
    icfg = cfg['inference']
    n_samples = int(icfg['n_theta_samples'])
    chunk = int(icfg['wls_chunk'])
    n_per = int(icfg['n_mt_draws_per_theta'])
    n_comp = len(cfg['processing']['components'])

    t0 = time.perf_counter()
    model.eval()
    wf = torch.as_tensor(ex['waveforms'],
                         dtype=torch.float32).unsqueeze(0).to(device)
    md = torch.as_tensor(ex['metadata'],
                         dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.as_tensor(np.asarray(ex['mask'], dtype=bool)
                           ).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.embed(wf, md, mask)
        thetas = model.sample_emb(emb, n_samples)[0]          # (n, 4)
        log_q = model.log_prob_emb(
            emb.expand(n_samples, -1), thetas).cpu().numpy()
    t_flow = time.perf_counter() - t0

    fm, data, weights = event_forward_model(
        ex, cfg, fetcher, np.arange(len(ex['station_codes'])), device)
    n_data = int(data.shape[0] * data.shape[1])
    k = 6 if cfg['wls']['constraint'] == 'full' else 5
    dof = max(n_data - k, 1)

    m_hats, covs, vrs, vr_traces, misfits, shifts = [], [], [], [], [], []
    with torch.no_grad():
        for i in range(0, n_samples, chunk):
            res, sh = fm.solve(thetas[i:i + chunk], data, weights)
            m_hats.append(res.m_hat.cpu().numpy())
            covs.append(res.cov.cpu().numpy())
            vrs.append(res.vr.cpu().numpy())
            vr_traces.append(res.vr_trace.cpu().numpy())
            misfits.append(res.misfit.cpu().numpy())
            shifts.append(sh.cpu().numpy() if sh is not None
                          else np.zeros((res.m_hat.shape[0],
                                         len(ex['station_codes']))))
    m_hats = np.concatenate(m_hats).astype(np.float64)
    covs = np.concatenate(covs).astype(np.float64)
    vr = np.concatenate(vrs)
    vr_trace = np.concatenate(vr_traces)
    misfit = np.concatenate(misfits).astype(np.float64)
    station_shift = np.concatenate(shifts)
    thetas_np = thetas.cpu().numpy().astype(np.float64)
    t_wls = time.perf_counter() - t0 - t_flow

    # importance log-weights (Rao-Blackwellized marginal likelihood, flat
    # MT prior; theta prior = uniform box -> in-box indicator)
    chi2 = misfit * n_data
    s2 = np.maximum(chi2 / dof, np.finfo(np.float64).tiny)
    sign, logdet_cov = np.linalg.slogdet(covs)
    logdet_h = k * np.log(s2) - logdet_cov            # |H| = s2^k / |cov|
    bounds = prior_bounds_si(cfg['priors'],
                             model.theta_names).numpy().astype(np.float64)
    in_box = np.all((thetas_np >= bounds[:, 0])
                    & (thetas_np <= bounds[:, 1]), axis=1)
    log_w = -0.5 * chi2 - 0.5 * logdet_h - log_q
    log_w = np.where(in_box & (sign > 0), log_w, -np.inf)
    if np.all(~np.isfinite(log_w)):
        logger.warning('all importance weights vanished — flow mass '
                       'outside the prior box? falling back to uniform')
        log_w = np.zeros_like(log_w)
    log_w = log_w - np.max(log_w[np.isfinite(log_w)])
    w_is = np.exp(log_w)
    w_is = w_is / w_is.sum()
    ess = float(1.0 / np.sum(w_is ** 2))

    # MT posterior draws (mixture of Gaussians)
    m6_samples = mt_posterior_draws(m_hats, covs, rng, n_per)

    j_map = int(np.argmax(log_q))                     # posterior mode proxy
    j_best_vr = int(np.argmax(vr))
    m6_mean = m_hats.mean(axis=0)
    m6_mean_is = _weighted_mean(m_hats, w_is)

    def _summ(m6: np.ndarray) -> dict[str, Any]:
        return {
            'm6': [float(x) for x in m6],
            'mw': float(m0_to_mw(scalar_moment(m6))),
            'decomposition': decompose_mt(m6),
        }

    elapsed = time.perf_counter() - t0
    result = {
        'event': str(name if name is not None
                     else ex.get('name', 'event')),
        'theta_names': list(model.theta_names),
        'theta_samples': thetas_np,
        'log_q': log_q,
        'm_hats': m_hats,
        'covs': covs,
        'm6_samples': m6_samples,
        'vr': vr,
        'vr_trace': vr_trace,
        'misfit': misfit,
        'station_shift': station_shift,
        'log_w_is': log_w,
        'w_is': w_is,
        'ess': ess,
        'station_codes': list(ex['station_codes']),
        'summary': {
            'mean': _summ(m6_mean),
            'mean_is': _summ(m6_mean_is),
            'map': _summ(m_hats[j_map]),
            'theta_map': [float(x) for x in thetas_np[j_map]],
            'theta_mean': [float(x) for x in thetas_np.mean(axis=0)],
            'theta_std': [float(x) for x in thetas_np.std(axis=0)],
            'theta_best_vr': [float(x) for x in thetas_np[j_best_vr]],
            'best_vr': float(vr[j_best_vr]),
            'vr_at_map': float(vr[j_map]),
            'ess': ess,
            'n_theta_samples': n_samples,
            'importance_primary': bool(icfg['importance_reweight']),
            'timing_s': {'flow': t_flow, 'wls': t_wls, 'total': elapsed},
            'band_hz': list(ex['band_hz']),
        },
        # per-station VR table at the MAP theta (n_sta, n_comp)
        'vr_station_map': vr_trace[j_map].reshape(-1, n_comp),
    }
    logger.info('event %s: %d theta samples in %.2f s (flow %.3f s, '
                'wls %.2f s), best VR %.3f, ESS %.0f',
                result['event'], n_samples, elapsed, t_flow, t_wls,
                float(vr[j_best_vr]), ess)
    return result


def save_results(result: dict[str, Any], outdir: str) -> str:
    """results/<event_id>/: posterior.h5 (samples + per-sample WLS output)
    and summary.json (point estimates, decomposition, timing)."""
    os.makedirs(outdir, exist_ok=True)
    h5_path = os.path.join(outdir, 'posterior.h5')
    with h5py.File(h5_path, 'w') as f:
        for key in ('theta_samples', 'log_q', 'm_hats', 'covs',
                    'm6_samples', 'vr', 'vr_trace', 'misfit',
                    'station_shift', 'log_w_is', 'w_is',
                    'vr_station_map'):
            f.create_dataset(key, data=np.asarray(result[key]))
        f.create_dataset('station_codes', data=np.array(
            result['station_codes'], dtype='S'))
        f.attrs['event'] = result['event']
        f.attrs['theta_names'] = list(result['theta_names'])
        f.attrs['ess'] = result['ess']
    with open(os.path.join(outdir, 'summary.json'), 'w') as f:
        json.dump(result['summary'], f, indent=2)
    return h5_path


def load_results(outdir: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with h5py.File(os.path.join(outdir, 'posterior.h5'), 'r') as f:
        for key in f.keys():
            out[key] = f[key][()]
        out['station_codes'] = [c.decode()
                                for c in out.pop('station_codes')]
        out['event'] = str(f.attrs['event'])
        out['theta_names'] = [str(n) for n in f.attrs['theta_names']]
    with open(os.path.join(outdir, 'summary.json')) as f:
        out['summary'] = json.load(f)
    return out
