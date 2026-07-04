"""Publication figures (PLAN §10, Phase 8). ALL figures live here.

Every figure is saved as PDF + PNG (config plots.dpi) through
:func:`savefig`, styled by the single rcParams block below. Maps use
cartopy when available (no pygmt in this environment) and fall back to
plain lon/lat axes.
"""
from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from . import KM, resolve_path  # noqa: E402
from .forward import (  # noqa: E402
    m0_to_mw, m6_to_matrix, project_deviatoric, scalar_moment)

logger = logging.getLogger('sbi_mt.plots')

#: the single publication style block (PLAN §10)
RC_PARAMS = {
    'font.size': 9,
    'font.family': 'sans-serif',
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.frameon': False,
    'lines.linewidth': 1.0,
    'figure.constrained_layout.use': True,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
}

COLOR_OBS = 'black'
COLOR_ENSEMBLE = '0.6'
COLOR_BEST = 'crimson'
COLOR_REF = 'steelblue'


def savefig(fig: plt.Figure, cfg: dict[str, Any], name: str) -> list[str]:
    """Save to <paths.figures>/<name>.{pdf,png} (config plots.formats)."""
    outdir = resolve_path(cfg, cfg['paths']['figures'])
    os.makedirs(outdir, exist_ok=True)
    dpi = int(cfg['plots']['dpi'])
    paths = []
    for ext in cfg['plots']['formats']:
        p = os.path.join(outdir, f'{name}.{ext}')
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    plt.close(fig)
    logger.info('figure %s -> %s', name, ', '.join(paths))
    return paths


def _pyrocko_mt(m6: np.ndarray) -> Any:
    from pyrocko import moment_tensor as pmt
    return pmt.MomentTensor(mnn=float(m6[0]), mee=float(m6[1]),
                            mdd=float(m6[2]), mne=float(m6[3]),
                            mnd=float(m6[4]), med=float(m6[5]))


def _fuzzy_beachball(ax: plt.Axes, m6_samples: np.ndarray,
                     best_m6: np.ndarray | None = None,
                     n_max: int = 200) -> None:
    from pyrocko.plot import beachball
    idx = np.linspace(0, len(m6_samples) - 1,
                      min(n_max, len(m6_samples))).astype(int)
    mts = [_pyrocko_mt(m6_samples[i]) for i in idx]
    kwargs = dict(beachball_type='full', size=100, size_units='data',
                  position=(0.0, 0.0), color_t='black',
                  grid_resolution=200)
    beachball.plot_fuzzy_beachball_mpl_pixmap(
        mts, ax, best_mt=_pyrocko_mt(best_m6)
        if best_m6 is not None else None, **kwargs)
    ax.set_xlim(-55, 55)
    ax.set_ylim(-55, 55)
    ax.set_aspect('equal')
    ax.axis('off')


def _solid_beachball(ax: plt.Axes, m6: np.ndarray,
                     color: str = COLOR_REF) -> None:
    from pyrocko.plot import beachball
    beachball.plot_beachball_mpl(
        _pyrocko_mt(m6), ax, beachball_type='full', size=100,
        size_units='data', position=(0.0, 0.0), color_t=color,
        linewidth=0.8)
    ax.set_xlim(-55, 55)
    ax.set_ylim(-55, 55)
    ax.set_aspect('equal')
    ax.axis('off')


# ----------------------------------------------------------------------------
# 1. Grond-style waveform fit (flagship)
# ----------------------------------------------------------------------------

def plot_waveform_fits(ex: dict[str, Any], result: dict[str, Any],
                       cfg: dict[str, Any], fetcher: Any,
                       name: str | None = None) -> plt.Figure:
    """Observed (black) vs posterior synthetics ensemble (gray) and MAP
    synthetic (red); per-panel station code, distance, azimuth, VR and the
    applied station time shift as a colored marker (blue = negative,
    red = positive, Grond style); shaded tapers; fuzzy beachball header."""
    from .train import event_forward_model
    plt.rcParams.update(RC_PARAMS)
    proc = cfg['processing']
    comps = list(proc['components'])
    n_comp = len(comps)
    n_draws = int(cfg['plots']['waveform_fit_n_draws'])

    order = np.argsort(np.asarray(ex['sta_distance']))
    n_sta = len(order)
    fm, data, _w = event_forward_model(
        ex, cfg, fetcher, np.arange(n_sta), 'cpu')
    obs = data.numpy()

    thetas = np.asarray(result['theta_samples'])
    m_hats = np.asarray(result['m_hats'])
    shifts = np.asarray(result['station_shift'])
    j_map = int(np.argmax(np.asarray(result['log_q'])))
    idx = np.linspace(0, len(thetas) - 1,
                      min(n_draws, len(thetas))).astype(int)
    with torch.no_grad():
        ens = fm.synthetics(
            torch.as_tensor(thetas[idx], dtype=torch.float32),
            torch.as_tensor(m_hats[idx], dtype=torch.float32),
            station_shifts=torch.as_tensor(shifts[idx],
                                           dtype=torch.float32)).numpy()
        best = fm.synthetics(
            torch.as_tensor(thetas[[j_map]], dtype=torch.float32),
            torch.as_tensor(m_hats[[j_map]], dtype=torch.float32),
            station_shifts=torch.as_tensor(
                shifts[[j_map]], dtype=torch.float32)).numpy()[0]

    vr_map = np.asarray(result['vr_trace'])[j_map]
    deltat = fm.deltat
    itmin = np.asarray(ex['itmin'])
    n_t = obs.shape[1]
    taper = float(proc['taper_fraction'])
    shift_max = float(cfg['wls']['station_shift_max_s']) or 1.0

    fig_h = 1.0 + 0.75 * n_sta
    fig, axes = plt.subplots(
        n_sta, n_comp, figsize=(7.2, fig_h), sharex=False, squeeze=False,
        gridspec_kw={'hspace': 0.55})
    for row, s in enumerate(order):
        t = (itmin[s] + np.arange(n_t)) * deltat
        row_scale = max(np.abs(obs[s * n_comp + np.arange(n_comp)]).max(),
                        np.finfo(float).tiny)
        for col in range(n_comp):
            ax = axes[row, col]
            k = s * n_comp + col
            ax.fill_betweenx([-1.15, 1.15], t[0],
                             t[0] + taper * n_t * deltat,
                             color='0.92', zorder=0)
            ax.fill_betweenx([-1.15, 1.15], t[-1] - taper * n_t * deltat,
                             t[-1], color='0.92', zorder=0)
            for e in ens:
                ax.plot(t, e[k] / row_scale, color=COLOR_ENSEMBLE,
                        lw=0.3, alpha=0.35, zorder=1)
            ax.plot(t, obs[k] / row_scale, color=COLOR_OBS, lw=0.8,
                    zorder=3)
            ax.plot(t, best[k] / row_scale, color=COLOR_BEST, lw=0.7,
                    zorder=2)
            # station time shift marker, Grond palette
            sh = float(shifts[j_map, s])
            ax.scatter([t[0] + 0.97 * (t[-1] - t[0])], [0.85],
                       s=14, marker='o',
                       color=plt.get_cmap('coolwarm')(
                           0.5 + 0.5 * np.clip(sh / shift_max, -1, 1)),
                       edgecolor='k', linewidth=0.3, zorder=4)
            ax.text(0.99, 0.02, f'VR {100 * vr_map[k]:.0f}%',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=6)
            if row == 0:
                ax.set_title(comps[col], fontsize=9)
            if col == 0:
                ax.text(0.01, 0.98,
                        f"{ex['station_codes'][s]}  "
                        f"{ex['sta_distance'][s] / KM:.0f} km  "
                        f"{np.degrees(ex['sta_azimuth'][s]):.0f}°  "
                        f"Δt {sh:+.1f} s",
                        transform=ax.transAxes, ha='left', va='top',
                        fontsize=6.5)
            ax.set_ylim(-1.15, 1.15)
            ax.set_yticks([])
            if row == n_sta - 1:
                ax.set_xlabel('time since origin [s]')
            else:
                ax.set_xticklabels([])
            for side in ('top', 'right', 'left'):
                ax.spines[side].set_visible(False)
        axes[row, 0].set_ylabel(f'{row_scale:.1e} m', fontsize=6)

    s = result['summary']
    fig.suptitle(
        f"{result['event']}   Mw {s['mean']['mw']:.2f}   "
        f"depth {(float(ex['depth']) + s['theta_mean'][2]) / KM:.0f} km   "
        f"best VR {s['best_vr']:.2f}",
        fontsize=10)
    # fuzzy beachball inset in the header
    try:
        ax_bb = fig.add_axes([0.008, 0.955, 0.04, 0.04])
        _fuzzy_beachball(ax_bb, np.asarray(result['m6_samples']),
                         best_m6=np.asarray(s['mean']['m6']), n_max=60)
    except Exception as exc:                      # cosmetic only
        logger.warning('beachball inset failed: %s', exc)
    if name:
        savefig(fig, cfg, name)
    return fig


# ----------------------------------------------------------------------------
# 2. fuzzy beachball vs reference
# ----------------------------------------------------------------------------

def plot_fuzzy_beachball(result: dict[str, Any], cfg: dict[str, Any],
                         m6_ref: np.ndarray | None = None,
                         ref_label: str = 'reference',
                         name: str | None = None) -> plt.Figure:
    plt.rcParams.update(RC_PARAMS)
    two = m6_ref is not None and np.all(np.isfinite(m6_ref))
    fig, axes = plt.subplots(1, 2 if two else 1, figsize=(5.0, 2.7),
                             squeeze=False)
    s = result['summary']
    _fuzzy_beachball(axes[0, 0], np.asarray(result['m6_samples']),
                     best_m6=np.asarray(s['mean']['m6']))
    axes[0, 0].set_title(
        f"SBI posterior\nMw {s['mean']['mw']:.2f}, "
        f"DC {s['mean']['decomposition']['dc']:.0f}%")
    if two:
        _solid_beachball(axes[0, 1], np.asarray(m6_ref))
        axes[0, 1].set_title(
            f'{ref_label}\nMw '
            f'{float(m0_to_mw(scalar_moment(np.asarray(m6_ref)))):.2f}')
    fig.suptitle(result['event'])
    if name:
        savefig(fig, cfg, name)
    return fig


# ----------------------------------------------------------------------------
# 3. lune / Hudson source-type plots
# ----------------------------------------------------------------------------

def lune_coordinates(m6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tape & Tape (2012) lune longitude gamma [deg, -30..30] and latitude
    delta [deg, -90..90] from the eigenvalues."""
    m = m6_to_matrix(np.asarray(m6, dtype=np.float64))
    lam = np.sort(np.linalg.eigvalsh(m), axis=-1)[..., ::-1]  # l1>=l2>=l3
    l1, l2, l3 = lam[..., 0], lam[..., 1], lam[..., 2]
    norm = np.sqrt(l1**2 + l2**2 + l3**2)
    norm = np.where(norm > 0, norm, 1.0)
    gamma = np.degrees(np.arctan2(-l1 + 2 * l2 - l3,
                                  np.sqrt(3.0) * (l1 - l3)))
    beta = np.arccos(np.clip((l1 + l2 + l3) / (np.sqrt(3.0) * norm),
                             -1.0, 1.0))
    return gamma, 90.0 - np.degrees(beta)


def hudson_coordinates(m6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hudson (u, v) skewed-diamond coordinates (u = tau(1-|k|), v = k)."""
    m = m6_to_matrix(np.asarray(m6, dtype=np.float64))
    iso = np.trace(m, axis1=-2, axis2=-1) / 3.0
    dev = np.sort(np.linalg.eigvalsh(
        m6_to_matrix(project_deviatoric(np.asarray(m6)))), axis=-1)
    e1, e2, e3 = dev[..., 0], dev[..., 1], dev[..., 2]
    denom = np.maximum(np.maximum(np.abs(e1), np.abs(e3)),
                       np.finfo(float).tiny)
    tau = -2.0 * e2 / denom
    k = iso / (np.abs(iso) + denom)
    return tau * (1.0 - np.abs(k)), k


def plot_lune_hudson(result: dict[str, Any], cfg: dict[str, Any],
                     m6_ref: np.ndarray | None = None,
                     name: str | None = None) -> plt.Figure:
    plt.rcParams.update(RC_PARAMS)
    samples = np.asarray(result['m6_samples'])
    fig, (ax_l, ax_h) = plt.subplots(1, 2, figsize=(7.0, 3.4))

    g, d = lune_coordinates(samples)
    ax_l.hexbin(g, d, gridsize=35, extent=(-30, 30, -90, 90),
                cmap='Greys', mincnt=1)
    ax_l.plot([-30, 30, 30, -30, -30], [-90, -90, 90, 90, -90],
              color='k', lw=0.8)
    for gg, dd, lab in ((0, 0, 'DC'), (-30, 0, '-CLVD'), (30, 0, '+CLVD'),
                        (0, 90, '+ISO'), (0, -90, '-ISO')):
        ax_l.annotate(lab, (gg, dd), fontsize=7, ha='center',
                      xytext=(0, 4), textcoords='offset points')
    ax_l.set_xlabel(r'lune longitude $\gamma$ [deg]')
    ax_l.set_ylabel(r'lune latitude $\delta$ [deg]')
    ax_l.set_xlim(-33, 33)
    ax_l.set_ylim(-95, 95)
    ax_l.set_title('Tape & Tape lune')

    u, v = hudson_coordinates(samples)
    ax_h.hexbin(u, v, gridsize=35, extent=(-4 / 3, 4 / 3, -1, 1),
                cmap='Greys', mincnt=1)
    ax_h.plot([0, 4 / 3, 0, -4 / 3, 0], [-1, 0, 1, 0, -1],
              color='k', lw=0.8)
    ax_h.set_xlabel('Hudson $u$')
    ax_h.set_ylabel('Hudson $v$ (ISO)')
    ax_h.set_title('Hudson diagram')

    if m6_ref is not None and np.all(np.isfinite(m6_ref)):
        gr, dr = lune_coordinates(np.asarray(m6_ref)[None, :])
        ur, vr_ = hudson_coordinates(np.asarray(m6_ref)[None, :])
        ax_l.plot(gr, dr, marker='*', ms=12, color=COLOR_BEST,
                  mec='k', mew=0.5, ls='none', label='reference')
        ax_h.plot(ur, vr_, marker='*', ms=12, color=COLOR_BEST,
                  mec='k', mew=0.5, ls='none', label='reference')
        ax_l.legend(loc='lower left')
    fig.suptitle(result['event'])
    if name:
        savefig(fig, cfg, name)
    return fig


# ----------------------------------------------------------------------------
# 4. nuisance corner plot
# ----------------------------------------------------------------------------

_THETA_LABELS = {'dn_km': r'$\Delta$N [km]', 'de_km': r'$\Delta$E [km]',
                 'dz_km': r'$\Delta$z [km]', 'dt0_s': r'$\Delta t_0$ [s]'}


def plot_nuisance_corner(result: dict[str, Any], cfg: dict[str, Any],
                         theta_true: np.ndarray | None = None,
                         name: str | None = None) -> plt.Figure:
    """Corner plot of q(theta | d); the catalog solution sits at the origin
    (theta is the perturbation from it)."""
    plt.rcParams.update(RC_PARAMS)
    thetas = np.asarray(result['theta_samples']).copy()
    names = list(result['theta_names'])
    k = thetas.shape[1]
    for i, nm in enumerate(names):        # SI -> km/s for display
        if nm.endswith('_km'):
            thetas[:, i] /= KM
    labels = [_THETA_LABELS.get(nm, nm) for nm in names]
    tt = None
    if theta_true is not None:
        tt = np.asarray(theta_true, dtype=float).copy()
        for i, nm in enumerate(names):
            if nm.endswith('_km'):
                tt[i] /= KM

    fig, axes = plt.subplots(k, k, figsize=(1.6 * k + 0.6, 1.6 * k + 0.6))
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if j > i:
                ax.axis('off')
                continue
            if i == j:
                ax.hist(thetas[:, i], bins=40, color='0.4',
                        histtype='stepfilled', alpha=0.8)
                ax.axvline(0.0, color=COLOR_REF, lw=0.8)   # catalog
                if tt is not None:
                    ax.axvline(tt[i], color=COLOR_BEST, lw=0.8)
                ax.set_yticks([])
            else:
                ax.hist2d(thetas[:, j], thetas[:, i], bins=40,
                          cmap='Greys')
                ax.axvline(0.0, color=COLOR_REF, lw=0.6)
                ax.axhline(0.0, color=COLOR_REF, lw=0.6)
                if tt is not None:
                    ax.plot(tt[j], tt[i], marker='x', color=COLOR_BEST,
                            ms=6, mew=1.2)
            if i == k - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])
            elif j != 0:
                ax.set_yticklabels([])
    fig.suptitle(f"{result['event']} — nuisance posterior "
                 '(blue: catalog, red: truth)')
    if name:
        savefig(fig, cfg, name)
    return fig


# ----------------------------------------------------------------------------
# 5. station map
# ----------------------------------------------------------------------------

def plot_station_map(ex: dict[str, Any], result: dict[str, Any],
                     cfg: dict[str, Any],
                     name: str | None = None) -> plt.Figure:
    """Epicenter posterior cloud, stations colored by VR at the MAP theta,
    beachball at the MAP location. Cartopy if available."""
    plt.rcParams.update(RC_PARAMS)
    lat0, lon0 = float(ex['lat']), float(ex['lon'])
    coslat = math.cos(math.radians(lat0))
    thetas = np.asarray(result['theta_samples'])
    lat_cloud = lat0 + thetas[:, 0] / 111195.0
    lon_cloud = lon0 + thetas[:, 1] / (111195.0 * coslat)
    dist = np.asarray(ex['sta_distance'])
    az = np.asarray(ex['sta_azimuth'])
    sta_lat = lat0 + dist * np.cos(az) / 111195.0
    sta_lon = lon0 + dist * np.sin(az) / (111195.0 * coslat)
    vr_sta = np.asarray(result['vr_station_map']).mean(axis=1)

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(6.0, 5.2))
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.add_feature(cfeature.LAND, facecolor='0.93')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='0.8')
        gl.top_labels = gl.right_labels = False
        tr = dict(transform=proj)
    except Exception:
        fig, ax = plt.subplots(figsize=(6.0, 5.2))
        ax.set_xlabel('longitude [deg]')
        ax.set_ylabel('latitude [deg]')
        ax.set_aspect(1.0 / coslat)
        tr = {}

    ax.scatter(lon_cloud, lat_cloud, s=2, color=COLOR_BEST, alpha=0.15,
               zorder=3, label='centroid posterior', **tr)
    sc = ax.scatter(sta_lon, sta_lat, c=vr_sta, cmap='viridis',
                    vmin=0.0, vmax=1.0, marker='^', s=55,
                    edgecolor='k', linewidth=0.5, zorder=4,
                    label='stations', **tr)
    for i, code in enumerate(ex['station_codes']):
        ax.annotate(code.split('.')[-1], (sta_lon[i], sta_lat[i]),
                    fontsize=6, xytext=(0, 5),
                    textcoords='offset points', ha='center')
    ax.plot(lon0, lat0, marker='*', ms=13, color='gold', mec='k',
            mew=0.6, ls='none', zorder=5, label='catalog', **tr)
    fig.colorbar(sc, ax=ax, shrink=0.7, label='VR at MAP')

    pad = 0.4
    ax.set_xlim(min(sta_lon.min(), lon0) - pad,
                max(sta_lon.max(), lon0) + pad)
    ax.set_ylim(min(sta_lat.min(), lat0) - pad,
                max(sta_lat.max(), lat0) + pad)

    # beachball at the MAP centroid (inset axes anchored in figure coords)
    try:
        theta_map = np.asarray(result['summary']['theta_map'])
        blat = lat0 + theta_map[0] / 111195.0
        blon = lon0 + theta_map[1] / (111195.0 * coslat)
        x, y = ax.transData.transform((blon, blat))
        xf, yf = fig.transFigure.inverted().transform((x, y))
        ax_bb = fig.add_axes([xf - 0.035, yf - 0.035, 0.07, 0.07])
        _solid_beachball(ax_bb, np.asarray(result['summary']['map']['m6']),
                         color=COLOR_BEST)
    except Exception as exc:
        logger.warning('map beachball failed: %s', exc)
    ax.legend(loc='upper left')
    ax.set_title(result['event'])
    if name:
        savefig(fig, cfg, name)
    return fig


# ----------------------------------------------------------------------------
# 6. SBC rank histograms
# ----------------------------------------------------------------------------

_M6_LABELS = [r'$m_{nn}$', r'$m_{ee}$', r'$m_{dd}$', r'$m_{ne}$',
              r'$m_{nd}$', r'$m_{ed}$']


def plot_sbc(arrays: dict[str, np.ndarray], cfg: dict[str, Any],
             n_bins: int = 20, name: str | None = None) -> plt.Figure:
    """Rank histograms per theta dim and MT component with the 99%
    binomial uniformity envelope."""
    from scipy import stats
    plt.rcParams.update(RC_PARAMS)
    theta_names = list(cfg['theta']['names'])
    ranks_all = [(arrays['ranks_theta'], theta_names),
                 (arrays['ranks_m6'], _M6_LABELS)]
    n_cols = max(len(lbl) for _, lbl in ranks_all)
    n_samp = int(arrays['n_theta_samples'])
    fig, axes = plt.subplots(2, n_cols, figsize=(1.7 * n_cols + 0.5, 3.6),
                             squeeze=False)
    for r, (ranks, labels) in enumerate(ranks_all):
        n_ev = ranks.shape[0]
        lo = stats.binom.ppf(0.005, n_ev, 1.0 / n_bins)
        hi = stats.binom.ppf(0.995, n_ev, 1.0 / n_bins)
        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(labels):
                ax.axis('off')
                continue
            ax.hist(ranks[:, c], bins=np.linspace(0, n_samp + 1,
                                                  n_bins + 1),
                    color='0.45', histtype='stepfilled')
            ax.axhspan(lo, hi, color=COLOR_REF, alpha=0.2, lw=0)
            ax.axhline(n_ev / n_bins, color=COLOR_REF, lw=0.7)
            ax.set_title(labels[c], fontsize=8)
            ax.set_yticks([])
            ax.set_xlim(0, n_samp + 1)
            if r == 1:
                ax.set_xlabel('rank')
    fig.suptitle('SBC rank histograms (band: 99% uniformity envelope)')
    if name:
        savefig(fig, cfg, name)
    return fig


# ----------------------------------------------------------------------------
# 7. coverage
# ----------------------------------------------------------------------------

def plot_coverage(arrays: dict[str, np.ndarray], cfg: dict[str, Any],
                  which: str = 'hits_m6',
                  name: str | None = None) -> plt.Figure:
    plt.rcParams.update(RC_PARAMS)
    levels = np.asarray(arrays['coverage_levels'], dtype=float)
    cov = np.asarray(arrays[which], dtype=float).mean(axis=0)  # (L, k)
    labels = (_M6_LABELS if which == 'hits_m6'
              else list(cfg['theta']['names']))
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], color='0.5', lw=0.8, ls='--')
    for k in range(cov.shape[1]):
        ax.plot(levels, cov[:, k], marker='o', ms=3, label=labels[k])
    ax.set_xlabel('nominal credible level')
    ax.set_ylabel('empirical coverage')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend(ncols=2)
    ax.set_title('MT component coverage' if which == 'hits_m6'
                 else 'nuisance coverage')
    if name:
        savefig(fig, cfg, name)
    return fig


# ----------------------------------------------------------------------------
# 8. accuracy curves
# ----------------------------------------------------------------------------

def plot_accuracy_curves(arrays: dict[str, np.ndarray],
                         cfg: dict[str, Any],
                         name: str | None = None) -> plt.Figure:
    from .evaluate import binned_kagan
    plt.rcParams.update(RC_PARAMS)
    fig, (ax_s, ax_n) = plt.subplots(1, 2, figsize=(6.8, 3.0))
    for ax, key, edges, xlabel, logx in (
            (ax_s, 'snr_median', cfg['evaluate']['snr_bins'],
             'median SNR', True),
            (ax_n, 'n_stations', cfg['evaluate']['station_bins'],
             'station count', False)):
        b = binned_kagan(arrays, key, edges)
        for ip, (label, color) in enumerate(
                (('reference model', COLOR_REF),
                 ('perturbed model', COLOR_BEST))):
            sel = b['count'][ip] > 0
            ax.plot(b['centers'][sel], b['median_kagan'][ip][sel],
                    marker='o', ms=4, color=color, label=label)
        if logx:
            ax.set_xscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('median Kagan angle [deg]')
        ax.legend()
    fig.suptitle('posterior-mean MT accuracy')
    if name:
        savefig(fig, cfg, name)
    return fig


# ----------------------------------------------------------------------------
# 9. benchmark vs Grond / reference
# ----------------------------------------------------------------------------

def plot_benchmark(rows: Sequence[dict[str, Any]], cfg: dict[str, Any],
                   name: str | None = None) -> plt.Figure:
    plt.rcParams.update(RC_PARAMS)
    rows = [r for r in rows if 'kagan_ref_deg' in r]
    events = [r['event'] for r in rows]
    x = np.arange(len(rows))
    fig, (ax_k, ax_w, ax_t) = plt.subplots(
        1, 3, figsize=(9.5, 3.2),
        gridspec_kw={'width_ratios': [2, 1.4, 1.4]})

    width = 0.38
    ax_k.bar(x - width / 2, [r['kagan_ref_deg'] for r in rows], width,
             color=COLOR_REF, label='SBI')
    kg = [r.get('kagan_grond_ref_deg', np.nan) for r in rows]
    ax_k.bar(x + width / 2, kg, width, color=COLOR_BEST, label='Grond')
    ax_k.set_xticks(x)
    ax_k.set_xticklabels(events, rotation=45, ha='right', fontsize=6)
    ax_k.set_ylabel('Kagan angle to reference [deg]')
    ax_k.legend()

    ci_s = [np.mean(r['ci68_sbi']) for r in rows]
    ci_g = [np.mean(r['ci68_grond']) if r.get('ci68_grond') else np.nan
            for r in rows]
    ax_w.scatter(ci_g, ci_s, s=18, color='0.3')
    lim = np.nanmax([*ci_s, *ci_g, 0.1]) * 1.15
    ax_w.plot([0, lim], [0, lim], color='0.6', lw=0.7, ls='--')
    ax_w.set_xlabel('Grond CI68 width / M0')
    ax_w.set_ylabel('SBI CI68 width / M0')
    ax_w.set_xlim(0, lim)
    ax_w.set_ylim(0, lim)

    t_s = [r['time_sbi_s'] for r in rows]
    t_g = [r.get('time_grond_s', np.nan) for r in rows]
    ax_t.bar(x - width / 2, t_s, width, color=COLOR_REF, label='SBI')
    ax_t.bar(x + width / 2, t_g, width, color=COLOR_BEST, label='Grond')
    ax_t.set_yscale('log')
    ax_t.set_xticks(x)
    ax_t.set_xticklabels(events, rotation=45, ha='right', fontsize=6)
    ax_t.set_ylabel('wall-clock per event [s]')
    ax_t.legend()
    fig.suptitle('real-event benchmark')
    if name:
        savefig(fig, cfg, name)
    return fig


# ----------------------------------------------------------------------------
# 10. ablation summary
# ----------------------------------------------------------------------------

def plot_ablation_summary(baseline: dict[str, np.ndarray],
                          ablations: dict[str, dict[str, np.ndarray]],
                          cfg: dict[str, Any],
                          name: str | None = None) -> plt.Figure:
    """(a) Gaussian-noise training -> coverage degradation; (b) no velocity
    perturbation -> bias on perturbed test events; (c) direct regression vs
    WLS layer -> accuracy gap. Missing ablations are skipped with a note."""
    plt.rcParams.update(RC_PARAMS)
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
    levels = np.asarray(baseline['coverage_levels'], dtype=float)

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color='0.5', lw=0.8, ls='--')
    ax.plot(levels, baseline['hits_m6'].mean(axis=(0, 2)), marker='o',
            ms=3, color=COLOR_REF, label='noise library')
    a = ablations.get('gaussian_noise')
    if a is not None:
        ax.plot(np.asarray(a['coverage_levels'], dtype=float),
                a['hits_m6'].mean(axis=(0, 2)), marker='s', ms=3,
                color=COLOR_BEST, label='Gaussian noise')
    else:
        ax.text(0.5, 0.2, 'ablation (a)\nnot available', ha='center',
                transform=ax.transAxes, fontsize=8)
    ax.set_xlabel('nominal level')
    ax.set_ylabel('empirical coverage (m6)')
    ax.set_title('(a) empirical noise')
    ax.legend()

    ax = axes[1]
    pert = np.asarray(baseline['perturbed'], dtype=bool)
    pairs = [('with vel. pert.', baseline)]
    b = ablations.get('no_velocity_perturbation')
    if b is not None:
        pairs.append(('reference-only', b))
    else:
        ax.text(0.5, 0.2, 'ablation (b)\nnot available', ha='center',
                transform=ax.transAxes, fontsize=8)
    xs = np.arange(len(pairs))
    for split, off, color in ((~pert, -0.18, COLOR_REF),
                              (pert, 0.18, COLOR_BEST)):
        vals = []
        for _, arr in pairs:
            p = np.asarray(arr['perturbed'], dtype=bool)
            sel = ~p if color == COLOR_REF else p
            k = np.asarray(arr['kagan_deg'], dtype=float)[sel]
            vals.append(np.median(k) if k.size else np.nan)
        ax.bar(xs + off, vals, 0.34, color=color,
               label='reference test' if color == COLOR_REF
               else 'perturbed test')
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in pairs], fontsize=8)
    ax.set_ylabel('median Kagan [deg]')
    ax.set_title('(b) velocity robustness')
    ax.legend()

    ax = axes[2]
    c = ablations.get('direct_regression')
    if c is not None and np.isfinite(
            np.asarray(c['kagan_direct_deg'], dtype=float)).any():
        for key, label, color in (
                ('kagan_deg', 'WLS layer (SBI)', COLOR_REF),
                ('kagan_direct_deg', 'direct regression', COLOR_BEST)):
            k = np.sort(np.asarray(c[key], dtype=float))
            k = k[np.isfinite(k)]
            ax.plot(k, np.linspace(0, 1, k.size), color=color,
                    label=label)
        ax.set_xlabel('Kagan angle [deg]')
        ax.set_ylabel('empirical CDF')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'ablation (c)\nnot available', ha='center',
                transform=ax.transAxes, fontsize=8)
    ax.set_title('(c) WLS vs regression')
    if name:
        savefig(fig, cfg, name)
    return fig


# ----------------------------------------------------------------------------
# 11. training diagnostics
# ----------------------------------------------------------------------------

def plot_training_diagnostics(run_dir: str, cfg: dict[str, Any],
                              name: str | None = None) -> plt.Figure:
    plt.rcParams.update(RC_PARAMS)
    with open(os.path.join(run_dir, 'log.jsonl')) as f:
        records = [json.loads(line) for line in f]
    train = [r for r in records if 'npe_loss' in r]
    evals = [r for r in records if 'val_loss' in r]

    fig, (ax_l, ax_k, ax_c) = plt.subplots(1, 3, figsize=(9.5, 2.9))
    ax_l.plot([r['step'] for r in train], [r['npe_loss'] for r in train],
              color='0.5', lw=0.7, label='train NPE')
    if evals:
        ax_l.plot([r['step'] for r in evals],
                  [r['val_loss'] for r in evals], color=COLOR_BEST,
                  marker='o', ms=3, label='val NPE')
    ax_l.set_xlabel('step')
    ax_l.set_ylabel(r'$-\log q(\theta \mid d)$')
    ax_l.legend()

    ev_k = [r for r in evals if 'val_kagan_deg' in r]
    if ev_k:
        ax_k.plot([r['step'] for r in ev_k],
                  [r['val_kagan_deg'] for r in ev_k], color=COLOR_REF,
                  marker='o', ms=3)
    ax_k.set_xlabel('step')
    ax_k.set_ylabel('val median Kagan [deg]')

    for key, color, label in (('val_coverage_68', COLOR_REF, '68%'),
                              ('val_coverage_90', COLOR_BEST, '90%')):
        ev_c = [r for r in evals if key in r]
        if ev_c:
            ax_c.plot([r['step'] for r in ev_c],
                      [r[key] for r in ev_c], color=color, marker='o',
                      ms=3, label=label)
    ax_c.axhline(0.68, color=COLOR_REF, lw=0.6, ls='--')
    ax_c.axhline(0.90, color=COLOR_BEST, lw=0.6, ls='--')
    ax_c.set_xlabel('step')
    ax_c.set_ylabel('val coverage')
    ax_c.set_ylim(0, 1)
    ax_c.legend()
    fig.suptitle(f'training diagnostics — {os.path.basename(run_dir)}')
    if name:
        savefig(fig, cfg, name)
    return fig
