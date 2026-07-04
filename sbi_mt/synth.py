"""Synthetic training-set generation (PLAN §5, Phase 3).

Training events are generated ON THE FLY in the dataloader; only the
validation split is cached (script 03). Per event, data are simulated from
(anchor + theta) with a DATA store that is velocity-perturbed for a
configurable fraction of samples, on a minimal 2x2-node point grid
(bit-identical bilinear interpolation to the full grid, ~KB instead of
~half a GB per event); the WLS layer and aux loss always use the REFERENCE
store — that mismatch is what teaches theta to absorb theory error.
"""
from __future__ import annotations

import glob
import logging
import math
import os
from typing import Any, Iterator, Sequence

import h5py
import numpy as np
import torch
from matplotlib.path import Path as _MplPath

from . import KM, resolve_path
from .data import GaussianNoise, NoiseLibrary, trace_metadata
from .forward import ForwardModel, matrix_to_m6, mw_to_m0, scalar_moment
from .gf import (
    EventInfo, GFCube, Station, StoreFetcher, build_point_grid, cube_path,
    data_windows, station_geometries)

logger = logging.getLogger('sbi_mt.synth')

Fetcher = GFCube | StoreFetcher

#: off-diagonal scale of the orthonormal Frobenius basis
#: (m11, m22, m33, sqrt(2) m12, sqrt(2) m13, sqrt(2) m23)
_OFF_DIAG = 1.0 / math.sqrt(2.0)


# ----------------------------------------------------------------------------
# priors
# ----------------------------------------------------------------------------

def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform (Haar) random rotation via sign-fixed QR of a Gaussian."""
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diagonal(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def sample_mt(rng: np.random.Generator,
              mw_range: Sequence[float],
              mode: str = 'uniform_tt',
              dc_fraction: float = 0.5) -> tuple[np.ndarray, float]:
    """Random moment tensor m6 (NED [Nm]) and its Mw.

    'uniform_tt': uniform on the Tape & Tape (2015) parametrization ==
    isotropic direction in the 6-D orthonormal Frobenius basis — sample a
    standard normal there (off-diagonals scaled by 1/sqrt(2) back to m6
    coordinates), normalize to unit scalar moment, scale by M0(U(mw_range)).
    'dc_dominant': with probability dc_fraction a pure DC with uniform
    random orientation, otherwise a 'uniform_tt' draw.
    """
    if mode not in ('uniform_tt', 'dc_dominant'):
        raise ValueError(f'unknown mt prior {mode!r}')
    mw = float(rng.uniform(float(mw_range[0]), float(mw_range[1])))
    if mode == 'dc_dominant' and rng.random() < dc_fraction:
        q = _random_rotation(rng)
        m6 = matrix_to_m6(q @ np.diag([-1.0, 0.0, 1.0]) @ q.T)
    else:
        x = rng.standard_normal(6)
        x /= np.linalg.norm(x)
        m6 = np.concatenate([x[:3], x[3:] * _OFF_DIAG])
    m6 = m6 / scalar_moment(m6)
    return m6 * float(mw_to_m0(mw)), mw


def sample_in_polygon(rng: np.random.Generator,
                      polygon: Sequence[Sequence[float]]
                      ) -> tuple[float, float]:
    """Uniform (lon, lat) inside the study polygon (bbox rejection)."""
    poly = np.asarray(polygon, dtype=float)
    path = _MplPath(poly)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    while True:
        p = rng.uniform(lo, hi)
        if path.contains_point(p):
            return float(p[0]), float(p[1])


def sample_nuisance(rng: np.random.Generator, priors: dict[str, Any],
                    anchor_depth: float, z_min: float, z_max: float,
                    max_tries: int = 1000) -> np.ndarray:
    """theta = (dn, de, dz, dt0) in SI units (m, m, m, s). dz is rejection-
    sampled so the perturbed depth stays inside the store depth range
    (the same truncation the inference grid imposes)."""
    dn = rng.uniform(*priors['dn_km']) * KM
    de = rng.uniform(*priors['de_km']) * KM
    for _ in range(max_tries):
        dz = rng.uniform(*priors['dz_km']) * KM
        if z_min <= anchor_depth + dz <= z_max:
            break
    else:
        raise ValueError(
            f'no admissible dz for anchor depth {anchor_depth} m in '
            f'[{z_min}, {z_max}] m')
    dt0 = rng.uniform(*priors['dt0_s'])
    return np.array([dn, de, dz, dt0], dtype=float)


def sample_station_dts(rng: np.random.Generator, priors: dict[str, Any],
                       n_sta: int) -> np.ndarray:
    """Per-station time shifts; all zero for a configurable fraction of
    samples (priors.station_dt_zero_fraction)."""
    if rng.random() < float(priors['station_dt_zero_fraction']):
        return np.zeros(n_sta)
    return rng.uniform(*priors['station_dt_s'], size=n_sta)


# ----------------------------------------------------------------------------
# fetcher discovery
# ----------------------------------------------------------------------------

def load_fetchers(cfg: dict[str, Any]) -> tuple[GFCube, list[GFCube]]:
    """Open the reference cube and all velocity-perturbed cubes. Perturbed
    ids come from gf.perturbed_ids, else by globbing <ref>_pert*_cube.h5
    next to the reference (script 01 writes them there)."""
    gcfg = cfg['gf']
    ref_id = gcfg['stores'][gcfg['store']]['id']
    ref = GFCube(cube_path(cfg, ref_id))
    ids: list[str] = list(gcfg.get('perturbed_ids') or [])
    if not ids:
        pattern = cube_path(cfg, f'{ref_id}_pert*')
        ids = [os.path.basename(p)[:-len('_cube.h5')]
               for p in sorted(glob.glob(pattern))]
    perturbed = [GFCube(cube_path(cfg, i)) for i in ids]
    logger.info('loaded reference cube %s + %d perturbed', ref_id,
                len(perturbed))
    return ref, perturbed


# ----------------------------------------------------------------------------
# generator
# ----------------------------------------------------------------------------

class SyntheticGenerator:
    """On-the-fly synthetic training events against the REAL station
    geometry (PLAN §5). One :meth:`generate` call returns one example dict:

    waveforms (n_tr, n_t) float32   preprocessed, per-trace RMS-normalized
    metadata  (n_tr, 9)  float32    see data.trace_metadata
    theta     (4,)                  (dn, de, dz, dt0) SI
    m6        (6,) [Nm], mw, mask (n_tr,), weights (n_tr,) = 1/sigma^2
    snr, norms (n_tr,), itmin/station_dt/sta_azimuth/sta_distance (n_sta,),
    station_codes, lat/lon/depth (anchor), perturbed, band_hz.
    """

    def __init__(self, cfg: dict[str, Any],
                 stations: Sequence[Station],
                 reference: Fetcher,
                 perturbed: Sequence[Fetcher] = (),
                 noise: GaussianNoise | NoiseLibrary | None = None):
        self.cfg = cfg
        self.stations = list(stations)
        self.reference = reference
        self.perturbed = list(perturbed)
        self.noise = noise if noise is not None else GaussianNoise()

        proc = cfg['processing']
        pri = cfg['priors']
        self.components: tuple[str, ...] = tuple(proc['components'])
        self.n_comp = len(self.components)
        self.snr_min = float(proc['snr_min'])
        self.polygon = np.asarray(pri['study_polygon'], dtype=float)

        d_nodes = np.asarray(reference.distances, dtype=float)
        z_nodes = np.asarray(reference.depths, dtype=float)
        dd = float(d_nodes[1] - d_nodes[0])
        # stations must keep their PERTURBED distance inside the store range
        disp_max = math.hypot(
            max(abs(float(pri['dn_km'][0])), abs(float(pri['dn_km'][1]))),
            max(abs(float(pri['de_km'][0])),
                abs(float(pri['de_km'][1])))) * KM
        self.dist_lo = float(d_nodes[0]) + disp_max + 2 * dd
        self.dist_hi = float(d_nodes[-1]) - disp_max - 2 * dd
        self.z_lo = float(z_nodes[0])
        self.z_hi = float(z_nodes[-1])

    def reopen(self) -> None:
        """Re-open all HDF5-backed fetchers (dataloader worker processes;
        h5py handles inherited across fork are not safe)."""
        for f in (self.reference, *self.perturbed):
            if hasattr(f, 'reopen'):
                f.reopen()

    def generate(self, rng: np.random.Generator,
                 max_tries: int = 100) -> dict[str, Any]:
        for _ in range(max_tries):
            ex = self._try_generate(rng)
            if ex is not None:
                return ex
        raise RuntimeError(
            f'no valid synthetic event in {max_tries} tries — check that '
            'the station geometry reaches the study polygon within the '
            'store distance range')

    def _try_generate(self, rng: np.random.Generator
                      ) -> dict[str, Any] | None:
        cfg = self.cfg
        pri = cfg['priors']
        scfg = cfg['synth']

        # anchor ("catalog") event in the study polygon
        lon, lat = sample_in_polygon(rng, self.polygon)
        depth = float(rng.uniform(*pri['depth_km'])) * KM
        anchor = EventInfo(name='synth', lat=lat, lon=lon, depth=depth)

        # station subset of the real inventory (PLAN §5.2)
        geoms_all = station_geometries(anchor, self.stations)
        eligible = [g for g in geoms_all
                    if self.dist_lo <= g.distance <= self.dist_hi]
        n_min, n_max = (int(x) for x in scfg['n_stations'])
        if len(eligible) < n_min:
            return None
        n_target = int(rng.integers(n_min, min(n_max, len(eligible)) + 1))
        sel = rng.choice(len(eligible), size=n_target, replace=False)
        keep = rng.random(n_target) >= float(scfg['station_dropout'])
        geoms = [eligible[i] for i, k in zip(sel, keep) if k]
        if len(geoms) < n_min:
            return None
        n_sta = len(geoms)

        # source + nuisance draws
        m6, mw = sample_mt(rng, pri['mw'], pri['mt_prior'],
                           float(pri['dc_dominant_fraction']))
        theta = sample_nuisance(rng, pri, depth, self.z_lo, self.z_hi)
        station_dt = sample_station_dts(rng, pri, n_sta)

        # data store: velocity-perturbed for a fraction of samples
        use_pert = bool(self.perturbed) and (
            rng.random() < float(scfg['velocity_perturbed_fraction']))
        fetcher = (self.perturbed[int(rng.integers(len(self.perturbed)))]
                   if use_pert else self.reference)

        # windows are ALWAYS defined from the reference store P times —
        # exactly what inference does with real data
        itmin, n_t = data_windows(self.reference, anchor, geoms, cfg)
        grid = build_point_grid(
            fetcher, anchor, geoms, cfg, theta[0], theta[1], theta[2],
            itmin=itmin, n_t=n_t)
        fm = ForwardModel(grid, cfg)

        codes = [g.code for g in geoms]
        with torch.no_grad():
            sig = fm.synthetics(
                torch.tensor(theta, dtype=torch.float32).unsqueeze(0),
                torch.tensor(m6, dtype=torch.float32),
                station_shifts=torch.tensor(
                    station_dt, dtype=torch.float32).unsqueeze(0),
            )[0].numpy()
            noise_f = fm.preprocess(torch.as_tensor(
                self.noise.sample(rng, codes, self.components, n_t))).numpy()

        # one event-wide noise scale so the MEDIAN per-trace SNR hits the
        # target — relative station noise levels are preserved (PLAN §5.4)
        sig_rms = np.sqrt(np.mean(sig.astype(np.float64) ** 2, axis=-1))
        noise_rms = np.sqrt(
            np.mean(noise_f.astype(np.float64) ** 2, axis=-1))
        if not np.all(noise_rms > 0):
            return None
        snr0 = sig_rms / noise_rms
        med = float(np.median(snr0))
        if not np.isfinite(med) or med <= 0:
            return None
        target = 10.0 ** rng.uniform(
            math.log10(float(scfg['snr_range'][0])),
            math.log10(float(scfg['snr_range'][1])))
        scale = med / target
        sigma = noise_rms * scale
        snr = snr0 / scale
        data = sig.astype(np.float64) + noise_f.astype(np.float64) * scale

        mask = snr >= self.snr_min
        if int(mask.sum()) < 4:
            return None
        weights = np.where(mask, 1.0 / sigma ** 2, 0.0)
        norms = np.sqrt(np.mean(data ** 2, axis=-1))
        rel_w = weights / weights.max()

        az_sta = np.array([g.azimuth for g in geoms])
        dist_sta = np.array([g.distance for g in geoms])
        comp_idx = np.tile(np.arange(self.n_comp), n_sta)
        metadata = trace_metadata(
            np.repeat(az_sta, self.n_comp),
            np.repeat(dist_sta, self.n_comp),
            comp_idx, self.n_comp, snr, rel_w, norms)

        return {
            'waveforms': (data / norms[:, None]).astype(np.float32),
            'metadata': metadata,
            'theta': theta,
            'm6': m6,
            'mw': mw,
            'mask': mask,
            'weights': weights,
            'snr': snr.astype(np.float32),
            'norms': norms.astype(np.float32),
            'itmin': itmin,
            'station_dt': station_dt,
            'station_codes': codes,
            'sta_azimuth': az_sta,
            'sta_distance': dist_sta,
            'lat': lat,
            'lon': lon,
            'depth': depth,
            'perturbed': use_pert,
            'band_hz': list(cfg['processing']['band_hz']),
        }


# ----------------------------------------------------------------------------
# validation cache (script 03) and torch plumbing
# ----------------------------------------------------------------------------

_ARRAY_KEYS = ('waveforms', 'metadata', 'theta', 'm6', 'mask', 'weights',
               'snr', 'norms', 'itmin', 'station_dt', 'sta_azimuth',
               'sta_distance')
_ATTR_KEYS = ('mw', 'lat', 'lon', 'depth', 'perturbed')


def write_event(grp: h5py.Group, ex: dict[str, Any]) -> None:
    for k in _ARRAY_KEYS:
        grp.create_dataset(k, data=np.asarray(ex[k]))
    grp.create_dataset(
        'station_codes', data=np.array(ex['station_codes'], dtype='S'))
    for k in _ATTR_KEYS:
        grp.attrs[k] = ex[k]
    grp.attrs['band_hz'] = list(ex['band_hz'])


def read_event(grp: h5py.Group) -> dict[str, Any]:
    ex: dict[str, Any] = {k: grp[k][()] for k in _ARRAY_KEYS}
    ex['mask'] = ex['mask'].astype(bool)
    ex['station_codes'] = [
        c.decode() for c in grp['station_codes'][()]]
    for k in _ATTR_KEYS:
        ex[k] = grp.attrs[k]
    ex['band_hz'] = list(grp.attrs['band_hz'])
    return ex


def cache_validation_set(cfg: dict[str, Any], gen: SyntheticGenerator,
                         path: str, n: int | None = None,
                         seed: int | None = None) -> None:
    """Fixed cached validation split (PLAN §7.3): n events (mixed
    reference/perturbed velocity per the generator's config)."""
    n = int(n if n is not None else cfg['synth']['n_validation'])
    seed = int(seed if seed is not None else cfg['seed'] + 1)
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.attrs['n_events'] = n
        f.attrs['seed'] = seed
        f.attrs['tier'] = str(cfg.get('tier', ''))
        f.attrs['band_hz'] = list(cfg['processing']['band_hz'])
        for i in range(n):
            write_event(f.create_group(f'event_{i:06d}'), gen.generate(rng))
            if (i + 1) % 200 == 0 or i + 1 == n:
                logger.info('cached %d / %d validation events', i + 1, n)


def iter_cached_events(path: str) -> Iterator[dict[str, Any]]:
    with h5py.File(path, 'r') as f:
        for key in sorted(f.keys()):
            yield read_event(f[key])


class SyntheticDataset(torch.utils.data.IterableDataset):
    """Infinite stream of on-the-fly synthetic events; each dataloader
    worker gets an independent seed stream."""

    def __init__(self, generator: SyntheticGenerator, seed: int):
        super().__init__()
        self.generator = generator
        self.seed = int(seed)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        if info is not None:
            self.generator.reopen()
        rng = np.random.default_rng(self.seed + 100_003 * (wid + 1))
        while True:
            yield self.generator.generate(rng)


def pad_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable trace counts to the batch maximum. 'mask' combines QC
    and padding (False = ignore trace). Full example dicts ride along under
    '_examples' for the aux misfit loss (train.py rebuilds grids per event)."""
    n_max = max(ex['waveforms'].shape[0] for ex in batch)
    n_t = batch[0]['waveforms'].shape[1]
    md_dim = batch[0]['metadata'].shape[1]
    B = len(batch)
    wf = torch.zeros(B, n_max, n_t)
    md = torch.zeros(B, n_max, md_dim)
    mask = torch.zeros(B, n_max, dtype=torch.bool)
    for i, ex in enumerate(batch):
        n = ex['waveforms'].shape[0]
        wf[i, :n] = torch.as_tensor(ex['waveforms'])
        md[i, :n] = torch.as_tensor(ex['metadata'])
        mask[i, :n] = torch.as_tensor(ex['mask'])
    return {
        'waveforms': wf,
        'metadata': md,
        'mask': mask,
        'theta': torch.stack([
            torch.as_tensor(ex['theta'], dtype=torch.float32)
            for ex in batch]),
        'm6': torch.stack([
            torch.as_tensor(ex['m6'], dtype=torch.float32)
            for ex in batch]),
        '_examples': batch,
    }
