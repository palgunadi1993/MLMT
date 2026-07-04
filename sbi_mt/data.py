"""Real-data loading/preprocessing and the empirical noise library (PLAN §5).

The preprocessing applied here is the SAME chain the training synthetics see:
instrument response removal to displacement and rotation to ZRT happen in
obspy, after which windowing uses the store P-time tables and taper +
zero-phase bandpass reuse the torch operators from forward.py verbatim —
real and synthetic data go through one code path.

obspy is imported lazily inside the functions that need it so that training
dataloader workers (which only use the noise classes below) never pay for it.
"""
from __future__ import annotations

import csv
import glob
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Sequence

import h5py
import numpy as np
import torch

from . import KM, resolve_path
from .forward import bandpass_zerophase, cosine_taper
from .gf import (
    EventInfo, GFCube, Station, StoreFetcher, data_windows,
    station_geometries)

logger = logging.getLogger('sbi_mt.data')

R2D = 180.0 / math.pi


# ----------------------------------------------------------------------------
# moment tensor convention conversion
# ----------------------------------------------------------------------------

def rtp_to_ned(m_rtp: np.ndarray) -> np.ndarray:
    """(mrr, mtt, mpp, mrt, mrp, mtp) in the spherical (r=up, t=south,
    p=east) convention of GCMT/USGS catalogs -> m6 NED
    (mnn, mee, mdd, mne, mnd, med). With n = -t and d = -r:
    mnn=mtt, mee=mpp, mdd=mrr, mne=-mtp, mnd=mrt, med=-mrp
    (verified against pyrocko.moment_tensor in tests)."""
    mrr, mtt, mpp, mrt, mrp, mtp = np.moveaxis(
        np.asarray(m_rtp, dtype=float), -1, 0)
    return np.stack([mtt, mpp, mrr, -mtp, mrt, -mrp], axis=-1)


# ----------------------------------------------------------------------------
# catalog and inventory
# ----------------------------------------------------------------------------

@dataclass
class CatalogEvent(EventInfo):
    """Validation event with optional reference MT (converted to NED [Nm])."""
    m6_ref: np.ndarray | None = None
    ref_source: str = ''


_MT_COLS = ('mrr', 'mtt', 'mpp', 'mrt', 'mrp', 'mtp')


def read_catalog(path: str) -> list[CatalogEvent]:
    """Read catalog.csv with columns: name, time (ISO 8601), lat, lon,
    depth_km, magnitude, mrr, mtt, mpp, mrt, mrp, mtp [Nm, RTP convention],
    ref_source. The six MT columns and ref_source may be empty."""
    from obspy import UTCDateTime

    events: list[CatalogEvent] = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            m6_ref = None
            if all(row.get(c, '').strip() for c in _MT_COLS):
                m6_ref = rtp_to_ned(
                    np.array([float(row[c]) for c in _MT_COLS]))
            events.append(CatalogEvent(
                name=row['name'].strip(),
                lat=float(row['lat']),
                lon=float(row['lon']),
                depth=float(row['depth_km']) * KM,
                time=float(UTCDateTime(row['time'].strip()).timestamp),
                magnitude=float(row['magnitude']),
                m6_ref=m6_ref,
                ref_source=row.get('ref_source', '').strip()))
    return events


def load_stations(path: str) -> tuple[list[Station], Any]:
    """StationXML -> unique NET.STA station list + the obspy Inventory
    (needed later for response removal)."""
    from obspy import read_inventory

    inv = read_inventory(path)
    stations: dict[str, Station] = {}
    for net in inv:
        for sta in net:
            code = f'{net.code}.{sta.code}'
            if code not in stations:
                stations[code] = Station(
                    code=code, lat=float(sta.latitude),
                    lon=float(sta.longitude))
    return list(stations.values()), inv


# ----------------------------------------------------------------------------
# shared per-trace conditioning features
# ----------------------------------------------------------------------------

def trace_metadata(
        azimuth: np.ndarray,
        distance: np.ndarray,
        comp_index: np.ndarray,
        n_components: int,
        snr: np.ndarray,
        rel_weight: np.ndarray,
        norm: np.ndarray) -> np.ndarray:
    """Per-trace metadata vector, dim = 6 + n_components
    (model.metadata_dim): [sin az, cos az, log10 dist_km,
    component one-hot, log10 SNR, relative weight, log10 norm].
    Used identically for synthetic and real events."""
    onehot = np.eye(n_components, dtype=float)[np.asarray(comp_index)]
    cols = [
        np.sin(azimuth), np.cos(azimuth),
        np.log10(np.clip(distance / KM, 1e-3, None)),
        *[onehot[:, k] for k in range(n_components)],
        np.log10(np.clip(snr, 1e-3, 1e6)),
        rel_weight,
        np.log10(np.clip(norm, 1e-30, None)),
    ]
    return np.stack(cols, axis=-1).astype(np.float32)


def preprocess_window(x: np.ndarray, deltat: float,
                      cfg: dict[str, Any]) -> np.ndarray:
    """Cosine taper + zero-phase bandpass on the last axis — the exact torch
    chain from forward.py, exposed for numpy windows (real data, noise)."""
    proc = cfg['processing']
    t = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float64)
    t = t * cosine_taper(t.shape[-1], float(proc['taper_fraction']),
                         t.dtype, t.device)
    t = bandpass_zerophase(
        t, deltat, float(proc['band_hz'][0]), float(proc['band_hz'][1]),
        int(proc['filter_order']))
    return t.numpy()


# ----------------------------------------------------------------------------
# noise sources for synthetic training data
# ----------------------------------------------------------------------------

class GaussianNoise:
    """Fallback noise source: unit-variance white noise, identical relative
    levels for all traces. TODO(user-data): replaced by :class:`NoiseLibrary`
    once continuous data exist and script 02 has built noise.h5."""

    def sample(self, rng: np.random.Generator,
               station_codes: Sequence[str],
               components: Sequence[str],
               n_t: int) -> np.ndarray:
        n_tr = len(station_codes) * len(components)
        return rng.standard_normal((n_tr, n_t)).astype(np.float32)


class NoiseLibrary:
    """Station-matched empirical noise windows from noise.h5 (script 02).

    Layout: one group per NET.STA with datasets 'Z' / 'H' of shape
    (n_windows, n_samples) [m, displacement at store rate] and companion
    '<class>_epoch' start times; file attrs band_hz, sample_rate_hz,
    window_length_s (a second high-frequency library can coexist — Tier B).

    Fallback chain per trace: exact station -> random station with the same
    channel class -> random station, other class. Never per-trace Gaussian,
    so relative station noise levels stay physical within an event.
    """

    def __init__(self, path: str):
        self.path = path
        self._f = h5py.File(path, 'r')
        self.band_hz = tuple(self._f.attrs['band_hz'])
        self.sample_rate_hz = float(self._f.attrs['sample_rate_hz'])
        self._index: dict[str, dict[str, h5py.Dataset]] = {}
        for code, grp in self._f.items():
            classes = {c: grp[c] for c in ('Z', 'H')
                       if c in grp and grp[c].shape[0] > 0}
            if classes:
                self._index[code] = classes
        if not self._index:
            raise ValueError(f'noise library {path} contains no windows')
        self._by_class: dict[str, list[h5py.Dataset]] = {
            c: [d[c] for d in self._index.values() if c in d]
            for c in ('Z', 'H')}

    def close(self) -> None:
        self._f.close()

    def _draw(self, rng: np.random.Generator, dset: h5py.Dataset,
              n_t: int) -> np.ndarray:
        w = np.asarray(dset[int(rng.integers(dset.shape[0]))], dtype=np.float32)
        if w.size < n_t:      # wrap: noise is stationary over the window
            w = np.resize(w, n_t)
        off = int(rng.integers(w.size - n_t + 1))
        return w[off:off + n_t]

    def _dataset_for(self, rng: np.random.Generator, code: str,
                     cls: str) -> h5py.Dataset:
        entry = self._index.get(code)
        if entry and cls in entry:
            return entry[cls]
        pool = self._by_class.get(cls) or self._by_class.get(
            'H' if cls == 'Z' else 'Z') or []
        if not pool:
            raise ValueError(f'no noise windows for class {cls!r}')
        return pool[int(rng.integers(len(pool)))]

    def sample(self, rng: np.random.Generator,
               station_codes: Sequence[str],
               components: Sequence[str],
               n_t: int) -> np.ndarray:
        out = np.empty((len(station_codes) * len(components), n_t),
                       dtype=np.float32)
        i = 0
        for code in station_codes:
            for comp in components:
                cls = 'Z' if comp == 'Z' else 'H'
                out[i] = self._draw(
                    rng, self._dataset_for(rng, code, cls), n_t)
                i += 1
        return out


def make_noise_source(cfg: dict[str, Any]) -> GaussianNoise | NoiseLibrary:
    """Config-selected noise source with graceful Gaussian fallback."""
    path = resolve_path(cfg, cfg['paths']['noise_library'])
    if cfg['synth']['noise'] == 'library':
        if os.path.exists(path):
            return NoiseLibrary(path)
        logger.warning(
            'noise library %s not found — falling back to Gaussian noise. '
            'TODO(user-data): run scripts/02_build_noise_library.py', path)
    return GaussianNoise()


# ----------------------------------------------------------------------------
# real-event preprocessing (PLAN §5.1-5.2) — identical output format to synth
# ----------------------------------------------------------------------------

def read_event_waveforms(cfg: dict[str, Any], event: CatalogEvent) -> Any:
    """Read the raw miniSEED of one validation event: either all files under
    waveforms/<event.name>/ or files waveforms/<event.name>.*"""
    from obspy import Stream, read

    base = resolve_path(cfg, cfg['paths']['waveforms'])
    paths = (sorted(glob.glob(os.path.join(base, event.name, '*')))
             or sorted(glob.glob(os.path.join(base, f'{event.name}.*'))))
    if not paths:
        raise FileNotFoundError(
            f'no waveforms for event {event.name} under {base}')
    st = Stream()
    for p in paths:
        st += read(p)
    return st


def _cut(tr: Any, t_start: float, n: int, allow_partial: bool = False
         ) -> np.ndarray | None:
    """Cut n samples starting at epoch t_start from an obspy trace; None on
    gaps/insufficient data. allow_partial: return the trailing overlap
    (>= 25% of n) instead — used for pre-event noise windows."""
    sr = float(tr.stats.sampling_rate)
    i0 = int(round((t_start - float(tr.stats.starttime.timestamp)) * sr))
    i1 = i0 + n
    if allow_partial:
        i0 = max(i0, 0)
        i1 = min(i1, tr.stats.npts)
        if i1 - i0 < n // 4:
            return None
    elif i0 < 0 or i1 > tr.stats.npts:
        return None
    data = np.asarray(tr.data[i0:i1], dtype=np.float64)
    if np.ma.isMaskedArray(tr.data) or not np.all(np.isfinite(data)):
        return None
    return data


def _pre_filt(cfg: dict[str, Any]) -> tuple[float, float, float, float]:
    f1, f2 = cfg['processing']['band_hz']
    ff = cfg['processing']['response_removal']['pre_filt_factor']
    return (ff[0] * f1, ff[1] * f1, ff[2] * f2, ff[3] * f2)


def prepare_event_data(
        cfg: dict[str, Any],
        event: CatalogEvent,
        stream: Any,
        inventory: Any,
        fetcher: GFCube | StoreFetcher,
        stations: Sequence[Station] | None = None) -> dict[str, Any]:
    """Real-event preprocessing to the SAME example dict the synthetic
    generator emits (minus 'theta'; 'm6' is the reference MT or NaNs).

    Chain (PLAN §5): demean/detrend -> response removal to displacement ->
    rotate to ZNE -> anti-alias lowpass + resample to store rate -> rotate
    ZRT with catalog back-azimuth -> P-window cut via the store P-time table
    -> torch taper + zero-phase bandpass -> sigma_noise from the pre-event
    window -> SNR QC mask and 1/sigma^2 weights.

    TODO(user-data): QC heuristics (partial noise windows, clip detection)
    to be tuned once real BMKG/IA waveforms are available.
    """
    proc = cfg['processing']
    components = tuple(proc['components'])
    n_comp = len(components)
    deltat = fetcher.deltat
    rate = 1.0 / deltat

    if stations is None:
        stations, _ = load_stations(  # pragma: no cover - convenience path
            resolve_path(cfg, cfg['paths']['stations_xml']))

    st = stream.copy()
    st.merge(method=1, fill_value=None)
    st.detrend('demean')
    st.detrend('linear')
    try:
        st.remove_response(
            inventory=inventory, output='DISP', pre_filt=_pre_filt(cfg),
            water_level=float(proc['response_removal']['water_level']))
    except Exception:
        # per-trace removal so one bad response does not kill the event
        good = []
        for tr in st:
            try:
                tr.remove_response(
                    inventory=inventory, output='DISP',
                    pre_filt=_pre_filt(cfg),
                    water_level=float(
                        proc['response_removal']['water_level']))
                good.append(tr)
            except Exception:
                logger.info('no response for %s — dropped', tr.id)
        st = type(st)(good)
    st.rotate('->ZNE', inventory=inventory)
    for tr in st:
        if abs(tr.stats.sampling_rate - rate) > 1e-6:
            tr.filter('lowpass', freq=0.4 * rate, corners=4, zerophase=True)
            tr.resample(rate, no_filter=True)

    have = {f'{tr.stats.network}.{tr.stats.station}' for tr in st}
    sel_stations = [s for s in stations if s.code in have]
    geoms_all = station_geometries(event, sel_stations)
    d_nodes = np.asarray(fetcher.distances, dtype=float)
    geoms = [g for g in geoms_all
             if d_nodes[0] <= g.distance <= d_nodes[-1]]
    if not geoms:
        raise ValueError(f'no stations in store range for {event.name}')
    itmin_all, n_t = data_windows(fetcher, event, geoms, cfg)

    wf_rows, sigma_rows, keep_geoms, keep_itmin = [], [], [], []
    for g, it0 in zip(geoms, itmin_all):
        net, sta = g.code.split('.')
        st_sta = st.select(network=net, station=sta).copy()
        if len(st_sta.select(component='Z')) != 1 or \
                len(st_sta.select(component='N')) != 1 or \
                len(st_sta.select(component='E')) != 1:
            logger.info('%s: incomplete ZNE — skipped', g.code)
            continue
        st_sta.rotate('NE->RT', back_azimuth=g.back_azimuth * R2D)
        t0 = event.time + float(it0) * deltat
        rows, sigmas, ok = [], [], True
        for comp in components:
            trs = st_sta.select(component=comp)
            sig = _cut(trs[0], t0, n_t) if len(trs) == 1 else None
            if sig is None:
                ok = False
                break
            noise = _cut(trs[0], t0 - n_t * deltat, n_t, allow_partial=True)
            rows.append(preprocess_window(sig, deltat, cfg))
            if noise is not None and noise.size:
                nf = preprocess_window(noise - noise.mean(), deltat, cfg)
                lo, hi = int(0.1 * nf.size), int(0.9 * nf.size)
                sigmas.append(float(np.sqrt(np.mean(nf[lo:hi] ** 2))))
            else:
                sigmas.append(np.nan)
        if not ok:
            logger.info('%s: window/gap failure — skipped', g.code)
            continue
        wf_rows.extend(rows)
        sigma_rows.extend(sigmas)
        keep_geoms.append(g)
        keep_itmin.append(int(it0))

    if not keep_geoms:
        raise ValueError(f'no usable traces for event {event.name}')

    data = np.asarray(wf_rows, dtype=np.float64)
    sigma = np.asarray(sigma_rows, dtype=np.float64)
    med = np.nanmedian(sigma)
    bad_sigma = ~np.isfinite(sigma)
    if np.isfinite(med):
        sigma[bad_sigma] = med
    else:
        raise ValueError(f'no pre-event noise estimate for {event.name}')

    sig_rms = np.sqrt(np.mean(data ** 2, axis=-1))
    snr = sig_rms / np.maximum(sigma, 1e-30)
    mask = (snr >= float(proc['snr_min'])) & ~bad_sigma
    weights = np.where(mask, 1.0 / np.maximum(sigma, 1e-30) ** 2, 0.0)
    norms = np.maximum(sig_rms, 1e-30)
    rel_w = weights / weights.max() if weights.max() > 0 else weights

    az_sta = np.array([g.azimuth for g in keep_geoms])
    dist_sta = np.array([g.distance for g in keep_geoms])
    comp_idx = np.tile(np.arange(n_comp), len(keep_geoms))
    metadata = trace_metadata(
        np.repeat(az_sta, n_comp), np.repeat(dist_sta, n_comp),
        comp_idx, n_comp, snr, rel_w, norms)

    m6 = (np.asarray(event.m6_ref, dtype=float)
          if event.m6_ref is not None else np.full(6, np.nan))
    return {
        'waveforms': (data / norms[:, None]).astype(np.float32),
        'metadata': metadata,
        'm6': m6,
        'mw': float(event.magnitude or np.nan),
        'mask': mask,
        'weights': weights,
        'snr': snr.astype(np.float32),
        'norms': norms.astype(np.float32),
        'itmin': np.asarray(keep_itmin, dtype=np.int64),
        'station_dt': np.zeros(len(keep_geoms)),
        'station_codes': [g.code for g in keep_geoms],
        'sta_azimuth': az_sta,
        'sta_distance': dist_sta,
        'lat': event.lat,
        'lon': event.lon,
        'depth': event.depth,
        'time': event.time,
        'perturbed': False,
        'band_hz': list(proc['band_hz']),
        'ref_source': event.ref_source,
    }


# ----------------------------------------------------------------------------
# noise library builder (script 02)
# ----------------------------------------------------------------------------

def build_noise_library(cfg: dict[str, Any],
                        out_path: str | None = None) -> bool:
    """Chop continuous data into event-free displacement windows and write
    noise.h5 (see :class:`NoiseLibrary` for the layout). Returns False when
    no continuous data are present (TODO(user-data))."""
    from obspy import read

    proc = cfg['processing']
    nb = cfg['noise_build']
    out_path = out_path or resolve_path(cfg, cfg['paths']['noise_library'])
    raw_dir = resolve_path(cfg, cfg['paths']['noise_raw'])
    files = sorted(
        p for p in glob.glob(os.path.join(raw_dir, '**', '*'), recursive=True)
        if os.path.isfile(p))
    if not files:
        logger.warning('no continuous data under %s — nothing to do. '
                       'TODO(user-data)', raw_dir)
        return False

    _, inv = load_stations(resolve_path(cfg, cfg['paths']['stations_xml']))
    reject: list[tuple[float, float]] = []
    cat_path = resolve_path(cfg, cfg['paths']['catalog'])
    if os.path.exists(cat_path):
        pre = float(nb['event_reject_pre_s'])
        post = float(nb['event_reject_post_s'])
        reject = [(ev.time - pre, ev.time + post)
                  for ev in read_catalog(cat_path)]

    rate = float(cfg['gf']['sample_rate_hz'])
    n_t = int(round(float(proc['window_length_s']) * rate))
    stride = max(1, int(round(n_t * float(nb['window_stride_fraction']))))
    cap = int(nb['max_windows_per_station'])
    windows: dict[str, dict[str, list[np.ndarray]]] = {}
    epochs: dict[str, dict[str, list[float]]] = {}

    for path in files:
        try:
            st = read(path)
        except Exception:
            logger.info('unreadable file %s — skipped', path)
            continue
        st.merge(method=1, fill_value=None)
        st.detrend('demean')
        st.detrend('linear')
        for tr in st:
            code = f'{tr.stats.network}.{tr.stats.station}'
            cls = 'Z' if tr.stats.channel[-1].upper() == 'Z' else 'H'
            if len(windows.get(code, {}).get(cls, ())) >= cap:
                continue
            try:
                tr.remove_response(
                    inventory=inv, output='DISP', pre_filt=_pre_filt(cfg),
                    water_level=float(
                        proc['response_removal']['water_level']))
            except Exception:
                logger.info('no response for %s — skipped', tr.id)
                continue
            if abs(tr.stats.sampling_rate - rate) > 1e-6:
                tr.filter('lowpass', freq=0.4 * rate, corners=4,
                          zerophase=True)
                tr.resample(rate, no_filter=True)
            t_tr = float(tr.stats.starttime.timestamp)
            dest = windows.setdefault(code, {}).setdefault(cls, [])
            edest = epochs.setdefault(code, {}).setdefault(cls, [])
            for i0 in range(0, tr.stats.npts - n_t + 1, stride):
                if len(dest) >= cap:
                    break
                t0 = t_tr + i0 / rate
                t1 = t0 + n_t / rate
                if any(t0 < e1 and t1 > e0 for e0, e1 in reject):
                    continue
                w = np.asarray(tr.data[i0:i0 + n_t], dtype=np.float64)
                if np.ma.isMaskedArray(tr.data) or \
                        not np.all(np.isfinite(w)):
                    continue
                dest.append((w - w.mean()).astype(np.float32))
                edest.append(t0)

    n_total = sum(len(v) for d in windows.values() for v in d.values())
    if n_total == 0:
        logger.warning('no windows survived QC — noise.h5 not written')
        return False

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        f.attrs['band_hz'] = list(proc['band_hz'])
        f.attrs['sample_rate_hz'] = rate
        f.attrs['window_length_s'] = float(proc['window_length_s'])
        for code, per_cls in windows.items():
            grp = f.create_group(code)
            for cls, wins in per_cls.items():
                grp.create_dataset(cls, data=np.stack(wins))
                grp.create_dataset(
                    f'{cls}_epoch', data=np.asarray(epochs[code][cls]))
    logger.info('wrote %d windows for %d stations to %s',
                n_total, len(windows), out_path)
    return True
