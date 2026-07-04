"""Green's function stores, dense GF cubes and differentiable G-matrix assembly.

Interpolation choice (PLAN §3.2): MULTILINEAR. The 'elastic10' scheme stores
10 Green's function components per (source_depth, distance) node; a
three-component seismogram for an arbitrary moment tensor is an analytic
combination with azimuth-dependent weights (pyrocko's f0..f5). We therefore

  1. extract per-station (depth x distance) blocks of the 10 raw components
     onto fixed per-station time windows (``GFGrid``), optionally from a dense
     HDF5 export of the whole store (``GFCube``) so that training never has to
     touch the pyrocko store;
  2. interpolate the blocks bilinearly in (depth, distance) in torch —
     identical to pyrocko's 'multilinear' target interpolation — which makes
     the assembled G matrix differentiable w.r.t. the centroid perturbation
     (dn, de, dz);
  3. apply the azimuth weights analytically (exact and differentiable in the
     perturbed source azimuth), and rotate the synthetic radial/transverse
     components into the FIXED data frame defined by the catalog back-azimuth.

Receiver geometry for perturbed centroids uses a local Cartesian frame
anchored at the catalog epicenter (receiver at distance*(cos az, sin az));
the geometry error of this flat approximation is negligible for |dn|,|de|
within the grid extent, and vanishes exactly at the catalog location.
"""
from __future__ import annotations

import importlib
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

import h5py
import numpy as np
import torch

from pyrocko import gf as pgf
from pyrocko import cake, orthodrome

from . import KM, resolve_path

logger = logging.getLogger('sbi_mt.gf')

D2R = math.pi / 180.0

#: elastic10 component indices entering radial, transverse and vertical sums
_RADIAL_COMPS = (0, 1, 2, 8)
_TRANSVERSE_COMPS = (3, 4)
_DOWN_COMPS = (5, 6, 7, 9)
#: rows of the azimuth-weight matrix used for each sum (f-indices)
_RADIAL_ROWS = (0, 1, 2, 5)
_TRANSVERSE_ROWS = (3, 4)
_DOWN_ROWS = (0, 1, 2, 5)


# ----------------------------------------------------------------------------
# basic containers
# ----------------------------------------------------------------------------

@dataclass
class EventInfo:
    """Catalog event: reference point for the nuisance parametrization."""
    name: str
    lat: float
    lon: float
    depth: float          # [m]
    time: float = 0.0     # epoch [s]
    magnitude: float | None = None


@dataclass
class Station:
    code: str             # 'NET.STA'
    lat: float
    lon: float


@dataclass
class StationGeometry:
    """Catalog-location geometry of one station (angles in radians)."""
    code: str
    distance: float       # [m], catalog epicenter -> station
    azimuth: float        # source->receiver azimuth at source
    back_azimuth: float   # receiver->source azimuth at receiver


def station_geometries(
        event: EventInfo, stations: Sequence[Station]) -> list[StationGeometry]:
    out = []
    for sta in stations:
        dist = float(orthodrome.distance_accurate50m_numpy(
            np.array([event.lat]), np.array([event.lon]),
            np.array([sta.lat]), np.array([sta.lon]))[0])
        azi, bazi = orthodrome.azibazi(event.lat, event.lon, sta.lat, sta.lon)
        out.append(StationGeometry(
            code=sta.code, distance=dist,
            azimuth=float(azi) * D2R, back_azimuth=float(bazi) * D2R))
    return out


# ----------------------------------------------------------------------------
# store creation (script 01)
# ----------------------------------------------------------------------------

def store_dir(cfg: dict[str, Any], store_id: str) -> str:
    return os.path.join(resolve_path(cfg, cfg['paths']['gf_stores']), store_id)


def load_velocity_model_str(cfg: dict[str, Any]) -> str:
    with open(resolve_path(cfg, cfg['gf']['earthmodel'])) as f:
        return f.read()


def perturb_velocity_model(
        nd_str: str, rng: np.random.Generator,
        pct_range: tuple[float, float]) -> str:
    """Perturb vp and vs of every layer by independent random factors with
    |dv/v| uniform in ``pct_range`` percent (sign random). Interface depths,
    densities and Q are kept; repeated-depth pairs (layer top/bottom written
    with the same depth) get consistent factors per layer line."""
    lines_out: list[str] = []
    for line in nd_str.splitlines():
        toks = line.split()
        if len(toks) not in (4, 6, 9):
            lines_out.append(line)
            continue
        vals = [float(x) for x in toks]
        for i in (1, 2):  # vp, vs
            pct = rng.uniform(pct_range[0], pct_range[1]) / 100.0
            sign = 1.0 if rng.random() < 0.5 else -1.0
            vals[i] = vals[i] * (1.0 + sign * pct)
        lines_out.append('  '.join(f'{v:.5g}' for v in vals))
    return '\n'.join(lines_out) + '\n'


def create_store(
        cfg: dict[str, Any], store_id: str, earthmodel_nd: str,
        force: bool = False, nworkers: int | None = None) -> str:
    """Create and build a fomosto store with the backend named in the config
    (qseis | qssp | ahfullgreen). Returns the store directory."""
    gcfg = cfg['gf']
    sdir = store_dir(cfg, store_id)
    backend = importlib.import_module(f"pyrocko.fomosto.{gcfg['backend']}")

    if os.path.exists(os.path.join(sdir, 'traces')) and not force:
        logger.info('store %s exists, skipping build', store_id)
        return sdir

    if not os.path.exists(os.path.join(sdir, 'config')):
        os.makedirs(os.path.dirname(sdir), exist_ok=True)
        config_params = dict(
            id=store_id,
            ncomponents=10,
            sample_rate=float(gcfg['sample_rate_hz']),
            receiver_depth=0.0,
            source_depth_min=float(gcfg['source_depth_min_km']) * KM,
            source_depth_max=float(gcfg['source_depth_max_km']) * KM,
            source_depth_delta=float(gcfg['source_depth_delta_km']) * KM,
            distance_min=float(gcfg['distance_min_km']) * KM,
            distance_max=float(gcfg['distance_max_km']) * KM,
            distance_delta=float(gcfg['distance_delta_km']) * KM,
            earthmodel_1d=cake.LayeredModel.from_scanlines(
                cake.read_nd_model_str(earthmodel_nd)),
            tabulated_phases=[
                pgf.meta.TPDef(id='anyP', definition='P,p,\\P,\\p'),
                pgf.meta.TPDef(id='anyS', definition='S,s,\\S,\\s')])
        backend.init(sdir, gcfg.get('backend_variant'),
                     config_params=config_params)

    backend.build(sdir, force=force, nworkers=nworkers)
    store = pgf.store.Store(sdir)
    store.make_travel_time_tables(force=force)
    store.close()
    return sdir


def cube_path(cfg: dict[str, Any], store_id: str) -> str:
    """Canonical location of a store's dense HDF5 export (script 01)."""
    return os.path.join(
        resolve_path(cfg, cfg['paths']['gf_stores']), f'{store_id}_cube.h5')


def get_engine(cfg: dict[str, Any]) -> pgf.LocalEngine:
    return pgf.LocalEngine(
        store_superdirs=[resolve_path(cfg, cfg['paths']['gf_stores'])])


def phase_time(store: pgf.store.Store, z: float, d: float,
               phase_id: str = 'anyP') -> float:
    """First-arrival time of the tabulated phase; hard error if unavailable."""
    t = store.t(f'{{stored:{phase_id}}}', (z, d))
    if t is None:
        raise ValueError(
            f'no {phase_id} arrival tabulated at depth={z} m, distance={d} m')
    return float(t)


# ----------------------------------------------------------------------------
# dense GF cube (HDF5 export of a whole store)
# ----------------------------------------------------------------------------

def _cube_time_windows(
        store: pgf.store.Store, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-distance start indices and the common sample count of the cube.

    The window must contain, for every depth node, the data window
    [t_P - pre_p, t_P - pre_p + length] shifted by up to the maximum
    |dt0| + |station dt| in either direction, plus interpolation slack across
    the grid extent. We anchor at the minimum P time over depths minus a
    conservative pad.
    """
    scfg = store.config
    deltat = float(scfg.deltat)
    proc = cfg['processing']
    depths = scfg.coords[0]
    dists = scfg.coords[1]

    pad_pre = (float(proc['window_pre_p_s'])
               + abs(float(cfg['priors']['dt0_s'][0]))
               + abs(float(cfg['priors']['station_dt_s'][0]))
               + 10.0)
    pad_post = (abs(float(cfg['priors']['dt0_s'][1]))
                + abs(float(cfg['priors']['station_dt_s'][1]))
                + 10.0)

    tp = np.empty((depths.size, dists.size))
    for j, z in enumerate(depths):
        for i, d in enumerate(dists):
            tp[j, i] = phase_time(store, z, d)
    tp_min = tp.min(axis=0)
    tp_max = tp.max(axis=0)

    it0 = np.floor((tp_min - pad_pre) / deltat).astype(np.int64)
    n_t = int(np.ceil(
        (np.max(tp_max - tp_min) + pad_pre + pad_post
         + float(proc['window_length_s'])) / deltat)) + 1
    return np.asarray(depths, dtype=float), np.asarray(dists, dtype=float), \
        tp, it0, n_t


def extract_cube(store: pgf.store.Store, cfg: dict[str, Any],
                 out_path: str) -> None:
    """Export the full store as a dense float32 array
    gf[n_z, n_d, 10, n_t] with per-distance start indices ``it0``."""
    depths, dists, tp, it0, n_t = _cube_time_windows(store, cfg)
    band = cfg['processing']['band_hz']
    logger.info('extracting cube %s: (%d, %d, 10, %d)',
                out_path, depths.size, dists.size, n_t)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        dset = f.create_dataset(
            'gf', shape=(depths.size, dists.size, 10, n_t), dtype='f4',
            chunks=(1, min(32, dists.size), 10, n_t))
        for iz, z in enumerate(depths):
            block = np.zeros((dists.size, 10, n_t), dtype=np.float32)
            for i_d, d in enumerate(dists):
                for k in range(10):
                    tr = store.get((z, d, k), itmin=int(it0[i_d]),
                                   nsamples=n_t)
                    block[i_d, k, :tr.data.size] = tr.data
                    if tr.data.size < n_t:
                        block[i_d, k, tr.data.size:] = (
                            tr.data[-1] if tr.data.size else 0.0)
            dset[iz] = block
        f.create_dataset('depths', data=depths)
        f.create_dataset('distances', data=dists)
        f.create_dataset('it0', data=it0)
        # P arrival table: lets dataloader workers derive data windows
        # without touching the pyrocko store
        f.create_dataset('tp', data=tp)
        f.attrs['deltat'] = float(store.config.deltat)
        f.attrs['store_id'] = store.config.id
        f.attrs['band_hz'] = band       # noise/GF band bookkeeping (Tier B)


class GFCube:
    """Memory-mapped reader for a dense GF export (see :func:`extract_cube`)."""

    def __init__(self, path: str):
        self.path = path
        self._f = h5py.File(path, 'r')
        self.gf = self._f['gf']
        self.depths: np.ndarray = self._f['depths'][()]
        self.distances: np.ndarray = self._f['distances'][()]
        self.it0: np.ndarray = self._f['it0'][()]
        self.tp: np.ndarray = self._f['tp'][()]
        self.deltat: float = float(self._f.attrs['deltat'])
        self.store_id: str = str(self._f.attrs['store_id'])

    def close(self) -> None:
        self._f.close()

    def reopen(self) -> None:
        """Re-open the HDF5 handle. Must be called in dataloader worker
        processes: h5py handles inherited across fork are not safe."""
        try:
            self._f.close()
        except Exception:
            pass
        self._f = h5py.File(self.path, 'r')
        self.gf = self._f['gf']

    def p_time(self, z: float, d: float) -> float:
        """Bilinear interpolation of the tabulated P arrival [s]."""
        fz = np.clip((z - self.depths[0])
                     / (self.depths[1] - self.depths[0]),
                     0, self.depths.size - 1 - 1e-9)
        fd = np.clip((d - self.distances[0])
                     / (self.distances[1] - self.distances[0]),
                     0, self.distances.size - 1 - 1e-9)
        j, i = int(fz), int(fd)
        wz, wd = fz - j, fd - i
        return float(
            (1 - wz) * ((1 - wd) * self.tp[j, i] + wd * self.tp[j, i + 1])
            + wz * ((1 - wd) * self.tp[j + 1, i] + wd * self.tp[j + 1, i + 1]))

    def fetch(self, iz: np.ndarray, i_d: np.ndarray,
              itmin: int, n_t: int) -> np.ndarray:
        """Return gf[len(iz), len(i_d), 10, n_t] cut to the absolute sample
        window [itmin, itmin + n_t) (relative to source time)."""
        out = np.zeros((iz.size, i_d.size, 10, n_t), dtype=np.float32)
        for b, j in enumerate(i_d):
            j = int(j)
            lo = itmin - int(self.it0[j])
            n_cube = self.gf.shape[3]
            if lo < 0:
                raise ValueError(
                    f'GF cube window starts too late at distance index {j}: '
                    f'need sample {itmin}, cube starts at {self.it0[j]}')
            hi = min(lo + n_t, n_cube)
            data = self.gf[iz.min():iz.max() + 1, j, :, lo:hi]
            data = data[iz - iz.min()]
            out[:, b, :, :hi - lo] = data
            if hi - lo < n_t:   # extend with last value (static offset)
                out[:, b, :, hi - lo:] = data[..., -1:]
        return out


class StoreFetcher:
    """Same interface as :class:`GFCube`.fetch but reading directly from a
    pyrocko store (slow path: tests and one-off inference without a cube)."""

    def __init__(self, store: pgf.store.Store):
        self.store = store
        self.depths = np.asarray(store.config.coords[0], dtype=float)
        self.distances = np.asarray(store.config.coords[1], dtype=float)
        self.deltat = float(store.config.deltat)
        self.store_id = store.config.id

    def p_time(self, z: float, d: float) -> float:
        """Tabulated P arrival [s] (exact store lookup)."""
        return phase_time(self.store, z, d)

    def fetch(self, iz: np.ndarray, i_d: np.ndarray,
              itmin: int, n_t: int) -> np.ndarray:
        out = np.zeros((iz.size, i_d.size, 10, n_t), dtype=np.float64)
        for a, izz in enumerate(iz):
            for b, jdd in enumerate(i_d):
                for k in range(10):
                    tr = self.store.get(
                        (self.depths[int(izz)], self.distances[int(jdd)], k),
                        itmin=itmin, nsamples=n_t)
                    out[a, b, k, :tr.data.size] = tr.data
                    if tr.data.size < n_t and tr.data.size:
                        out[a, b, k, tr.data.size:] = tr.data[-1]
        return out


# ----------------------------------------------------------------------------
# per-event differentiable GF grid
# ----------------------------------------------------------------------------

@dataclass
class GFGrid:
    """Per-event GF block: everything :func:`assemble_G` needs, as tensors.

    Axes: gf[n_sta, n_z, n_d, 10, n_t]. Depth nodes are absolute [m] and
    common to all stations; distance nodes are per-station absolute [m]
    (uniform spacing). Each station has its own time window starting at
    sample ``itmin[s]`` (relative to source time, step ``deltat``).
    """
    gf: torch.Tensor            # (n_sta, n_z, n_d, 10, n_t)
    depths: torch.Tensor        # (n_z,)
    dists: torch.Tensor         # (n_sta, n_d)
    itmin: torch.Tensor         # (n_sta,) long
    deltat: float
    depth_ref: float            # catalog depth [m]
    dist_ref: torch.Tensor      # (n_sta,) catalog distances [m]
    azimuth_ref: torch.Tensor   # (n_sta,) [rad]
    sta_north: torch.Tensor     # (n_sta,) receiver coords, local frame [m]
    sta_east: torch.Tensor      # (n_sta,)
    codes: list[str] = field(default_factory=list)
    components: tuple[str, ...] = ('Z', 'R', 'T')

    @property
    def n_sta(self) -> int:
        return int(self.gf.shape[0])

    @property
    def n_t(self) -> int:
        return int(self.gf.shape[4])

    def to(self, device: torch.device | str) -> 'GFGrid':
        for name in ('gf', 'depths', 'dists', 'itmin', 'dist_ref',
                     'azimuth_ref', 'sta_north', 'sta_east'):
            setattr(self, name, getattr(self, name).to(device))
        return self


def data_windows(
        fetcher: GFCube | StoreFetcher,
        event: EventInfo,
        geoms: Sequence[StationGeometry],
        cfg: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Per-station data windows from the CATALOG-location P arrival:
    start sample ``itmin[s]`` (relative to source time) and common length
    ``n_t``. Uses ``fetcher.p_time`` so dataloader workers never touch the
    pyrocko store."""
    proc = cfg['processing']
    deltat = fetcher.deltat
    n_t = int(round(float(proc['window_length_s']) / deltat))
    itmin = np.array([
        int(math.floor(
            (fetcher.p_time(event.depth, g.distance)
             - float(proc['window_pre_p_s'])) / deltat))
        for g in geoms], dtype=np.int64)
    return itmin, n_t


def build_gf_grid(
        fetcher: GFCube | StoreFetcher,
        event: EventInfo,
        geoms: Sequence[StationGeometry],
        cfg: dict[str, Any],
        itmin: np.ndarray | None = None,
        n_t: int | None = None,
        dtype: torch.dtype = torch.float32,
        store: pgf.store.Store | None = None) -> GFGrid:
    """Cut per-station (depth x distance x component x time) blocks around
    the catalog location.

    ``itmin``/``n_t`` define the per-station data windows (samples relative
    to source time). If omitted they are derived from the P arrival at the
    catalog location using the config window (requires ``store`` for the
    traveltime lookup).
    """
    gcfg = cfg['gf']['grid']
    proc = cfg['processing']
    deltat = fetcher.deltat
    z_nodes_all = fetcher.depths
    d_nodes_all = fetcher.distances
    dz_store = float(z_nodes_all[1] - z_nodes_all[0])
    dd_store = float(d_nodes_all[1] - d_nodes_all[0])

    ext_z = float(gcfg['extent_depth_km']) * KM
    ext_h = float(gcfg['extent_horizontal_km']) * KM
    # slack: horizontal grid extent shifts distances by at most sqrt(2)*ext_h
    ext_d = ext_h * math.sqrt(2.0) + dd_store

    iz_lo = max(0, int(np.floor((event.depth - ext_z - z_nodes_all[0])
                                / dz_store)))
    iz_hi = min(z_nodes_all.size - 1,
                int(np.ceil((event.depth + ext_z - z_nodes_all[0])
                            / dz_store)))
    iz = np.arange(iz_lo, iz_hi + 1)

    n_d = int(np.ceil(2 * ext_d / dd_store)) + 2

    if n_t is None:
        n_t = int(round(float(proc['window_length_s']) / deltat))

    if itmin is None:
        if store is not None:
            itmin = np.array([
                int(math.floor(
                    (phase_time(store, event.depth, g.distance)
                     - float(proc['window_pre_p_s'])) / deltat))
                for g in geoms], dtype=np.int64)
        else:
            itmin, _ = data_windows(fetcher, event, geoms, cfg)
    itmin = np.asarray(itmin, dtype=np.int64)

    blocks, dists_nodes = [], []
    for s, g in enumerate(geoms):
        j_c = int(round((g.distance - d_nodes_all[0]) / dd_store))
        j_lo = max(0, min(j_c - n_d // 2, d_nodes_all.size - n_d))
        jd = np.arange(j_lo, j_lo + n_d)
        blocks.append(fetcher.fetch(iz, jd, int(itmin[s]), n_t))
        dists_nodes.append(d_nodes_all[jd])

    gf_t = torch.as_tensor(np.stack(blocks), dtype=dtype)
    az = torch.tensor([g.azimuth for g in geoms], dtype=dtype)
    dist_ref = torch.tensor([g.distance for g in geoms], dtype=dtype)

    return GFGrid(
        gf=gf_t,
        depths=torch.as_tensor(z_nodes_all[iz], dtype=dtype),
        dists=torch.as_tensor(np.stack(dists_nodes), dtype=dtype),
        itmin=torch.as_tensor(itmin),
        deltat=deltat,
        depth_ref=event.depth,
        dist_ref=dist_ref,
        azimuth_ref=az,
        sta_north=dist_ref * torch.cos(az),
        sta_east=dist_ref * torch.sin(az),
        codes=[g.code for g in geoms],
        components=tuple(proc.get('components', ['Z', 'R', 'T'])))


def build_point_grid(
        fetcher: GFCube | StoreFetcher,
        event: EventInfo,
        geoms: Sequence[StationGeometry],
        cfg: dict[str, Any],
        dn: float,
        de: float,
        dz: float,
        itmin: np.ndarray | None = None,
        n_t: int | None = None,
        dtype: torch.dtype = torch.float32) -> GFGrid:
    """Minimal 2x2-node GFGrid bracketing ONE perturbed source location.

    Training-data generation needs G at exactly one (dn, de, dz); the two
    bracketing depth nodes and two bracketing distance nodes per station give
    bit-identical bilinear interpolation to the full grid at a fraction of
    the memory (~KB/event vs ~half a GB for the full extent grid, which is
    reserved for inference and the aux misfit loss).
    """
    proc = cfg['processing']
    deltat = fetcher.deltat
    z_nodes = np.asarray(fetcher.depths, dtype=float)
    d_nodes = np.asarray(fetcher.distances, dtype=float)
    dz_store = float(z_nodes[1] - z_nodes[0])
    dd_store = float(d_nodes[1] - d_nodes[0])

    if n_t is None:
        n_t = int(round(float(proc['window_length_s']) / deltat))
    if itmin is None:
        itmin, _ = data_windows(fetcher, event, geoms, cfg)
    itmin = np.asarray(itmin, dtype=np.int64)

    z_pert = event.depth + float(dz)
    j = int(np.clip(math.floor((z_pert - z_nodes[0]) / dz_store),
                    0, z_nodes.size - 2))
    iz = np.array([j, j + 1])

    blocks, dists_nodes = [], []
    for s, g in enumerate(geoms):
        rn = g.distance * math.cos(g.azimuth) - float(dn)
        re = g.distance * math.sin(g.azimuth) - float(de)
        dist_p = math.hypot(rn, re)
        i = int(np.clip(math.floor((dist_p - d_nodes[0]) / dd_store),
                        0, d_nodes.size - 2))
        jd = np.array([i, i + 1])
        blocks.append(fetcher.fetch(iz, jd, int(itmin[s]), n_t))
        dists_nodes.append(d_nodes[jd])

    gf_t = torch.as_tensor(np.stack(blocks), dtype=dtype)
    az = torch.tensor([g.azimuth for g in geoms], dtype=dtype)
    dist_ref = torch.tensor([g.distance for g in geoms], dtype=dtype)

    return GFGrid(
        gf=gf_t,
        depths=torch.as_tensor(z_nodes[iz], dtype=dtype),
        dists=torch.as_tensor(np.stack(dists_nodes), dtype=dtype),
        itmin=torch.as_tensor(itmin),
        deltat=deltat,
        depth_ref=event.depth,
        dist_ref=dist_ref,
        azimuth_ref=az,
        sta_north=dist_ref * torch.cos(az),
        sta_east=dist_ref * torch.sin(az),
        codes=[g.code for g in geoms],
        components=tuple(proc.get('components', ['Z', 'R', 'T'])))


def _azimuth_weight_matrix(az: torch.Tensor) -> torch.Tensor:
    """Weight matrix A with f = A @ m6 (pyrocko elastic10 convention,
    m6 = (m_nn, m_ee, m_dd, m_ne, m_nd, m_ed)). Shape (..., 6, 6):
    rows f0..f5, columns m6 components."""
    sa, ca = torch.sin(az), torch.cos(az)
    s2a, c2a = torch.sin(2 * az), torch.cos(2 * az)
    zero = torch.zeros_like(az)
    one = torch.ones_like(az)
    rows = [
        torch.stack([ca**2, sa**2, zero, s2a, zero, zero], dim=-1),        # f0
        torch.stack([zero, zero, zero, zero, ca, sa], dim=-1),             # f1
        torch.stack([zero, zero, one, zero, zero, zero], dim=-1),          # f2
        torch.stack([-0.5*s2a, 0.5*s2a, zero, c2a, zero, zero], dim=-1),   # f3
        torch.stack([zero, zero, zero, zero, -sa, ca], dim=-1),            # f4
        torch.stack([sa**2, ca**2, zero, -s2a, zero, zero], dim=-1),       # f5
    ]
    return torch.stack(rows, dim=-2)


def _interp_bilinear(grid: GFGrid, dist: torch.Tensor,
                     depth: torch.Tensor) -> torch.Tensor:
    """Bilinear interpolation of grid.gf at per-(sample, station) queries.

    dist, depth: (B, n_sta) absolute [m]. Returns (B, n_sta, 10, n_t).
    """
    n_z = grid.depths.shape[0]
    n_d = grid.dists.shape[1]
    z0, dz = grid.depths[0], grid.depths[1] - grid.depths[0]
    d0 = grid.dists[:, 0].unsqueeze(0)                      # (1, n_sta)
    dd = (grid.dists[:, 1] - grid.dists[:, 0]).unsqueeze(0)

    fz = ((depth - z0) / dz).clamp(0.0, n_z - 1 - 1e-6)
    fd = ((dist - d0) / dd).clamp(0.0, n_d - 1 - 1e-6)
    iz0 = fz.floor().long()
    id0 = fd.floor().long()
    wz = (fz - iz0).unsqueeze(-1).unsqueeze(-1)
    wd = (fd - id0).unsqueeze(-1).unsqueeze(-1)

    s_idx = torch.arange(grid.n_sta, device=grid.gf.device).unsqueeze(0)
    s_idx = s_idx.expand_as(iz0)

    def corner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return grid.gf[s_idx, a, b]                          # (B, n_sta, 10, n_t)

    g00 = corner(iz0, id0)
    g01 = corner(iz0, id0 + 1)
    g10 = corner(iz0 + 1, id0)
    g11 = corner(iz0 + 1, id0 + 1)
    return ((1 - wz) * ((1 - wd) * g00 + wd * g01)
            + wz * ((1 - wd) * g10 + wd * g11))


def assemble_G(grid: GFGrid,
               dn: torch.Tensor,
               de: torch.Tensor,
               dz: torch.Tensor) -> torch.Tensor:
    """Assemble the G matrix at perturbed centroids.

    dn, de, dz: tensors of shape (B,) [m], differentiable. Returns
    G of shape (B, n_traces, n_t, 6) with traces ordered station-major,
    components (Z, R, T), and MT basis (m_nn, m_ee, m_dd, m_ne, m_nd, m_ed)
    in Nm (unit moment per column).
    """
    dn, de, dz = torch.atleast_1d(dn), torch.atleast_1d(de), torch.atleast_1d(dz)
    B = dn.shape[0]

    rn = grid.sta_north.unsqueeze(0) - dn.unsqueeze(1)       # (B, n_sta)
    re = grid.sta_east.unsqueeze(0) - de.unsqueeze(1)
    dist = torch.sqrt(rn**2 + re**2)
    az = torch.atan2(re, rn)
    # back-azimuth tracked as catalog value + azimuth change (exact at the
    # catalog location; flat-frame approximation elsewhere)
    delta_az = az - grid.azimuth_ref.unsqueeze(0)
    depth = (grid.depth_ref + dz).unsqueeze(1).expand_as(dist)

    g = _interp_bilinear(grid, dist, depth)                  # (B, n_sta, 10, n_t)
    A = _azimuth_weight_matrix(az)                           # (B, n_sta, 6, 6)

    def combine(comp_idx: tuple[int, ...],
                row_idx: tuple[int, ...]) -> torch.Tensor:
        gg = g[:, :, list(comp_idx)]                         # (B, n_sta, c, n_t)
        ww = A[:, :, list(row_idx)]                          # (B, n_sta, c, 6)
        return torch.einsum('bscn,bsck->bskn', gg, ww)       # (B, n_sta, 6, n_t)

    R = combine(_RADIAL_COMPS, _RADIAL_ROWS)
    T = combine(_TRANSVERSE_COMPS, _TRANSVERSE_ROWS)
    D = combine(_DOWN_COMPS, _DOWN_ROWS)

    # rotate (R, T) of the perturbed source into the fixed data frame
    cd = torch.cos(delta_az).unsqueeze(-1).unsqueeze(-1)
    sd = torch.sin(delta_az).unsqueeze(-1).unsqueeze(-1)
    R_fix = cd * R - sd * T
    T_fix = sd * R + cd * T
    Z = -D                                                   # up positive

    comp_map = {'Z': Z, 'R': R_fix, 'T': T_fix}
    stacked = torch.stack([comp_map[c] for c in grid.components], dim=2)
    # (B, n_sta, n_comp, 6, n_t) -> (B, n_traces, n_t, 6)
    B_, n_sta, n_comp, six, n_t = stacked.shape
    return stacked.permute(0, 1, 2, 4, 3).reshape(
        B_, n_sta * n_comp, n_t, six)
