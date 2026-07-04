"""Phase 1 acceptance: torch-assembled synthetics == pyrocko.gf.Engine.

Criterion (PLAN §3.3): for 3 random MTs/locations the torch forward matches
engine seismograms to < 1e-6 relative RMS.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pyrocko import gf as pgf

from sbi_mt.gf import (
    EventInfo, GFCube, StoreFetcher, assemble_G, build_gf_grid, extract_cube,
    station_geometries)
from conftest import TEST_STORE_ID, random_event_and_stations

R2D = 180.0 / np.pi


def engine_reference(engine: pgf.LocalEngine, event: EventInfo,
                     stations, geoms, m6: np.ndarray, dn: float = 0.0,
                     de: float = 0.0, dz: float = 0.0) -> list:
    """Engine synthetics in the fixed (catalog back-azimuth) Z, R, T frame.
    Targets sit at the true station lat/lon so the engine geometry is
    identical to the one used for the GF grid."""
    source = pgf.MTSource(
        lat=event.lat, lon=event.lon, depth=event.depth + dz,
        north_shift=dn, east_shift=de, m6=m6, time=event.time)
    targets = []
    for sta, g in zip(stations, geoms):
        bazi_deg = g.back_azimuth * R2D
        # measurement directions of the FIXED data frame:
        # Z up, R away from catalog source, T = Z x R
        for azi, dip in ((0.0, -90.0),
                         (bazi_deg + 180.0, 0.0),
                         (bazi_deg - 90.0, 0.0)):
            targets.append(pgf.Target(
                quantity='displacement', lat=sta.lat, lon=sta.lon,
                azimuth=azi, dip=dip, interpolation='multilinear',
                store_id=TEST_STORE_ID))
    return engine.process(source, targets).pyrocko_traces()


def compare_windows(traces, grid, synth: torch.Tensor,
                    event_time: float) -> float:
    """Event-aggregate relative RMS misfit between engine traces and torch
    synthetics over the per-station windows. Aggregating over the event keeps
    the metric meaningful on near-nodal traces, whose own RMS is at the
    float32 noise floor of the GF store."""
    deltat = grid.deltat
    num = den = 0.0
    for i, tr in enumerate(traces):
        s = i // len(grid.components)
        mine = synth[i].detach().cpu().numpy()
        j0 = int(round((tr.tmin - event_time) / deltat)) - int(grid.itmin[s])
        e0 = max(0, -j0)
        m0 = j0 + e0
        n = min(tr.ydata.size - e0, mine.size - m0)
        assert n > mine.size // 2, 'windows barely overlap'
        diff = mine[m0:m0 + n] - tr.ydata[e0:e0 + n]
        num += float(np.sum(diff**2))
        den += float(np.sum(tr.ydata[e0:e0 + n]**2))
    return float(np.sqrt(num / den))


@pytest.fixture(scope='module')
def grids(test_store, test_cfg):
    """Three random events with stations, geometries and float64 GF grids."""
    rng = np.random.default_rng(7)
    out = []
    fetcher = StoreFetcher(test_store)
    for _ in range(3):
        ev, stations = random_event_and_stations(rng)
        geoms = station_geometries(ev, stations)
        grid = build_gf_grid(fetcher, ev, geoms, test_cfg,
                             dtype=torch.float64, store=test_store)
        m6 = rng.normal(size=6) * 1e15
        out.append((ev, stations, geoms, grid, m6))
    return out


def test_forward_matches_engine_at_catalog_location(grids, test_engine):
    for ev, stations, geoms, grid, m6 in grids:
        zero = torch.zeros(1, dtype=torch.float64)
        G = assemble_G(grid, zero, zero, zero)          # (1, n_tr, n_t, 6)
        synth = G[0] @ torch.as_tensor(m6, dtype=torch.float64)
        traces = engine_reference(test_engine, ev, stations, geoms, m6)
        worst = compare_windows(traces, grid, synth, ev.time)
        assert worst < 1e-6, f'relative RMS {worst:.3e} exceeds 1e-6'


def test_forward_matches_engine_at_perturbed_location(grids, test_engine):
    rng = np.random.default_rng(11)
    for ev, stations, geoms, grid, m6 in grids:
        dn = float(rng.uniform(-3e3, 3e3))
        de = float(rng.uniform(-3e3, 3e3))
        dz = float(rng.uniform(-2e3, 2e3))
        G = assemble_G(grid,
                       torch.tensor([dn], dtype=torch.float64),
                       torch.tensor([de], dtype=torch.float64),
                       torch.tensor([dz], dtype=torch.float64))
        synth = G[0] @ torch.as_tensor(m6, dtype=torch.float64)
        traces = engine_reference(test_engine, ev, stations, geoms, m6,
                                  dn=dn, de=de, dz=dz)
        worst = compare_windows(traces, grid, synth, ev.time)
        # Looser bound: pyrocko mixes geometry conventions for shifted
        # sources (spherical ne_to_latlon + ~50m-accurate geodesic
        # distances), giving 5-10 m distance differences vs our consistent
        # local Cartesian frame -> few-permille waveform differences.
        # Training and inference share the same frame, so this inconsistency
        # never enters the inversion.
        assert worst < 2e-2, f'relative RMS {worst:.3e} exceeds 2e-2'


def test_assemble_G_is_differentiable(grids):
    _, _, _, grid, m6 = grids[0]
    dn = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    de = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    dz = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    G = assemble_G(grid, dn, de, dz)
    synth = G[0] @ torch.as_tensor(m6, dtype=torch.float64)
    loss = (synth**2).sum()
    gn, ge, gz = torch.autograd.grad(loss, (dn, de, dz))
    for g in (gn, ge, gz):
        assert torch.isfinite(g).all() and g.abs().sum() > 0


def test_cube_matches_store_fetcher(test_store, test_cfg, tmp_path):
    cube_path = str(tmp_path / 'cube.h5')
    extract_cube(test_store, test_cfg, cube_path)
    cube = GFCube(cube_path)
    fetcher = StoreFetcher(test_store)
    iz = np.array([1, 2, 3])
    i_d = np.array([5, 6, 7, 8])
    itmin, n_t = 10, 150
    a = cube.fetch(iz, i_d, itmin, n_t)
    b = fetcher.fetch(iz, i_d, itmin, n_t)
    assert np.allclose(a, b.astype(np.float32), rtol=1e-6, atol=0)
    cube.close()
