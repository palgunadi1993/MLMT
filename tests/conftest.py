"""Shared fixtures: a small analytical (ahfullgreen) GF store so the full
torch forward chain can be validated against pyrocko without external
modelling codes. The store is cached in tests/_cache across runs."""
from __future__ import annotations

import copy
import os
from typing import Any

import numpy as np
import pytest

from pyrocko import gf as pgf

from sbi_mt import load_config, seed_everything
from sbi_mt.gf import create_store

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, 'tests', '_cache')
TEST_STORE_ID = 'test_ahfull'

#: homogeneous model — the ahfullgreen backend requires it
HOMOGENEOUS_ND = """0.0   6.0  3.464  2.70  1264.0  600.0
30.0  6.0  3.464  2.70  1264.0  600.0
"""


@pytest.fixture(scope='session')
def test_cfg() -> dict[str, Any]:
    cfg = load_config(os.path.join(ROOT, 'config', 'default.yaml'))
    cfg['paths']['gf_stores'] = os.path.join(CACHE, 'gf_stores')
    cfg['paths']['noise_library'] = os.path.join(CACHE, 'noise.h5')
    cfg['paths']['runs'] = os.path.join(CACHE, 'runs')
    cfg['paths']['results'] = os.path.join(CACHE, 'results')
    cfg['gf'].update(
        backend='ahfullgreen', backend_variant=None,
        store='reference', sample_rate_hz=4.0,
        distance_min_km=2.0, distance_max_km=60.0, distance_delta_km=2.0,
        source_depth_min_km=2.0, source_depth_max_km=12.0,
        source_depth_delta_km=2.0)
    cfg['gf']['stores'] = {'reference': {'id': TEST_STORE_ID}}
    cfg['gf']['grid'].update(extent_horizontal_km=4.0, extent_depth_km=3.0,
                             spacing_horizontal_km=None, spacing_depth_km=None)
    cfg['processing'].update(
        band_hz=[0.04, 0.4], window_pre_p_s=5.0, window_length_s=45.0,
        taper_fraction=0.05)
    cfg['priors'].update(
        mw=[4.0, 5.0], depth_km=[5.0, 9.0], dt0_s=[-3.0, 3.0],
        station_dt_s=[-1.5, 1.5], dn_km=[-4.0, 4.0], de_km=[-4.0, 4.0],
        dz_km=[-3.0, 3.0],
        study_polygon=[[114.8, -8.2], [115.2, -8.2],
                       [115.2, -7.8], [114.8, -7.8]])
    return cfg


@pytest.fixture(scope='session')
def test_store(test_cfg: dict[str, Any]) -> pgf.store.Store:
    seed_everything(1)
    sdir = create_store(test_cfg, TEST_STORE_ID, HOMOGENEOUS_ND, nworkers=1)
    engine = pgf.LocalEngine(store_superdirs=[os.path.dirname(sdir)])
    return engine.get_store(TEST_STORE_ID)


@pytest.fixture(scope='session')
def test_engine(test_store: pgf.store.Store,
                test_cfg: dict[str, Any]) -> pgf.LocalEngine:
    return pgf.LocalEngine(
        store_superdirs=[os.path.join(test_cfg['paths']['gf_stores'])])


def random_event_and_stations(
        rng: np.random.Generator, n_sta: int = 4
) -> tuple['EventInfo', list['Station']]:
    from sbi_mt.gf import EventInfo, Station
    ev = EventInfo(
        name='test', lat=-8.0 + float(rng.uniform(-0.02, 0.02)),
        lon=115.0 + float(rng.uniform(-0.02, 0.02)),
        depth=float(rng.uniform(6.0, 8.0)) * 1000.0, time=0.0)
    stations = []
    for i in range(n_sta):
        azi = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(18e3, 42e3)
        dlat = dist * np.cos(azi) / 111195.0
        dlon = dist * np.sin(azi) / (111195.0 * np.cos(np.deg2rad(ev.lat)))
        stations.append(Station(
            code=f'XX.S{i:02d}', lat=ev.lat + dlat, lon=ev.lon + dlon))
    return ev, stations
