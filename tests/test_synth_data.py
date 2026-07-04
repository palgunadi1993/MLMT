"""Phase 3 acceptance: MT prior sampling, RTP->NED conversion vs pyrocko,
point-grid == full-grid interpolation, the end-to-end synthetic generator
against the analytical test store (WLS at true theta recovers the true MT),
noise-library station matching and the validation-cache round trip."""
from __future__ import annotations

import copy

import h5py
import numpy as np
import pytest
import torch

from sbi_mt.data import GaussianNoise, NoiseLibrary, rtp_to_ned
from sbi_mt.forward import ForwardModel, decompose_mt, mw_to_m0, scalar_moment
from sbi_mt.gf import (
    EventInfo, StoreFetcher, assemble_G, build_gf_grid, build_point_grid,
    data_windows, station_geometries)
from sbi_mt.synth import (
    SyntheticGenerator, cache_validation_set, iter_cached_events,
    pad_collate, sample_in_polygon, sample_mt, sample_nuisance)
from conftest import random_event_and_stations


# ----------------------------------------------------------------------------
# priors
# ----------------------------------------------------------------------------

def test_sample_mt_scalar_moment_matches_mw():
    rng = np.random.default_rng(0)
    for mode in ('uniform_tt', 'dc_dominant'):
        for _ in range(50):
            m6, mw = sample_mt(rng, (4.0, 6.0), mode, dc_fraction=0.5)
            assert scalar_moment(m6) == pytest.approx(
                float(mw_to_m0(mw)), rel=1e-9)


def test_sample_mt_dc_dominant_is_pure_dc():
    rng = np.random.default_rng(1)
    for _ in range(20):
        m6, _ = sample_mt(rng, (5.0, 5.0), 'dc_dominant', dc_fraction=1.0)
        dec = decompose_mt(m6)
        assert dec['dc'] > 99.9
        assert abs(m6[0] + m6[1] + m6[2]) < 1e-9 * scalar_moment(m6)


def test_sample_in_polygon():
    rng = np.random.default_rng(2)
    poly = [[110.0, -12.0], [120.0, -12.0], [120.0, -5.0], [110.0, -5.0]]
    for _ in range(100):
        lon, lat = sample_in_polygon(rng, poly)
        assert 110.0 <= lon <= 120.0 and -12.0 <= lat <= -5.0


def test_sample_nuisance_respects_depth_bounds():
    rng = np.random.default_rng(3)
    priors = dict(dn_km=[-4.0, 4.0], de_km=[-4.0, 4.0], dz_km=[-3.0, 3.0],
                  dt0_s=[-3.0, 3.0])
    for _ in range(200):
        th = sample_nuisance(rng, priors, 3000.0, 2000.0, 12000.0)
        assert 2000.0 <= 3000.0 + th[2] <= 12000.0
        assert abs(th[0]) <= 4000.0 and abs(th[3]) <= 3.0


def test_rtp_to_ned_matches_pyrocko():
    from pyrocko import moment_tensor as pmt
    rng = np.random.default_rng(4)
    for _ in range(10):
        mrr, mtt, mpp, mrt, mrp, mtp = rng.standard_normal(6)
        m_use = np.array([[mrr, mrt, mrp],
                          [mrt, mtt, mtp],
                          [mrp, mtp, mpp]])
        mt = pmt.MomentTensor(m_up_south_east=m_use)
        ours = rtp_to_ned(np.array([mrr, mtt, mpp, mrt, mrp, mtp]))
        np.testing.assert_allclose(ours, mt.m6(), atol=1e-12)


# ----------------------------------------------------------------------------
# point grid == full grid
# ----------------------------------------------------------------------------

def test_point_grid_matches_full_grid(test_store, test_cfg):
    rng = np.random.default_rng(5)
    ev, stations = random_event_and_stations(rng, n_sta=3)
    geoms = station_geometries(ev, stations)
    fetcher = StoreFetcher(test_store)
    itmin, n_t = data_windows(fetcher, ev, geoms, test_cfg)

    dn, de, dz = 1500.0, -1800.0, 800.0     # inside the test grid extents
    grid_full = build_gf_grid(fetcher, ev, geoms, test_cfg, itmin=itmin,
                              n_t=n_t, dtype=torch.float64)
    grid_pt = build_point_grid(fetcher, ev, geoms, test_cfg, dn, de, dz,
                               itmin=itmin, n_t=n_t, dtype=torch.float64)
    assert grid_pt.gf.shape[1:3] == (2, 2)

    args = tuple(torch.tensor([v], dtype=torch.float64)
                 for v in (dn, de, dz))
    G_full = assemble_G(grid_full, *args)
    G_pt = assemble_G(grid_pt, *args)
    scale = G_full.abs().max()
    assert (G_full - G_pt).abs().max() / scale < 1e-12


# ----------------------------------------------------------------------------
# end-to-end generator
# ----------------------------------------------------------------------------

@pytest.fixture(scope='module')
def synth_cfg(test_cfg):
    cfg = copy.deepcopy(test_cfg)
    # shrink polygon/priors so the conftest station pool stays in range
    cfg['priors'].update(
        study_polygon=[[114.95, -8.05], [115.05, -8.05],
                       [115.05, -7.95], [114.95, -7.95]],
        depth_km=[5.0, 9.0], dn_km=[-2.5, 2.5], de_km=[-2.5, 2.5],
        dz_km=[-2.0, 2.0], dt0_s=[-2.0, 2.0],
        station_dt_zero_fraction=1.0)
    cfg['synth'].update(
        n_stations=[4, 6], station_dropout=0.0, snr_range=[50.0, 200.0],
        noise='gaussian', velocity_perturbed_fraction=0.0)
    return cfg


@pytest.fixture(scope='module')
def generator(test_store, synth_cfg):
    rng = np.random.default_rng(6)
    _, stations = random_event_and_stations(rng, n_sta=8)
    return SyntheticGenerator(
        synth_cfg, stations, StoreFetcher(test_store),
        noise=GaussianNoise()), stations


def test_generator_example_structure(generator, synth_cfg):
    gen, _ = generator
    rng = np.random.default_rng(7)
    ex = gen.generate(rng)
    n_tr, n_t = ex['waveforms'].shape
    n_sta = len(ex['station_codes'])
    assert n_tr == 3 * n_sta and 4 <= n_sta <= 6
    assert n_t == int(round(45.0 * 4.0))       # window_length_s * rate
    assert ex['metadata'].shape == (n_tr, 9)   # model.metadata_dim
    assert ex['metadata'].dtype == np.float32
    assert ex['theta'].shape == (4,) and ex['m6'].shape == (6,)
    assert ex['mask'].dtype == bool and ex['weights'][~ex['mask']].sum() == 0
    # per-trace RMS normalization, norm stored
    rms = np.sqrt(np.mean(ex['waveforms'].astype(np.float64) ** 2, axis=-1))
    np.testing.assert_allclose(rms, 1.0, rtol=1e-4)
    # median per-trace SNR hits the target range by construction
    assert 45.0 <= np.median(ex['snr']) <= 220.0
    assert scalar_moment(ex['m6']) == pytest.approx(
        float(mw_to_m0(ex['mw'])), rel=1e-6)


def test_generator_wls_recovers_mt(generator, synth_cfg, test_store):
    """The final Phase 3 smoke test: WLS on the REFERENCE full grid at the
    TRUE theta must recover the true MT from a generated high-SNR example."""
    gen, stations = generator
    rng = np.random.default_rng(8)
    ex = gen.generate(rng)

    ev = EventInfo(name='synth', lat=float(ex['lat']), lon=float(ex['lon']),
                   depth=float(ex['depth']))
    by_code = {s.code: s for s in stations}
    sel = [by_code[c] for c in ex['station_codes']]
    geoms = station_geometries(ev, sel)
    fetcher = StoreFetcher(test_store)
    n_t = ex['waveforms'].shape[1]
    grid = build_gf_grid(fetcher, ev, geoms, synth_cfg, itmin=ex['itmin'],
                         n_t=n_t, dtype=torch.float64)
    fm = ForwardModel(grid, synth_cfg)

    data = torch.as_tensor(
        ex['waveforms'].astype(np.float64) * ex['norms'][:, None])
    weights = torch.as_tensor(ex['weights'])
    theta = torch.as_tensor(ex['theta'], dtype=torch.float64).unsqueeze(0)
    res, _ = fm.solve(theta, data, weights, align_stations=False)

    m_hat = res.m_hat[0].numpy()
    rel = np.linalg.norm(m_hat - ex['m6']) / np.linalg.norm(ex['m6'])
    assert rel < 0.05                       # float32 generation + SNR >= 50
    assert float(res.vr[0]) > 0.9


def test_generator_is_deterministic(generator):
    gen, _ = generator
    ex1 = gen.generate(np.random.default_rng(9))
    ex2 = gen.generate(np.random.default_rng(9))
    np.testing.assert_array_equal(ex1['waveforms'], ex2['waveforms'])
    np.testing.assert_array_equal(ex1['theta'], ex2['theta'])


# ----------------------------------------------------------------------------
# noise library
# ----------------------------------------------------------------------------

def test_noise_library_station_matching(tmp_path):
    path = str(tmp_path / 'noise.h5')
    with h5py.File(path, 'w') as f:
        f.attrs['band_hz'] = [0.04, 0.4]
        f.attrs['sample_rate_hz'] = 4.0
        f.attrs['window_length_s'] = 50.0
        g = f.create_group('XX.S00')
        g.create_dataset('Z', data=np.full((3, 200), 1.0, dtype='f4'))
        g.create_dataset('Z_epoch', data=np.zeros(3))
        g.create_dataset('H', data=np.full((3, 200), 2.0, dtype='f4'))
        g.create_dataset('H_epoch', data=np.zeros(3))

    lib = NoiseLibrary(path)
    rng = np.random.default_rng(10)
    out = lib.sample(rng, ['XX.S00', 'YY.S99'], ('Z', 'R', 'T'), 150)
    assert out.shape == (6, 150) and out.dtype == np.float32
    np.testing.assert_array_equal(out[0], 1.0)      # matched Z
    np.testing.assert_array_equal(out[1:3], 2.0)    # matched horizontals
    np.testing.assert_array_equal(out[3], 1.0)      # fallback station, Z
    np.testing.assert_array_equal(out[4:6], 2.0)
    # windows shorter than n_t wrap instead of failing
    assert lib.sample(rng, ['XX.S00'], ('Z',), 300).shape == (1, 300)
    lib.close()


# ----------------------------------------------------------------------------
# validation cache + collate
# ----------------------------------------------------------------------------

def test_validation_cache_roundtrip(generator, synth_cfg, tmp_path):
    gen, _ = generator
    path = str(tmp_path / 'val.h5')
    cache_validation_set(synth_cfg, gen, path, n=3, seed=11)
    events = list(iter_cached_events(path))
    assert len(events) == 3

    rng = np.random.default_rng(11)
    for ex in events:
        ref = gen.generate(rng)
        np.testing.assert_array_equal(ex['waveforms'], ref['waveforms'])
        np.testing.assert_array_equal(ex['theta'], ref['theta'])
        np.testing.assert_array_equal(ex['mask'], ref['mask'])
        assert ex['station_codes'] == ref['station_codes']
        assert ex['mw'] == ref['mw']
        assert ex['band_hz'] == ref['band_hz']

    batch = pad_collate(events)
    n_max = max(e['waveforms'].shape[0] for e in events)
    assert batch['waveforms'].shape == (3, n_max, events[0]['waveforms'].shape[1])
    assert batch['metadata'].shape[2] == 9
    assert batch['theta'].shape == (3, 4)
    for i, ex in enumerate(events):
        n = ex['waveforms'].shape[0]
        assert not batch['mask'][i, n:].any()
