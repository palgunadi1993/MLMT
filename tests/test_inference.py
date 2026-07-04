"""Phase 6 smoke test: run_event on synthetic events against the test
store — theta sampling, chunked full-grid WLS, Gaussian-mixture MT draws,
importance weights, results round trip. Uses an untrained model (the
mechanics, not the accuracy, are under test)."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from sbi_mt.data import GaussianNoise
from sbi_mt.gf import StoreFetcher
from sbi_mt.inference import load_results, run_event, save_results
from sbi_mt.model import PosteriorModel
from sbi_mt.synth import SyntheticGenerator
from conftest import random_event_and_stations


@pytest.fixture(scope='module')
def inf_cfg(test_cfg):
    cfg = copy.deepcopy(test_cfg)
    cfg['priors'].update(
        study_polygon=[[114.95, -8.05], [115.05, -8.05],
                       [115.05, -7.95], [114.95, -7.95]],
        depth_km=[5.0, 9.0], dn_km=[-2.5, 2.5], de_km=[-2.5, 2.5],
        dz_km=[-2.0, 2.0], dt0_s=[-2.0, 2.0],
        station_dt_zero_fraction=1.0)
    cfg['synth'].update(
        n_stations=[4, 6], station_dropout=0.0, snr_range=[10.0, 50.0],
        noise='gaussian', velocity_perturbed_fraction=0.0)
    cfg['inference'].update(n_theta_samples=64, wls_chunk=32,
                            n_mt_draws_per_theta=2)
    return cfg


def test_run_event_and_roundtrip(inf_cfg, test_store, tmp_path):
    torch.manual_seed(3)
    rng = np.random.default_rng(21)
    _, stations = random_event_and_stations(rng, n_sta=8)
    fetcher = StoreFetcher(test_store)
    gen = SyntheticGenerator(inf_cfg, stations, fetcher,
                             noise=GaussianNoise())
    ex = gen.generate(rng)

    model = PosteriorModel(inf_cfg)
    res = run_event(model, ex, inf_cfg, fetcher, 'cpu', rng, name='synth0')

    n = 64
    n_sta = len(ex['station_codes'])
    assert res['theta_samples'].shape == (n, 4)
    assert res['m_hats'].shape == (n, 6)
    assert res['covs'].shape == (n, 6, 6)
    assert res['m6_samples'].shape == (n * 2, 6)
    assert res['vr_trace'].shape == (n, 3 * n_sta)
    assert res['vr_station_map'].shape == (n_sta, 3)
    assert np.isfinite(res['m_hats']).all()
    assert np.isfinite(res['log_q']).all()
    w = res['w_is']
    assert w.shape == (n,) and w.sum() == pytest.approx(1.0)
    assert 1.0 <= res['ess'] <= n + 1e-6
    s = res['summary']
    for key in ('mean', 'mean_is', 'map'):
        assert len(s[key]['m6']) == 6
        dec = s[key]['decomposition']
        assert dec['iso'] + dec['dc'] + dec['clvd'] == pytest.approx(
            100.0, abs=1e-6)
    assert s['timing_s']['total'] > 0

    outdir = str(tmp_path / 'synth0')
    save_results(res, outdir)
    back = load_results(outdir)
    np.testing.assert_allclose(back['m_hats'], res['m_hats'])
    np.testing.assert_allclose(back['w_is'], res['w_is'])
    assert back['event'] == 'synth0'
    assert back['summary']['map']['mw'] == s['map']['mw']


def test_evaluate_synthetic_smoke(inf_cfg, test_store, tmp_path):
    """Phase 7 pipeline: SBC ranks / coverage / Kagan arrays over a few
    events, plus HDF5 round trip."""
    import copy
    from sbi_mt.evaluate import (
        evaluate_synthetic, load_evaluation, save_evaluation)
    cfg = copy.deepcopy(inf_cfg)
    cfg['evaluate'].update(n_theta_samples=48,
                           coverage_levels=[0.5, 0.9])
    torch.manual_seed(4)
    rng = np.random.default_rng(31)
    _, stations = random_event_and_stations(rng, n_sta=8)
    fetcher = StoreFetcher(test_store)
    gen = SyntheticGenerator(cfg, stations, fetcher, noise=GaussianNoise())
    events = [gen.generate(rng) for _ in range(3)]

    model = PosteriorModel(cfg)
    arrays = evaluate_synthetic(model, events, cfg, fetcher, 'cpu', rng)
    assert arrays['ranks_theta'].shape == (3, 4)
    assert arrays['ranks_m6'].shape == (3, 6)
    assert (arrays['ranks_theta'] <= 48).all()
    assert arrays['hits_m6'].shape == (3, 2, 6)
    assert arrays['kagan_deg'].shape == (3,)
    assert np.isfinite(arrays['kagan_deg']).all()
    assert np.isnan(arrays['kagan_direct_deg']).all()   # no ablation head

    path = str(tmp_path / 'eval.h5')
    save_evaluation(arrays, path)
    back = load_evaluation(path)
    np.testing.assert_allclose(back['kagan_deg'], arrays['kagan_deg'])
