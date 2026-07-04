"""Phase 8 smoke test: every figure function renders and writes PDF+PNG.
Content is synthetic (untrained model) — layout/IO, not science, is under
test here."""
from __future__ import annotations

import copy
import json
import os

import numpy as np
import pytest
import torch

from sbi_mt.data import GaussianNoise
from sbi_mt.gf import StoreFetcher
from sbi_mt.inference import run_event
from sbi_mt.model import PosteriorModel
from sbi_mt.synth import SyntheticGenerator
from sbi_mt import plots
from conftest import random_event_and_stations


@pytest.fixture(scope='module')
def plot_setup(test_cfg, test_store, tmp_path_factory):
    cfg = copy.deepcopy(test_cfg)
    cfg['paths']['figures'] = str(tmp_path_factory.mktemp('figures'))
    cfg['priors'].update(
        study_polygon=[[114.95, -8.05], [115.05, -8.05],
                       [115.05, -7.95], [114.95, -7.95]],
        depth_km=[5.0, 9.0], dn_km=[-2.5, 2.5], de_km=[-2.5, 2.5],
        dz_km=[-2.0, 2.0], dt0_s=[-2.0, 2.0],
        station_dt_zero_fraction=1.0)
    cfg['synth'].update(
        n_stations=[4, 5], station_dropout=0.0, snr_range=[10.0, 50.0],
        noise='gaussian', velocity_perturbed_fraction=0.0)
    cfg['inference'].update(n_theta_samples=40, wls_chunk=20)
    cfg['plots']['waveform_fit_n_draws'] = 8

    torch.manual_seed(5)
    rng = np.random.default_rng(41)
    _, stations = random_event_and_stations(rng, n_sta=8)
    fetcher = StoreFetcher(test_store)
    gen = SyntheticGenerator(cfg, stations, fetcher, noise=GaussianNoise())
    ex = gen.generate(rng)
    model = PosteriorModel(cfg)
    result = run_event(model, ex, cfg, fetcher, 'cpu', rng, name='plotev')
    return cfg, ex, result, fetcher


def _check_written(cfg, name):
    outdir = cfg['paths']['figures']
    for ext in ('pdf', 'png'):
        p = os.path.join(outdir, f'{name}.{ext}')
        assert os.path.exists(p) and os.path.getsize(p) > 0, p


def test_waveform_fits(plot_setup):
    cfg, ex, result, fetcher = plot_setup
    plots.plot_waveform_fits(ex, result, cfg, fetcher, name='wf')
    _check_written(cfg, 'wf')


def test_fuzzy_beachball(plot_setup):
    cfg, ex, result, _ = plot_setup
    plots.plot_fuzzy_beachball(result, cfg, m6_ref=np.asarray(ex['m6']),
                               name='bb')
    _check_written(cfg, 'bb')


def test_lune_hudson(plot_setup):
    cfg, ex, result, _ = plot_setup
    plots.plot_lune_hudson(result, cfg, m6_ref=np.asarray(ex['m6']),
                           name='lune')
    _check_written(cfg, 'lune')
    # lune coordinate sanity: pure DC and pure ISO land on the landmarks
    g, d = plots.lune_coordinates(
        np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]))
    assert abs(g[0]) < 1e-9 and abs(d[0]) < 1e-9
    g, d = plots.lune_coordinates(np.array([[1.0, 1.0, 1.0, 0, 0, 0]]))
    assert d[0] == pytest.approx(90.0, abs=1e-9)


def test_nuisance_corner(plot_setup):
    cfg, ex, result, _ = plot_setup
    plots.plot_nuisance_corner(result, cfg,
                               theta_true=np.asarray(ex['theta']),
                               name='corner')
    _check_written(cfg, 'corner')


def test_station_map(plot_setup):
    cfg, ex, result, _ = plot_setup
    plots.plot_station_map(ex, result, cfg, name='map')
    _check_written(cfg, 'map')


def test_sbc_coverage_accuracy_ablation(plot_setup):
    cfg = plot_setup[0]
    rng = np.random.default_rng(7)
    n_ev, n_samp = 60, 40
    levels = [0.5, 0.68, 0.9]
    arrays = {
        'ranks_theta': rng.integers(0, n_samp + 1, size=(n_ev, 4)),
        'ranks_m6': rng.integers(0, n_samp + 1, size=(n_ev, 6)),
        'hits_theta': rng.random((n_ev, len(levels), 4)) < 0.7,
        'hits_m6': rng.random((n_ev, len(levels), 6)) < 0.7,
        'kagan_deg': rng.uniform(0, 60, n_ev),
        'kagan_direct_deg': rng.uniform(10, 80, n_ev),
        'snr_median': 10 ** rng.uniform(0.2, 1.3, n_ev),
        'n_stations': rng.integers(4, 6, n_ev),
        'perturbed': rng.random(n_ev) < 0.5,
        'coverage_levels': np.asarray(levels),
        'n_theta_samples': np.asarray(n_samp),
    }
    cfg2 = copy.deepcopy(cfg)
    cfg2['evaluate']['snr_bins'] = [1.5, 5.0, 20.0]
    cfg2['evaluate']['station_bins'] = [4, 5, 6]
    plots.plot_sbc(arrays, cfg2, name='sbc')
    _check_written(cfg2, 'sbc')
    plots.plot_coverage(arrays, cfg2, which='hits_m6', name='cov')
    _check_written(cfg2, 'cov')
    plots.plot_accuracy_curves(arrays, cfg2, name='acc')
    _check_written(cfg2, 'acc')
    plots.plot_ablation_summary(
        arrays, {'gaussian_noise': arrays,
                 'no_velocity_perturbation': arrays,
                 'direct_regression': arrays}, cfg2, name='abl')
    _check_written(cfg2, 'abl')


def test_benchmark_and_diagnostics(plot_setup, tmp_path):
    cfg = plot_setup[0]
    rows = [{'event': f'ev{i}', 'kagan_ref_deg': 10.0 + i,
             'kagan_grond_ref_deg': 12.0 + i, 'mw_sbi': 5.0,
             'time_sbi_s': 0.8, 'time_grond_s': 3600.0,
             'ci68_sbi': [0.1] * 6, 'ci68_grond': [0.12] * 6}
            for i in range(4)]
    plots.plot_benchmark(rows, cfg, name='bench')
    _check_written(cfg, 'bench')

    run_dir = str(tmp_path)
    with open(os.path.join(run_dir, 'log.jsonl'), 'w') as f:
        for step in range(1, 6):
            f.write(json.dumps({'step': step * 10,
                                'npe_loss': 30.0 - step}) + '\n')
            f.write(json.dumps({
                'step': step * 10, 'val_loss': 31.0 - step,
                'val_kagan_deg': 40.0 - 3 * step,
                'val_coverage_68': 0.6, 'val_coverage_90': 0.88}) + '\n')
    plots.plot_training_diagnostics(run_dir, cfg, name='diag')
    _check_written(cfg, 'diag')
