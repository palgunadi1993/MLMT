"""Phase 5 acceptance (PLAN §11.4): the training loop must overfit a small
fixed event set — NPE loss drops by several nats from initialization — with
the aux misfit loss and validation metrics (Kagan, coverage) exercised
end-to-end against the analytical test store."""
from __future__ import annotations

import copy
import json
import os

import numpy as np
import pytest
import torch

from sbi_mt.data import GaussianNoise
from sbi_mt.gf import StoreFetcher
from sbi_mt.synth import SyntheticGenerator
from sbi_mt.train import fit, fixed_event_batches
from conftest import random_event_and_stations


@pytest.fixture(scope='module')
def train_cfg(test_cfg):
    cfg = copy.deepcopy(test_cfg)
    cfg['priors'].update(
        study_polygon=[[114.95, -8.05], [115.05, -8.05],
                       [115.05, -7.95], [114.95, -7.95]],
        depth_km=[5.0, 9.0], dn_km=[-2.5, 2.5], de_km=[-2.5, 2.5],
        dz_km=[-2.0, 2.0], dt0_s=[-2.0, 2.0],
        station_dt_zero_fraction=1.0)
    cfg['synth'].update(
        n_stations=[4, 6], station_dropout=0.0, snr_range=[5.0, 50.0],
        noise='gaussian', velocity_perturbed_fraction=0.0)
    cfg['train'].update(
        batch_size=8, lr=1.0e-3, eval_every_steps=60, log_every_steps=20,
        early_stop_patience=50, num_workers=0, amp=False,
        aux_misfit_loss=True, aux_misfit_weight=0.1,
        aux_misfit_samples=2, aux_misfit_events_per_batch=1,
        aux_misfit_stations=3,
        val_metrics_events=4, val_theta_samples=32, val_wls_chunk=32)
    return cfg


def test_overfit_small_event_set(train_cfg, test_store, tmp_path_factory):
    torch.manual_seed(0)
    rng = np.random.default_rng(11)
    _, stations = random_event_and_stations(rng, n_sta=8)
    fetcher = StoreFetcher(test_store)
    gen = SyntheticGenerator(train_cfg, stations, fetcher,
                             noise=GaussianNoise())
    events = [gen.generate(rng) for _ in range(24)]

    run_dir = str(tmp_path_factory.mktemp('overfit_run'))
    batches = fixed_event_batches(events, 8, seed=5)
    summary = fit(train_cfg, batches, events, fetcher, run_dir,
                  max_steps=180, device=torch.device('cpu'))

    with open(os.path.join(run_dir, 'log.jsonl')) as f:
        records = [json.loads(line) for line in f]
    first_train = next(r for r in records if 'npe_loss' in r)
    evals = [r for r in records if 'val_loss' in r]
    assert len(evals) >= 2

    # memorization: val(==train) loss well below the initial training loss
    assert summary['best_val_loss'] < first_train['npe_loss'] - 1.5, (
        first_train['npe_loss'], summary['best_val_loss'])
    # loss decreases over evals (allow noise: last <= first)
    assert evals[-1]['val_loss'] <= evals[0]['val_loss']

    # val metrics were produced and are sane
    m = summary['last_metrics']
    assert 0.0 <= m['val_kagan_deg'] <= 120.0
    assert 0.0 <= m['val_coverage_68'] <= 1.0
    assert 0.0 <= m['val_coverage_90'] <= 1.0

    # checkpoints reload into a working model
    from sbi_mt.train import load_checkpoint
    model, ckpt = load_checkpoint(os.path.join(run_dir, 'ckpt_best.pt'))
    assert ckpt['metrics']['val_loss'] == pytest.approx(
        summary['best_val_loss'])
    from sbi_mt.synth import pad_collate
    b = pad_collate(events[:2])
    with torch.no_grad():
        lp = model.log_prob(b['waveforms'], b['metadata'], b['mask'],
                            b['theta'])
    assert torch.isfinite(lp).all()
