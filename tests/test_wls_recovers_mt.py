"""Phase 2 acceptance (PLAN §4.3): noise-free synthetic with known m and
correct theta -> recovered m error < 1e-8; wrong theta (km-scale centroid
error) -> visibly degraded variance reduction."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from sbi_mt.forward import (
    ForwardModel, align_station_shifts, bandpass_zerophase, cosine_taper,
    project_deviatoric, time_shift, wls_solve)
from sbi_mt.gf import StoreFetcher, build_gf_grid, station_geometries
from conftest import random_event_and_stations


@pytest.fixture(scope='module')
def setup(test_store, test_cfg):
    cfg = copy.deepcopy(test_cfg)
    cfg['wls']['tikhonov'] = 1e-14        # unbiased solve for recovery tests
    cfg['wls']['station_shift_enable'] = False
    rng = np.random.default_rng(3)
    ev, stations = random_event_and_stations(rng, n_sta=5)
    geoms = station_geometries(ev, stations)
    grid = build_gf_grid(StoreFetcher(test_store), ev, geoms, cfg,
                         dtype=torch.float64, store=test_store)
    fm = ForwardModel(grid, cfg)
    m6 = torch.as_tensor(rng.normal(size=6) * 1e15, dtype=torch.float64)
    weights = torch.ones(grid.n_sta * 3, dtype=torch.float64)
    return fm, m6, weights


def theta(dn=0.0, de=0.0, dz=0.0, dt0=0.0) -> torch.Tensor:
    return torch.tensor([[dn, de, dz, dt0]], dtype=torch.float64)


def test_noise_free_recovery_below_1e8(setup):
    fm, m6, weights = setup
    data = fm.synthetics(theta(), m6)[0]
    res, _ = fm.solve(theta(), data, weights)
    rel = float(torch.linalg.norm(res.m_hat[0] - m6)
                / torch.linalg.norm(m6))
    assert rel < 1e-8, f'relative MT error {rel:.3e} exceeds 1e-8'
    assert float(res.vr[0]) > 1.0 - 1e-12


def test_wrong_centroid_degrades_vr(setup):
    fm, m6, weights = setup
    data = fm.synthetics(theta(), m6)[0]
    res_true, _ = fm.solve(theta(), data, weights)
    # 5 km centroid error: (3, 4) km horizontal (grid extent is +-4 km)
    res_wrong, _ = fm.solve(theta(dn=3e3, de=4e3), data, weights)
    vr_t, vr_w = float(res_true.vr[0]), float(res_wrong.vr[0])
    print(f'\n[sanity] VR correct theta: {vr_t:.4f}   '
          f'VR 5 km centroid error: {vr_w:.4f}')
    assert vr_t > 0.999
    assert vr_w < vr_t - 0.05, 'wrong centroid should visibly degrade VR'


def test_deviatoric_constraint(setup):
    fm, m6, weights = setup
    m6_dev = torch.as_tensor(project_deviatoric(m6.numpy()))
    data = fm.synthetics(theta(), m6_dev)[0]
    fm_dev = copy.copy(fm)
    fm_dev.constraint = 'deviatoric'
    res, _ = fm_dev.solve(theta(), data, weights)
    m_hat = res.m_hat[0]
    assert abs(float(m_hat[0] + m_hat[1] + m_hat[2])) < 1e-6 * float(
        torch.linalg.norm(m_hat))
    rel = float(torch.linalg.norm(m_hat - m6_dev) / torch.linalg.norm(m6_dev))
    assert rel < 1e-8


def test_dt0_shifts_are_consistent(setup):
    fm, m6, weights = setup
    data = fm.synthetics(theta(dt0=1.3), m6)[0]
    res_good, _ = fm.solve(theta(dt0=1.3), data, weights)
    res_bad, _ = fm.solve(theta(), data, weights)
    rel = float(torch.linalg.norm(res_good.m_hat[0] - m6)
                / torch.linalg.norm(m6))
    assert rel < 1e-8
    assert float(res_bad.vr[0]) < float(res_good.vr[0]) - 0.05


def test_station_alignment_recovers_shifts(setup):
    fm, m6, weights = setup
    true = torch.tensor([[0.6, -0.9, 0.3, 0.0, -0.45]], dtype=torch.float64)
    data = fm.synthetics(theta(), m6, station_shifts=true)[0]
    res0, _ = fm.solve(theta(), data, weights, align_stations=False)
    shifts = align_station_shifts(res0.synth, data.unsqueeze(0),
                                  fm.deltat, max_shift=1.5)
    assert torch.allclose(shifts, true, atol=0.1), (
        f'measured {shifts.tolist()} vs true {true.tolist()}')
    res1, _ = fm.solve(theta(), data, weights, align_stations=True)
    assert float(res1.vr[0]) > float(res0.vr[0])
    assert float(res1.vr[0]) > 0.98


def test_waveform_ops():
    # taper: ends ~0, middle 1
    w = cosine_taper(200, 0.1, torch.float64)
    assert float(w[0]) < 1e-12 and float(w[100]) == 1.0
    # zero-phase bandpass: impulse response symmetric about the impulse
    x = torch.zeros(1, 512, dtype=torch.float64)
    x[0, 256] = 1.0
    y = bandpass_zerophase(x, 0.25, 0.04, 0.4, 4)[0]
    left, right = y[:256].flip(0), y[257:]
    assert torch.allclose(left[:200], right[:200], atol=1e-12)
    # time_shift: shifting a smooth pulse by k samples == roll
    t = torch.arange(512, dtype=torch.float64) * 0.25
    g = torch.exp(-((t - 60.0) / 5.0) ** 2).unsqueeze(0)
    shifted = time_shift(g, 0.25, torch.tensor([[2.0]], dtype=torch.float64))
    assert torch.allclose(shifted[0], torch.roll(g[0], 8), atol=1e-9)