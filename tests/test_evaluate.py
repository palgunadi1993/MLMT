"""Phase 7 acceptance: Kagan angle vs pyrocko, SBC rank uniformity for an
exact sampler, coverage of a known Gaussian, mixture draws moments, and the
binned-accuracy / benchmark plumbing."""
from __future__ import annotations

import numpy as np
import pytest

from sbi_mt.evaluate import (
    binned_kagan, central_interval_hits, empirical_coverage, kagan_angle,
    mt_posterior_draws, rank_uniformity_pvalue, sbc_ranks)
from sbi_mt.forward import matrix_to_m6
from sbi_mt.synth import _random_rotation


def test_kagan_angle_matches_pyrocko():
    from pyrocko import moment_tensor as pmt
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.standard_normal(6) * 1e17
        b = rng.standard_normal(6) * 1e17
        mta = pmt.MomentTensor(mnn=a[0], mee=a[1], mdd=a[2], mne=a[3],
                               mnd=a[4], med=a[5])
        mtb = pmt.MomentTensor(mnn=b[0], mee=b[1], mdd=b[2], mne=b[3],
                               mnd=b[4], med=b[5])
        expect = pmt.kagan_angle(mta, mtb)
        assert kagan_angle(a, b) == pytest.approx(expect, abs=0.05)


def test_kagan_angle_known_cases():
    dc = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])       # pure strike-slip
    # arccos near +1 amplifies roundoff to ~1e-6 deg — tolerance reflects that
    assert kagan_angle(dc, dc) == pytest.approx(0.0, abs=1e-4)
    assert kagan_angle(dc, 2.5 * dc) == pytest.approx(0.0, abs=1e-4)
    # rotating a DC by a known small angle about a principal axis
    rng = np.random.default_rng(1)
    q = _random_rotation(rng)
    m = q @ np.diag([-1.0, 0.0, 1.0]) @ q.T
    ang = np.deg2rad(25.0)
    c, s = np.cos(ang), np.sin(ang)
    # rotate about the B (middle) eigenvector => Kagan angle == 25 deg
    w, v = np.linalg.eigh(m)
    axis = v[:, 1]
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    m_rot = R @ m @ R.T
    assert kagan_angle(matrix_to_m6(m), matrix_to_m6(m_rot)) == (
        pytest.approx(25.0, abs=0.1))


def test_sbc_ranks_uniform_for_exact_sampler():
    rng = np.random.default_rng(2)
    n_sim, n_samp, k = 400, 100, 3
    ranks = np.stack([
        sbc_ranks(rng.standard_normal((n_samp, k)),
                  rng.standard_normal(k))
        for _ in range(n_sim)])
    for d in range(k):
        assert rank_uniformity_pvalue(ranks[:, d], n_samp) > 0.005
    # and clearly non-uniform for a biased sampler
    ranks_bad = np.stack([
        sbc_ranks(rng.standard_normal((n_samp, 1)) + 2.0,
                  rng.standard_normal(1))
        for _ in range(n_sim)])
    assert rank_uniformity_pvalue(ranks_bad[:, 0], n_samp) < 1e-6


def test_coverage_of_exact_gaussian():
    rng = np.random.default_rng(3)
    levels = [0.5, 0.9]
    hits = np.stack([
        central_interval_hits(rng.standard_normal((400, 2)),
                              rng.standard_normal(2), levels)
        for _ in range(600)])
    cov = hits.mean(axis=0)                          # (n_levels, 2)
    assert np.allclose(cov[0], 0.5, atol=0.06)
    assert np.allclose(cov[1], 0.9, atol=0.04)


def test_mt_posterior_draws_moments():
    rng = np.random.default_rng(4)
    m_hat = np.tile(np.array([1.0, -2.0, 1.0, 0.5, 0.0, -0.5]), (3, 1))
    cov = np.stack([np.diag([0.1, 0.2, 0.1, 0.05, 0.05, 0.05]) ** 2] * 3)
    draws = mt_posterior_draws(m_hat, cov, rng, n_per=4000)
    assert draws.shape == (12000, 6)
    np.testing.assert_allclose(draws.mean(axis=0), m_hat[0], atol=0.01)
    np.testing.assert_allclose(draws.std(axis=0),
                               np.sqrt(np.diag(cov[0])), rtol=0.05)


def test_binned_kagan_and_empirical_coverage():
    arrays = {
        'snr_median': np.array([2.0, 2.5, 8.0, 9.0]),
        'kagan_deg': np.array([30.0, 40.0, 5.0, 15.0]),
        'perturbed': np.array([False, True, False, True]),
        'hits_m6': np.ones((4, 2, 6), dtype=bool),
    }
    b = binned_kagan(arrays, 'snr_median', [1.5, 6.0, 20.0])
    assert b['median_kagan'][0].tolist() == [30.0, 5.0]     # reference
    assert b['median_kagan'][1].tolist() == [40.0, 15.0]    # perturbed
    assert empirical_coverage(arrays, 'hits_m6').shape == (2, 6)
    assert empirical_coverage(arrays, 'hits_m6').mean() == 1.0
