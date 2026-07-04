"""Phase 4 acceptance: shapes, permutation invariance, mask correctness,
parameter budget, and gradient flow through rsample (aux-loss path)."""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import pytest
import torch

from sbi_mt import load_config, seed_everything
from sbi_mt.model import PosteriorModel, prior_bounds_si

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

B, N_TR, N_T = 3, 12, 160


@pytest.fixture(scope='module')
def cfg() -> dict[str, Any]:
    return load_config(os.path.join(ROOT, 'config', 'default.yaml'))


@pytest.fixture(scope='module')
def model(cfg: dict[str, Any]) -> PosteriorModel:
    seed_everything(7)
    return PosteriorModel(cfg)


def _batch(cfg: dict[str, Any], seed: int = 0
           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                      torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    md_dim = int(cfg['model']['metadata_dim'])
    wf = torch.randn(B, N_TR, N_T, generator=g)
    md = torch.randn(B, N_TR, md_dim, generator=g)
    mask = torch.ones(B, N_TR, dtype=torch.bool)
    bounds = prior_bounds_si(cfg['priors'], cfg['theta']['names'])
    u = torch.rand(B, bounds.shape[0], generator=g)
    theta = bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])
    return wf, md, mask, theta


def test_shapes(model: PosteriorModel, cfg: dict[str, Any]) -> None:
    wf, md, mask, theta = _batch(cfg)
    emb = model.embed(wf, md, mask)
    assert emb.shape == (B, int(cfg['model']['set_transformer']['dim']))
    lp = model.log_prob(wf, md, mask, theta)
    assert lp.shape == (B,)
    assert torch.isfinite(lp).all()
    s = model.sample(wf, md, mask, n=17)
    assert s.shape == (B, 17, model.n_theta)
    assert torch.isfinite(s).all()


def test_permutation_invariance(model: PosteriorModel,
                                cfg: dict[str, Any]) -> None:
    wf, md, mask, theta = _batch(cfg, seed=1)
    perm = torch.randperm(N_TR, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        lp = model.log_prob(wf, md, mask, theta)
        lp_perm = model.log_prob(
            wf[:, perm], md[:, perm], mask[:, perm], theta)
    assert torch.allclose(lp, lp_perm, atol=1e-4), (lp, lp_perm)


def test_mask_correctness(model: PosteriorModel,
                          cfg: dict[str, Any]) -> None:
    """Padded traces must not influence the posterior: same valid traces
    with different padding lengths/garbage give identical log_prob."""
    wf, md, mask, theta = _batch(cfg, seed=3)
    n_valid = 5
    mask = torch.zeros(B, N_TR, dtype=torch.bool)
    mask[:, :n_valid] = True

    wf2 = wf.clone()
    md2 = md.clone()
    wf2[:, n_valid:] = 1e3 * torch.randn(
        B, N_TR - n_valid, N_T, generator=torch.Generator().manual_seed(4))
    md2[:, n_valid:] = -7.0

    with torch.no_grad():
        lp_a = model.log_prob(wf, md, mask, theta)
        lp_b = model.log_prob(wf2, md2, mask, theta)
        lp_c = model.log_prob(wf[:, :n_valid], md[:, :n_valid],
                              mask[:, :n_valid], theta)
    assert torch.allclose(lp_a, lp_b, atol=1e-4)
    assert torch.allclose(lp_a, lp_c, atol=1e-4)


def test_variable_trace_counts(model: PosteriorModel,
                               cfg: dict[str, Any]) -> None:
    wf, md, mask, theta = _batch(cfg, seed=5)
    mask = mask.clone()
    mask[0, 4:] = False
    mask[1, 9:] = False
    lp = model.log_prob(wf, md, mask, theta)
    assert torch.isfinite(lp).all()


def test_parameter_budget(model: PosteriorModel) -> None:
    n = model.n_parameters()
    assert 1_000_000 <= n <= 3_000_000, f'{n:,} outside PLAN budget 1-3 M'


def test_theta_standardization_roundtrip(model: PosteriorModel,
                                         cfg: dict[str, Any]) -> None:
    bounds = prior_bounds_si(cfg['priors'], cfg['theta']['names'])
    theta = bounds[:, 0] + torch.rand(50, model.n_theta) * (
        bounds[:, 1] - bounds[:, 0])
    back = model.destandardize(model.standardize(theta))
    assert torch.allclose(back, theta, rtol=1e-5, atol=1e-3)
    z = model.standardize(theta)
    # std = range/sqrt(12) puts the uniform box inside [-sqrt(3), sqrt(3)]
    assert z.abs().max() <= math.sqrt(3.0) + 1e-5


def test_rsample_gradients_reach_encoder(model: PosteriorModel,
                                         cfg: dict[str, Any]) -> None:
    """The aux misfit loss backpropagates through flow rsamples into the
    encoder — the whole point of the fit-aware auxiliary term."""
    wf, md, mask, _ = _batch(cfg, seed=6)
    model.zero_grad(set_to_none=True)
    emb = model.embed(wf, md, mask)
    theta, log_q = model.rsample_emb(emb, n=3)
    assert theta.shape == (B, 3, model.n_theta)
    assert log_q.shape == (B, 3)
    (theta.pow(2).mean() + log_q.mean()).backward()
    conv_grad = model.encoder.blocks[0].conv1.weight.grad
    assert conv_grad is not None and conv_grad.abs().sum() > 0
    flow_grads = [p.grad for p in model.flow.parameters()
                  if p.grad is not None]
    assert len(flow_grads) > 0
