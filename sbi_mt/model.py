"""Neural posterior estimator (PLAN §6, Phase 4).

Per-trace 1D CNN encoder with FiLM conditioning on the trace metadata,
masked Set Transformer aggregation over the variable-size station/component
set (permutation invariant), and a conditional neural spline flow (zuko)
over the nuisance parameters theta = (dn, de, dz, dt0).

The moment tensor is NOT modeled by the flow — given theta it is solved in
closed form by forward.wls_solve (Rao-Blackwellization). DirectMTHead exists
only for ablation (c) of PLAN §9.4 (direct regression instead of WLS).

theta convention: SI units (m, m, m, s) at every public interface; the flow
operates on standardized coordinates with an affine layer built from the
prior bounds (mean = box center, std = range / sqrt(12)).
"""
from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import zuko

from . import KM

logger = logging.getLogger('sbi_mt.model')


def prior_bounds_si(priors: dict[str, Any],
                    names: Sequence[str]) -> torch.Tensor:
    """(n_theta, 2) lower/upper prior bounds in SI units. Config prior keys
    carry their unit as a suffix (dn_km, dt0_s, ...)."""
    rows = []
    for name in names:
        lo, hi = (float(x) for x in priors[name])
        if name.endswith('_km'):
            lo, hi = lo * KM, hi * KM
        rows.append((lo, hi))
    return torch.tensor(rows, dtype=torch.float32)


# ----------------------------------------------------------------------------
# per-trace encoder: residual 1D CNN with FiLM conditioning
# ----------------------------------------------------------------------------

class ResBlock1d(nn.Module):
    """Stride-2 residual block: Conv-GN-GELU-Conv-GN + 1x1 skip, GELU."""

    def __init__(self, c_in: int, c_out: int, kernel: int, groups: int):
        super().__init__()
        pad = kernel // 2
        g = math.gcd(groups, c_out)
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, stride=2, padding=pad)
        self.gn1 = nn.GroupNorm(g, c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, stride=1, padding=pad)
        self.gn2 = nn.GroupNorm(g, c_out)
        self.skip = nn.Conv1d(c_in, c_out, 1, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return F.gelu(h + self.skip(x))


class TraceEncoder(nn.Module):
    """Waveform (n_t,) + metadata (metadata_dim,) -> embedding.

    FiLM: one shared metadata trunk, per-block heads producing (dgamma,
    beta); features are scaled by (1 + dgamma) after each block so the
    untrained modulation starts at identity."""

    def __init__(self, channels: Sequence[int], kernel_sizes: Sequence[int],
                 embedding_dim: int, groupnorm_groups: int,
                 metadata_dim: int, film_hidden: int = 64):
        super().__init__()
        assert len(channels) == len(kernel_sizes)
        c_prev = 1
        blocks = []
        for c, k in zip(channels, kernel_sizes):
            blocks.append(ResBlock1d(c_prev, c, k, groupnorm_groups))
            c_prev = c
        self.blocks = nn.ModuleList(blocks)
        self.film_trunk = nn.Sequential(
            nn.Linear(metadata_dim, film_hidden), nn.GELU())
        self.film_heads = nn.ModuleList(
            [nn.Linear(film_hidden, 2 * c) for c in channels])
        for head in self.film_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.out = nn.Linear(2 * channels[-1], embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, wf: torch.Tensor, md: torch.Tensor) -> torch.Tensor:
        """wf (N, n_t), md (N, metadata_dim) -> (N, embedding_dim)."""
        x = wf.unsqueeze(1)                              # (N, 1, n_t)
        film = self.film_trunk(md)
        for block, head in zip(self.blocks, self.film_heads):
            x = block(x)
            dgamma, beta = head(film).chunk(2, dim=-1)
            x = (1.0 + dgamma).unsqueeze(-1) * x + beta.unsqueeze(-1)
        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=-1)
        return self.out(pooled)


# ----------------------------------------------------------------------------
# masked Set Transformer (Lee et al. 2019): MAB / ISAB / PMA
# ----------------------------------------------------------------------------

class MAB(nn.Module):
    """Multihead attention block. kv_mask (B, n_kv) True = valid; masked
    keys receive exactly zero attention, so padded traces cannot influence
    any valid output row."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln0 = nn.LayerNorm(dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                kv_mask: torch.Tensor | None = None) -> torch.Tensor:
        kpm = None if kv_mask is None else ~kv_mask
        a, _ = self.attn(q, kv, kv, key_padding_mask=kpm,
                         need_weights=False)
        h = self.ln0(q + a)
        return self.ln1(h + self.ff(h))


class ISAB(nn.Module):
    """Induced set attention. The second MAB needs no mask: its keys are the
    inducing summaries (built from valid elements only) and outputs at padded
    query rows are discarded by the masks of downstream blocks."""

    def __init__(self, dim: int, n_heads: int, n_inducing: int):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, n_inducing, dim)
                                     / math.sqrt(dim))
        self.mab0 = MAB(dim, n_heads)
        self.mab1 = MAB(dim, n_heads)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.mab0(self.inducing.expand(x.shape[0], -1, -1), x, mask)
        return self.mab1(x, h)


class PMA(nn.Module):
    """Pooling by multihead attention onto one learned seed vector."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, dim) / math.sqrt(dim))
        self.mab = MAB(dim, n_heads)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.mab(self.seed.expand(x.shape[0], -1, -1), x,
                        mask).squeeze(1)


class SetEncoder(nn.Module):
    """Trace embeddings (+ raw metadata, concatenated) -> permutation-
    invariant event embedding of size `dim`."""

    def __init__(self, in_dim: int, dim: int, n_isab: int, n_heads: int,
                 n_inducing: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)
        self.isabs = nn.ModuleList(
            [ISAB(dim, n_heads, n_inducing) for _ in range(n_isab)])
        self.pma = PMA(dim, n_heads)
        self.dim = dim

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.proj(x)
        for isab in self.isabs:
            h = isab(h, mask)
        return self.pma(h, mask)


# ----------------------------------------------------------------------------
# ablation (c) head — direct MT regression, no WLS layer (PLAN §9.4)
# ----------------------------------------------------------------------------

class DirectMTHead(nn.Module):
    """Event embedding -> (unit-Frobenius m6 direction (6,), log10 M0)."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 7))

    def forward(self, emb: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(emb)
        direction = F.normalize(out[..., :6], dim=-1)
        return direction, out[..., 6]


# ----------------------------------------------------------------------------
# full posterior model
# ----------------------------------------------------------------------------

class PosteriorModel(nn.Module):
    """encoder -> set transformer -> conditional NSF over standardized theta.

    Public interfaces take/return theta in SI units. `embed` is exposed
    separately so training can reuse one event embedding for the NPE loss
    and the aux misfit loss (flow rsamples)."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        mcfg = cfg['model']
        enc = mcfg['encoder']
        st = mcfg['set_transformer']
        fl = mcfg['flow']
        if str(fl.get('type', 'nsf')) != 'nsf':
            raise ValueError(f"unsupported flow type {fl['type']!r}")

        self.theta_names: tuple[str, ...] = tuple(cfg['theta']['names'])
        self.n_theta = len(self.theta_names)
        bounds = prior_bounds_si(cfg['priors'], self.theta_names)
        loc = bounds.mean(dim=1)
        scale = (bounds[:, 1] - bounds[:, 0]) / math.sqrt(12.0)
        self.register_buffer('theta_loc', loc)
        self.register_buffer('theta_scale', scale)

        md_dim = int(mcfg['metadata_dim'])
        self.encoder = TraceEncoder(
            channels=[int(c) for c in enc['channels']],
            kernel_sizes=[int(k) for k in enc['kernel_sizes']],
            embedding_dim=int(enc['embedding_dim']),
            groupnorm_groups=int(enc['groupnorm_groups']),
            metadata_dim=md_dim)
        self.set_encoder = SetEncoder(
            in_dim=int(enc['embedding_dim']) + md_dim,
            dim=int(st['dim']), n_isab=int(st['n_isab']),
            n_heads=int(st['n_heads']), n_inducing=int(st['n_inducing']))
        self.flow = zuko.flows.NSF(
            features=self.n_theta, context=int(st['dim']),
            transforms=int(fl['transforms']),
            hidden_features=tuple(int(h) for h in fl['hidden']),
            bins=int(fl['bins']))

        self.direct_mt = (DirectMTHead(int(st['dim']))
                          if bool(cfg.get('evaluate', {}).get(
                              'ablations', {}).get('direct_regression'))
                          else None)

        n = self.n_parameters()
        logger.info('PosteriorModel: %.2f M parameters (budget 1-3 M)',
                    n / 1e6)
        print(f'PosteriorModel: {n:,} parameters ({n / 1e6:.2f} M)')

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- embeddings ----------------------------------------------------------

    def embed(self, waveforms: torch.Tensor, metadata: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
        """waveforms (B, n_tr, n_t), metadata (B, n_tr, md), mask (B, n_tr)
        True = valid -> event embedding (B, set_dim)."""
        B, n_tr, n_t = waveforms.shape
        tr = self.encoder(waveforms.reshape(B * n_tr, n_t),
                          metadata.reshape(B * n_tr, -1))
        tr = torch.cat([tr.reshape(B, n_tr, -1), metadata], dim=-1)
        return self.set_encoder(tr, mask)

    # -- theta standardization ----------------------------------------------

    def standardize(self, theta: torch.Tensor) -> torch.Tensor:
        return (theta - self.theta_loc) / self.theta_scale

    def destandardize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.theta_scale + self.theta_loc

    @property
    def _log_det_scale(self) -> torch.Tensor:
        return torch.log(self.theta_scale).sum()

    # -- flow interfaces (theta in SI) ---------------------------------------

    def log_prob_emb(self, emb: torch.Tensor,
                     theta: torch.Tensor) -> torch.Tensor:
        dist = self.flow(emb)
        return dist.log_prob(self.standardize(theta)) - self._log_det_scale

    def log_prob(self, waveforms: torch.Tensor, metadata: torch.Tensor,
                 mask: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return self.log_prob_emb(self.embed(waveforms, metadata, mask),
                                 theta)

    def sample_emb(self, emb: torch.Tensor, n: int) -> torch.Tensor:
        """(B, set_dim) -> theta samples (B, n, n_theta), no grad."""
        with torch.no_grad():
            z = self.flow(emb).sample((n,))              # (n, B, n_theta)
        return self.destandardize(z).permute(1, 0, 2)

    def sample(self, waveforms: torch.Tensor, metadata: torch.Tensor,
               mask: torch.Tensor, n: int) -> torch.Tensor:
        return self.sample_emb(self.embed(waveforms, metadata, mask), n)

    def rsample_emb(self, emb: torch.Tensor, n: int
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparametrized samples for the aux misfit loss: theta (B, n,
        n_theta) with gradients into the flow AND the encoders, plus
        log q(theta) (B, n)."""
        z, log_q = self.flow(emb).rsample_and_log_prob((n,))
        theta = self.destandardize(z).permute(1, 0, 2)
        return theta, (log_q - self._log_det_scale).permute(1, 0)


def build_model(cfg: dict[str, Any],
                device: torch.device | str = 'cpu') -> PosteriorModel:
    model = PosteriorModel(cfg).to(device)
    return model
