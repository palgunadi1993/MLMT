"""SBI-MT: amortized Bayesian moment tensor inversion.

Neural posterior estimation over nuisance parameters (centroid perturbation,
time shift) combined with a closed-form (Rao-Blackwellized) weighted linear
least-squares moment tensor solve against a Green's function library.
"""
from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
import yaml

__version__ = '0.1.0'

KM = 1000.0


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config file. All hyperparameters, paths, bands and priors
    live in the config — modules must not hardcode any of them (Tier B is a
    config change, not a code change)."""
    with open(path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    cfg['_config_path'] = os.path.abspath(path)
    cfg['_root'] = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    return cfg


def resolve_path(cfg: dict[str, Any], path: str) -> str:
    """Resolve a config-relative path against the repository root."""
    if os.path.isabs(path):
        return path
    return os.path.join(cfg.get('_root', '.'), path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict[str, Any]) -> torch.device:
    dev = cfg.get('device', 'auto')
    if dev == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(dev)
