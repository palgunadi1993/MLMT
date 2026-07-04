#!/usr/bin/env python3
"""Cache the fixed synthetic validation split (PLAN §5, §7). Training data
are generated on the fly in the dataloader; only validation is cached.

    python scripts/03_generate_training_data.py --config config/default.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sbi_mt import load_config, resolve_path, seed_everything  # noqa: E402
from sbi_mt.data import load_stations, make_noise_source  # noqa: E402
from sbi_mt.synth import (  # noqa: E402
    SyntheticGenerator, cache_validation_set, load_fetchers)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config/default.yaml')
    ap.add_argument('--n', type=int, default=None,
                    help='override synth.n_validation')
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(name)s: %(message)s')
    cfg = load_config(args.config)
    seed_everything(int(cfg['seed']))

    stations, _ = load_stations(
        resolve_path(cfg, cfg['paths']['stations_xml']))
    reference, perturbed = load_fetchers(cfg)
    gen = SyntheticGenerator(
        cfg, stations, reference, perturbed, noise=make_noise_source(cfg))
    out = resolve_path(cfg, cfg['paths']['validation_cache'])
    cache_validation_set(cfg, gen, out, n=args.n)
    print('validation cache written to', out)


if __name__ == '__main__':
    main()
