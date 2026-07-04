#!/usr/bin/env python3
"""Build the empirical noise library noise.h5 from continuous miniSEED under
data/raw/noise, rejecting windows that overlap catalog events (PLAN §5).

    python scripts/02_build_noise_library.py --config config/default.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sbi_mt import load_config, seed_everything  # noqa: E402
from sbi_mt.data import build_noise_library  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config/default.yaml')
    ap.add_argument('--out', default=None,
                    help='override paths.noise_library output path')
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(name)s: %(message)s')
    cfg = load_config(args.config)
    seed_everything(int(cfg['seed']))
    if not build_noise_library(cfg, out_path=args.out):
        print('No noise library written. TODO(user-data): place continuous '
              'miniSEED under', cfg['paths']['noise_raw'],
              '- training falls back to Gaussian noise until then.')
        sys.exit(1)


if __name__ == '__main__':
    main()
