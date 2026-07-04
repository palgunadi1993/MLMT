#!/usr/bin/env python3
"""Train the amortized posterior (PLAN §7). On-the-fly synthetic training
against the real station inventory; validation on the cached split from
script 03. Run the overfit sanity check before any full training:

    python scripts/04_train.py --config config/default.yaml --overfit 100
    python scripts/04_train.py --config config/default.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from sbi_mt import get_device, load_config, resolve_path, seed_everything  # noqa: E402
from sbi_mt.data import load_stations, make_noise_source  # noqa: E402
from sbi_mt.synth import (  # noqa: E402
    SyntheticGenerator, iter_cached_events, load_fetchers)
from sbi_mt.train import fit, fixed_event_batches, make_train_loader  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config/default.yaml')
    ap.add_argument('--run-name', default='sbi_mt')
    ap.add_argument('--max-steps', type=int, default=None,
                    help='override train.n_train_events / batch_size')
    ap.add_argument('--overfit', type=int, default=None, metavar='N',
                    help='sanity mode: train on N fixed events (PLAN §11)')
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(name)s: %(message)s')
    cfg = load_config(args.config)
    seed_everything(int(cfg['seed']))
    device = get_device(cfg)

    stations, _ = load_stations(
        resolve_path(cfg, cfg['paths']['stations_xml']))
    reference, perturbed = load_fetchers(cfg)
    gen = SyntheticGenerator(
        cfg, stations, reference, perturbed, noise=make_noise_source(cfg))

    if args.overfit:
        rng = np.random.default_rng(int(cfg['seed']))
        events = [gen.generate(rng) for _ in range(args.overfit)]
        batches = fixed_event_batches(
            events, int(cfg['train']['batch_size']), int(cfg['seed']) + 1)
        val_events = events                    # overfit: val == train
        run_dir = os.path.join(resolve_path(cfg, cfg['paths']['runs']),
                               f'{args.run_name}_overfit{args.overfit}')
        max_steps = args.max_steps or 2000
        summary = fit(cfg, batches, val_events, reference, run_dir,
                      max_steps=max_steps, device=device)
        print('overfit sanity:', summary)
        return

    val_path = resolve_path(cfg, cfg['paths']['validation_cache'])
    if not os.path.exists(val_path):
        sys.exit(f'validation cache {val_path} missing — run '
                 'scripts/03_generate_training_data.py first')
    val_events = list(iter_cached_events(val_path))
    loader = make_train_loader(cfg, gen)
    run_dir = os.path.join(resolve_path(cfg, cfg['paths']['runs']),
                           args.run_name)
    summary = fit(cfg, loader, val_events, reference, run_dir,
                  max_steps=args.max_steps, device=device)
    print('training done:', summary)


if __name__ == '__main__':
    main()
