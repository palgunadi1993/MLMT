#!/usr/bin/env python3
"""Amortized inference on catalog events (PLAN §8). Preprocesses real
waveforms with the identical pipeline used for training synthetics, then
writes results/<event_id>/{posterior.h5, summary.json} per event.

    python scripts/05_run_inference.py --config config/default.yaml \
        --checkpoint runs/sbi_mt/ckpt_best.pt [--events EV1 EV2 ...]

Without real waveforms, --validation N runs on the first N cached
validation events (synthetic end-to-end check).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from sbi_mt import get_device, load_config, resolve_path, seed_everything  # noqa: E402
from sbi_mt.inference import run_event, save_results  # noqa: E402
from sbi_mt.synth import iter_cached_events, load_fetchers  # noqa: E402
from sbi_mt.train import load_checkpoint  # noqa: E402

logger = logging.getLogger('sbi_mt.scripts.05')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config/default.yaml')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--events', nargs='*', default=None,
                    help='catalog event names (default: all)')
    ap.add_argument('--validation', type=int, default=None, metavar='N',
                    help='run on N cached validation events instead')
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(name)s: %(message)s')
    cfg = load_config(args.config)
    seed_everything(int(cfg['seed']))
    device = get_device(cfg)
    rng = np.random.default_rng(int(cfg['seed']) + 55)

    model, _ = load_checkpoint(args.checkpoint, device)
    reference, _ = load_fetchers(cfg)
    results_dir = resolve_path(cfg, cfg['paths']['results'])

    if args.validation:
        val_path = resolve_path(cfg, cfg['paths']['validation_cache'])
        for i, ex in enumerate(iter_cached_events(val_path)):
            if i >= args.validation:
                break
            name = f'val_{i:06d}'
            res = run_event(model, ex, cfg, reference, device, rng,
                            name=name)
            save_results(res, os.path.join(results_dir, name))
        return

    from sbi_mt.data import (
        load_stations, prepare_event_data, read_catalog,
        read_event_waveforms)
    catalog = read_catalog(resolve_path(cfg, cfg['paths']['catalog']))
    if args.events:
        catalog = [ev for ev in catalog if ev.name in set(args.events)]
        if not catalog:
            sys.exit(f'no catalog events match {args.events}')
    stations, inventory = load_stations(
        resolve_path(cfg, cfg['paths']['stations_xml']))

    for ev in catalog:
        try:
            stream = read_event_waveforms(cfg, ev)
            ex = prepare_event_data(cfg, ev, stream, inventory, reference,
                                    stations=stations)
        except Exception as exc:
            logger.warning('skipping %s: %s', ev.name, exc)
            continue
        res = run_event(model, ex, cfg, reference, device, rng,
                        name=ev.name)
        save_results(res, os.path.join(results_dir, ev.name))


if __name__ == '__main__':
    main()
