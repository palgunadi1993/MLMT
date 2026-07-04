#!/usr/bin/env python3
"""Validation & benchmark (PLAN §9).

Synthetic suite (needs a checkpoint + the cached validation split):
SBC ranks, coverage, Kagan binned by SNR / station count / velocity
perturbation -> results/synthetic_eval[_<tag>].h5.

Real-event benchmark (needs script 05 outputs in results/): per event,
Kagan/Mw/depth vs the catalog reference MT, plus Grond comparison where
<paths.grond_runs>/<event>.yaml exists (keys: m6 [NED, Nm], mw, depth_km,
wall_clock_s, optional ci68) -> results/benchmark.json.

    python scripts/06_benchmark_vs_grond.py --config config/default.yaml \
        --checkpoint runs/sbi_mt/ckpt_best.pt [--n-events 1000] [--tag ablation_a]
    python scripts/06_benchmark_vs_grond.py --config config/default.yaml --real-only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from sbi_mt import get_device, load_config, resolve_path, seed_everything  # noqa: E402
from sbi_mt.evaluate import (  # noqa: E402
    benchmark_event, binned_kagan, empirical_coverage, evaluate_synthetic,
    rank_uniformity_pvalue, save_evaluation)
from sbi_mt.synth import iter_cached_events, load_fetchers  # noqa: E402

logger = logging.getLogger('sbi_mt.scripts.06')


def synthetic_suite(cfg, args) -> None:
    from sbi_mt.train import load_checkpoint
    device = get_device(cfg)
    rng = np.random.default_rng(int(cfg['seed']) + 66)
    model, _ = load_checkpoint(args.checkpoint, device)
    reference, _ = load_fetchers(cfg)

    n = int(args.n_events or cfg['evaluate']['n_sbc'])
    val_path = resolve_path(cfg, cfg['paths']['validation_cache'])
    events = []
    for i, ex in enumerate(iter_cached_events(val_path)):
        if i >= n:
            break
        events.append(ex)
    logger.info('evaluating %d held-out synthetic events', len(events))
    arrays = evaluate_synthetic(model, events, cfg, reference, device, rng)

    tag = f'_{args.tag}' if args.tag else ''
    out = os.path.join(resolve_path(cfg, cfg['paths']['results']),
                       f'synthetic_eval{tag}.h5')
    save_evaluation(arrays, out)
    print('synthetic evaluation written to', out)

    n_ts = int(arrays['n_theta_samples'])
    for name, key in (('theta', 'ranks_theta'), ('m6', 'ranks_m6')):
        pvals = [rank_uniformity_pvalue(arrays[key][:, d], n_ts)
                 for d in range(arrays[key].shape[1])]
        print(f'SBC {name} uniformity p-values:',
              ['%.3f' % p for p in pvals])
    print('median Kagan [deg]: %.2f' % float(np.median(arrays['kagan_deg'])))
    cov = empirical_coverage(arrays, 'hits_m6')
    for il, lev in enumerate(arrays['coverage_levels']):
        print('coverage m6 @ %.0f%%: %.3f'
              % (100 * lev, float(cov[il].mean())))
    for key, edges in (('snr_median', cfg['evaluate']['snr_bins']),
                       ('n_stations', cfg['evaluate']['station_bins'])):
        b = binned_kagan(arrays, key, edges)
        print(f'median Kagan by {key}: centers {b["centers"].tolist()} '
              f'ref {np.round(b["median_kagan"][0], 2).tolist()} '
              f'pert {np.round(b["median_kagan"][1], 2).tolist()}')


def real_benchmark(cfg) -> None:
    from sbi_mt.data import read_catalog
    from sbi_mt.inference import load_results
    results_dir = resolve_path(cfg, cfg['paths']['results'])
    grond_dir = resolve_path(cfg, cfg['paths']['grond_runs'])
    catalog = read_catalog(resolve_path(cfg, cfg['paths']['catalog']))
    rows = []
    for ev in catalog:
        outdir = os.path.join(results_dir, ev.name)
        if not os.path.exists(os.path.join(outdir, 'posterior.h5')):
            logger.info('no inference results for %s — skipping', ev.name)
            continue
        result = load_results(outdir)
        grond = None
        gpath = os.path.join(grond_dir, f'{ev.name}.yaml')
        if os.path.exists(gpath):
            with open(gpath) as f:
                grond = yaml.safe_load(f)
        rows.append(benchmark_event(ev, result, grond))
    if not rows:
        print('no benchmark rows — run scripts/05_run_inference.py first')
        return
    out = os.path.join(results_dir, 'benchmark.json')
    with open(out, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'benchmark for {len(rows)} events written to {out}')
    kag = [r['kagan_ref_deg'] for r in rows if 'kagan_ref_deg' in r]
    if kag:
        print('median Kagan vs reference: %.2f deg' % float(np.median(kag)))
    t = [r['time_sbi_s'] for r in rows]
    print('median SBI inference time: %.2f s/event' % float(np.median(t)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config/default.yaml')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--n-events', type=int, default=None,
                    help='override evaluate.n_sbc')
    ap.add_argument('--tag', default=None,
                    help='suffix for synthetic_eval output (ablations)')
    ap.add_argument('--real-only', action='store_true')
    ap.add_argument('--synthetic-only', action='store_true')
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(name)s: %(message)s')
    cfg = load_config(args.config)
    seed_everything(int(cfg['seed']))

    if not args.real_only:
        if not args.checkpoint:
            sys.exit('--checkpoint required for the synthetic suite '
                     '(or pass --real-only)')
        synthetic_suite(cfg, args)
    if not args.synthetic_only:
        real_benchmark(cfg)


if __name__ == '__main__':
    main()
