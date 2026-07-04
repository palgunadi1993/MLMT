#!/usr/bin/env python3
"""Generate all figures (PLAN §10) from whatever outputs exist:

- results/<event>/posterior.h5      -> waveform fits, beachballs, lune,
                                       corner, station map (figures 1-5)
- results/synthetic_eval.h5         -> SBC, coverage, accuracy (6-8)
- results/synthetic_eval_<abl>.h5   -> ablation summary (10)
- results/benchmark.json            -> benchmark figure (9)
- runs/<run-name>/log.jsonl         -> training diagnostics (11)

    python scripts/07_make_figures.py --config config/default.yaml \
        [--run-name sbi_mt] [--events EV1 ...]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from sbi_mt import load_config, resolve_path, seed_everything  # noqa: E402
from sbi_mt.evaluate import ABLATIONS, load_evaluation  # noqa: E402
from sbi_mt.inference import load_results  # noqa: E402
from sbi_mt.synth import iter_cached_events, load_fetchers  # noqa: E402
from sbi_mt import plots  # noqa: E402

logger = logging.getLogger('sbi_mt.scripts.07')


def event_figures(cfg, args) -> None:
    results_dir = resolve_path(cfg, cfg['paths']['results'])
    if not os.path.isdir(results_dir):
        logger.info('no results directory — skipping per-event figures')
        return
    names = sorted(
        d for d in os.listdir(results_dir)
        if os.path.exists(os.path.join(results_dir, d, 'posterior.h5')))
    if args.events:
        names = [n for n in names if n in set(args.events)]
    if not names:
        logger.info('no per-event results found')
        return

    reference, _ = load_fetchers(cfg)
    val_events = None
    catalog = stations = inventory = None

    for ev_name in names:
        result = load_results(os.path.join(results_dir, ev_name))
        ex = m6_ref = theta_true = None
        if ev_name.startswith('val_'):
            if val_events is None:
                val_path = resolve_path(cfg,
                                        cfg['paths']['validation_cache'])
                val_events = list(iter_cached_events(val_path))
            ex = val_events[int(ev_name.split('_')[1])]
            m6_ref = np.asarray(ex['m6'])
            theta_true = np.asarray(ex['theta'])
        else:
            from sbi_mt.data import (
                load_stations, prepare_event_data, read_catalog,
                read_event_waveforms)
            if catalog is None:
                catalog = {e.name: e for e in read_catalog(
                    resolve_path(cfg, cfg['paths']['catalog']))}
                stations, inventory = load_stations(
                    resolve_path(cfg, cfg['paths']['stations_xml']))
            ev = catalog.get(ev_name)
            if ev is None:
                logger.warning('%s not in catalog — skipping', ev_name)
                continue
            try:
                stream = read_event_waveforms(cfg, ev)
                ex = prepare_event_data(cfg, ev, stream, inventory,
                                        reference, stations=stations)
            except Exception as exc:
                logger.warning('cannot rebuild data for %s: %s',
                               ev_name, exc)
                continue
            if ev.m6_ref is not None:
                m6_ref = np.asarray(ev.m6_ref)

        plots.plot_waveform_fits(ex, result, cfg, reference,
                                 name=f'{ev_name}_waveform_fits')
        plots.plot_fuzzy_beachball(result, cfg, m6_ref=m6_ref,
                                   name=f'{ev_name}_beachball')
        plots.plot_lune_hudson(result, cfg, m6_ref=m6_ref,
                               name=f'{ev_name}_lune_hudson')
        plots.plot_nuisance_corner(result, cfg, theta_true=theta_true,
                                   name=f'{ev_name}_nuisance_corner')
        plots.plot_station_map(ex, result, cfg,
                               name=f'{ev_name}_station_map')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config/default.yaml')
    ap.add_argument('--run-name', default='sbi_mt')
    ap.add_argument('--events', nargs='*', default=None)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(name)s: %(message)s')
    cfg = load_config(args.config)
    seed_everything(int(cfg['seed']))
    results_dir = resolve_path(cfg, cfg['paths']['results'])

    event_figures(cfg, args)

    eval_path = os.path.join(results_dir, 'synthetic_eval.h5')
    baseline = None
    if os.path.exists(eval_path):
        baseline = load_evaluation(eval_path)
        plots.plot_sbc(baseline, cfg, name='sbc_ranks')
        plots.plot_coverage(baseline, cfg, which='hits_m6',
                            name='coverage_m6')
        plots.plot_coverage(baseline, cfg, which='hits_theta',
                            name='coverage_theta')
        plots.plot_accuracy_curves(baseline, cfg, name='accuracy_curves')
    else:
        logger.info('no synthetic_eval.h5 — skipping figures 6-8 '
                    '(run scripts/06 with --checkpoint)')

    if baseline is not None:
        ablations = {}
        for abl in ABLATIONS:
            p = os.path.join(results_dir, f'synthetic_eval_{abl}.h5')
            if os.path.exists(p):
                ablations[abl] = load_evaluation(p)
        if ablations:
            plots.plot_ablation_summary(baseline, ablations, cfg,
                                        name='ablation_summary')
        else:
            logger.info('no ablation evaluations — skipping figure 10')

    bench_path = os.path.join(results_dir, 'benchmark.json')
    if os.path.exists(bench_path):
        with open(bench_path) as f:
            rows = json.load(f)
        plots.plot_benchmark(rows, cfg, name='benchmark')
    else:
        logger.info('no benchmark.json — skipping figure 9')

    run_dir = os.path.join(resolve_path(cfg, cfg['paths']['runs']),
                           args.run_name)
    if os.path.exists(os.path.join(run_dir, 'log.jsonl')):
        plots.plot_training_diagnostics(run_dir, cfg,
                                        name='training_diagnostics')
    else:
        logger.info('no training log in %s — skipping figure 11', run_dir)


if __name__ == '__main__':
    main()
