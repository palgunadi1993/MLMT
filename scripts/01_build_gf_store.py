#!/usr/bin/env python3
"""Build the reference fomosto store, N velocity-perturbed stores for
training robustness, and export each as a dense GF cube (Phase 1 + PLAN §5.3).

    python scripts/01_build_gf_store.py --config config/default.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyrocko import gf as pgf  # noqa: E402

from sbi_mt import load_config, seed_everything  # noqa: E402
from sbi_mt.gf import (  # noqa: E402
    create_store, cube_path, extract_cube, load_velocity_model_str,
    perturb_velocity_model, store_dir)

logger = logging.getLogger('sbi_mt.scripts.01')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config/default.yaml')
    ap.add_argument('--force', action='store_true',
                    help='rebuild stores and cubes even if present')
    # nworkers > 1 can hang fomosto builds in some environments
    ap.add_argument('--nworkers', type=int, default=1)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(name)s: %(message)s')
    cfg = load_config(args.config)
    seed_everything(int(cfg['seed']))
    gcfg = cfg['gf']
    ref_id = gcfg['stores'][gcfg['store']]['id']
    nd_ref = load_velocity_model_str(cfg)

    rng = np.random.default_rng(int(cfg['seed']))
    pct = tuple(float(x) for x in gcfg['velocity_perturbation_percent'])
    builds: list[tuple[str, str]] = [(ref_id, nd_ref)]
    for k in range(int(gcfg['n_perturbed_stores'])):
        builds.append(
            (f'{ref_id}_pert{k}', perturb_velocity_model(nd_ref, rng, pct)))

    for sid, nd in builds:
        logger.info('=== store %s ===', sid)
        create_store(cfg, sid, nd, force=args.force, nworkers=args.nworkers)
        cpath = cube_path(cfg, sid)
        if os.path.exists(cpath) and not args.force:
            logger.info('cube %s exists, skipping', cpath)
            continue
        store = pgf.store.Store(store_dir(cfg, sid))
        extract_cube(store, cfg, cpath)
        store.close()
    logger.info('done: %d stores + cubes under %s', len(builds),
                os.path.dirname(cube_path(cfg, ref_id)))


if __name__ == '__main__':
    main()
