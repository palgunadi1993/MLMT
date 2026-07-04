"""Training loop (PLAN §7, Phase 5).

Loss: NPE = -log q(theta_true | data), plus an optional auxiliary term
(config train.aux_misfit_loss): the expected waveform misfit of the WLS
moment tensor solution at reparametrized flow samples, differentiable
through forward.py — this makes the posterior fit-aware. The aux term runs
on a subset of events/stations per batch because it needs full-extent GF
grids; the WLS always uses the REFERENCE store.

Validation: NPE loss on the cached split, plus Kagan angle of the
posterior-mean MT and 68/90% coverage of the MT components on a subset
(config train.val_metrics_*). Checkpoint best-val, early stopping.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import yaml

from . import get_device, seed_everything
from .evaluate import posterior_metrics, wls_over_thetas
from .forward import ForwardModel
from .gf import EventInfo, StationGeometry, build_gf_grid
from .model import PosteriorModel
from .synth import SyntheticDataset, SyntheticGenerator, pad_collate

logger = logging.getLogger('sbi_mt.train')

Fetcher = Any  # GFCube | StoreFetcher


# ----------------------------------------------------------------------------
# per-event forward-model reconstruction (aux loss + val metrics)
# ----------------------------------------------------------------------------

def event_forward_model(
        ex: dict[str, Any], cfg: dict[str, Any], fetcher: Fetcher,
        station_idx: np.ndarray, device: torch.device | str
) -> tuple[ForwardModel, torch.Tensor, torch.Tensor]:
    """Rebuild a full-extent-grid ForwardModel for a subset of an example's
    stations, plus its UN-normalized preprocessed data and per-trace weights
    (trace order station-major, matching the grid)."""
    n_comp = len(cfg['processing']['components'])
    station_idx = np.asarray(station_idx, dtype=np.int64)
    geoms = [StationGeometry(
        code=str(ex['station_codes'][s]),
        distance=float(ex['sta_distance'][s]),
        azimuth=float(ex['sta_azimuth'][s]),
        back_azimuth=0.0)                # unused by the grid/G assembly
        for s in station_idx]
    anchor = EventInfo(name='ev', lat=float(ex['lat']),
                       lon=float(ex['lon']), depth=float(ex['depth']))
    itmin = np.asarray(ex['itmin'], dtype=np.int64)[station_idx]
    n_t = int(ex['waveforms'].shape[1])
    grid = build_gf_grid(fetcher, anchor, geoms, cfg,
                         itmin=itmin, n_t=n_t).to(device)
    fm = ForwardModel(grid, cfg)
    tr = (station_idx[:, None] * n_comp + np.arange(n_comp)).ravel()
    wf = np.asarray(ex['waveforms'])[tr] * np.asarray(ex['norms'])[tr, None]
    data = torch.as_tensor(wf, dtype=torch.float32, device=device)
    weights = torch.as_tensor(np.asarray(ex['weights'])[tr],
                              dtype=torch.float32, device=device)
    return fm, data, weights


def aux_misfit_loss(model: PosteriorModel, emb: torch.Tensor,
                    batch: dict[str, Any], cfg: dict[str, Any],
                    fetcher: Fetcher, rng: np.random.Generator,
                    device: torch.device | str) -> torch.Tensor:
    """Expected weighted waveform misfit of the WLS solution at flow
    rsamples, averaged over a random subset of the batch's events and
    stations (full grids are too big for every event every step)."""
    tcfg = cfg['train']
    ex_list: list[dict[str, Any]] = batch['_examples']
    n_ev = min(int(tcfg['aux_misfit_events_per_batch']), len(ex_list))
    n_samp = int(tcfg['aux_misfit_samples'])
    n_comp = len(cfg['processing']['components'])
    idx_ev = rng.choice(len(ex_list), size=n_ev, replace=False)
    losses = []
    for i in idx_ev:
        ex = ex_list[int(i)]
        sta_mask = np.asarray(ex['mask']).reshape(-1, n_comp)
        valid = np.flatnonzero(sta_mask.any(axis=1))
        if valid.size < 2:
            continue
        k = min(int(tcfg['aux_misfit_stations']), valid.size)
        sel = np.sort(rng.choice(valid, size=k, replace=False))
        fm, data, weights = event_forward_model(ex, cfg, fetcher, sel,
                                                device)
        theta, _ = model.rsample_emb(emb[int(i)].unsqueeze(0), n_samp)
        res, _ = fm.solve(theta.reshape(n_samp, -1), data, weights)
        # misfit normalized by weighted data power (= 1 - VR): bounded and
        # scale-free, so the 0.1 weight is meaningful against the NPE term
        losses.append((1.0 - res.vr).mean())
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


# ----------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------

def validate(model: PosteriorModel, val_events: Sequence[dict[str, Any]],
             cfg: dict[str, Any], fetcher: Fetcher,
             device: torch.device | str,
             rng: np.random.Generator) -> dict[str, float]:
    """Val NPE loss on all events; Kagan + coverage on the first
    train.val_metrics_events events (full-grid WLS at flow samples)."""
    tcfg = cfg['train']
    bs = int(tcfg['batch_size'])
    model.eval()
    nll = []
    with torch.no_grad():
        for i in range(0, len(val_events), bs):
            b = pad_collate(list(val_events[i:i + bs]))
            lp = model.log_prob(
                b['waveforms'].to(device), b['metadata'].to(device),
                b['mask'].to(device), b['theta'].to(device))
            nll.append(-lp.cpu())
    out: dict[str, float] = {'val_loss': float(torch.cat(nll).mean())}

    n_me = min(int(tcfg['val_metrics_events']), len(val_events))
    n_ts = int(tcfg['val_theta_samples'])
    chunk = int(tcfg['val_wls_chunk'])
    levels = (0.68, 0.90)
    kagans, hits = [], []
    with torch.no_grad():
        for ex in val_events[:n_me]:
            b = pad_collate([ex])
            emb = model.embed(
                b['waveforms'].to(device), b['metadata'].to(device),
                b['mask'].to(device))
            thetas = model.sample_emb(emb, n_ts)[0]
            fm, data, weights = event_forward_model(
                ex, cfg, fetcher,
                np.arange(len(ex['station_codes'])), device)
            m_hats, covs, _ = wls_over_thetas(fm, thetas, data, weights,
                                              chunk)
            met = posterior_metrics(np.asarray(ex['m6'], dtype=np.float64),
                                    m_hats, covs, rng, levels)
            kagans.append(met['kagan_deg'])
            hits.append(met['coverage_hits'])
    model.train()
    if kagans:
        cov = np.stack(hits).mean(axis=(0, 2))       # per level
        out['val_kagan_deg'] = float(np.median(kagans))
        out['val_coverage_68'] = float(cov[0])
        out['val_coverage_90'] = float(cov[1])
    return out


# ----------------------------------------------------------------------------
# data plumbing
# ----------------------------------------------------------------------------

def make_train_loader(cfg: dict[str, Any], generator: SyntheticGenerator
                      ) -> torch.utils.data.DataLoader:
    tcfg = cfg['train']
    nw = int(tcfg['num_workers'])
    return torch.utils.data.DataLoader(
        SyntheticDataset(generator, seed=int(cfg['seed'])),
        batch_size=int(tcfg['batch_size']), collate_fn=pad_collate,
        num_workers=nw, persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None)


def fixed_event_batches(events: Sequence[dict[str, Any]], batch_size: int,
                        seed: int) -> Iterator[dict[str, Any]]:
    """Infinite shuffled minibatches over a FIXED event list (the
    overfit-on-100-events sanity check of PLAN §11)."""
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(len(events))
        for i in range(0, len(events) - batch_size + 1, batch_size):
            yield pad_collate([events[j] for j in order[i:i + batch_size]])


# ----------------------------------------------------------------------------
# checkpoints
# ----------------------------------------------------------------------------

def _clean_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if not k.startswith('_')}


def save_checkpoint(path: str, model: PosteriorModel,
                    optimizer: torch.optim.Optimizer, step: int,
                    metrics: dict[str, float], cfg: dict[str, Any]) -> None:
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'step': step,
        'metrics': metrics,
        'config': _clean_cfg(cfg),
    }, path)


def load_checkpoint(path: str, device: torch.device | str = 'cpu'
                    ) -> tuple[PosteriorModel, dict[str, Any]]:
    """Rebuild the model from the config stored inside the checkpoint (the
    architecture is fully config-defined). Returns (model.eval(), ckpt)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = PosteriorModel(ckpt['config']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt


# ----------------------------------------------------------------------------
# main loop
# ----------------------------------------------------------------------------

def fit(cfg: dict[str, Any],
        train_batches: Iterator[dict[str, Any]] | Any,
        val_events: Sequence[dict[str, Any]],
        fetcher: Fetcher,
        run_dir: str,
        max_steps: int | None = None,
        device: torch.device | None = None) -> dict[str, Any]:
    """Train an NPE model. train_batches: iterable of pad_collate batches
    (DataLoader or fixed_event_batches). fetcher: REFERENCE store cube for
    the aux loss and val metrics. Returns a summary dict."""
    device = device if device is not None else get_device(cfg)
    tcfg = cfg['train']
    seed_everything(int(cfg['seed']))

    model = PosteriorModel(cfg).to(device)
    total_steps = int(max_steps if max_steps is not None
                      else max(1, int(tcfg['n_train_events'])
                               // int(tcfg['batch_size'])))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(tcfg['lr']),
        weight_decay=float(tcfg['weight_decay']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps)
    use_amp = bool(tcfg['amp']) and device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    aux_on = bool(tcfg['aux_misfit_loss'])
    aux_w = float(tcfg['aux_misfit_weight'])
    rng = np.random.default_rng(int(cfg['seed']) + 987)

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(_clean_cfg(cfg), f, sort_keys=False)
    log_f = open(os.path.join(run_dir, 'log.jsonl'), 'a')

    def log_record(rec: dict[str, Any]) -> None:
        log_f.write(json.dumps(rec) + '\n')
        log_f.flush()

    best_val = math.inf
    bad_evals = 0
    last_metrics: dict[str, float] = {}
    t0 = time.time()
    loss_acc: list[float] = []
    it = iter(train_batches)

    for step in range(1, total_steps + 1):
        batch = next(it)
        wf = batch['waveforms'].to(device)
        md = batch['metadata'].to(device)
        mask = batch['mask'].to(device)
        theta = batch['theta'].to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            emb = model.embed(wf, md, mask)
        emb = emb.float()
        npe = -model.log_prob_emb(emb, theta).mean()
        loss = npe
        if model.direct_mt is not None:
            # ablation (c): direct MT regression instead of the WLS layer —
            # unit-direction MSE + log10 M0 MSE (PLAN §9.4)
            m6 = batch['m6'].to(device)
            m0 = torch.sqrt(((m6[:, :3] ** 2).sum(-1)
                             + 2.0 * (m6[:, 3:] ** 2).sum(-1)) / 2.0)
            direction, log_m0 = model.direct_mt(emb)
            loss = loss + (
                (direction
                 - torch.nn.functional.normalize(m6, dim=-1)).pow(2).mean()
                + (log_m0 - torch.log10(m0)).pow(2).mean())
        aux_val = float('nan')
        if aux_on and aux_w > 0:
            aux = aux_misfit_loss(model, emb, batch, cfg, fetcher, rng,
                                  device)
            loss = npe + aux_w * aux
            aux_val = float(aux.detach())

        if not torch.isfinite(loss):
            logger.warning('non-finite loss at step %d (npe %.3g, aux %.3g)'
                           ' — skipping step', step, float(npe), aux_val)
            scheduler.step()
            continue
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(tcfg['grad_clip']))
        if torch.isfinite(grad_norm):
            scaler.step(optimizer)
        else:
            logger.warning('non-finite grad norm at step %d — skipping '
                           'optimizer step', step)
        scaler.update()
        scheduler.step()
        loss_acc.append(float(npe.detach()))

        if step % int(tcfg['log_every_steps']) == 0 or step == 1:
            rec = {'step': step, 'npe_loss': float(np.mean(loss_acc)),
                   'aux_loss': aux_val,
                   'lr': float(scheduler.get_last_lr()[0]),
                   'elapsed_s': time.time() - t0}
            logger.info(
                'step %6d/%d  npe %.4f  aux %s  lr %.2e', step,
                total_steps, rec['npe_loss'],
                f'{aux_val:.4f}' if math.isfinite(aux_val) else '-',
                rec['lr'])
            log_record(rec)
            loss_acc = []

        if (step % int(tcfg['eval_every_steps']) == 0
                or step == total_steps) and len(val_events) > 0:
            metrics = validate(model, val_events, cfg, fetcher, device,
                               rng)
            metrics['step'] = step
            last_metrics = metrics
            logger.info('eval @ %d: %s', step, {
                k: round(v, 4) for k, v in metrics.items()})
            log_record(metrics)
            save_checkpoint(os.path.join(run_dir, 'ckpt_last.pt'),
                            model, optimizer, step, metrics, cfg)
            if metrics['val_loss'] < best_val:
                best_val = metrics['val_loss']
                bad_evals = 0
                save_checkpoint(os.path.join(run_dir, 'ckpt_best.pt'),
                                model, optimizer, step, metrics, cfg)
            else:
                bad_evals += 1
                if bad_evals >= int(tcfg['early_stop_patience']):
                    logger.info('early stop at step %d (patience %d)',
                                step, bad_evals)
                    break

    if not os.path.exists(os.path.join(run_dir, 'ckpt_best.pt')):
        save_checkpoint(os.path.join(run_dir, 'ckpt_best.pt'), model,
                        optimizer, total_steps, last_metrics, cfg)
    log_f.close()
    return {'run_dir': run_dir, 'best_val_loss': best_val,
            'last_metrics': last_metrics,
            'elapsed_s': time.time() - t0}
