# SBI-MT — Amortized Bayesian Moment Tensor Inversion

Simulation-based inference for regional moment tensors (Sunda–Banda arc /
Indonesian network). A neural posterior estimator learns the nuisance
parameters — centroid perturbation (ΔN, ΔE, Δz) and centroid time shift
(Δt0) — from waveforms; the moment tensor itself is never learned: for each
nuisance sample it is solved in closed form by weighted linear least squares
against a Green's function library (Rao-Blackwellization). The MT posterior
is a mixture of analytic Gaussians, one per nuisance sample.

Key properties:

- **Amortized**: after training once on synthetics, inference per event is a
  single network pass plus ~2000 linear solves (target < 1 s on GPU).
- **Fit-aware**: an optional auxiliary loss backpropagates the waveform
  misfit of the WLS solution through the differentiable forward operator.
- **Robust to theory error**: training data are generated from
  velocity-perturbed GF stores while the solve always uses the reference
  store, teaching the nuisance posterior to absorb model error.
- **Config-driven**: bands, stores, priors, and constraint modes all live in
  `config/*.yaml`; a local high-frequency tier (Mw 1–3.8) is a config
  change, not a code change.

## Layout

```
config/default.yaml     all hyperparameters, paths, band, priors
sbi_mt/
  gf.py                 fomosto store creation, GF cubes, differentiable G assembly
  forward.py            torch taper/filter/shift ops, WLS solve, station alignment
  data.py               real-data pipeline, noise library, catalog/StationXML IO
  synth.py              on-the-fly synthetic training events, validation cache
  model.py              CNN encoder + FiLM, masked Set Transformer, zuko NSF
  train.py              NPE + aux loss training loop, validation metrics
  inference.py          per-event posterior, importance reweighting, HDF5 output
  evaluate.py           Kagan angle, SBC, coverage, ablations, Grond benchmark
  plots.py              all publication figures (PDF + PNG)
scripts/01..07          the pipeline, each runnable with --config
tests/                  pytest suite against a small analytical GF store
```

## Setup

Python ≥ 3.11 with: pyrocko, obspy, torch, zuko, h5py, numpy, scipy,
matplotlib, pyyaml (optional: cartopy for maps). `pip install -e .` or use
an existing conda environment providing these. Building real GF stores
additionally needs a fomosto backend (QSEIS/QSSP).

Place inputs under `data/raw/` (see `config/default.yaml` → `paths`):
`stations.xml`, `catalog.csv`, `waveforms/`, `noise/`, a regional 1D model
at `config/velocity_model.nd`, and optionally `grond_runs/` summaries.

## Workflow

```bash
python scripts/01_build_gf_store.py       --config config/default.yaml   # stores + cubes
python scripts/02_build_noise_library.py  --config config/default.yaml   # noise.h5
python scripts/03_generate_training_data.py --config config/default.yaml # val cache
python scripts/04_train.py --overfit 100  --config config/default.yaml   # sanity check
python scripts/04_train.py                --config config/default.yaml   # full training
python scripts/05_run_inference.py --checkpoint runs/sbi_mt/ckpt_best.pt \
                                          --config config/default.yaml
python scripts/06_benchmark_vs_grond.py --checkpoint runs/sbi_mt/ckpt_best.pt \
                                          --config config/default.yaml
python scripts/07_make_figures.py         --config config/default.yaml
```

## Tests

```bash
python -m pytest tests/ -q
```

The suite validates the torch forward model against `pyrocko.gf.Engine`
(< 1e-6 relative RMS), machine-precision MT recovery of the WLS layer,
the synthetic generator, flow shapes/permutation-invariance/masking, an
overfitting sanity check of the training loop, inference round trips,
metric correctness (Kagan angle vs pyrocko), and figure rendering.
