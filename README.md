# CCKD-SSVEP

**Cross-Channel Knowledge Distillation for Few-Electrode SSVEP Recognition**

Distilling knowledge from a 9-electrode teacher to 1–3 electrode students for
subject-independent SSVEP classification, framed as **privileged-information
distillation**: the teacher sees all 9 channels (training time only), the student sees a
subset of the **same trial**.

## Results

FB-tCNN, 1.0 s window, fixed subject-independent split, 3 seeds, paired per-subject Wilcoxon signed-rank.

| Dataset | Channels | Baseline | CCKD | Δ (pp) | p |
|---|---|---|---|---|---|
| Benchmark | 9 (teacher) | 64.60 | — | — | — |
| Benchmark | 3 | 33.79 | 34.76 | +0.97 | 0.160 |
| Benchmark | 1 | 28.01 | 29.10 | +1.08 | 0.049 |
| BETA | 9 (teacher) | 27.72 | — | — | — |
| BETA | 3 | 17.99 | 19.25 | **+1.26** | **0.015** |
| BETA | 1 | 12.35 | 13.29 | **+0.94** | **0.027** |

Three findings:
1. **Structural (relational) distillation gives ~+1 pp**, replicated on BETA (held out, no tuning).
2. **Logit KD hurts** (−0.75 pp at 3C, −2.04 pp at 1C) — the teacher's decisions rest on information the student cannot observe.
3. **One channel is input-limited** — its gain does not exceed the 3-channel case.

> The effect is **small (~1 pp)**. The value lies in the cross-dataset consistency and the
> negative logit-KD result, not in the magnitude. The contribution is **electrode
> reduction** (66.7 % / 88.9 %), not reduced compute.

## Layout

```
models/backbones.py   # FB-tCNN, EEGNet + filter bank
models/kd_losses.py   # logit KD, similarity-preserving KD
cckd_pipeline.py      # split, train, eval, ITR, statistics
run_cckd_full.py      # runner (batch-safe)
ssvep_cckd.pbs        # PBS job script
```

## Usage

```bash
pip install torch numpy scipy
python run_cckd_full.py --use_dummy    # test the pipeline
python run_cckd_full.py                # full run
```

Edit the config block at the top of `run_cckd_full.py`: `DATASET`, `WINDOW_S`,
`BACKBONE`, `MODES`, `SEEDS`. Data goes in `data/raw/benchmark/` and `data/raw/beta/`.

**Sampling-rate caveat:** the two loaders use different conventions — Benchmark
`t_end = 0.14 + W/4`, BETA `t_end = 0.14 + W`. Both are physically 250 Hz. The runner
handles this; always check the `n_samples=…` line printed at the start of each run.

## Method

```
L = L_CE + λ_kd · L_logit(τ) + λ_sp · L_structural
```
Defaults: `λ_sp=100`, `λ_kd=0` (proposed configuration). `L_structural` is
similarity-preserving KD (Tung & Mori, ICCV 2019) applied at two feature layers — it
matches the **pairwise similarity structure across the batch** rather than forcing
features to be equal. The first convolution collapses the channel dimension, so features
are independent of channel count and no projection head is needed.

## References

- Wang et al., IEEE TNSRE 2017 (Benchmark) · Liu et al., Front. Neurosci. 2020 (BETA)
- Ding et al., IEEE TNSRE 2021 ([FB-tCNN](https://github.com/DingWenl/FB-tCNN)) · Lawhern et al., J. Neural Eng. 2018 (EEGNet)
- Tung & Mori, ICCV 2019 (Similarity-Preserving KD)
