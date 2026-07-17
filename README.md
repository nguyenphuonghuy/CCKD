[README.md](https://github.com/user-attachments/files/30110316/README.md)
# CCKD-SSVEP v0.1.0 — Module 1

Independent project for the 9-channel FB-tCNN teacher baseline.
The raw dataset stays anywhere on disk. **Do not copy it into this project.**

## 1. Install
```bash
python -m venv .venv
source .venv/bin/activate       # Linux
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## 2. CPU smoke test
```bash
python tests/smoke_test.py
```
Expected: `SMOKE TEST PASSED`.

## 3. Point to the external dataset
Either edit only `dataset.data_root` in `configs/benchmark.yaml`, or leave it unchanged and pass:
```bash
--data_root /absolute/path/to/Benchmark
```
CLI override has priority.

## 4. Inspect one subject before training
```bash
python scripts/inspect_dataset.py \
  --config configs/benchmark.yaml \
  --data_root /absolute/path/to/Benchmark \
  --subject 1
```
Expected Benchmark shape for one subject at 1 s: `(240, 64, 250)` and selected teacher shape `(240, 9, 250)`.
Expected BETA shape: `(160, 64, 250)` and selected teacher shape `(160, 9, 250)`.

## 5. One-fold, five-epoch test
Linux:
```bash
bash scripts/run_benchmark_smoke.sh /absolute/path/to/Benchmark
```
Cross-platform direct command:
```bash
python training/train_teacher.py \
  --config configs/benchmark.yaml \
  --data_root /absolute/path/to/Benchmark \
  --n_folds 1 --epochs 5 \
  --output_dir results/module1/benchmark_smoke
```

## 6. Files to send back after the smoke run
```text
results/module1/benchmark_smoke/training.log
results/module1/benchmark_smoke/teacher_results.json
```
Also send the terminal traceback if an error occurs.

## 7. Full run (only after smoke validation)
```bash
python training/train_teacher.py \
  --config configs/benchmark.yaml \
  --data_root /absolute/path/to/Benchmark
```

## Scientific safeguards
- LOSO test subject is never used for early stopping.
- Validation subjects are drawn only from the remaining training subjects.
- Test subject is evaluated once after restoring the best validation checkpoint.
- Window samples are computed as `round(time_seconds * sfreq)`.
