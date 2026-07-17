#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT=${1:?Usage: bash scripts/run_beta_smoke.sh /absolute/path/to/BETA}
python scripts/inspect_dataset.py --config configs/beta.yaml --data_root "$DATA_ROOT" --subject 1
python training/train_teacher.py --config configs/beta.yaml --data_root "$DATA_ROOT" --n_folds 1 --epochs 5 --output_dir results/module1/beta_smoke
