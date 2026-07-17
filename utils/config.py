from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(p)
    return cfg


def deep_get(cfg: Dict[str, Any], key: str, default=None):
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def deep_set(cfg: Dict[str, Any], key: str, value: Any) -> None:
    cur = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out = deepcopy(cfg)
    mapping = {
        "data_root": "dataset.data_root",
        "output_dir": "output.output_dir",
        "cache_dir": "output.cache_dir",
        "n_subjects": "dataset.n_subjects",
        "n_folds": "evaluation.n_folds",
        "epochs": "training.epochs",
        "batch_size": "training.batch_size",
        "device": "runtime.device",
        "workers": "runtime.workers",
        "seed": "runtime.seed",
    }
    for arg_name, cfg_key in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            deep_set(out, cfg_key, value)
    return out


def resolve_project_path(value: str | Path, project_root: str | Path) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (Path(project_root) / p).resolve()
