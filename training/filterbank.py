"""Filter-bank preprocessing utilities for FB-tCNN."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
from scipy.signal import cheby1, sosfiltfilt

Band = Tuple[float, float]


def validate_bands(bands: Sequence[Band], sfreq: float) -> None:
    nyq = sfreq / 2.0
    if not bands:
        raise ValueError("At least one sub-band is required")
    for low, high in bands:
        if not (0 < low < high < nyq):
            raise ValueError(f"Invalid band ({low}, {high}) for sfreq={sfreq}")


def apply_filterbank(
    X: np.ndarray,
    sfreq: float,
    bands: Sequence[Band],
    order: int = 4,
    ripple_db: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    """Convert raw trials [N,C,T] to filter-bank trials [N,B,C,T]."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"X must have shape [N,C,T], got {X.shape}")
    validate_bands(bands, sfreq)

    outputs = []
    for low, high in bands:
        sos = cheby1(
            order, ripple_db, [low, high], btype="bandpass",
            fs=sfreq, output="sos"
        )
        filtered = sosfiltfilt(sos, X, axis=-1).astype(np.float32)
        if normalize:
            mean = filtered.mean(axis=-1, keepdims=True)
            std = filtered.std(axis=-1, keepdims=True)
            filtered = (filtered - mean) / (std + 1e-6)
        outputs.append(filtered)
    return np.stack(outputs, axis=1).astype(np.float32)


def cache_key(dataset: str, shape: Iterable[int], sfreq: float,
              bands: Sequence[Band], channel_indices: Sequence[int]) -> str:
    payload = {
        "dataset": dataset,
        "shape": list(shape),
        "sfreq": sfreq,
        "bands": [list(b) for b in bands],
        "channels": list(channel_indices),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def load_or_create_cache(
    X: np.ndarray,
    cache_dir: str | Path,
    dataset: str,
    sfreq: float,
    bands: Sequence[Band],
    channel_indices: Sequence[int],
) -> np.ndarray:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(dataset, X.shape, sfreq, bands, channel_indices)
    path = cache_dir / f"fbtcnn_{dataset}_{key}.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    X_fb = apply_filterbank(X, sfreq=sfreq, bands=bands)
    np.save(path, X_fb)
    return np.load(path, mmap_mode="r")
