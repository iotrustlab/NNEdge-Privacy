#!/usr/bin/env python3
"""Shared helpers for cross-dataset transfer experiments."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent


def _resolve_existing_dir(*candidates: object) -> Path:
    """Return the first existing directory from a list of candidates."""
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_dir():
            return path
    raise FileNotFoundError(f"No valid dataset directory found in candidates: {candidates}")


def get_stm_root() -> Path:
    """Resolve the STM dataset root for this machine."""
    return _resolve_existing_dir(
        os.environ.get("NNEDGE_PRIVACY_STM_ROOT"),
        os.environ.get("NNEDGE_PRIVACY_DATA_ROOT"),
        REPO_ROOT / "STM_Dataset_Six_Activities",
        REPO_ROOT / "Data",
    )


def get_motionsense_root() -> Path:
    """Resolve the MotionSense dataset root for this machine."""
    return _resolve_existing_dir(
        os.environ.get("NNEDGE_PRIVACY_MOTIONSENSE_ROOT"),
        REPO_ROOT / "MotionSense",
    )


def calculate_vulnerability(y_pred_proba) -> float:
    """Average max posterior probability over samples."""
    posterior = np.asarray(y_pred_proba, dtype=float)
    if posterior.ndim != 2 or posterior.size == 0:
        return 0.0
    return float(np.mean(np.max(posterior, axis=1)))
