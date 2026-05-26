#!/usr/bin/env python3
"""Path helpers for cross-sensor STM/UCI/MotionSense experiments."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def _resolve_existing_dir(*candidates: object) -> Path:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_dir():
            return path
    raise FileNotFoundError(f"No valid dataset directory found in candidates: {candidates}")


def get_stm_uc_root() -> Path:
    return _resolve_existing_dir(
        os.environ.get("NNEDGE_PRIVACY_STM_ROOT"),
        REPO_ROOT / "STM-UC",
        REPO_ROOT / "Data",
    )


def get_stm_six_root() -> Path:
    return _resolve_existing_dir(
        os.environ.get("NNEDGE_PRIVACY_STM_SIX_ROOT"),
        os.environ.get("NNEDGE_PRIVACY_STM_ROOT"),
        REPO_ROOT / "STM_Dataset_Six_Activities",
        REPO_ROOT / "Data",
    )


def get_uci_root() -> Path:
    return _resolve_existing_dir(
        os.environ.get("NNEDGE_PRIVACY_UCI_ROOT"),
        REPO_ROOT / "UCI_HAR",
    )


def get_motionsense_root() -> Path:
    return _resolve_existing_dir(
        os.environ.get("NNEDGE_PRIVACY_MOTIONSENSE_ROOT"),
        REPO_ROOT / "MotionSense",
    )


def get_results_root(*parts: str, lower: bool = False) -> Path:
    env_root = os.environ.get("NNEDGE_PRIVACY_RESULTS_ROOT")
    if env_root:
        base = Path(env_root).expanduser()
    else:
        base = REPO_ROOT / ("results" if lower else "Results")
    return base.joinpath(*parts)
