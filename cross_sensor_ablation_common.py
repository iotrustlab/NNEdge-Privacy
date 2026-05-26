#!/usr/bin/env python3
"""Shared helpers for the cross-sensor ablation scripts."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "Results" / "Identity-placement-ablation" / "50Hz"
DEFAULT_SPLIT_SPECS = {
    "8_3": {"train_size": 8, "test_size": 3},
    "6_5": {"train_size": 6, "test_size": 5},
    "4_7": {"train_size": 4, "test_size": 7},
    "1_10": {"train_size": 1, "test_size": 10},
}
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_COMBINATIONS = 10
SOURCE_FREQUENCY_HZ = 50


@dataclass(frozen=True)
class CrossSensorRunConfig:
    split_name: str
    max_combinations: int
    random_seed: int
    results_root: Path
    frequency_hz: int = SOURCE_FREQUENCY_HZ


def build_run_config(experiment_name: str) -> CrossSensorRunConfig:
    parser = argparse.ArgumentParser(description=f"Run {experiment_name} cross-sensor ablation.")
    parser.add_argument(
        "--split",
        choices=("8_3",),
        default="8_3",
        help="Train/test split to evaluate. This execution path is restricted to 8_3 only.",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=DEFAULT_MAX_COMBINATIONS,
        help="Maximum number of train/test user combinations to evaluate.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used when sampling combinations.",
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help="Optional override for the results root.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve() if args.results_root else DEFAULT_RESULTS_ROOT.resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    if args.max_combinations < 1:
        parser.error("--max-combinations must be at least 1.")

    return CrossSensorRunConfig(
        split_name=args.split,
        max_combinations=args.max_combinations,
        random_seed=args.random_seed,
        results_root=results_root,
    )


def build_results_dir(results_root: Path, experiment_name: str) -> Path:
    results_dir = Path(results_root) / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def sample_combinations(combinations_list, max_combinations: int, seed: int):
    combinations_list = list(combinations_list)
    if len(combinations_list) <= max_combinations:
        return combinations_list

    rng = random.Random(seed)
    sampled = rng.sample(combinations_list, k=max_combinations)
    return sorted(sampled, key=lambda item: (tuple(item[0]), tuple(item[1])))


def calculate_vulnerability(y_pred_proba) -> float:
    posterior = np.asarray(y_pred_proba)
    if posterior.ndim != 2 or posterior.shape[0] == 0:
        return 0.0
    return float(np.mean(np.max(posterior, axis=1)))
