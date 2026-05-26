#!/usr/bin/env python3
"""Shared configuration helpers for the anthropometrics + placements experiments."""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SOURCE_FREQUENCY_HZ = 50
SUPPORTED_FREQUENCIES_HZ = (5, 10, 25, 50)
DEFAULT_RANDOM_SEED = 42
SUPPORTED_SPLIT_NAMES = ("8_3",)
SUPPORTED_MODEL_LABELS = ("RF", "DT", "NB", "CNN", "RNN", "Transformer")

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = REPO_ROOT / "Data"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "Results" / "Identity-placement-anthrop"


@dataclass(frozen=True)
class AnthropPlacementExperimentConfig:
    data_root: Path
    results_root: Path
    frequency_hz: int
    max_combinations: int
    random_seed: int
    split_name: str
    model_labels: tuple[str, ...]


def _parse_env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw_value!r}.") from exc


def _parse_model_labels(raw_value: str | None) -> tuple[str, ...]:
    if raw_value is None:
        return SUPPORTED_MODEL_LABELS

    model_labels = tuple(label.strip() for label in raw_value.split(",") if label.strip())
    if not model_labels:
        raise ValueError("NNEDGE_PRIVACY_MODELS must contain at least one model label.")

    invalid_labels = sorted(set(model_labels) - set(SUPPORTED_MODEL_LABELS))
    if invalid_labels:
        raise ValueError(
            f"Unsupported model labels in NNEDGE_PRIVACY_MODELS: {invalid_labels}. "
            f"Supported labels are {SUPPORTED_MODEL_LABELS}."
        )

    return model_labels


def build_experiment_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--frequency-hz",
        type=int,
        default=_parse_env_int("NNEDGE_PRIVACY_FREQUENCY_HZ", 25),
        choices=SUPPORTED_FREQUENCIES_HZ,
        help="Target sampling frequency in Hz. Source data is 50 Hz and is downsampled in memory.",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=_parse_env_int("NNEDGE_PRIVACY_MAX_COMBINATIONS", 5),
        help="Maximum number of train/test user combinations to evaluate per split.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=_parse_env_int("NNEDGE_PRIVACY_RANDOM_SEED", DEFAULT_RANDOM_SEED),
        help="Random seed used when sampling a subset of combinations.",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("NNEDGE_PRIVACY_DATA_ROOT"),
        help="Optional override for the dataset root. Defaults to repo-local Data/.",
    )
    parser.add_argument(
        "--results-root",
        default=os.environ.get("NNEDGE_PRIVACY_RESULTS_ROOT"),
        help="Optional override for the results root. Defaults to repo-local Results/Identity-placement-anthrop/.",
    )
    parser.add_argument(
        "--split",
        default=os.environ.get("NNEDGE_PRIVACY_SPLIT", SUPPORTED_SPLIT_NAMES[0]),
        choices=SUPPORTED_SPLIT_NAMES,
        help="Train/test user split to evaluate for this branch.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(_parse_model_labels(os.environ.get("NNEDGE_PRIVACY_MODELS"))),
        choices=SUPPORTED_MODEL_LABELS,
        help="Model families to evaluate. Defaults to all branch-required model labels.",
    )
    return parser


def resolve_experiment_config(description: str) -> AnthropPlacementExperimentConfig:
    parser = build_experiment_arg_parser(description)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else DEFAULT_DATA_ROOT.resolve()
    results_root = (
        Path(args.results_root).expanduser().resolve() if args.results_root else DEFAULT_RESULTS_ROOT.resolve()
    )

    if not data_root.exists():
        parser.error(f"Data root does not exist: {data_root}")

    if args.max_combinations < 1:
        parser.error("--max-combinations must be at least 1.")

    results_root.mkdir(parents=True, exist_ok=True)

    return AnthropPlacementExperimentConfig(
        data_root=data_root,
        results_root=results_root,
        frequency_hz=args.frequency_hz,
        max_combinations=args.max_combinations,
        random_seed=args.random_seed,
        split_name=args.split,
        model_labels=tuple(args.models),
    )


def build_frequency_results_dir(results_root: Path, frequency_hz: int, experiment_name: str) -> Path:
    results_dir = Path(results_root) / f"{frequency_hz}Hz" / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def build_model_results_dir(results_root: Path, frequency_hz: int, experiment_name: str, model_label: str) -> Path:
    results_dir = build_frequency_results_dir(results_root, frequency_hz, experiment_name) / model_label
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def user_processed_dir(data_root: Path, user_id: int) -> Path:
    return Path(data_root) / f"User {user_id}" / "Processed"


def user_measurements_file(data_root: Path, user_id: int) -> Path:
    return Path(data_root) / f"User {user_id}" / "measurements.txt"


def load_experiment_csv(file_path: os.PathLike[str] | str, frequency_hz: int) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    if frequency_hz == SOURCE_FREQUENCY_HZ:
        return df

    if frequency_hz <= 0 or SOURCE_FREQUENCY_HZ % frequency_hz != 0:
        raise ValueError(
            f"Unsupported target frequency {frequency_hz} Hz. "
            f"Supported values are {SUPPORTED_FREQUENCIES_HZ}."
        )

    stride = SOURCE_FREQUENCY_HZ // frequency_hz
    return df.iloc[::stride].reset_index(drop=True)


def sample_experiment_combinations(combinations_list, max_combinations: int, seed: int):
    combinations_list = list(combinations_list)

    if len(combinations_list) <= max_combinations:
        return combinations_list

    rng = random.Random(seed)
    sampled = rng.sample(combinations_list, k=max_combinations)
    return sorted(sampled, key=_combination_sort_key)


def _combination_sort_key(combination):
    if isinstance(combination, dict):
        return (
            tuple(combination.get("train_users", [])),
            tuple(combination.get("test_users", [])),
        )

    if isinstance(combination, tuple) and len(combination) == 2:
        return (tuple(combination[0]), tuple(combination[1]))

    return repr(combination)
