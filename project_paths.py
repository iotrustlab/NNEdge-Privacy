import os
import argparse
import sys
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent


def _looks_like_dataset(path: Path) -> bool:
    return path.is_dir() and any(
        child.is_dir() and child.name.startswith("User ") for child in path.iterdir()
    )


def get_repo_root() -> str:
    return str(REPO_ROOT)


def get_data_root() -> str:
    candidates = []
    env_path = os.environ.get("NNEDGE_PRIVACY_DATA_ROOT")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend(
        [
            REPO_ROOT / "Data",
        ]
    )

    tried = []
    for candidate in candidates:
        tried.append(candidate)
        if _looks_like_dataset(candidate):
            return str(candidate.resolve())

    tried_str = "\n".join(f"  - {path}" for path in tried)
    raise FileNotFoundError(
        "Could not locate the NNEdge-Privacy dataset.\n"
        "Set NNEDGE_PRIVACY_DATA_ROOT to your dataset directory or place the dataset at "
        f"'{REPO_ROOT / 'Data'}'.\n"
        f"Tried:\n{tried_str}"
    )


def get_results_root(*parts: str) -> str:
    env_path = os.environ.get("NNEDGE_PRIVACY_RESULTS_ROOT")
    base = Path(env_path).expanduser() if env_path else REPO_ROOT / "Results"
    frequency_dir = get_frequency_dir_name()
    if not re.fullmatch(r"\d+Hz", base.name):
        base = base / frequency_dir
    return str(base.joinpath(*parts))


_CLI_OPTIONS = None


def _get_cli_options():
    global _CLI_OPTIONS
    if _CLI_OPTIONS is None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--window-sizes", nargs="+")
        parser.add_argument("--split-configs", nargs="+")
        parser.add_argument("--train-test-ratios", nargs="+")
        parser.add_argument("--total-users", type=int)
        parser.add_argument("--frequency-hz", type=int)
        parser.add_argument("--max-combinations", type=int)
        _CLI_OPTIONS, _ = parser.parse_known_args(sys.argv[1:])
    return _CLI_OPTIONS


def _split_raw_values(raw):
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [raw]

    values = []
    for item in raw:
        for chunk in str(item).split(","):
            chunk = chunk.strip()
            if chunk:
                values.append(chunk)
    return values or None


def _parse_pair_values(raw_values, cast_type):
    values = []
    for raw in _split_raw_values(raw_values) or []:
        normalized = raw.replace("/", ":")
        left, sep, right = normalized.partition(":")
        if not sep:
            raise ValueError(f"Expected pair formatted like 'a:b', got '{raw}'.")
        values.append((cast_type(left.strip()), cast_type(right.strip())))
    return values


def _get_total_users(defaults):
    options = _get_cli_options()
    if options.total_users:
        return options.total_users

    env_total_users = os.environ.get("NNEDGE_PRIVACY_TOTAL_USERS")
    if env_total_users:
        return int(env_total_users)

    if defaults:
        return int(defaults[0][0] + defaults[0][1])

    raise ValueError(
        "Unable to infer total users. Pass --total-users or set NNEDGE_PRIVACY_TOTAL_USERS."
    )


def get_window_sizes(defaults) -> list[int]:
    options = _get_cli_options()
    raw_values = _split_raw_values(options.window_sizes)
    if raw_values:
        return [int(value) for value in raw_values]

    raw = os.environ.get("NNEDGE_PRIVACY_WINDOW_SIZES")
    if not raw:
        return list(defaults)

    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))

    if not values:
        raise ValueError(
            "NNEDGE_PRIVACY_WINDOW_SIZES was set but no valid integers were provided."
        )

    return values


def get_frequency_hz(default: int = 50) -> int:
    options = _get_cli_options()
    if options.frequency_hz:
        return int(options.frequency_hz)

    env_frequency_hz = os.environ.get("NNEDGE_PRIVACY_FREQUENCY_HZ")
    if env_frequency_hz:
        return int(env_frequency_hz)

    return int(default)


def get_frequency_dir_name(default: int = 50) -> str:
    return f"{get_frequency_hz(default)}Hz"


def get_max_combinations(default: int = 2) -> int:
    options = _get_cli_options()
    if options.max_combinations:
        return int(options.max_combinations)

    env_max_combinations = os.environ.get("NNEDGE_PRIVACY_MAX_COMBINATIONS")
    if env_max_combinations:
        return int(env_max_combinations)

    return int(default)


def load_experiment_csv(file_path, source_frequency_hz: int = 50):
    df = pd.read_csv(file_path)
    target_frequency_hz = get_frequency_hz(source_frequency_hz)

    if target_frequency_hz == source_frequency_hz:
        return df

    if target_frequency_hz <= 0:
        raise ValueError("Target frequency must be positive.")

    if target_frequency_hz > source_frequency_hz:
        raise ValueError(
            f"Target frequency {target_frequency_hz} Hz exceeds source frequency {source_frequency_hz} Hz."
        )

    if source_frequency_hz % target_frequency_hz != 0:
        raise ValueError(
            f"Target frequency {target_frequency_hz} Hz must evenly divide source frequency {source_frequency_hz} Hz."
        )

    step = source_frequency_hz // target_frequency_hz
    return df.iloc[::step].reset_index(drop=True)


def get_split_configs(defaults) -> list[tuple[int, int]]:
    options = _get_cli_options()

    if options.split_configs:
        return _parse_pair_values(options.split_configs, int)

    env_split_configs = os.environ.get("NNEDGE_PRIVACY_SPLIT_CONFIGS")
    if env_split_configs:
        return _parse_pair_values(env_split_configs, int)

    ratio_values = None
    if options.train_test_ratios:
        ratio_values = _parse_pair_values(options.train_test_ratios, float)
    else:
        env_train_test_ratios = os.environ.get("NNEDGE_PRIVACY_TRAIN_TEST_RATIOS")
        if env_train_test_ratios:
            ratio_values = _parse_pair_values(env_train_test_ratios, float)

    if ratio_values:
        total_users = _get_total_users(defaults)
        split_configs = []
        for train_ratio, test_ratio in ratio_values:
            total_ratio = train_ratio + test_ratio
            if total_ratio <= 0:
                raise ValueError(
                    f"Invalid train/test ratio '{train_ratio}:{test_ratio}'."
                )
            normalized_train = train_ratio / total_ratio
            n_train = int(round(normalized_train * total_users))
            n_train = min(max(n_train, 1), total_users - 1)
            n_test = total_users - n_train
            split_configs.append((n_train, n_test))
        return split_configs

    return list(defaults)
