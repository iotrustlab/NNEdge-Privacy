from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import (
    get_data_root,
    get_frequency_dir_name,
    get_frequency_hz,
    get_max_combinations,
    get_results_root,
    get_split_configs,
    get_window_sizes,
    load_experiment_csv,
)


__all__ = [
    "get_data_root",
    "get_frequency_dir_name",
    "get_frequency_hz",
    "get_max_combinations",
    "get_results_root",
    "get_split_configs",
    "get_window_sizes",
    "load_experiment_csv",
]
