#!/usr/bin/env python3
"""Game-1 comprehensive Identity + Placement analysis for split 8_3."""

from __future__ import annotations

import json
import os
from datetime import datetime
from itertools import combinations
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from identity_placement_common import (
    MODEL_LABELS,
    build_aggregate_summary,
    ensure_tensorflow_gpu,
    make_combination_result,
    set_global_seeds,
    train_tabular_model,
)
from project_paths import (
    SOURCE_FREQUENCY_HZ,
    build_model_results_dir,
    load_experiment_csv,
    resolve_experiment_config,
    sample_experiment_combinations,
    user_processed_dir,
)

warnings.filterwarnings("ignore")

print("GAME-1 COMPREHENSIVE SPLIT ANALYSIS")
print("=" * 40)
print("Split target: 8_3 only")
print("Model families: RF, DT, NB, CNN, RNN, Transformer")
print()


class Game1ComprehensiveSplitAnalyzer:
    """Run the branch-specific Game-1 sweep for all requested model families."""

    def __init__(self, config):
        self.config = config
        self.activities = ["Downstairs", "Jogging", "Laying", "Sitting", "Standing", "Upstairs", "Walking"]
        self.target_placement = "left-wrist"
        self.all_users = list(range(1, 12))
        self.binary_col = "dec_tree_out_1"
        self.model_labels = MODEL_LABELS
        self.split_ratios = {self.config.split_name: {"train_size": 8, "test_size": 3}}
        self.experiment_name = "game1_comprehensive_analysis"

        print(f"Frequency: {self.config.frequency_hz} Hz (source: {SOURCE_FREQUENCY_HZ} Hz)")
        print(f"Split: {self.config.split_name}")
        print(f"Max combinations per split: {self.config.max_combinations}")
        print(f"Data root: {self.config.data_root}")
        print(f"Results root: {self.config.results_root}")
        print()

    def build_rf_model(self, random_seed: int) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_seed,
            class_weight="balanced",
        )

    def model_results_dir(self, model_label: str) -> str:
        return str(
            build_model_results_dir(
                self.config.results_root,
                self.config.frequency_hz,
                self.experiment_name,
                model_label,
            )
        )

    def generate_combinations(self) -> list[dict]:
        split_config = self.split_ratios[self.config.split_name]
        all_test_combinations = list(combinations(self.all_users, split_config["test_size"]))

        train_test_pairs = []
        for test_users in all_test_combinations:
            train_users = [user_id for user_id in self.all_users if user_id not in test_users]
            train_test_pairs.append({"train_users": train_users, "test_users": list(test_users)})

        selected_pairs = sample_experiment_combinations(
            train_test_pairs,
            max_combinations=self.config.max_combinations,
            seed=self.config.random_seed,
        )

        print(
            f"Split {self.config.split_name}: selected {len(selected_pairs)} combinations "
            f"out of {len(train_test_pairs)} possible"
        )
        return selected_pairs

    def load_user_binary_data(self, user_id: int) -> dict[str, np.ndarray]:
        user_dir = user_processed_dir(self.config.data_root, user_id)
        if not os.path.exists(user_dir):
            print(f"Warning: user {user_id} data not found")
            return {}

        user_data: dict[str, np.ndarray] = {}
        for activity in self.activities:
            activity_dir = os.path.join(user_dir, activity)
            if not os.path.exists(activity_dir):
                continue

            activity_sequences = []

            if activity in ["Standing", "Sitting", "Laying", "Walking", "Jogging"]:
                placement_file = os.path.join(activity_dir, f"{self.target_placement}.csv")
                if os.path.exists(placement_file):
                    try:
                        dataframe = load_experiment_csv(placement_file, self.config.frequency_hz)
                        if self.binary_col in dataframe.columns:
                            activity_sequences.append(dataframe[self.binary_col].values)
                    except Exception:
                        continue
            else:
                placement_subdir = os.path.join(activity_dir, self.target_placement)
                if os.path.exists(placement_subdir):
                    csv_files = sorted(
                        file_name for file_name in os.listdir(placement_subdir) if file_name.endswith(".csv")
                    )
                    for csv_file in csv_files:
                        file_path = os.path.join(placement_subdir, csv_file)
                        try:
                            dataframe = load_experiment_csv(file_path, self.config.frequency_hz)
                            if self.binary_col in dataframe.columns:
                                activity_sequences.append(dataframe[self.binary_col].values)
                        except Exception:
                            continue

            if activity_sequences:
                user_data[activity] = np.concatenate(activity_sequences)

        return user_data

    def extract_binary_features(self, binary_sequence: np.ndarray) -> dict[str, float]:
        if len(binary_sequence) == 0:
            return {}

        features: dict[str, float] = {}
        features["activity_ratio"] = np.mean(binary_sequence)
        features["total_length"] = len(binary_sequence)
        features["active_samples"] = np.sum(binary_sequence)

        def get_runs(sequence: np.ndarray, value: int) -> list[int]:
            runs = []
            current_run = 0
            for bit in sequence:
                if bit == value:
                    current_run += 1
                else:
                    if current_run > 0:
                        runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                runs.append(current_run)
            return runs

        active_runs = get_runs(binary_sequence, 1)
        inactive_runs = get_runs(binary_sequence, 0)

        if active_runs:
            features["num_active_bursts"] = len(active_runs)
            features["avg_active_duration"] = np.mean(active_runs)
            features["max_active_duration"] = np.max(active_runs)
            features["min_active_duration"] = np.min(active_runs)
            features["std_active_duration"] = np.std(active_runs)
            features["median_active_duration"] = np.median(active_runs)
            features["active_regularity"] = (
                1.0 / (1.0 + np.std(active_runs) / np.mean(active_runs)) if len(active_runs) > 1 else 1.0
            )
        else:
            features["num_active_bursts"] = 0.0
            features["avg_active_duration"] = 0.0
            features["max_active_duration"] = 0.0
            features["min_active_duration"] = 0.0
            features["std_active_duration"] = 0.0
            features["median_active_duration"] = 0.0
            features["active_regularity"] = 0.0

        if inactive_runs:
            features["num_rest_periods"] = len(inactive_runs)
            features["avg_rest_duration"] = np.mean(inactive_runs)
            features["max_rest_duration"] = np.max(inactive_runs)
            features["std_rest_duration"] = np.std(inactive_runs)
            features["median_rest_duration"] = np.median(inactive_runs)
        else:
            features["num_rest_periods"] = 0.0
            features["avg_rest_duration"] = 0.0
            features["max_rest_duration"] = 0.0
            features["std_rest_duration"] = 0.0
            features["median_rest_duration"] = 0.0

        if len(binary_sequence) > 1:
            transitions = np.sum(np.abs(np.diff(binary_sequence.astype(int))))
            state_changes = np.diff(binary_sequence.astype(int))
            on_transitions = np.sum(state_changes == 1)
            off_transitions = np.sum(state_changes == -1)

            features["total_transitions"] = transitions
            features["transition_rate"] = transitions / len(binary_sequence)
            features["on_transitions"] = on_transitions
            features["off_transitions"] = off_transitions
            features["transition_balance"] = abs(on_transitions - off_transitions) / max(
                1, on_transitions + off_transitions
            )
        else:
            features["total_transitions"] = 0.0
            features["transition_rate"] = 0.0
            features["on_transitions"] = 0.0
            features["off_transitions"] = 0.0
            features["transition_balance"] = 0.0

        all_runs = active_runs + inactive_runs
        if len(all_runs) > 1:
            features["overall_pattern_variance"] = np.var(all_runs)
            features["pattern_complexity"] = len(set(all_runs))
            features["rhythm_consistency"] = 1.0 / (1.0 + np.std(all_runs) / np.mean(all_runs))
        else:
            features["overall_pattern_variance"] = 0.0
            features["pattern_complexity"] = 1.0
            features["rhythm_consistency"] = 0.0

        if len(binary_sequence) >= 10:
            window_size = len(binary_sequence) // 10
            window_activities = []
            for start_index in range(0, len(binary_sequence) - window_size, window_size):
                window = binary_sequence[start_index : start_index + window_size]
                window_activities.append(np.mean(window))

            if len(window_activities) > 1:
                features["temporal_activity_variance"] = np.var(window_activities)
                features["temporal_activity_trend"] = np.corrcoef(
                    range(len(window_activities)), window_activities
                )[0, 1]
                features["peak_activity_window"] = np.max(window_activities)
                features["min_activity_window"] = np.min(window_activities)
            else:
                features["temporal_activity_variance"] = 0.0
                features["temporal_activity_trend"] = 0.0
                features["peak_activity_window"] = features["activity_ratio"]
                features["min_activity_window"] = features["activity_ratio"]
        else:
            features["temporal_activity_variance"] = 0.0
            features["temporal_activity_trend"] = 0.0
            features["peak_activity_window"] = features["activity_ratio"]
            features["min_activity_window"] = features["activity_ratio"]

        cleaned_features: dict[str, float] = {}
        for key, value in features.items():
            if np.isnan(value) or np.isinf(value):
                cleaned_features[key] = 0.0
            else:
                cleaned_features[key] = float(value)

        return cleaned_features

    def create_dataset(self, user_list: list[int]):
        X = []
        y = []
        feature_names = None

        for user_id in user_list:
            user_data = self.load_user_binary_data(user_id)
            for activity, binary_sequence in user_data.items():
                features = self.extract_binary_features(binary_sequence)
                if not features:
                    continue

                if feature_names is None:
                    feature_names = sorted(features.keys())

                X.append([features.get(name, 0.0) for name in feature_names])
                y.append(activity)

        return np.asarray(X, dtype=np.float64), np.asarray(y), feature_names or []

    def prepare_combination_data(self, train_users: list[int], test_users: list[int]) -> dict | None:
        X_train, y_train, feature_names = self.create_dataset(train_users)
        if len(X_train) == 0:
            return None

        X_test = []
        y_test = []
        for user_id in test_users:
            user_data = self.load_user_binary_data(user_id)
            for activity, binary_sequence in user_data.items():
                features = self.extract_binary_features(binary_sequence)
                if features:
                    X_test.append([features.get(name, 0.0) for name in feature_names])
                    y_test.append(activity)

        if not X_test:
            return None

        X_test_array = np.asarray(X_test, dtype=np.float64)
        y_test_array = np.asarray(y_test)

        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test_array)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0))
        X_test_scaled = scaler.transform(np.nan_to_num(X_test_array, nan=0.0, posinf=0.0, neginf=0.0))

        return {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train_encoded,
            "y_test": y_test_encoded,
            "label_encoder": label_encoder,
            "feature_names": feature_names,
            "train_samples": len(X_train_scaled),
            "test_samples": len(X_test_scaled),
        }

    def evaluate_model_combination(
        self,
        model_label: str,
        train_users: list[int],
        test_users: list[int],
        combination_seed: int,
    ) -> dict | None:
        prepared = self.prepare_combination_data(train_users, test_users)
        if prepared is None:
            return None

        set_global_seeds(combination_seed)
        y_pred_encoded, y_pred_proba = train_tabular_model(
            model_label=model_label,
            X_train=prepared["X_train"],
            y_train=prepared["y_train"],
            X_test=prepared["X_test"],
            random_seed=combination_seed,
            rf_builder=lambda: self.build_rf_model(combination_seed),
        )

        result = make_combination_result(
            model_label=model_label,
            train_users=train_users,
            test_users=test_users,
            train_samples=prepared["train_samples"],
            test_samples=prepared["test_samples"],
            n_features=len(prepared["feature_names"]),
            y_true_encoded=prepared["y_test"],
            y_pred_encoded=y_pred_encoded,
            y_pred_proba=y_pred_proba,
        )
        result["y_true"] = prepared["label_encoder"].inverse_transform(prepared["y_test"]).tolist()
        result["y_pred"] = prepared["label_encoder"].inverse_transform(y_pred_encoded).tolist()
        return result

    def analyze_model(self, model_label: str, combinations_list: list[dict], model_index: int) -> dict | None:
        print(f"Running {model_label} for split {self.config.split_name}")
        results = []

        for combo_index, combo in enumerate(combinations_list):
            combination_seed = self.config.random_seed + (model_index * 1000) + combo_index
            result = self.evaluate_model_combination(
                model_label,
                combo["train_users"],
                combo["test_users"],
                combination_seed,
            )
            if result:
                results.append(result)

        if not results:
            print(f"No valid {model_label} results for split {self.config.split_name}")
            return None

        model_results_dir = self.model_results_dir(model_label)
        split_dir = os.path.join(model_results_dir, f"split_{self.config.split_name}")
        os.makedirs(split_dir, exist_ok=True)

        split_summary = build_aggregate_summary(
            combination_results=results,
            model_label=model_label,
            split_name=self.config.split_name,
            frequency_hz=self.config.frequency_hz,
            source_frequency_hz=SOURCE_FREQUENCY_HZ,
            max_combinations_requested=self.config.max_combinations,
            data_root=str(self.config.data_root),
            results_dir=model_results_dir,
            extra_fields={
                "analysis_type": "comprehensive_split_analysis",
                "game": "Game-1",
                "constraint": "binary_only",
                "placement": self.target_placement,
                "total_combinations_available": len(list(combinations(self.all_users, 3))),
                "timestamp": datetime.now().isoformat(),
            },
        )

        with open(os.path.join(split_dir, "comprehensive_analysis.json"), "w", encoding="utf-8") as file_handle:
            json.dump(split_summary, file_handle, indent=2)

        split_summary_without_details = {
            key: value for key, value in split_summary.items() if key != "detailed_results"
        }
        with open(os.path.join(split_dir, "summary_statistics.json"), "w", encoding="utf-8") as file_handle:
            json.dump(split_summary_without_details, file_handle, indent=2)

        master_summary = {
            "analysis_type": "comprehensive_split_analysis",
            "game": "Game-1",
            "model_label": model_label,
            "constraint": "binary_only",
            "placement": self.target_placement,
            "frequency_hz": self.config.frequency_hz,
            "source_frequency_hz": SOURCE_FREQUENCY_HZ,
            "split_name": self.config.split_name,
            "max_combinations_requested": self.config.max_combinations,
            "data_root": str(self.config.data_root),
            "results_dir": model_results_dir,
            "split_results": {self.config.split_name: split_summary_without_details},
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(model_results_dir, "master_summary.json"), "w", encoding="utf-8") as file_handle:
            json.dump(master_summary, file_handle, indent=2)

        print(
            f"  {model_label}: accuracy max={split_summary['accuracy_max']:.4f}, "
            f"f1 max={split_summary['f1_max']:.4f}, "
            f"vulnerability max={split_summary['vulnerability_max']:.4f}"
        )
        return split_summary_without_details

    def run_comprehensive_analysis(self) -> dict[str, dict]:
        print("Starting Game-1 sweep")
        print("-" * 40)
        ensure_tensorflow_gpu(random_seed=self.config.random_seed, require_gpu=True)

        combinations_list = self.generate_combinations()
        all_results = {}

        for model_index, model_label in enumerate(self.model_labels):
            model_result = self.analyze_model(model_label, combinations_list, model_index)
            if model_result:
                all_results[model_label] = model_result

        print()
        print("Game-1 sweep complete")
        print(f"Models completed: {len(all_results)}")
        return all_results


if __name__ == "__main__":
    configuration = resolve_experiment_config("Run the Game-1 identity + placement experiment.")
    analyzer = Game1ComprehensiveSplitAnalyzer(configuration)
    results = analyzer.run_comprehensive_analysis()

    if results:
        print("Analysis completed successfully.")
    else:
        print("No results generated.")
