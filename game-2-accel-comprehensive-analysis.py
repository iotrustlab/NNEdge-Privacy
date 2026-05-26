#!/usr/bin/env python3
"""Game-2 accelerometer comprehensive Identity + Placement analysis for split 8_3."""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from identity_placement_common import (
    MODEL_LABELS,
    build_aggregate_summary,
    ensure_tensorflow_gpu,
    make_combination_result,
    make_rf_voting_baseline,
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

print("GAME-2 ACCELEROMETER COMPREHENSIVE ANALYSIS")
print("=" * 46)
print("Split target: 8_3 only")
print("Model families: RF, DT, NB, CNN, RNN, Transformer")
print()


class Game2AccelComprehensiveAnalyzer:
    """Run the branch-specific Game-2 accelerometer sweep for all requested models."""

    def __init__(self, config):
        self.config = config
        self.activities = ["Downstairs", "Jogging", "Laying", "Sitting", "Standing", "Upstairs", "Walking"]
        self.all_users = list(range(1, 12))
        self.strategic_placements = ["left-ankle", "right-ankle", "left-wrist", "right-wrist"]
        self.accel_cols = ["acc_x[mg]", "acc_y[mg]", "acc_z[mg]"]
        self.binary_col = "dec_tree_out_1"
        self.available_cols = self.accel_cols + [self.binary_col]
        self.model_labels = MODEL_LABELS
        self.split_ratios = {self.config.split_name: {"train_size": 8, "test_size": 3}}
        self.experiment_name = "game2_accel_comprehensive_analysis"

        print(f"Frequency: {self.config.frequency_hz} Hz (source: {SOURCE_FREQUENCY_HZ} Hz)")
        print(f"Split: {self.config.split_name}")
        print(f"Max combinations per split: {self.config.max_combinations}")
        print(f"Data root: {self.config.data_root}")
        print(f"Results root: {self.config.results_root}")
        print()

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

    def load_multi_placement_data(self, user_id: int) -> dict[str, dict[str, pd.DataFrame]]:
        user_dir = user_processed_dir(self.config.data_root, user_id)
        if not os.path.exists(user_dir):
            return {}

        user_data: dict[str, dict[str, pd.DataFrame]] = {}
        for activity in self.activities:
            activity_dir = os.path.join(user_dir, activity)
            if not os.path.exists(activity_dir):
                continue

            activity_data: dict[str, pd.DataFrame] = {}
            for placement in self.strategic_placements:
                placement_data = []

                direct_file = os.path.join(activity_dir, f"{placement}.csv")
                if os.path.exists(direct_file):
                    try:
                        placement_data.append(load_experiment_csv(direct_file, self.config.frequency_hz))
                    except Exception:
                        continue

                placement_dir = os.path.join(activity_dir, placement)
                if os.path.exists(placement_dir):
                    csv_files = glob.glob(os.path.join(placement_dir, "*.csv"))
                    for csv_file in sorted(csv_files):
                        try:
                            placement_data.append(load_experiment_csv(csv_file, self.config.frequency_hz))
                        except Exception:
                            continue

                if placement_data:
                    activity_data[placement] = pd.concat(placement_data, ignore_index=True)

            if activity_data:
                user_data[activity] = activity_data

        return user_data

    def extract_placement_specific_features(self, dataframe: pd.DataFrame, placement: str) -> dict[str, float]:
        if len(dataframe) == 0:
            return {}

        features: dict[str, float] = {}
        available_cols = [column for column in self.available_cols if column in dataframe.columns]
        if not available_cols:
            return {}

        for column in [col for col in available_cols if col in self.accel_cols]:
            data = dataframe[column].values
            axis = column.replace("[mg]", "").replace("acc_", "")
            prefix = f"{placement}_{axis}"

            features[f"{prefix}_mean"] = np.mean(data)
            features[f"{prefix}_std"] = np.std(data)
            features[f"{prefix}_max"] = np.max(data)
            features[f"{prefix}_min"] = np.min(data)
            features[f"{prefix}_median"] = np.median(data)
            features[f"{prefix}_range"] = np.max(data) - np.min(data)
            features[f"{prefix}_energy"] = np.sum(data**2)

            if len(data) > 1:
                features[f"{prefix}_skew"] = pd.Series(data).skew()
                features[f"{prefix}_kurtosis"] = pd.Series(data).kurtosis()
                features[f"{prefix}_rms"] = np.sqrt(np.mean(data**2))

                diff_data = np.diff(data)
                features[f"{prefix}_diff_mean"] = np.mean(diff_data)
                features[f"{prefix}_diff_std"] = np.std(diff_data)
                features[f"{prefix}_diff_max"] = np.max(np.abs(diff_data))
                features[f"{prefix}_q25"] = np.percentile(data, 25)
                features[f"{prefix}_q75"] = np.percentile(data, 75)
                features[f"{prefix}_iqr"] = np.percentile(data, 75) - np.percentile(data, 25)
                features[f"{prefix}_zero_crossings"] = np.sum(np.diff(np.sign(data)) != 0)

        if all(column in dataframe.columns for column in self.accel_cols):
            accel_data = dataframe[self.accel_cols].values
            magnitude = np.linalg.norm(accel_data, axis=1)
            prefix = f"{placement}_magnitude"

            features[f"{prefix}_mean"] = np.mean(magnitude)
            features[f"{prefix}_std"] = np.std(magnitude)
            features[f"{prefix}_max"] = np.max(magnitude)
            features[f"{prefix}_min"] = np.min(magnitude)
            features[f"{prefix}_median"] = np.median(magnitude)
            features[f"{prefix}_range"] = np.max(magnitude) - np.min(magnitude)
            features[f"{prefix}_energy"] = np.sum(magnitude**2)

        if self.binary_col in dataframe.columns:
            binary_data = dataframe[self.binary_col].values
            prefix = f"{placement}_binary"
            features[f"{prefix}_activity_ratio"] = np.mean(binary_data)

            if len(binary_data) > 1:
                transitions = np.sum(np.abs(np.diff(binary_data.astype(int))))
                features[f"{prefix}_transitions"] = transitions
                features[f"{prefix}_transition_rate"] = transitions / len(binary_data)
            else:
                features[f"{prefix}_transitions"] = 0.0
                features[f"{prefix}_transition_rate"] = 0.0

        cleaned_features: dict[str, float] = {}
        for key, value in features.items():
            if np.isnan(value) or np.isinf(value):
                cleaned_features[key] = 0.0
            else:
                cleaned_features[key] = float(value)

        return cleaned_features

    def build_user_features(self, user_data: dict[str, dict[str, pd.DataFrame]]) -> dict[str, dict[str, float]]:
        all_features: dict[str, dict[str, float]] = {}

        for activity, activity_data in user_data.items():
            placement_features: dict[str, float] = {}

            for placement, dataframe in activity_data.items():
                placement_features.update(self.extract_placement_specific_features(dataframe, placement))

            if len(activity_data) >= 2:
                placements = list(activity_data.keys())
                for index, placement_one in enumerate(placements):
                    for placement_two in placements[index + 1 :]:
                        dataframe_one = activity_data[placement_one]
                        dataframe_two = activity_data[placement_two]
                        if self.accel_cols[0] not in dataframe_one.columns or self.accel_cols[0] not in dataframe_two.columns:
                            continue

                        for axis_col in self.accel_cols:
                            if axis_col not in dataframe_one.columns or axis_col not in dataframe_two.columns:
                                continue
                            try:
                                min_length = min(len(dataframe_one), len(dataframe_two))
                                correlation = np.corrcoef(
                                    dataframe_one[axis_col].values[:min_length],
                                    dataframe_two[axis_col].values[:min_length],
                                )[0, 1]
                                if not np.isnan(correlation):
                                    axis = axis_col.replace("[mg]", "").replace("acc_", "")
                                    placement_features[f"{placement_one}_{placement_two}_{axis}_correlation"] = (
                                        float(correlation)
                                    )
                            except Exception:
                                continue

            if placement_features:
                all_features[activity] = placement_features

        return all_features

    def prepare_combination_data(self, train_users: list[int], test_users: list[int]) -> dict | None:
        X_train = []
        y_train = []

        for user_id in train_users:
            user_features = self.build_user_features(self.load_multi_placement_data(user_id))
            for activity, features in user_features.items():
                X_train.append(features)
                y_train.append(activity)

        if not X_train:
            return None

        X_test = []
        y_test = []
        for user_id in test_users:
            user_features = self.build_user_features(self.load_multi_placement_data(user_id))
            for activity, features in user_features.items():
                X_test.append(features)
                y_test.append(activity)

        if not X_test:
            return None

        feature_names = sorted({key for feature_map in X_train + X_test for key in feature_map.keys()})
        X_train_array = np.asarray([[feature_map.get(name, 0.0) for name in feature_names] for feature_map in X_train])
        X_test_array = np.asarray([[feature_map.get(name, 0.0) for name in feature_names] for feature_map in X_test])
        y_train_array = np.asarray(y_train)
        y_test_array = np.asarray(y_test)

        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train_array)
        y_test_encoded = label_encoder.transform(y_test_array)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(np.nan_to_num(X_train_array, nan=0.0, posinf=0.0, neginf=0.0))
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
            rf_builder=lambda: make_rf_voting_baseline(combination_seed),
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
                "game": "Game-2 Accelerometer",
                "sensor_modality": "accelerometer_binary_multi_placement",
                "strategic_placements": self.strategic_placements,
                "total_combinations_available": len(list(combinations(self.all_users, 3))),
                "timestamp": datetime.now().isoformat(),
            },
        )

        with open(os.path.join(split_dir, "comprehensive_analysis.json"), "w", encoding="utf-8") as file_handle:
            json.dump(split_summary, file_handle, indent=2)

        split_summary_without_details = {
            key: value for key, value in split_summary.items() if key != "detailed_results"
        }

        summary_payload = {
            "analysis_type": "comprehensive_split_analysis",
            "game": "Game-2 Accelerometer",
            "model_label": model_label,
            "frequency_hz": self.config.frequency_hz,
            "source_frequency_hz": SOURCE_FREQUENCY_HZ,
            "split_name": self.config.split_name,
            "max_combinations_requested": self.config.max_combinations,
            "data_root": str(self.config.data_root),
            "results_dir": model_results_dir,
            "split_results": {self.config.split_name: split_summary_without_details},
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(model_results_dir, "summary_comparison.json"), "w", encoding="utf-8") as file_handle:
            json.dump(summary_payload, file_handle, indent=2)

        print(
            f"  {model_label}: accuracy max={split_summary['accuracy_max']:.4f}, "
            f"f1 max={split_summary['f1_max']:.4f}, "
            f"vulnerability max={split_summary['vulnerability_max']:.4f}"
        )
        return split_summary_without_details

    def run_comprehensive_analysis(self) -> dict[str, dict]:
        print("Starting Game-2 accelerometer sweep")
        print("-" * 46)
        ensure_tensorflow_gpu(random_seed=self.config.random_seed, require_gpu=True)

        combinations_list = self.generate_combinations()
        all_results = {}

        for model_index, model_label in enumerate(self.model_labels):
            model_result = self.analyze_model(model_label, combinations_list, model_index)
            if model_result:
                all_results[model_label] = model_result

        print()
        print("Game-2 accelerometer sweep complete")
        print(f"Models completed: {len(all_results)}")
        return all_results


if __name__ == "__main__":
    configuration = resolve_experiment_config("Run the Game-2 accelerometer identity + placement experiment.")
    analyzer = Game2AccelComprehensiveAnalyzer(configuration)
    results = analyzer.run_comprehensive_analysis()

    if results:
        print("Analysis completed successfully.")
    else:
        print("No results generated.")
