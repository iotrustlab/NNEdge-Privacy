#!/usr/bin/env python3
"""Game-3 comprehensive Identity + Placement analysis for split 8_3."""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
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

print("GAME-3 COMPREHENSIVE ANALYSIS")
print("=" * 30)
print("Split target: 8_3 only")
print("Model families: RF, DT, NB, CNN, RNN, Transformer")
print()


class Game3ComprehensiveSplitAnalyzer:
    """Run the branch-specific Game-3 sweep for all requested model families."""

    def __init__(self, config):
        self.config = config
        self.activities = ["Downstairs", "Jogging", "Laying", "Sitting", "Standing", "Upstairs", "Walking"]
        self.target_placement = "right-ankle"
        self.all_users = list(range(1, 12))
        self.accel_cols = ["acc_x[mg]", "acc_y[mg]", "acc_z[mg]"]
        self.gyro_cols = ["gyro_x[mdps]", "gyro_y[mdps]", "gyro_z[mdps]"]
        self.binary_col = "dec_tree_out_1"
        self.all_sensor_cols = self.accel_cols + self.gyro_cols + [self.binary_col]
        self.model_labels = MODEL_LABELS
        self.split_ratios = {self.config.split_name: {"train_size": 8, "test_size": 3}}
        self.experiment_name = "game3_comprehensive_analysis"

        print(f"Frequency: {self.config.frequency_hz} Hz (source: {SOURCE_FREQUENCY_HZ} Hz)")
        print(f"Split: {self.config.split_name}")
        print(f"Max combinations per split: {self.config.max_combinations}")
        print(f"Data root: {self.config.data_root}")
        print(f"Results root: {self.config.results_root}")
        print()

    def build_rf_model(self, random_seed: int) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=100,
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

    def load_user_data(self, user_id: int) -> dict[str, pd.DataFrame]:
        user_dir = user_processed_dir(self.config.data_root, user_id)
        if not os.path.exists(user_dir):
            print(f"Warning: user {user_id} data not found")
            return {}

        user_data: dict[str, pd.DataFrame] = {}
        for activity in self.activities:
            activity_dir = os.path.join(user_dir, activity)
            if not os.path.exists(activity_dir):
                continue

            placement_data = []
            direct_file = os.path.join(activity_dir, f"{self.target_placement}.csv")
            if os.path.exists(direct_file):
                try:
                    placement_data.append(load_experiment_csv(direct_file, self.config.frequency_hz))
                except Exception:
                    continue

            placement_dir = os.path.join(activity_dir, self.target_placement)
            if os.path.exists(placement_dir):
                csv_files = glob.glob(os.path.join(placement_dir, "*.csv"))
                for csv_file in sorted(csv_files):
                    try:
                        placement_data.append(load_experiment_csv(csv_file, self.config.frequency_hz))
                    except Exception:
                        continue

            if placement_data:
                user_data[activity] = pd.concat(placement_data, ignore_index=True)

        return user_data

    def extract_features(self, dataframe: pd.DataFrame) -> dict[str, float]:
        if len(dataframe) == 0:
            return {}

        features: dict[str, float] = {}
        available_cols = [column for column in self.all_sensor_cols if column in dataframe.columns]

        for column in available_cols:
            data = dataframe[column].values
            prefix = column.replace("[mg]", "").replace("[mdps]", "").replace("_out_1", "")

            features[f"{prefix}_mean"] = np.mean(data)
            features[f"{prefix}_std"] = np.std(data)
            features[f"{prefix}_min"] = np.min(data)
            features[f"{prefix}_max"] = np.max(data)
            features[f"{prefix}_median"] = np.median(data)
            features[f"{prefix}_range"] = np.max(data) - np.min(data)

            if column != self.binary_col:
                features[f"{prefix}_energy"] = np.sum(data**2)
                features[f"{prefix}_skew"] = pd.Series(data).skew()
                features[f"{prefix}_kurtosis"] = pd.Series(data).kurtosis()
                features[f"{prefix}_rms"] = np.sqrt(np.mean(data**2))
                features[f"{prefix}_zero_crossings"] = np.sum(np.diff(np.sign(data)) != 0) if len(data) > 1 else 0.0
                features[f"{prefix}_q25"] = np.percentile(data, 25)
                features[f"{prefix}_q75"] = np.percentile(data, 75)
                features[f"{prefix}_iqr"] = np.percentile(data, 75) - np.percentile(data, 25)
            else:
                features[f"{prefix}_activity_ratio"] = np.mean(data)
                if len(data) > 1:
                    transitions = np.sum(np.abs(np.diff(data.astype(int))))
                    features[f"{prefix}_transitions"] = transitions
                    features[f"{prefix}_transition_rate"] = transitions / len(data)
                else:
                    features[f"{prefix}_transitions"] = 0.0
                    features[f"{prefix}_transition_rate"] = 0.0

        if all(column in dataframe.columns for column in self.accel_cols):
            accel_data = dataframe[self.accel_cols].values
            accel_magnitude = np.linalg.norm(accel_data, axis=1)
            features["accel_magnitude_mean"] = np.mean(accel_magnitude)
            features["accel_magnitude_std"] = np.std(accel_magnitude)
            features["accel_magnitude_max"] = np.max(accel_magnitude)
            features["accel_magnitude_min"] = np.min(accel_magnitude)
            features["accel_magnitude_energy"] = np.sum(accel_magnitude**2)

        if all(column in dataframe.columns for column in self.gyro_cols):
            gyro_data = dataframe[self.gyro_cols].values
            gyro_magnitude = np.linalg.norm(gyro_data, axis=1)
            features["gyro_magnitude_mean"] = np.mean(gyro_magnitude)
            features["gyro_magnitude_std"] = np.std(gyro_magnitude)
            features["gyro_magnitude_max"] = np.max(gyro_magnitude)
            features["gyro_magnitude_min"] = np.min(gyro_magnitude)
            features["gyro_magnitude_energy"] = np.sum(gyro_magnitude**2)

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
            user_data = self.load_user_data(user_id)
            for activity, dataframe in user_data.items():
                features = self.extract_features(dataframe)
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
            user_data = self.load_user_data(user_id)
            for activity, dataframe in user_data.items():
                features = self.extract_features(dataframe)
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
                "game": "Game-3",
                "placement": self.target_placement,
                "sensor_modality": "accelerometer_gyroscope_binary_single_placement",
                "total_combinations_available": len(list(combinations(self.all_users, 3))),
                "timestamp": datetime.now().isoformat(),
            },
        )

        with open(os.path.join(split_dir, "comprehensive_analysis.json"), "w", encoding="utf-8") as file_handle:
            json.dump(split_summary, file_handle, indent=2)

        split_summary_without_details = {
            key: value for key, value in split_summary.items() if key != "detailed_results"
        }
        comparison_payload = {
            "analysis_type": "comprehensive_split_analysis",
            "game": "Game-3",
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
        with open(os.path.join(model_results_dir, "overall_comparison.json"), "w", encoding="utf-8") as file_handle:
            json.dump(comparison_payload, file_handle, indent=2)

        comparison_dataframe = pd.DataFrame(
            [
                {
                    "model_label": model_label,
                    "split": self.config.split_name,
                    "tested_combinations": split_summary_without_details["tested_combinations"],
                    "accuracy_max": split_summary_without_details["accuracy_max"],
                    "accuracy_std": split_summary_without_details["accuracy_std"],
                    "f1_max": split_summary_without_details["f1_max"],
                    "f1_std": split_summary_without_details["f1_std"],
                    "nmi_max": split_summary_without_details["nmi_max"],
                    "nmi_std": split_summary_without_details["nmi_std"],
                    "nrkl_max": split_summary_without_details["nrkl_max"],
                    "nrkl_std": split_summary_without_details["nrkl_std"],
                    "vulnerability_max": split_summary_without_details["vulnerability_max"],
                    "vulnerability_std": split_summary_without_details["vulnerability_std"],
                }
            ]
        )
        comparison_dataframe.to_csv(os.path.join(model_results_dir, "overall_comparison.csv"), index=False)

        print(
            f"  {model_label}: accuracy max={split_summary['accuracy_max']:.4f}, "
            f"f1 max={split_summary['f1_max']:.4f}, "
            f"vulnerability max={split_summary['vulnerability_max']:.4f}"
        )
        return split_summary_without_details

    def run_comprehensive_analysis(self) -> dict[str, dict]:
        print("Starting Game-3 sweep")
        print("-" * 30)
        ensure_tensorflow_gpu(random_seed=self.config.random_seed, require_gpu=True)

        combinations_list = self.generate_combinations()
        all_results = {}

        for model_index, model_label in enumerate(self.model_labels):
            model_result = self.analyze_model(model_label, combinations_list, model_index)
            if model_result:
                all_results[model_label] = model_result

        print()
        print("Game-3 sweep complete")
        print(f"Models completed: {len(all_results)}")
        return all_results


if __name__ == "__main__":
    configuration = resolve_experiment_config("Run the Game-3 identity + placement experiment.")
    analyzer = Game3ComprehensiveSplitAnalyzer(configuration)
    results = analyzer.run_comprehensive_analysis()

    if results:
        print("Analysis completed successfully.")
    else:
        print("No results generated.")
