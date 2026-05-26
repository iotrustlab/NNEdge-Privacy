#!/usr/bin/env python3
"""Game-2 gyroscope comprehensive Identity + Placement analysis for split 8_3."""

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

print("GAME-2 GYROSCOPE COMPREHENSIVE ANALYSIS")
print("=" * 42)
print("Split target: 8_3 only")
print("Model families: RF, DT, NB, CNN, RNN, Transformer")
print()


class Game2GyroComprehensiveSplitAnalyzer:
    """Run the branch-specific Game-2 gyroscope sweep for all requested models."""

    def __init__(self, config):
        self.config = config
        self.activities = ["Downstairs", "Jogging", "Laying", "Sitting", "Standing", "Upstairs", "Walking"]
        self.all_users = list(range(1, 12))
        self.strategic_placements = ["left-ankle", "right-ankle", "left-wrist", "right-wrist"]
        self.gyro_cols = ["gyro_x[mdps]", "gyro_y[mdps]", "gyro_z[mdps]"]
        self.binary_col = "dec_tree_out_1"
        self.available_cols = self.gyro_cols + [self.binary_col]
        self.model_labels = MODEL_LABELS
        self.split_ratios = {self.config.split_name: {"train_size": 8, "test_size": 3}}
        self.experiment_name = "game2_gyro_comprehensive_analysis"

        self.gyro_semantics = {
            "left-ankle": {
                "rotation_patterns": ["gait_rotation", "heel_strike_rotation", "foot_clearance"],
                "primary_axes": ["pitch", "roll"],
                "motion_signature": "rhythmic_gait_rotation",
            },
            "right-ankle": {
                "rotation_patterns": ["gait_rotation", "heel_strike_rotation", "foot_clearance"],
                "primary_axes": ["pitch", "roll"],
                "motion_signature": "rhythmic_gait_rotation",
            },
            "left-wrist": {
                "rotation_patterns": ["arm_swing_rotation", "gesture_rotation", "coordination"],
                "primary_axes": ["yaw", "pitch"],
                "motion_signature": "coordinated_arm_motion",
            },
            "right-wrist": {
                "rotation_patterns": ["arm_swing_rotation", "gesture_rotation", "coordination"],
                "primary_axes": ["yaw", "pitch"],
                "motion_signature": "coordinated_arm_motion",
            },
        }
        self.activity_gyro_patterns = {
            "Walking": {"rotation_intensity": "moderate", "rhythm": "regular", "axis_dominance": "pitch"},
            "Jogging": {"rotation_intensity": "high", "rhythm": "fast_regular", "axis_dominance": "pitch_roll"},
            "Upstairs": {"rotation_intensity": "high", "rhythm": "irregular", "axis_dominance": "pitch"},
            "Downstairs": {"rotation_intensity": "variable", "rhythm": "controlled", "axis_dominance": "pitch_roll"},
            "Standing": {"rotation_intensity": "minimal", "rhythm": "none", "axis_dominance": "stabilization"},
            "Sitting": {"rotation_intensity": "minimal", "rhythm": "none", "axis_dominance": "micro_adjustments"},
            "Laying": {"rotation_intensity": "none", "rhythm": "none", "axis_dominance": "none"},
        }

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
            print(f"Warning: user {user_id} data not found")
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

    def extract_gyroscope_semantic_features(
        self,
        dataframe: pd.DataFrame,
        placement: str,
        activity: str | None = None,
    ) -> dict[str, float]:
        if len(dataframe) == 0:
            return {}

        features: dict[str, float] = {}
        available_cols = [column for column in self.available_cols if column in dataframe.columns]
        if not available_cols:
            return {}

        for column in [col for col in available_cols if col in self.gyro_cols]:
            data = dataframe[column].values
            axis = column.replace("[mdps]", "").replace("gyro_", "")
            prefix = f"{placement}_{axis}"

            features[f"{prefix}_mean"] = np.mean(data)
            features[f"{prefix}_std"] = np.std(data)
            features[f"{prefix}_min"] = np.min(data)
            features[f"{prefix}_max"] = np.max(data)
            features[f"{prefix}_range"] = np.max(data) - np.min(data)
            features[f"{prefix}_median"] = np.median(data)
            features[f"{prefix}_q25"] = np.percentile(data, 25)
            features[f"{prefix}_q75"] = np.percentile(data, 75)
            features[f"{prefix}_iqr"] = features[f"{prefix}_q75"] - features[f"{prefix}_q25"]
            features[f"{prefix}_skew"] = pd.Series(data).skew()
            features[f"{prefix}_kurtosis"] = pd.Series(data).kurtosis()
            features[f"{prefix}_rms"] = np.sqrt(np.mean(data**2))
            features[f"{prefix}_energy"] = np.sum(data**2)
            features[f"{prefix}_mad"] = np.mean(np.abs(data - np.mean(data)))

            if "ankle" in placement:
                if axis == "x":
                    features[f"{prefix}_foot_roll_intensity"] = np.std(data)
                    features[f"{prefix}_lateral_stability"] = 1.0 / (1.0 + np.std(data))
                elif axis == "y":
                    features[f"{prefix}_gait_pitch_range"] = np.max(data) - np.min(data)
                    features[f"{prefix}_heel_strike_intensity"] = np.std(data)
                elif axis == "z":
                    features[f"{prefix}_foot_yaw_variation"] = np.var(data)
                    features[f"{prefix}_direction_changes"] = np.sum(np.abs(np.diff(np.sign(data))))
            elif "wrist" in placement:
                if axis == "x":
                    features[f"{prefix}_wrist_roll_gesture"] = np.std(data)
                elif axis == "y":
                    features[f"{prefix}_arm_swing_rotation"] = np.max(data) - np.min(data)
                    features[f"{prefix}_coordination_smoothness"] = 1.0 / (1.0 + np.std(np.diff(data)))
                elif axis == "z":
                    features[f"{prefix}_arm_coordination"] = np.std(data)

            if len(data) > 1:
                features[f"{prefix}_zero_crossings"] = np.sum(np.diff(np.sign(data)) != 0)
                features[f"{prefix}_rotation_changes"] = np.sum(np.abs(np.diff(data)) > np.std(data))
                features[f"{prefix}_diff_mean"] = np.mean(np.diff(data))
                features[f"{prefix}_diff_std"] = np.std(np.diff(data))
                features[f"{prefix}_angular_acceleration"] = np.mean(np.abs(np.diff(data)))

                if len(data) > 10:
                    threshold = np.mean(np.abs(data)) + np.std(np.abs(data))
                    rotation_peaks = np.sum(np.abs(data) > threshold)
                    features[f"{prefix}_rotation_peaks"] = rotation_peaks
                    features[f"{prefix}_rotation_peak_rate"] = rotation_peaks / len(data)

            if len(data) > 10:
                fft_values = np.abs(np.fft.fft(data))[: len(data) // 2]
                if len(fft_values) > 0:
                    features[f"{prefix}_fft_mean"] = np.mean(fft_values)
                    features[f"{prefix}_fft_std"] = np.std(fft_values)
                    features[f"{prefix}_spectral_energy"] = np.sum(fft_values**2)
                    features[f"{prefix}_dominant_freq"] = np.argmax(fft_values)
                    features[f"{prefix}_rotation_rhythm_freq"] = np.argmax(fft_values)

                    low_freq = np.sum(fft_values[: len(fft_values) // 4])
                    mid_freq = np.sum(fft_values[len(fft_values) // 4 : 3 * len(fft_values) // 4])
                    high_freq = np.sum(fft_values[3 * len(fft_values) // 4 :])
                    total_power = low_freq + mid_freq + high_freq

                    if total_power > 0:
                        features[f"{prefix}_low_freq_ratio"] = low_freq / total_power
                        features[f"{prefix}_mid_freq_ratio"] = mid_freq / total_power
                        features[f"{prefix}_high_freq_ratio"] = high_freq / total_power

        gyro_data = None
        if all(column in dataframe.columns for column in self.gyro_cols):
            gyro_data = dataframe[self.gyro_cols].values
            rotation_magnitude = np.linalg.norm(gyro_data, axis=1)
            prefix = f"{placement}_rotation"
            features[f"{prefix}_magnitude_mean"] = np.mean(rotation_magnitude)
            features[f"{prefix}_magnitude_std"] = np.std(rotation_magnitude)
            features[f"{prefix}_magnitude_max"] = np.max(rotation_magnitude)
            features[f"{prefix}_magnitude_range"] = np.max(rotation_magnitude) - np.min(rotation_magnitude)

            if "ankle" in placement:
                features[f"{placement}_gait_rotation_regularity"] = 1.0 / (1.0 + np.std(rotation_magnitude))
                features[f"{placement}_rotational_asymmetry"] = np.abs(np.mean(gyro_data[:, 0]))
                rotation_threshold = np.mean(rotation_magnitude) + np.std(rotation_magnitude)
                gait_rotations = rotation_magnitude > rotation_threshold
                features[f"{placement}_gait_rotation_frequency"] = np.sum(gait_rotations) / len(rotation_magnitude)
            elif "wrist" in placement:
                features[f"{placement}_arm_rotation_complexity"] = np.std(rotation_magnitude) / np.mean(
                    rotation_magnitude
                )
                features[f"{placement}_gesture_rotation_intensity"] = np.mean(rotation_magnitude)
                features[f"{placement}_arm_swing_consistency"] = 1.0 / (1.0 + np.std(rotation_magnitude))

            if activity and activity in self.activity_gyro_patterns:
                activity_pattern = self.activity_gyro_patterns[activity]
                measured_intensity = np.mean(rotation_magnitude)

                if activity_pattern["rotation_intensity"] == "high":
                    features[f"{placement}_intensity_match"] = 1.0 if measured_intensity > 50 else 0.0
                elif activity_pattern["rotation_intensity"] == "moderate":
                    features[f"{placement}_intensity_match"] = 1.0 if 20 <= measured_intensity <= 60 else 0.0
                elif activity_pattern["rotation_intensity"] == "minimal":
                    features[f"{placement}_intensity_match"] = 1.0 if measured_intensity < 30 else 0.0
                else:
                    features[f"{placement}_intensity_match"] = 1.0 if measured_intensity < 10 else 0.0

        if self.binary_col in dataframe.columns:
            binary_data = dataframe[self.binary_col].values
            prefix = f"{placement}_binary"
            features[f"{prefix}_activity_ratio"] = np.mean(binary_data)

            if len(binary_data) > 1:
                transitions = np.sum(np.abs(np.diff(binary_data.astype(int))))
                features[f"{prefix}_transitions"] = transitions
                features[f"{prefix}_transition_rate"] = transitions / len(binary_data)

                if gyro_data is not None and len(gyro_data) == len(binary_data):
                    for axis_index, axis_name in enumerate(["x", "y", "z"]):
                        correlation = np.corrcoef(gyro_data[:, axis_index], binary_data)[0, 1]
                        features[f"{prefix}_{axis_name}_correlation"] = 0.0 if np.isnan(correlation) else correlation

        cleaned_features: dict[str, float] = {}
        for key, value in features.items():
            if np.isnan(value) or np.isinf(value):
                cleaned_features[key] = 0.0
            else:
                cleaned_features[key] = float(value)

        return cleaned_features

    def extract_cross_placement_gyro_features(self, user_activity_data: dict[str, pd.DataFrame]) -> dict[str, float]:
        cross_features: dict[str, float] = {}
        available_placements = list(user_activity_data.keys())
        if len(available_placements) < 2:
            return cross_features

        placement_gyro_data = {}
        for placement in available_placements:
            dataframe = user_activity_data[placement]
            if all(column in dataframe.columns for column in self.gyro_cols):
                placement_gyro_data[placement] = dataframe[self.gyro_cols].values

        placement_pairs = [
            ("left-ankle", "right-ankle"),
            ("left-wrist", "right-wrist"),
            ("left-ankle", "left-wrist"),
            ("right-ankle", "right-wrist"),
        ]

        for placement_one, placement_two in placement_pairs:
            if placement_one not in placement_gyro_data or placement_two not in placement_gyro_data:
                continue

            data_one = placement_gyro_data[placement_one]
            data_two = placement_gyro_data[placement_two]
            min_length = min(len(data_one), len(data_two))
            data_one = data_one[:min_length]
            data_two = data_two[:min_length]

            for axis_index, axis_name in enumerate(["x", "y", "z"]):
                correlation = np.corrcoef(data_one[:, axis_index], data_two[:, axis_index])[0, 1]
                cross_features[f"{placement_one}_{placement_two}_{axis_name}_rotation_correlation"] = (
                    0.0 if np.isnan(correlation) else float(correlation)
                )

            magnitude_one = np.linalg.norm(data_one, axis=1)
            magnitude_two = np.linalg.norm(data_two, axis=1)
            magnitude_correlation = np.corrcoef(magnitude_one, magnitude_two)[0, 1]
            cross_features[f"{placement_one}_{placement_two}_rotation_magnitude_correlation"] = (
                0.0 if np.isnan(magnitude_correlation) else float(magnitude_correlation)
            )

            if len(data_one) > 1:
                peaks_one = np.where(magnitude_one > np.mean(magnitude_one) + np.std(magnitude_one))[0]
                peaks_two = np.where(magnitude_two > np.mean(magnitude_two) + np.std(magnitude_two))[0]
                if len(peaks_one) > 0 and len(peaks_two) > 0:
                    avg_phase_difference = np.mean([abs(p1 - p2) for p1 in peaks_one[:5] for p2 in peaks_two[:5]])
                    cross_features[f"{placement_one}_{placement_two}_rotation_phase_difference"] = float(
                        avg_phase_difference
                    )

        placement_rotation_intensities = {}
        for placement, data in placement_gyro_data.items():
            placement_rotation_intensities[placement] = np.mean(np.linalg.norm(data, axis=1))

        if placement_rotation_intensities:
            total_rotation = sum(placement_rotation_intensities.values())
            if total_rotation > 0:
                for placement, intensity in placement_rotation_intensities.items():
                    cross_features[f"{placement}_rotation_dominance_ratio"] = float(intensity / total_rotation)

        return cross_features

    def build_comprehensive_gyro_features(
        self,
        user_activity_data: dict[str, pd.DataFrame],
        activity: str | None = None,
    ) -> dict[str, float]:
        all_features: dict[str, float] = {}

        for placement, dataframe in user_activity_data.items():
            all_features.update(self.extract_gyroscope_semantic_features(dataframe, placement, activity))

        all_features.update(self.extract_cross_placement_gyro_features(user_activity_data))
        all_features["num_available_gyro_placements"] = float(len(user_activity_data))
        all_features["has_ankle_gyros"] = 1.0 if any("ankle" in placement for placement in user_activity_data) else 0.0
        all_features["has_wrist_gyros"] = 1.0 if any("wrist" in placement for placement in user_activity_data) else 0.0
        all_features["has_bilateral_ankle_gyros"] = (
            1.0 if {"left-ankle", "right-ankle"}.issubset(user_activity_data.keys()) else 0.0
        )
        all_features["has_bilateral_wrist_gyros"] = (
            1.0 if {"left-wrist", "right-wrist"}.issubset(user_activity_data.keys()) else 0.0
        )
        return all_features

    def prepare_combination_data(self, train_users: list[int], test_users: list[int]) -> dict | None:
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for user_id in train_users:
            user_data = self.load_multi_placement_data(user_id)
            for activity, activity_data in user_data.items():
                if not activity_data:
                    continue
                features = self.build_comprehensive_gyro_features(activity_data, activity=activity)
                if features:
                    X_train.append(features)
                    y_train.append(activity)

        if not X_train:
            return None

        for user_id in test_users:
            user_data = self.load_multi_placement_data(user_id)
            for activity, activity_data in user_data.items():
                if not activity_data:
                    continue
                features = self.build_comprehensive_gyro_features(activity_data, activity=None)
                if features:
                    X_test.append(features)
                    y_test.append(activity)

        if not X_test:
            return None

        feature_names = sorted({key for feature_map in X_train + X_test for key in feature_map.keys()})
        X_train_array = np.asarray([[feature_map.get(name, 0.0) for name in feature_names] for feature_map in X_train])
        X_test_array = np.asarray([[feature_map.get(name, 0.0) for name in feature_names] for feature_map in X_test])
        X_train_array = np.nan_to_num(X_train_array, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_array = np.nan_to_num(X_test_array, nan=0.0, posinf=0.0, neginf=0.0)

        y_train_array = np.asarray(y_train)
        y_test_array = np.asarray(y_test)

        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train_array)
        y_test_encoded = label_encoder.transform(y_test_array)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_array)
        X_test_scaled = scaler.transform(X_test_array)

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
                "analysis_type": "game2_gyro_comprehensive_analysis",
                "game": "Game-2 Gyroscope",
                "sensor_modality": "gyroscope_binary_multi_placement",
                "strategic_placements": self.strategic_placements,
                "model_architecture": "VotingClassifier(RandomForest+LogisticRegression+SVC)"
                if model_label == "RF"
                else model_label,
                "total_combinations_available": len(list(combinations(self.all_users, 3))),
                "timestamp": datetime.now().isoformat(),
            },
        )

        with open(os.path.join(split_dir, "comprehensive_analysis.json"), "w", encoding="utf-8") as file_handle:
            json.dump(split_summary, file_handle, indent=2)

        split_summary_without_details = {
            key: value for key, value in split_summary.items() if key != "detailed_results"
        }
        comprehensive_results = {
            "analysis_type": "game2_gyro_comprehensive_analysis",
            "description": "Comprehensive train/test combination analysis for Game-2 gyroscope adversary",
            "game": "Game-2 Gyroscope",
            "model_label": model_label,
            "sensor_modality": "gyroscope_binary_multi_placement",
            "strategic_placements": self.strategic_placements,
            "frequency_hz": self.config.frequency_hz,
            "source_frequency_hz": SOURCE_FREQUENCY_HZ,
            "split_name": self.config.split_name,
            "max_combinations_requested": self.config.max_combinations,
            "data_root": str(self.config.data_root),
            "results_dir": model_results_dir,
            "split_results": {self.config.split_name: split_summary_without_details},
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(model_results_dir, "comprehensive_results.json"), "w", encoding="utf-8") as file_handle:
            json.dump(comprehensive_results, file_handle, indent=2)

        report_lines = [
            "GAME-2-GYRO COMPREHENSIVE ANALYSIS SUMMARY",
            "=" * 50,
            "",
            f"Model: {model_label}",
            "Sensors: Gyroscope + Binary (multi-placement)",
            "Placements: left-ankle, right-ankle, left-wrist, right-wrist",
            "Features: Semantic gyroscope rotation analysis",
            f"Frequency: {self.config.frequency_hz} Hz",
            f"Split: {self.config.split_name}",
            "",
            "Headline metrics:",
            f"Accuracy max/std: {split_summary['accuracy_max']:.6f} / {split_summary['accuracy_std']:.6f}",
            f"F1 max/std: {split_summary['f1_max']:.6f} / {split_summary['f1_std']:.6f}",
            f"NMI max/std: {split_summary['nmi_max']:.6f} / {split_summary['nmi_std']:.6f}",
            f"NRKL max/std: {split_summary['nrkl_max']:.6f} / {split_summary['nrkl_std']:.6f}",
            f"Vulnerability max/std: {split_summary['vulnerability_max']:.6f} / {split_summary['vulnerability_std']:.6f}",
            "",
            f"Tested combinations: {split_summary['tested_combinations']}",
            f"Generated: {datetime.now().isoformat()}",
            "",
        ]
        with open(os.path.join(model_results_dir, "summary_report.txt"), "w", encoding="utf-8") as file_handle:
            file_handle.write("\n".join(report_lines))

        print(
            f"  {model_label}: accuracy max={split_summary['accuracy_max']:.4f}, "
            f"f1 max={split_summary['f1_max']:.4f}, "
            f"vulnerability max={split_summary['vulnerability_max']:.4f}"
        )
        return split_summary_without_details

    def run_comprehensive_analysis(self) -> dict[str, dict]:
        print("Starting Game-2 gyroscope sweep")
        print("-" * 42)
        ensure_tensorflow_gpu(random_seed=self.config.random_seed, require_gpu=True)

        combinations_list = self.generate_combinations()
        all_results = {}

        for model_index, model_label in enumerate(self.model_labels):
            model_result = self.analyze_model(model_label, combinations_list, model_index)
            if model_result:
                all_results[model_label] = model_result

        print()
        print("Game-2 gyroscope sweep complete")
        print(f"Models completed: {len(all_results)}")
        return all_results


if __name__ == "__main__":
    configuration = resolve_experiment_config("Run the Game-2 gyroscope identity + placement experiment.")
    analyzer = Game2GyroComprehensiveSplitAnalyzer(configuration)
    results = analyzer.run_comprehensive_analysis()

    if results:
        print("Analysis completed successfully.")
    else:
        print("No results generated.")
