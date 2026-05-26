#!/usr/bin/env python3
"""
Game-2 Improved Gyroscope Cross-Sensor Cross-Dataset Study (Placement-Agnostic Features)
========================================================================================

This script implements Game-2 cross-sensor cross-dataset HAR using IMPROVED placement-agnostic
gyroscope features based on comprehensive transferability analysis.

FEATURES: Top 15 Placement-Agnostic Gyroscope Features (0.013-0.026 transferability)
1. gyro_total_rotation_intensity (0.026) - BEST!
2. gyro_avg_rotation_intensity (0.026) - BEST!  
3. gyro_magnitude_mean (0.025)
4. gyro_magnitude_q25 (0.023)
5. gyro_magnitude_median (0.020)
6. gyro_magnitude_rms (0.020)
7. gyro_magnitude_sma (0.019)
8. gyro_magnitude_q10 (0.018)
9. gyro_magnitude_q75 (0.018)
10. gyro_magnitude_change_rate (0.015)
11. gyro_magnitude_derivative_std (0.014)
12. gyro_smoothness (0.013)
13. gyro_magnitude_q90 (0.013)
14. gyro_magnitude_std (0.013)
15. gyro_magnitude_min (0.013)

IMPROVEMENT: Orientation-invariant features for better cross-sensor transferability
BASELINE: Original Game-2 achieved 45.1% (STM→MS), 38.5% (MS→STM)
GOAL: Improve cross-sensor performance using placement-agnostic features
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
from datetime import datetime
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.fft import fft, fftfreq
import warnings

from cross_dataset_utils import calculate_vulnerability, get_motionsense_root, get_stm_root

warnings.filterwarnings('ignore')

print("🎭 GAME-2 IMPROVED GYROSCOPE CROSS-SENSOR CROSS-DATASET STUDY")
print("=" * 70)
print("🔬 Using TOP 15 PLACEMENT-AGNOSTIC Gyroscope Features")
print("🎯 Goal: Improve cross-sensor performance with orientation-invariant features")
print("📊 Features: Magnitude-based statistical and temporal dynamics")
print()

def calculate_mutual_information(y_true, y_pred):
    """
    Calculate Mutual Information between true and predicted labels
    
    Returns:
        mi_value: Raw mutual information value
        nmi_percentage: Normalized mutual information as percentage (0-100%)
    """
    try:
        # Calculate raw mutual information
        mi_value = mutual_info_score(y_true, y_pred)
        
        # Calculate entropies for normalization
        # Calculate probability distributions
        y_true_counts = np.bincount(y_true)
        y_pred_counts = np.bincount(y_pred)
        
        # Handle case where y_pred might have different number of classes
        n_classes = max(len(y_true_counts), len(y_pred_counts))
        
        # Pad with zeros if necessary
        if len(y_true_counts) < n_classes:
            y_true_counts = np.pad(y_true_counts, (0, n_classes - len(y_true_counts)))
        if len(y_pred_counts) < n_classes:
            y_pred_counts = np.pad(y_pred_counts, (0, n_classes - len(y_pred_counts)))
        
        # Calculate entropies (using natural logarithm)
        h_true = entropy(y_true_counts + 1e-10)  # Add small epsilon to avoid log(0)
        h_pred = entropy(y_pred_counts + 1e-10)
        
        # Normalize MI using geometric mean normalization
        nmi_value = mi_value / np.sqrt(h_true * h_pred) if h_true > 0 and h_pred > 0 else 0
        nmi_percentage = nmi_value * 100.0
        
        return mi_value, nmi_percentage
        
    except Exception as e:
        print(f"⚠️ Error calculating MI: {e}")
        return 0.0, 0.0

def calculate_kl_divergence(y_true, y_pred_proba, num_classes):
    """
    Calculate KL Divergence and Normalized KL metrics
    
    Args:
        y_true: True labels (integer encoded)
        y_pred_proba: Predicted probabilities (N x C matrix)
        num_classes: Number of classes
    
    Returns:
        kl_avg: Average KL divergence across all samples
        kl_normalized: KL divergence normalized by theoretical maximum
        reverse_kl_percentage: Reverse KL as percentage (0-100%, higher is better)
    """
    try:
        # Convert true labels to one-hot encoding
        y_true_one_hot = np.zeros((len(y_true), num_classes))
        y_true_one_hot[np.arange(len(y_true)), y_true] = 1
        
        # Calculate KL divergence for each sample
        kl_values = []
        for i in range(len(y_true)):
            true_dist = y_true_one_hot[i]
            pred_dist = y_pred_proba[i]
            
            # Add small epsilon to avoid log(0)
            pred_dist = pred_dist + 1e-10
            pred_dist = pred_dist / np.sum(pred_dist)  # Renormalize
            
            # Calculate KL divergence: KL(P||Q) = Σ P(x) * log(P(x) / Q(x))
            kl_value = 0.0
            for j in range(num_classes):
                if true_dist[j] > 0:  # Only consider non-zero entries in true distribution
                    kl_value += true_dist[j] * np.log(true_dist[j] / pred_dist[j])
            
            kl_values.append(kl_value)
        
        # Average KL divergence
        kl_avg = np.mean(kl_values)
        
        # Theoretical maximum KL divergence (uniform prediction when true is one-hot)
        kl_max_theoretical = np.log(num_classes)
        
        # Practical maximum (use observed max or theoretical max, whichever is larger)
        kl_max_practical = max(kl_max_theoretical, np.max(kl_values))
        
        # Normalize KL divergence
        kl_normalized = min(1.0, kl_avg / kl_max_practical)
        
        # Reverse KL percentage (higher is better)
        reverse_kl_percentage = (1.0 - kl_normalized) * 100.0
        
        return kl_avg, kl_normalized, reverse_kl_percentage
        
    except Exception as e:
        print(f"⚠️ Error calculating KL divergence: {e}")
        return 0.0, 1.0, 0.0

class Game2ImprovedGyroAnalyzer:
    """
    Improved Game-2 analyzer using top placement-agnostic gyroscope features
    Based on comprehensive transferability analysis results
    """
    
    def __init__(self, experiment_direction="stm_to_motionsense"):
        self.experiment_direction = experiment_direction
        
        # Dataset paths
        self.stm_path = str(get_stm_root())
        self.motionsense_path = str(get_motionsense_root())
        
        # Common attributes (same as Game-1)
        self.activities = ['walking', 'jogging', 'standing', 'sitting', 'upstairs', 'downstairs']
        self.stm_placements = ['left-ankle', 'left-wrist', 'right-ankle', 'right-pocket', 'right-wrist']
        
        # Improved features description
        self.feature_description = "Top 15 Placement-Agnostic Gyroscope Features (0.013-0.026 transferability)"
        
        if experiment_direction == "stm_to_motionsense":
            self.results_dir = "results/game-2-gyro-improved/STM_MultiPlacement_to_MotionSense"
            self.train_dataset = "STM (Individual + Combined Placements)"
            self.test_dataset = "MotionSense (Smartphone)"
            self.experiment_title = "Game-2 Improved Gyro Transfer: STM → MotionSense"
            self.experiment_description = "Train on STM placements, test cross-dataset on MotionSense (IMPROVED)"
        elif experiment_direction == "motionsense_to_stm":
            self.results_dir = "results/game-2-gyro-improved/MotionSense_to_STM_Placements"
            self.train_dataset = "MotionSense (Smartphone)" 
            self.test_dataset = "STM (Individual Placements)"
            self.experiment_title = "Game-2 Improved Gyro Transfer: MotionSense → STM Placements"
            self.experiment_description = "Train once on MotionSense, test on each STM placement (IMPROVED)"
        else:
            raise ValueError("experiment_direction must be 'stm_to_motionsense' or 'motionsense_to_stm'")
        
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Add combined placement for training on all placements
        self.all_placements = self.stm_placements + ['combined']
        
        print(f"🎯 Target: Cross-sensor cross-dataset adversarial HAR (Game-2 Improved)")
        print(f"📊 Activities: {len(self.activities)} common activities")
        print(f"🎭 STM Placements: {len(self.stm_placements)} placements")
        print(f"🔑 Features: {self.feature_description}")
        print(f"📍 Direction: {self.train_dataset} → {self.test_dataset}")
        print(f"📁 Results: {self.results_dir}")
        print()
        
        # Activity mappings for consistent naming
        self.activity_mapping = {
            'activity_walking': 'walking',
            'activity_jogging': 'jogging',
            'activity_standing': 'standing', 
            'activity_sitting': 'sitting',
            'activity_upstairs': 'upstairs',
            'activity_downstairs': 'downstairs'
        }

    def extract_improved_placement_agnostic_gyro_features(self, data_df):
        """
        Extract TOP 15 placement-agnostic gyroscope features based on transferability analysis
        
        Only the highest transferability features (0.013-0.026):
        Focus on magnitude-based statistical and temporal features
        """
        features = {}
        
        if not all(col in data_df.columns for col in ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']):
            return features
        
        x = data_df['gyro_x[mdps]'].values
        y = data_df['gyro_y[mdps]'].values
        z = data_df['gyro_z[mdps]'].values
        
        # Calculate magnitude (PLACEMENT-AGNOSTIC: independent of sensor orientation)
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        
        if len(magnitude) == 0:
            return features
        
        # ============ TOP 15 PLACEMENT-AGNOSTIC FEATURES ============
        
        # 1. Total rotation intensity (0.026 transferability) - BEST!
        features['gyro_total_rotation_intensity'] = np.sum(magnitude)
        
        # 2. Average rotation intensity (0.026 transferability) - BEST!
        features['gyro_avg_rotation_intensity'] = np.mean(magnitude)
        
        # 3. Magnitude mean (0.025 transferability)
        features['gyro_magnitude_mean'] = np.mean(magnitude)
        
        # 4. Magnitude Q25 (0.023 transferability)
        features['gyro_magnitude_q25'] = np.percentile(magnitude, 25)
        
        # 5. Magnitude median (0.020 transferability)
        features['gyro_magnitude_median'] = np.median(magnitude)
        
        # 6. Magnitude RMS (0.020 transferability)
        features['gyro_magnitude_rms'] = np.sqrt(np.mean(magnitude ** 2))
        
        # 7. Signal Magnitude Area (0.019 transferability)
        features['gyro_magnitude_sma'] = np.sum(np.abs(magnitude))
        
        # 8. Magnitude Q10 (0.018 transferability)
        features['gyro_magnitude_q10'] = np.percentile(magnitude, 10)
        
        # 9. Magnitude Q75 (0.018 transferability)
        features['gyro_magnitude_q75'] = np.percentile(magnitude, 75)
        
        # 10. Magnitude change rate (0.015 transferability)
        if len(magnitude) > 1:
            magnitude_changes = np.abs(np.diff(magnitude))
            features['gyro_magnitude_change_rate'] = np.mean(magnitude_changes)
        else:
            features['gyro_magnitude_change_rate'] = 0
        
        # 11. Magnitude derivative std (0.014 transferability)
        if len(magnitude) > 1:
            magnitude_derivative = np.diff(magnitude)
            features['gyro_magnitude_derivative_std'] = np.std(magnitude_derivative)
        else:
            features['gyro_magnitude_derivative_std'] = 0
        
        # 12. Smoothness (0.013 transferability)
        if len(magnitude) > 2:
            second_derivative = np.diff(magnitude, n=2)
            features['gyro_smoothness'] = -np.sum(second_derivative ** 2)  # Negative because smoother = less 2nd derivative
        else:
            features['gyro_smoothness'] = 0
        
        # 13. Magnitude Q90 (0.013 transferability)
        features['gyro_magnitude_q90'] = np.percentile(magnitude, 90)
        
        # 14. Magnitude std (0.013 transferability)
        features['gyro_magnitude_std'] = np.std(magnitude)
        
        # 15. Magnitude min (0.013 transferability)
        features['gyro_magnitude_min'] = np.min(magnitude)
        
        return features

    def load_stm_placement_data(self, placement):
        """Load STM data for a specific placement - handles both structures including stairs"""
        print(f"📊 Loading STM {placement} data...")
        
        placement_data = []
        
        for user_folder in os.listdir(self.stm_path):
            if user_folder.startswith('User '):
                user_path = os.path.join(self.stm_path, user_folder)
                processed_path = os.path.join(user_path, 'Processed')
                
                if os.path.exists(processed_path):
                    # Load data from each activity folder
                    for activity_folder in os.listdir(processed_path):
                        activity_path = os.path.join(processed_path, activity_folder)
                        
                        if os.path.isdir(activity_path):
                            # Map activity folder names to standard names
                            activity_mapping = {
                                'Walking': 'walking',
                                'Jogging': 'jogging', 
                                'Standing': 'standing',
                                'Sitting': 'sitting',
                                'Upstairs': 'upstairs',
                                'Downstairs': 'downstairs'
                            }
                            
                            if activity_folder in activity_mapping:
                                activity_name = activity_mapping[activity_folder]
                                
                                # Check for stairs structure: Activity/Placement/multiple_files.csv
                                if activity_folder in ['Upstairs', 'Downstairs']:
                                    placement_dir = os.path.join(activity_path, placement)
                                    if os.path.exists(placement_dir):
                                        # Load all CSV files in the placement directory
                                        csv_files = [f for f in os.listdir(placement_dir) if f.endswith('.csv')]
                                        for csv_file in csv_files:
                                            file_path = os.path.join(placement_dir, csv_file)
                                            try:
                                                df = pd.read_csv(file_path)
                                                if all(col in df.columns for col in ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']):
                                                    df['activity'] = activity_name
                                                    df['user_id'] = user_folder
                                                    df['placement'] = placement
                                                    placement_data.append(df)
                                            except Exception as e:
                                                print(f"⚠️ Error loading {file_path}: {e}")
                                
                                # Check for regular structure: Activity/placement.csv
                                else:
                                    placement_file = os.path.join(activity_path, f"{placement}.csv")
                                    if os.path.exists(placement_file):
                                        try:
                                            df = pd.read_csv(placement_file)
                                            # Check for gyroscope columns
                                            if all(col in df.columns for col in ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']):
                                                df['activity'] = activity_name
                                                df['user_id'] = user_folder
                                                df['placement'] = placement
                                                placement_data.append(df)
                                        except Exception as e:
                                            print(f"⚠️ Error loading {placement_file}: {e}")
        
        if placement_data:
            combined_data = pd.concat(placement_data, ignore_index=True)
            print(f"✅ {len(combined_data)} samples from {len(placement_data)} files")
            
            # Print activity breakdown for verification
            activity_counts = combined_data['activity'].value_counts()
            print(f"   📊 Activities: {dict(activity_counts)}")
            
            return combined_data
        else:
            print(f"❌ No data found for {placement}")
            return pd.DataFrame()

    def load_stm_combined_data(self):
        """
        Load gyroscope data from ALL STM placements combined
        """
        print(f"📊 Loading STM combined (all placements) data...", end="")
        
        all_data = []
        placement_counts = {placement: 0 for placement in self.stm_placements}
        
        for placement in self.stm_placements:
            placement_df = self.load_stm_placement_data(placement)
            if len(placement_df) > 0:
                placement_counts[placement] = len(placement_df)
                all_data.append(placement_df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f" ✅ {len(combined_df)} total samples")
            
            # Print breakdown by placement
            print(f"   📍 Placement breakdown:")
            for placement, count in placement_counts.items():
                if count > 0:
                    print(f"      {placement}: {count} samples")
            
            return combined_df
        else:
            print(f" ❌ No data found")
            return pd.DataFrame()

    def load_motionsense_data(self):
        """Load MotionSense smartphone gyroscope data"""
        print("📊 Loading MotionSense smartphone gyroscope data...")
        
        ms_data = []
        
        for subject_folder in os.listdir(self.motionsense_path):
            if subject_folder.startswith('subject_'):
                subject_path = os.path.join(self.motionsense_path, subject_folder)
                
                for activity_folder in os.listdir(subject_path):
                    if activity_folder in self.activity_mapping:
                        activity_name = self.activity_mapping[activity_folder]
                        data_file = os.path.join(subject_path, activity_folder, 'data.csv')
                        
                        if os.path.exists(data_file):
                            try:
                                df = pd.read_csv(data_file)
                                # Check for gyroscope columns
                                if all(col in df.columns for col in ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']):
                                    df['activity'] = activity_name
                                    df['user_id'] = subject_folder
                                    df['placement'] = 'smartphone'
                                    ms_data.append(df)
                            except Exception as e:
                                print(f"⚠️ Error loading {data_file}: {e}")
        
        if ms_data:
            combined_data = pd.concat(ms_data, ignore_index=True)
            print(f"✅ {len(combined_data)} samples from {len(ms_data)} files")
            
            # Print activity breakdown
            activity_counts = combined_data['activity'].value_counts()
            print(f"   📊 Activities: {dict(activity_counts)}")
            
            return combined_data
        else:
            print("❌ No MotionSense gyroscope data found")
            return pd.DataFrame()

    def create_dataset_from_dataframe(self, df, label='unknown'):
        """Create feature matrix from dataframe using improved placement-agnostic gyroscope features"""
        if len(df) == 0:
            return None, None, None
        
        X = []
        y = []
        metadata = []
        
        # Group by user and activity for windowing
        grouped = df.groupby(['user_id', 'activity'])
        
        for (user_id, activity), group in grouped:
            if len(group) == 0:
                continue
            
            # Create sliding windows (50 samples with 50% overlap)
            window_size = 50
            step_size = 25
            
            for i in range(0, len(group) - window_size + 1, step_size):
                window = group.iloc[i:i+window_size]
                
                # Extract improved placement-agnostic gyroscope features
                features = self.extract_improved_placement_agnostic_gyro_features(window)
                
                if len(features) > 0:
                    X.append(list(features.values()))
                    y.append(activity)
                    metadata.append({
                        'user': user_id,
                        'activity': activity,
                        'window_start': i,
                        'window_end': i + window_size,
                        'features': list(features.keys())
                    })
        
        if len(X) > 0:
            X_df = pd.DataFrame(X, columns=list(features.keys()))
            print(f"   📈 {label} dataset: {len(X)} samples, {len(features)} features")
            return X_df, np.array(y), metadata
        else:
            print(f"   ❌ No features extracted for {label}")
            return None, None, None

    def run_cross_sensor_experiment(self):
        """Run the appropriate cross-sensor experiment based on direction"""
        if self.experiment_direction == "stm_to_motionsense":
            return self.run_stm_to_motionsense()
        else:
            return self.run_motionsense_to_stm()

    def run_stm_to_motionsense(self):
        """STM → MotionSense experiment with improved placement-agnostic features"""
        print("📊 CASE 1: STM (Multi-Placement) → MotionSense (IMPROVED FEATURES)")
        print("="*70)
        
        # Load MotionSense test data
        ms_data = self.load_motionsense_data()
        if ms_data.empty:
            print("❌ Failed to load MotionSense data")
            return None
        
        ms_X, ms_y, ms_metadata = self.create_dataset_from_dataframe(ms_data, 'MotionSense')
        if ms_X is None:
            print("❌ Failed to create MotionSense dataset")
            return None
        
        results = []
        
        # First, determine common activities between STM and MotionSense once
        sample_stm_data = self.load_stm_placement_data(self.stm_placements[0])
        if not sample_stm_data.empty:
            sample_stm_X, sample_stm_y, _ = self.create_dataset_from_dataframe(sample_stm_data, 'sample')
            if sample_stm_X is not None:
                global_common_activities = list(set(sample_stm_y) & set(ms_y))
                print(f"📊 Global common activities: {global_common_activities}")
            else:
                print("❌ Could not determine common activities")
                return None
        else:
            print("❌ Could not load sample STM data")
            return None
        
        # Test each STM placement
        for placement in self.stm_placements:
            print(f"\n📍 Training on STM {placement}...")
            
            # Load STM placement data
            stm_data = self.load_stm_placement_data(placement)
            if stm_data.empty:
                continue
            
            stm_X, stm_y, stm_metadata = self.create_dataset_from_dataframe(stm_data, f'STM_{placement}')
            if stm_X is None:
                continue
            
            # Find common features (activities are already determined globally)
            common_activities = global_common_activities
            common_features = list(set(stm_X.columns) & set(ms_X.columns))
            
            if len(common_activities) < 2 or len(common_features) < 5:
                print(f"   ⚠️  Insufficient overlap: {len(common_activities)} activities, {len(common_features)} features")
                continue
            
            print(f"   📊 Common activities: {common_activities}")
            print(f"   📊 Common features: {len(common_features)}")
            
            # Filter to common activities and features
            stm_mask = np.isin(stm_y, common_activities)
            ms_mask = np.isin(ms_y, common_activities)
            
            stm_X_filtered = stm_X.loc[stm_mask, common_features]
            stm_y_filtered = stm_y[stm_mask]
            ms_X_filtered = ms_X.loc[ms_mask, common_features]
            ms_y_filtered = ms_y[ms_mask]
            
            # Train and test
            scaler = StandardScaler()
            stm_X_scaled = scaler.fit_transform(stm_X_filtered)
            ms_X_scaled = scaler.transform(ms_X_filtered)
            
            # Train Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(stm_X_scaled, stm_y_filtered)
            
            # Test on MotionSense
            ms_pred = rf.predict(ms_X_scaled)
            ms_pred_proba = rf.predict_proba(ms_X_scaled)
            
            # Standard metrics
            accuracy = accuracy_score(ms_y_filtered, ms_pred)
            f1 = f1_score(ms_y_filtered, ms_pred, average='weighted')
            precision = precision_score(ms_y_filtered, ms_pred, average='weighted', zero_division=0)
            recall = recall_score(ms_y_filtered, ms_pred, average='weighted', zero_division=0)
            
            # Calculate information-theoretic metrics
            unique_labels = sorted(list(set(ms_y_filtered)))
            le = LabelEncoder()
            ms_y_encoded = le.fit_transform(ms_y_filtered)
            ms_pred_encoded = le.transform(ms_pred)
            
            mi_value, nmi_percentage = calculate_mutual_information(ms_y_encoded, ms_pred_encoded)
            kl_avg, kl_normalized, reverse_kl_percentage = calculate_kl_divergence(
                ms_y_encoded, ms_pred_proba, len(unique_labels)
            )
            vulnerability = calculate_vulnerability(ms_pred_proba)
            
            # Generate confusion matrix
            cm = confusion_matrix(ms_y_filtered, ms_pred)
            
            # Determine if this is same-sensor or cross-sensor
            is_same_sensor = (placement == 'right-pocket')  # Assuming smartphone ≈ right-pocket
            
            result = {
                'placement': placement,
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'mutual_information': mi_value,
                'nmi_percentage': nmi_percentage,
                'kl_divergence': kl_avg,
                'kl_normalized': kl_normalized,
                'reverse_kl_percentage': reverse_kl_percentage,
                'vulnerability': vulnerability,
                'n_train': len(stm_X_scaled),
                'n_test': len(ms_X_scaled),
                'n_features': len(common_features),
                'n_classes': len(unique_labels),
                'is_same_sensor': is_same_sensor,
                'activities': common_activities,
                'confusion_matrix': cm.tolist(),
                'y_true': ms_y_filtered.tolist(),
                'y_pred': ms_pred.tolist(),
                'y_pred_proba': ms_pred_proba.tolist(),
                'feature_importance': rf.feature_importances_.tolist(),
                'feature_names': common_features
            }
            
            results.append(result)
            
            print(f"   🤖 Training: {len(stm_X_scaled)} samples, {len(common_features)} features")
            print(f"   📊 Testing: {len(ms_X_scaled)} samples")
            print(f"   🎯 Accuracy: {accuracy:.3f}")
            print(f"   📈 F1-Score: {f1:.3f}")
            print(f"   🧠 NMI: {nmi_percentage:.1f}%")
            print(f"   🎯 Reverse KL: {reverse_kl_percentage:.1f}%")
            print(f"   🔐 Vulnerability: {vulnerability:.3f}")
            print(f"   🔍 Sensor Type: {'Same-Sensor' if is_same_sensor else 'Cross-Sensor'}")
        
        # COMBINED TRAINING - Train on all STM placements together
        print(f"\n📍 Training on STM COMBINED (all placements)...")
        
        stm_combined_data = self.load_stm_combined_data()
        if not stm_combined_data.empty:
            stm_combined_X, stm_combined_y, stm_combined_metadata = self.create_dataset_from_dataframe(stm_combined_data, 'STM_combined')
            if stm_combined_X is not None:
                # Find common features (activities are already determined globally)
                common_features = [f for f in stm_combined_X.columns if f in ms_X.columns]
                common_activities = global_common_activities
                
                if common_features and common_activities:
                    # Filter data
                    stm_mask = np.isin(stm_combined_y, common_activities)
                    ms_mask = np.isin(ms_y, common_activities)
                    
                    stm_combined_X_filtered = stm_combined_X.loc[stm_mask, common_features]
                    stm_combined_y_filtered = stm_combined_y[stm_mask]
                    ms_X_filtered = ms_X.loc[ms_mask, common_features]
                    ms_y_filtered = ms_y[ms_mask]
                    
                    # Train and test
                    scaler = StandardScaler()
                    stm_combined_X_scaled = scaler.fit_transform(stm_combined_X_filtered)
                    ms_X_scaled = scaler.transform(ms_X_filtered)
                    
                    # Train Random Forest
                    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    rf.fit(stm_combined_X_scaled, stm_combined_y_filtered)
                    
                    # Test on MotionSense
                    ms_pred = rf.predict(ms_X_scaled)
                    ms_pred_proba = rf.predict_proba(ms_X_scaled)
                    
                    # Standard metrics
                    accuracy = accuracy_score(ms_y_filtered, ms_pred)
                    f1 = f1_score(ms_y_filtered, ms_pred, average='weighted')
                    precision = precision_score(ms_y_filtered, ms_pred, average='weighted', zero_division=0)
                    recall = recall_score(ms_y_filtered, ms_pred, average='weighted', zero_division=0)
                    
                    # Calculate information-theoretic metrics
                    unique_labels = sorted(list(set(ms_y_filtered)))
                    le = LabelEncoder()
                    ms_y_encoded = le.fit_transform(ms_y_filtered)
                    ms_pred_encoded = le.transform(ms_pred)
                    
                    mi_value, nmi_percentage = calculate_mutual_information(ms_y_encoded, ms_pred_encoded)
                    kl_avg, kl_normalized, reverse_kl_percentage = calculate_kl_divergence(
                        ms_y_encoded, ms_pred_proba, len(unique_labels)
                    )
                    vulnerability = calculate_vulnerability(ms_pred_proba)
                    
                    # Generate confusion matrix
                    cm = confusion_matrix(ms_y_filtered, ms_pred)
                    
                    result = {
                        'placement': 'combined',
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'precision': precision,
                        'recall': recall,
                        'mutual_information': mi_value,
                        'nmi_percentage': nmi_percentage,
                        'kl_divergence': kl_avg,
                        'kl_normalized': kl_normalized,
                        'reverse_kl_percentage': reverse_kl_percentage,
                        'vulnerability': vulnerability,
                        'n_train': len(stm_combined_X_scaled),
                        'n_test': len(ms_X_scaled),
                        'n_features': len(common_features),
                        'n_classes': len(unique_labels),
                        'is_same_sensor': False,  # Combined is always cross-sensor
                        'activities': common_activities,
                        'confusion_matrix': cm.tolist(),
                        'y_true': ms_y_filtered.tolist(),
                        'y_pred': ms_pred.tolist(),
                        'y_pred_proba': ms_pred_proba.tolist(),
                        'feature_importance': rf.feature_importances_.tolist(),
                        'feature_names': common_features
                    }
                    
                    results.append(result)
                    
                    print(f"   🤖 Training: {len(stm_combined_X_scaled)} samples, {len(common_features)} features")
                    print(f"   📊 Testing: {len(ms_X_scaled)} samples")
                    print(f"   🎯 Accuracy: {accuracy:.3f}")
                    print(f"   📈 F1-Score: {f1:.3f}")
                    print(f"   🧠 NMI: {nmi_percentage:.1f}%")
                    print(f"   🎯 Reverse KL: {reverse_kl_percentage:.1f}%")
                    print(f"   🔐 Vulnerability: {vulnerability:.3f}")
                    print(f"   🔍 Sensor Type: Cross-Sensor (Combined)")
        
        # Analysis and visualization
        if results:
            self.analyze_cross_sensor_results(results)
            self.create_comprehensive_plots(results)
            self.save_experiment_results(results)
            
        return results
    
    def run_motionsense_to_stm(self):
        """MotionSense → STM experiment with improved placement-agnostic features"""
        print("📊 CASE 2: MotionSense (Smartphone) → STM (Multi-Placement) [IMPROVED FEATURES]")
        print("="*70)
        
        # Load MotionSense training data
        ms_data = self.load_motionsense_data()
        if ms_data.empty:
            print("❌ Failed to load MotionSense data")
            return None
        
        ms_X, ms_y, ms_metadata = self.create_dataset_from_dataframe(ms_data, 'MotionSense')
        if ms_X is None:
            print("❌ Failed to create MotionSense dataset")
            return None
        
        results = []
        
        # First, determine common activities between MotionSense and STM once
        sample_stm_data = self.load_stm_placement_data(self.stm_placements[0])
        if not sample_stm_data.empty:
            sample_stm_X, sample_stm_y, _ = self.create_dataset_from_dataframe(sample_stm_data, 'sample')
            if sample_stm_X is not None:
                global_common_activities = list(set(sample_stm_y) & set(ms_y))
                print(f"📊 Global common activities: {global_common_activities}")
            else:
                print("❌ Could not determine common activities")
                return None
        else:
            print("❌ Could not load sample STM data")
            return None
        
        # Test on each STM placement
        for placement in self.stm_placements:
            print(f"\n📍 Testing on STM {placement}...")
            
            # Load STM placement data
            stm_data = self.load_stm_placement_data(placement)
            if stm_data.empty:
                continue
            
            stm_X, stm_y, stm_metadata = self.create_dataset_from_dataframe(stm_data, f'STM_{placement}')
            if stm_X is None:
                continue
            
            # Find common features (activities are already determined globally) 
            common_activities = global_common_activities
            common_features = list(set(stm_X.columns) & set(ms_X.columns))
            
            if len(common_activities) < 2 or len(common_features) < 5:
                print(f"   ⚠️  Insufficient overlap: {len(common_activities)} activities, {len(common_features)} features")
                continue
            
            print(f"   📊 Common activities: {common_activities}")
            print(f"   📊 Common features: {len(common_features)}")
            
            # Filter to common activities and features
            ms_mask = np.isin(ms_y, common_activities)
            stm_mask = np.isin(stm_y, common_activities)
            
            ms_X_filtered = ms_X.loc[ms_mask, common_features]
            ms_y_filtered = ms_y[ms_mask]
            stm_X_filtered = stm_X.loc[stm_mask, common_features]
            stm_y_filtered = stm_y[stm_mask]
            
            # Train and test
            scaler = StandardScaler()
            ms_X_scaled = scaler.fit_transform(ms_X_filtered)
            stm_X_scaled = scaler.transform(stm_X_filtered)
            
            # Train Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(ms_X_scaled, ms_y_filtered)
            
            # Test on STM placement
            stm_pred = rf.predict(stm_X_scaled)
            stm_pred_proba = rf.predict_proba(stm_X_scaled)
            
            # Standard metrics
            accuracy = accuracy_score(stm_y_filtered, stm_pred)
            f1 = f1_score(stm_y_filtered, stm_pred, average='weighted')
            precision = precision_score(stm_y_filtered, stm_pred, average='weighted', zero_division=0)
            recall = recall_score(stm_y_filtered, stm_pred, average='weighted', zero_division=0)
            
            # Calculate information-theoretic metrics
            unique_labels = sorted(list(set(stm_y_filtered)))
            le = LabelEncoder()
            stm_y_encoded = le.fit_transform(stm_y_filtered)
            stm_pred_encoded = le.transform(stm_pred)
            
            mi_value, nmi_percentage = calculate_mutual_information(stm_y_encoded, stm_pred_encoded)
            kl_avg, kl_normalized, reverse_kl_percentage = calculate_kl_divergence(
                stm_y_encoded, stm_pred_proba, len(unique_labels)
            )
            vulnerability = calculate_vulnerability(stm_pred_proba)
            
            # Generate confusion matrix
            cm = confusion_matrix(stm_y_filtered, stm_pred)
            
            # Determine if this is same-sensor or cross-sensor
            is_same_sensor = (placement == 'right-pocket')  # Assuming smartphone ≈ right-pocket
            
            result = {
                'placement': placement,
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'mutual_information': mi_value,
                'nmi_percentage': nmi_percentage,
                'kl_divergence': kl_avg,
                'kl_normalized': kl_normalized,
                'reverse_kl_percentage': reverse_kl_percentage,
                'vulnerability': vulnerability,
                'n_train': len(ms_X_scaled),
                'n_test': len(stm_X_scaled),
                'n_features': len(common_features),
                'n_classes': len(unique_labels),
                'is_same_sensor': is_same_sensor,
                'activities': common_activities,
                'confusion_matrix': cm.tolist(),
                'y_true': stm_y_filtered.tolist(),
                'y_pred': stm_pred.tolist(),
                'y_pred_proba': stm_pred_proba.tolist(),
                'feature_importance': rf.feature_importances_.tolist(),
                'feature_names': common_features
            }
            
            results.append(result)
            
            print(f"   🤖 Training: {len(ms_X_scaled)} samples, {len(common_features)} features")
            print(f"   📊 Testing: {len(stm_X_scaled)} samples")
            print(f"   🎯 Accuracy: {accuracy:.3f}")
            print(f"   📈 F1-Score: {f1:.3f}")
            print(f"   🧠 NMI: {nmi_percentage:.1f}%")
            print(f"   🎯 Reverse KL: {reverse_kl_percentage:.1f}%")
            print(f"   🔐 Vulnerability: {vulnerability:.3f}")
            print(f"   🔍 Sensor Type: {'Same-Sensor' if is_same_sensor else 'Cross-Sensor'}")
        
        # COMBINED TESTING - Test on all STM placements together
        print(f"\n📍 Testing on STM COMBINED (all placements)...")
        
        stm_combined_data = self.load_stm_combined_data()
        if not stm_combined_data.empty:
            stm_combined_X, stm_combined_y, stm_combined_metadata = self.create_dataset_from_dataframe(stm_combined_data, 'STM_combined')
            if stm_combined_X is not None:
                # Find common features (activities are already determined globally)
                common_features = [f for f in ms_X.columns if f in stm_combined_X.columns]
                common_activities = global_common_activities
                
                if common_features and common_activities:
                    # Filter data
                    ms_mask = np.isin(ms_y, common_activities)
                    stm_mask = np.isin(stm_combined_y, common_activities)
                    
                    ms_X_filtered = ms_X.loc[ms_mask, common_features]
                    ms_y_filtered = ms_y[ms_mask]
                    stm_combined_X_filtered = stm_combined_X.loc[stm_mask, common_features]
                    stm_combined_y_filtered = stm_combined_y[stm_mask]
                    
                    # Train and test
                    scaler = StandardScaler()
                    ms_X_scaled = scaler.fit_transform(ms_X_filtered)
                    stm_combined_X_scaled = scaler.transform(stm_combined_X_filtered)
                    
                    # Train Random Forest
                    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    rf.fit(ms_X_scaled, ms_y_filtered)
                    
                    # Test on STM combined
                    stm_pred = rf.predict(stm_combined_X_scaled)
                    stm_pred_proba = rf.predict_proba(stm_combined_X_scaled)
                    
                    # Standard metrics
                    accuracy = accuracy_score(stm_combined_y_filtered, stm_pred)
                    f1 = f1_score(stm_combined_y_filtered, stm_pred, average='weighted')
                    precision = precision_score(stm_combined_y_filtered, stm_pred, average='weighted', zero_division=0)
                    recall = recall_score(stm_combined_y_filtered, stm_pred, average='weighted', zero_division=0)
                    
                    # Calculate information-theoretic metrics
                    unique_labels = sorted(list(set(stm_combined_y_filtered)))
                    le = LabelEncoder()
                    stm_y_encoded = le.fit_transform(stm_combined_y_filtered)
                    stm_pred_encoded = le.transform(stm_pred)
                    
                    mi_value, nmi_percentage = calculate_mutual_information(stm_y_encoded, stm_pred_encoded)
                    kl_avg, kl_normalized, reverse_kl_percentage = calculate_kl_divergence(
                        stm_y_encoded, stm_pred_proba, len(unique_labels)
                    )
                    vulnerability = calculate_vulnerability(stm_pred_proba)
                    
                    # Generate confusion matrix
                    cm = confusion_matrix(stm_combined_y_filtered, stm_pred)
                    
                    result = {
                        'placement': 'combined',
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'precision': precision,
                        'recall': recall,
                        'mutual_information': mi_value,
                        'nmi_percentage': nmi_percentage,
                        'kl_divergence': kl_avg,
                        'kl_normalized': kl_normalized,
                        'reverse_kl_percentage': reverse_kl_percentage,
                        'vulnerability': vulnerability,
                        'n_train': len(ms_X_scaled),
                        'n_test': len(stm_combined_X_scaled),
                        'n_features': len(common_features),
                        'n_classes': len(unique_labels),
                        'is_same_sensor': False,  # Combined is always cross-sensor
                        'activities': common_activities,
                        'confusion_matrix': cm.tolist(),
                        'y_true': stm_combined_y_filtered.tolist(),
                        'y_pred': stm_pred.tolist(),
                        'y_pred_proba': stm_pred_proba.tolist(),
                        'feature_importance': rf.feature_importances_.tolist(),
                        'feature_names': common_features
                    }
                    
                    results.append(result)
                    
                    print(f"   🤖 Training: {len(ms_X_scaled)} samples, {len(common_features)} features")
                    print(f"   📊 Testing: {len(stm_combined_X_scaled)} samples")
                    print(f"   🎯 Accuracy: {accuracy:.3f}")
                    print(f"   📈 F1-Score: {f1:.3f}")
                    print(f"   🧠 NMI: {nmi_percentage:.1f}%")
                    print(f"   🎯 Reverse KL: {reverse_kl_percentage:.1f}%")
                    print(f"   🔐 Vulnerability: {vulnerability:.3f}")
                    print(f"   📈 F1-Score: {f1:.3f}")
                    print(f"   🔍 Sensor Type: Cross-Sensor (Combined)")
        
        # Analysis and visualization
        if results:
            self.analyze_cross_sensor_results(results)
            self.create_comprehensive_plots(results)
            self.save_experiment_results(results)
            
        return results
    
    def analyze_cross_sensor_results(self, results):
        """Analyze cross-sensor vs same-sensor performance with MI and KL metrics"""
        print("\n🔍 CROSS-SENSOR ANALYSIS (IMPROVED FEATURES):")
        print("="*50)
        
        df = pd.DataFrame(results)
        
        same_sensor = df[df['is_same_sensor'] == True]
        cross_sensor = df[df['is_same_sensor'] == False]
        
        if len(same_sensor) > 0:
            print(f"📱 Same-Sensor (right-pocket):")
            print(f"   Accuracy: {same_sensor['accuracy'].mean():.3f} ± {same_sensor['accuracy'].std():.3f}")
            print(f"   NMI: {same_sensor['nmi_percentage'].mean():.1f}% ± {same_sensor['nmi_percentage'].std():.1f}%")
            print(f"   Reverse KL: {same_sensor['reverse_kl_percentage'].mean():.1f}% ± {same_sensor['reverse_kl_percentage'].std():.1f}%")
            print(f"   Vulnerability: {same_sensor['vulnerability'].mean():.3f} ± {same_sensor['vulnerability'].std():.3f}")
        
        if len(cross_sensor) > 0:
            print(f"🔄 Cross-Sensor (others):")
            print(f"   Accuracy: {cross_sensor['accuracy'].mean():.3f} ± {cross_sensor['accuracy'].std():.3f}")
            print(f"   NMI: {cross_sensor['nmi_percentage'].mean():.1f}% ± {cross_sensor['nmi_percentage'].std():.1f}%")
            print(f"   Reverse KL: {cross_sensor['reverse_kl_percentage'].mean():.1f}% ± {cross_sensor['reverse_kl_percentage'].std():.1f}%")
            print(f"   Vulnerability: {cross_sensor['vulnerability'].mean():.3f} ± {cross_sensor['vulnerability'].std():.3f}")
        
        # Best and worst performers
        best_placement = df.loc[df['accuracy'].idxmax()]
        worst_placement = df.loc[df['accuracy'].idxmin()]
        
        print(f"\n🏆 Best: {best_placement['placement']} - Acc: {best_placement['accuracy']:.3f}, NMI: {best_placement['nmi_percentage']:.1f}%")
        print(f"📉 Worst: {worst_placement['placement']} - Acc: {worst_placement['accuracy']:.3f}, NMI: {worst_placement['nmi_percentage']:.1f}%")
        
        # Overall performance
        print(f"\n📊 Overall Performance:")
        print(f"   Accuracy: {df['accuracy'].mean():.3f} ± {df['accuracy'].std():.3f}")
        print(f"   NMI: {df['nmi_percentage'].mean():.1f}% ± {df['nmi_percentage'].std():.1f}%")
        print(f"   Reverse KL: {df['reverse_kl_percentage'].mean():.1f}% ± {df['reverse_kl_percentage'].std():.1f}%")
        print(f"   Vulnerability: {df['vulnerability'].mean():.3f} ± {df['vulnerability'].std():.3f}")
        
        # Quality assessment
        print(f"\n🎯 Information-Theoretic Quality Assessment:")
        avg_nmi = df['nmi_percentage'].mean()
        avg_kl = df['reverse_kl_percentage'].mean()
        
        if avg_nmi >= 90:
            print(f"   📊 NMI Quality: EXCELLENT ({avg_nmi:.1f}%)")
        elif avg_nmi >= 80:
            print(f"   📊 NMI Quality: GOOD ({avg_nmi:.1f}%)")
        elif avg_nmi >= 70:
            print(f"   📊 NMI Quality: MODERATE ({avg_nmi:.1f}%)")
        else:
            print(f"   📊 NMI Quality: NEEDS IMPROVEMENT ({avg_nmi:.1f}%)")
        
        if avg_kl >= 90:
            print(f"   🎯 Prediction Quality: EXCELLENT ({avg_kl:.1f}%)")
        elif avg_kl >= 80:
            print(f"   🎯 Prediction Quality: GOOD ({avg_kl:.1f}%)")
        elif avg_kl >= 70:
            print(f"   🎯 Prediction Quality: MODERATE ({avg_kl:.1f}%)")
        else:
            print(f"   🎯 Prediction Quality: NEEDS IMPROVEMENT ({avg_kl:.1f}%)")
        
        # Compare with baseline (if available)
        print(f"\n📈 IMPROVEMENT ANALYSIS:")
        print(f"🎯 Current Results: {df['accuracy'].mean():.3f} average accuracy")
        print(f"📊 Expected Baseline: ~42% (original Game-2)")
        if df['accuracy'].mean() > 0.42:
            improvement = (df['accuracy'].mean() - 0.42) * 100
            print(f"✅ Improvement: +{improvement:.1f} percentage points!")
        else:
            decline = (0.42 - df['accuracy'].mean()) * 100
            print(f"⚠️ Performance: -{decline:.1f} percentage points")
    
    def create_comprehensive_plots(self, results):
        """Create comprehensive plots including MI and KL metrics"""
        print("\n📊 Creating comprehensive plots...")
        
        # 1. Create confusion matrix for each model
        for result in results:
            placement = result['placement']
            cm = np.array(result['confusion_matrix'])
            activities = result['activities']
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=activities, yticklabels=activities)
            plt.title(f'Improved Gyroscope Confusion Matrix - {placement.title()}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            
            plot_file = os.path.join(self.results_dir, f"confusion_matrix_{placement}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Confusion matrix saved: {plot_file}")
        
        # 2. Feature importance plot for each model
        for result in results:
            placement = result['placement']
            importance = result['feature_importance']
            feature_names = result['feature_names']
            
            # Get all 15 features (should all be present)
            importance_data = list(zip(feature_names, importance))
            importance_data.sort(key=lambda x: x[1], reverse=True)
            
            if importance_data:
                plt.figure(figsize=(12, 8))
                features, importances = zip(*importance_data)
                y_pos = np.arange(len(features))
                
                plt.barh(y_pos, importances, color='green', alpha=0.7)
                plt.yticks(y_pos, features)
                plt.xlabel('Feature Importance')
                plt.title(f'Improved Gyroscope Feature Importance - {placement.title()}')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                
                plot_file = os.path.join(self.results_dir, f"feature_importance_{placement}.png")
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ Feature importance saved: {plot_file}")
        
        # 3. Comprehensive metrics comparison
        df = pd.DataFrame(results)
        placements = df['placement'].tolist()
        
        # Create comprehensive comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Colors for same vs cross sensor
        colors = ['gold' if placement == 'right-pocket' else 'lightblue' if placement == 'combined' else 'lightgreen' 
                 for placement in placements]
        
        # Accuracy plot
        bars1 = ax1.bar(placements, df['accuracy'], color=colors, alpha=0.8)
        ax1.set_title('Model Accuracy by Placement')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        for bar, acc in zip(bars1, df['accuracy']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        
        # F1-Score plot
        bars2 = ax2.bar(placements, df['f1_score'], color=colors, alpha=0.8)
        ax2.set_title('Model F1-Score by Placement')
        ax2.set_ylabel('F1-Score')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        for bar, f1 in zip(bars2, df['f1_score']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=8)
        
        # NMI Percentage plot
        bars3 = ax3.bar(placements, df['nmi_percentage'], color=colors, alpha=0.8)
        ax3.set_title('Normalized Mutual Information by Placement')
        ax3.set_ylabel('NMI (%)')
        ax3.set_ylim(0, 100)
        ax3.tick_params(axis='x', rotation=45)
        for bar, nmi in zip(bars3, df['nmi_percentage']):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{nmi:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Reverse KL Percentage plot
        bars4 = ax4.bar(placements, df['reverse_kl_percentage'], color=colors, alpha=0.8)
        ax4.set_title('Prediction Quality (Reverse KL) by Placement')
        ax4.set_ylabel('Reverse KL (%)')
        ax4.set_ylim(0, 100)
        ax4.tick_params(axis='x', rotation=45)
        for bar, kl in zip(bars4, df['reverse_kl_percentage']):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{kl:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plot_file = os.path.join(self.results_dir, "comprehensive_metrics_comparison.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Comprehensive metrics comparison saved: {plot_file}")
        
        # 4. Information-theoretic metrics correlation
        plt.figure(figsize=(10, 8))
        
        # Scatter plot of NMI vs Accuracy
        plt.subplot(2, 2, 1)
        plt.scatter(df['nmi_percentage'], df['accuracy'], c=[0 if p == 'right-pocket' else 1 for p in placements], 
                   cmap='viridis', alpha=0.7, s=100)
        plt.xlabel('NMI (%)')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs NMI')
        for i, placement in enumerate(placements):
            plt.annotate(placement, (df['nmi_percentage'].iloc[i], df['accuracy'].iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Scatter plot of Reverse KL vs Accuracy
        plt.subplot(2, 2, 2)
        plt.scatter(df['reverse_kl_percentage'], df['accuracy'], c=[0 if p == 'right-pocket' else 1 for p in placements], 
                   cmap='viridis', alpha=0.7, s=100)
        plt.xlabel('Reverse KL (%)')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Reverse KL')
        for i, placement in enumerate(placements):
            plt.annotate(placement, (df['reverse_kl_percentage'].iloc[i], df['accuracy'].iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # NMI vs Reverse KL correlation
        plt.subplot(2, 2, 3)
        plt.scatter(df['nmi_percentage'], df['reverse_kl_percentage'], c=[0 if p == 'right-pocket' else 1 for p in placements], 
                   cmap='viridis', alpha=0.7, s=100)
        plt.xlabel('NMI (%)')
        plt.ylabel('Reverse KL (%)')
        plt.title('NMI vs Reverse KL')
        for i, placement in enumerate(placements):
            plt.annotate(placement, (df['nmi_percentage'].iloc[i], df['reverse_kl_percentage'].iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Distribution plot
        plt.subplot(2, 2, 4)
        metrics = ['Accuracy', 'F1-Score', 'NMI%/100', 'RKL%/100']
        values = [df['accuracy'].mean(), df['f1_score'].mean(), 
                 df['nmi_percentage'].mean()/100, df['reverse_kl_percentage'].mean()/100]
        bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'orange', 'pink'], alpha=0.7)
        plt.title('Average Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        for bar, val in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plot_file = os.path.join(self.results_dir, "information_theoretic_analysis.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Information-theoretic analysis saved: {plot_file}")
        
        print("📊 All comprehensive plots created successfully!")
    
    def save_experiment_results(self, results):
        """Save experiment results with MI and KL metrics to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = os.path.join(self.results_dir, f"improved_gyro_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary with enhanced metrics
        summary_file = os.path.join(self.results_dir, f"improved_gyro_summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GAME-2 IMPROVED GYROSCOPE CROSS-SENSOR CROSS-DATASET RESULTS\n")
            f.write("With Information-Theoretic Analysis (MI & KL Divergence)\n")
            f.write("="*80 + "\n\n")
            f.write(f"Experiment: {self.experiment_title}\n")
            f.write(f"Features: {self.feature_description}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            df = pd.DataFrame(results)
            
            f.write("COMPREHENSIVE RESULTS BY PLACEMENT:\n")
            f.write("-" * 60 + "\n")
            for _, row in df.iterrows():
                sensor_type = "SAME-SENSOR" if row['is_same_sensor'] else "CROSS-SENSOR"
                f.write(f"{row['placement']:15} ({sensor_type:12}):\n")
                f.write(f"  Accuracy: {row['accuracy']:.3f}, F1: {row['f1_score']:.3f}\n")
                f.write(f"  NMI: {row['nmi_percentage']:.1f}%, Reverse KL: {row['reverse_kl_percentage']:.1f}%\n")
                f.write(f"  Vulnerability: {row.get('vulnerability', 0.0):.3f}\n")
                f.write(f"  Features: {row['n_features']}, Classes: {row['n_classes']}\n\n")
            
            f.write(f"OVERALL STATISTICS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Average Accuracy: {df['accuracy'].mean():.3f} ± {df['accuracy'].std():.3f}\n")
            f.write(f"Average F1-Score: {df['f1_score'].mean():.3f} ± {df['f1_score'].std():.3f}\n")
            f.write(f"Average NMI: {df['nmi_percentage'].mean():.1f}% ± {df['nmi_percentage'].std():.1f}%\n")
            f.write(f"Average Reverse KL: {df['reverse_kl_percentage'].mean():.1f}% ± {df['reverse_kl_percentage'].std():.1f}%\n\n")
            f.write(f"Average Vulnerability: {df['vulnerability'].mean():.3f} ± {df['vulnerability'].std():.3f}\n\n")
            
            same_sensor = df[df['is_same_sensor'] == True]
            cross_sensor = df[df['is_same_sensor'] == False]
            
            if len(same_sensor) > 0:
                f.write(f"SAME-SENSOR PERFORMANCE:\n")
                f.write(f"  Accuracy: {same_sensor['accuracy'].mean():.3f} ± {same_sensor['accuracy'].std():.3f}\n")
                f.write(f"  NMI: {same_sensor['nmi_percentage'].mean():.1f}% ± {same_sensor['nmi_percentage'].std():.1f}%\n")
                f.write(f"  Reverse KL: {same_sensor['reverse_kl_percentage'].mean():.1f}% ± {same_sensor['reverse_kl_percentage'].std():.1f}%\n\n")
                f.write(f"  Vulnerability: {same_sensor['vulnerability'].mean():.3f} ± {same_sensor['vulnerability'].std():.3f}\n\n")
            
            if len(cross_sensor) > 0:
                f.write(f"CROSS-SENSOR PERFORMANCE:\n")
                f.write(f"  Accuracy: {cross_sensor['accuracy'].mean():.3f} ± {cross_sensor['accuracy'].std():.3f}\n")
                f.write(f"  NMI: {cross_sensor['nmi_percentage'].mean():.1f}% ± {cross_sensor['nmi_percentage'].std():.1f}%\n")
                f.write(f"  Reverse KL: {cross_sensor['reverse_kl_percentage'].mean():.1f}% ± {cross_sensor['reverse_kl_percentage'].std():.1f}%\n\n")
                f.write(f"  Vulnerability: {cross_sensor['vulnerability'].mean():.3f} ± {cross_sensor['vulnerability'].std():.3f}\n\n")
            
            # Information-theoretic quality assessment
            avg_nmi = df['nmi_percentage'].mean()
            avg_kl = df['reverse_kl_percentage'].mean()
            
            f.write(f"INFORMATION-THEORETIC QUALITY ASSESSMENT:\n")
            f.write("-" * 45 + "\n")
            
            if avg_nmi >= 90:
                f.write(f"NMI Quality: EXCELLENT ({avg_nmi:.1f}%)\n")
            elif avg_nmi >= 80:
                f.write(f"NMI Quality: GOOD ({avg_nmi:.1f}%)\n")
            elif avg_nmi >= 70:
                f.write(f"NMI Quality: MODERATE ({avg_nmi:.1f}%)\n")
            else:
                f.write(f"NMI Quality: NEEDS IMPROVEMENT ({avg_nmi:.1f}%)\n")
            
            if avg_kl >= 90:
                f.write(f"Prediction Quality: EXCELLENT ({avg_kl:.1f}%)\n")
            elif avg_kl >= 80:
                f.write(f"Prediction Quality: GOOD ({avg_kl:.1f}%)\n")
            elif avg_kl >= 70:
                f.write(f"Prediction Quality: MODERATE ({avg_kl:.1f}%)\n")
            else:
                f.write(f"Prediction Quality: NEEDS IMPROVEMENT ({avg_kl:.1f}%)\n")
            
            f.write(f"\nIMPROVEMENT ANALYSIS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Baseline Performance: ~42% (original Game-2)\n")
            f.write(f"Improved Performance: {df['accuracy'].mean():.3f}\n")
            if df['accuracy'].mean() > 0.42:
                improvement = (df['accuracy'].mean() - 0.42) * 100
                f.write(f"Improvement: +{improvement:.1f} percentage points\n")
            else:
                decline = (0.42 - df['accuracy'].mean()) * 100
                f.write(f"Performance Change: -{decline:.1f} percentage points\n")
                
            f.write(f"\nFEATURE ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Number of Features Used: {df['n_features'].iloc[0] if not df.empty else 'N/A'}\n")
            f.write(f"Feature Type: Top 15 placement-agnostic gyroscope features\n")
            f.write(f"Cross-sensor Transfer: MotionSense (smartphone) → STM (multi-placement)\n")
        
        print(f"✅ Enhanced results saved to {results_file}")
        print(f"✅ Summary with MI/KL metrics saved to {summary_file}")
        print(f"✅ Summary saved to {summary_file}")

def main():
    print("🚀 Starting Game-2 Improved Gyroscope Cross-Sensor Cross-Dataset Analysis")
    print("🔬 Using top 15 placement-agnostic features with highest transferability")
    print("🔄 Running both experimental directions...\n")
    
    # Case 1: STM → MotionSense
    print("="*70)
    print("🎯 CASE 1: STM → MotionSense (IMPROVED)")
    print("="*70)
    analyzer_stm_to_ms = Game2ImprovedGyroAnalyzer(experiment_direction="stm_to_motionsense")
    results_stm_to_ms = analyzer_stm_to_ms.run_cross_sensor_experiment()
    
    # Case 2: MotionSense → STM
    print("\n" + "="*70)
    print("🎯 CASE 2: MotionSense → STM (IMPROVED)")
    print("="*70)
    analyzer_ms_to_stm = Game2ImprovedGyroAnalyzer(experiment_direction="motionsense_to_stm")
    results_ms_to_stm = analyzer_ms_to_stm.run_cross_sensor_experiment()
    
    print("\n🎉 Game-2 Improved Gyroscope Cross-Sensor Cross-Dataset Analysis Complete!")
    print("📊 Check the results directories for detailed analysis and plots.")
    print("🔬 Compare with original Game-2 results to see improvement!")

if __name__ == "__main__":
    main()
