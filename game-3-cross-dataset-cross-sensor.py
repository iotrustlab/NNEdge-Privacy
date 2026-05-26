#!/usr/bin/env python3
"""
Game-3 Multi-Modal Cross-Sensor Cross-Dataset Study (Accelerometer + Gyroscope Combined)
========================================================================================

This script implements Game-3 cross-sensor cross-dataset HAR using COMBINED features:
- ALL 15 scientifically validated accelerometer binary features (0.262-0.389 transferability)
- TOP 5 placement-agnostic gyroscope features (0.020-0.026 transferability)

ACCELEROMETER FEATURES (15 binary features):
1. binary_dynamic_activity (0.389) - Movement detection
2. binary_activity_detected (0.386) - General activity detection  
3. binary_very_low_intensity (0.386) - Energy-based classification
4. binary_z_energy_dominant (0.354) - Axis energy analysis
5. binary_compact_distribution (0.352) - Statistical distribution
6. binary_very_high_intensity (0.337) - Energy intensity
7. binary_low_freq_dominant (0.306) - Frequency domain
8. binary_high_crossing_rate (0.299) - Temporal dynamics
9. binary_high_dominant_freq (0.282) - Frequency analysis
10. binary_high_intensity (0.282) - Energy analysis
11. binary_predictable_pattern (0.281) - Pattern recognition
12. binary_magnitude_skewed (0.279) - Statistical shape
13. binary_low_crossing_rate (0.272) - Temporal pattern
14. binary_xz_positive_corr (0.269) - Axis correlation
15. binary_strong_autocorr (0.262) - Pattern consistency

GYROSCOPE FEATURES (5 best placement-agnostic features):
1. gyro_total_rotation_intensity (0.026) - BEST!
2. gyro_avg_rotation_intensity (0.026) - BEST!  
3. gyro_magnitude_mean (0.025)
4. gyro_magnitude_q25 (0.023)
5. gyro_magnitude_median (0.020)

TOTAL: 20 features (15 accelerometer binary + 5 gyroscope continuous)

EXPERIMENTAL DESIGN:
Case-1: STM (5 placements + combined) → MotionSense
Case-2: MotionSense → STM (5 placements + combined)

HYPOTHESIS: Multi-modal sensor fusion should improve cross-sensor transferability
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
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
import warnings

from cross_dataset_utils import calculate_vulnerability, get_motionsense_root, get_stm_root

warnings.filterwarnings('ignore')

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
        y_true: True labels (encoded as integers)
        y_pred_proba: Predicted probabilities (n_samples, n_classes)
        num_classes: Number of classes
    
    Returns:
        kl_avg: Average KL divergence
        kl_normalized: Normalized KL divergence (0-1 scale)
        reverse_kl_percentage: Reverse KL as percentage (0-100%)
    """
    try:
        # Convert true labels to one-hot encoding
        y_true_onehot = np.zeros((len(y_true), num_classes))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1
        
        # Calculate empirical class distribution
        class_counts = np.bincount(y_true, minlength=num_classes)
        true_distribution = class_counts / len(y_true)
        
        # Average predicted distribution
        avg_pred_distribution = np.mean(y_pred_proba, axis=0)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        true_distribution = true_distribution + epsilon
        avg_pred_distribution = avg_pred_distribution + epsilon
        
        # Normalize to ensure they sum to 1
        true_distribution = true_distribution / np.sum(true_distribution)
        avg_pred_distribution = avg_pred_distribution / np.sum(avg_pred_distribution)
        
        # Calculate KL divergence: KL(true || pred)
        kl_forward = np.sum(true_distribution * np.log(true_distribution / avg_pred_distribution))
        
        # Calculate reverse KL divergence: KL(pred || true)  
        kl_reverse = np.sum(avg_pred_distribution * np.log(avg_pred_distribution / true_distribution))
        
        # Use reverse KL as it's more sensitive to prediction quality
        kl_avg = kl_reverse
        
        # Normalize to [0,1] using sigmoid-like function
        kl_normalized = 1 / (1 + kl_avg)
        
        # Convert to percentage (higher is better)
        reverse_kl_percentage = kl_normalized * 100.0
        
        return kl_avg, kl_normalized, reverse_kl_percentage
        
    except Exception as e:
        print(f"⚠️ Error calculating KL divergence: {e}")
        return float('inf'), 0.0, 0.0

print("🎭 Game-3 MULTI-MODAL ACCELEROMETER + GYROSCOPE COMBINED")
print("=" * 65)
print("🔬 Using 15 accelerometer binary + 5 gyroscope features")
print("🏆 Total: 20 features from dual-sensor fusion")
print("🎯 Case-1: STM → MotionSense | Case-2: MotionSense → STM")
print("📈 Hypothesis: Multi-modal fusion improves transferability")
print()

class MultiModalAccelGyroCombinedHAR:
    """
    Multi-modal Game-3 HAR using combined accelerometer and gyroscope features
    """
    
    def __init__(self, results_dir="results/game-3-multimodal-combined"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # Dataset paths
        self.stm_path = str(get_stm_root())
        self.motionsense_path = str(get_motionsense_root())
        
        # Activity mappings (6 common activities)
        self.activity_mapping = {
            'downstairs': 0, 'jogging': 1, 'sitting': 2, 
            'standing': 3, 'upstairs': 4, 'walking': 5
        }
        
        # STM placements
        self.stm_placements = ['left-ankle', 'left-wrist', 'right-ankle', 'right-pocket', 'right-wrist']
        
        print(f"📁 Results directory: {results_dir}")
        print(f"🎯 Target activities: {list(self.activity_mapping.keys())}")
        print(f"📍 STM placements: {self.stm_placements}")
        print()

    def extract_accelerometer_binary_features(self, accel_data):
        """
        Extract ALL 15 scientifically validated accelerometer binary features
        """
        if len(accel_data) == 0:
            return {}
            
        # Handle different column naming conventions
        if 'acc_x[mg]' in accel_data.columns:
            # STM dataset format
            ax = np.array(accel_data['acc_x[mg]'])
            ay = np.array(accel_data['acc_y[mg]']) 
            az = np.array(accel_data['acc_z[mg]'])
        elif 'ax' in accel_data.columns:
            # Legacy format
            ax = np.array(accel_data['ax'])
            ay = np.array(accel_data['ay']) 
            az = np.array(accel_data['az'])
        else:
            return {}
        
        # Calculate magnitude
        magnitude = np.sqrt(ax**2 + ay**2 + az**2)
        
        features = {}
        
        # 1. binary_dynamic_activity (0.389) - Movement vs stationary detection
        magnitude_std = np.std(magnitude)
        features['binary_dynamic_activity'] = 1 if magnitude_std > np.median(magnitude) * 0.1 else 0
        
        # 2. binary_activity_detected (0.386) - General activity presence
        activity_threshold = np.mean(magnitude) + 0.5 * np.std(magnitude)
        features['binary_activity_detected'] = 1 if np.max(magnitude) > activity_threshold else 0
        
        # 3. binary_very_low_intensity (0.386) - Low energy state
        energy = np.sum(magnitude**2) / len(magnitude)
        low_energy_threshold = np.percentile(magnitude, 25)**2
        features['binary_very_low_intensity'] = 1 if energy < low_energy_threshold else 0
        
        # 4. binary_z_energy_dominant (0.354) - Vertical axis dominance
        z_energy = np.sum(az**2)
        total_energy = np.sum(ax**2) + np.sum(ay**2) + np.sum(az**2)
        features['binary_z_energy_dominant'] = 1 if z_energy > 0.4 * total_energy else 0
        
        # 5. binary_compact_distribution (0.352) - Magnitude distribution spread
        magnitude_iqr = np.percentile(magnitude, 75) - np.percentile(magnitude, 25)
        magnitude_range = np.max(magnitude) - np.min(magnitude)
        features['binary_compact_distribution'] = 1 if magnitude_iqr < 0.3 * magnitude_range else 0
        
        # 6. binary_very_high_intensity (0.337) - High energy state
        high_energy_threshold = np.percentile(magnitude, 90)**2
        features['binary_very_high_intensity'] = 1 if energy > high_energy_threshold else 0
        
        # 7. binary_low_freq_dominant (0.306) - Low frequency content dominance
        freqs, psd = signal.welch(magnitude, fs=50, nperseg=min(256, len(magnitude)//4))
        low_freq_power = np.sum(psd[freqs <= 2])
        total_power = np.sum(psd)
        features['binary_low_freq_dominant'] = 1 if low_freq_power > 0.6 * total_power else 0
        
        # 8. binary_high_crossing_rate (0.299) - High zero-crossing rate
        magnitude_centered = magnitude - np.mean(magnitude)
        crossings = np.sum(np.diff(np.signbit(magnitude_centered)))
        crossing_rate = crossings / len(magnitude)
        features['binary_high_crossing_rate'] = 1 if crossing_rate > 0.1 else 0
        
        # 9. binary_high_dominant_freq (0.282) - High dominant frequency
        dominant_freq_idx = np.argmax(psd)
        dominant_freq = freqs[dominant_freq_idx] if len(freqs) > dominant_freq_idx else 0
        features['binary_high_dominant_freq'] = 1 if dominant_freq > 5 else 0
        
        # 10. binary_high_intensity (0.282) - High intensity state
        intensity_threshold = np.percentile(magnitude, 75)
        features['binary_high_intensity'] = 1 if np.mean(magnitude) > intensity_threshold else 0
        
        # 11. binary_predictable_pattern (0.281) - Pattern predictability
        autocorr = np.correlate(magnitude, magnitude, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        if len(autocorr) > 10:
            pattern_predictability = np.max(autocorr[1:min(11, len(autocorr))]) / autocorr[0]
            features['binary_predictable_pattern'] = 1 if pattern_predictability > 0.7 else 0
        else:
            features['binary_predictable_pattern'] = 0
        
        # 12. binary_magnitude_skewed (0.279) - Magnitude distribution skewness
        magnitude_skewness = abs(stats.skew(magnitude))
        features['binary_magnitude_skewed'] = 1 if magnitude_skewness > 0.5 else 0
        
        # 13. binary_low_crossing_rate (0.272) - Low zero-crossing rate
        features['binary_low_crossing_rate'] = 1 if crossing_rate < 0.05 else 0
        
        # 14. binary_xz_positive_corr (0.269) - Positive X-Z correlation
        if np.std(ax) > 0 and np.std(az) > 0:
            xz_corr = np.corrcoef(ax, az)[0, 1]
            features['binary_xz_positive_corr'] = 1 if xz_corr > 0.2 else 0
        else:
            features['binary_xz_positive_corr'] = 0
        
        # 15. binary_strong_autocorr (0.262) - Strong autocorrelation
        if len(autocorr) > 5:
            strong_autocorr = np.max(autocorr[1:min(6, len(autocorr))]) / autocorr[0]
            features['binary_strong_autocorr'] = 1 if strong_autocorr > 0.8 else 0
        else:
            features['binary_strong_autocorr'] = 0
        
        return features

    def extract_gyroscope_top5_features(self, gyro_data):
        """
        Extract TOP 5 placement-agnostic gyroscope features
        """
        features = {}
        
        if not all(col in gyro_data.columns for col in ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']):
            # Return zeros if gyroscope data not available
            features['gyro_total_rotation_intensity'] = 0
            features['gyro_avg_rotation_intensity'] = 0
            features['gyro_magnitude_mean'] = 0
            features['gyro_magnitude_q25'] = 0
            features['gyro_magnitude_median'] = 0
            return features
        
        x = gyro_data['gyro_x[mdps]'].values
        y = gyro_data['gyro_y[mdps]'].values
        z = gyro_data['gyro_z[mdps]'].values
        
        # Calculate magnitude (PLACEMENT-AGNOSTIC: independent of sensor orientation)
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        
        if len(magnitude) == 0:
            features['gyro_total_rotation_intensity'] = 0
            features['gyro_avg_rotation_intensity'] = 0
            features['gyro_magnitude_mean'] = 0
            features['gyro_magnitude_q25'] = 0
            features['gyro_magnitude_median'] = 0
            return features
        
        # ============ TOP 5 GYROSCOPE FEATURES ============
        
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
        
        return features

    def extract_multimodal_features(self, data_window):
        """
        Extract combined accelerometer + gyroscope features
        Total: 20 features (15 accel binary + 5 gyro continuous)
        """
        # Extract accelerometer binary features
        accel_features = self.extract_accelerometer_binary_features(data_window)
        
        # Extract top 5 gyroscope features
        gyro_features = self.extract_gyroscope_top5_features(data_window)
        
        # Combine all features
        combined_features = {**accel_features, **gyro_features}
        
        return combined_features

    def load_motionsense_data(self):
        """Load MotionSense smartphone accelerometer and gyroscope data"""
        print("📊 Loading MotionSense smartphone multi-modal data...")
        
        base_path = self.motionsense_path
        combined_data = []
        files_loaded = 0
        activities_loaded = set()
        
        # Activity mapping for MotionSense
        activity_folder_mapping = {
            'activity_walking': 'walking',
            'activity_jogging': 'jogging', 
            'activity_standing': 'standing',
            'activity_sitting': 'sitting',
            'activity_upstairs': 'upstairs',
            'activity_downstairs': 'downstairs'
        }
        
        for subject_folder in sorted(os.listdir(base_path)):
            if subject_folder.startswith('subject_'):
                subject_path = os.path.join(base_path, subject_folder)
                
                for activity_folder in os.listdir(subject_path):
                    if activity_folder in activity_folder_mapping:
                        activity_name = activity_folder_mapping[activity_folder]
                        
                        data_file = os.path.join(subject_path, activity_folder, 'data.csv')
                        if os.path.exists(data_file):
                            try:
                                df = pd.read_csv(data_file)
                                
                                # Check for both accelerometer and gyroscope columns
                                accel_cols = ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]']
                                gyro_cols = ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']
                                
                                if all(col in df.columns for col in accel_cols):
                                    # Add gyroscope columns with zeros if not present
                                    for col in gyro_cols:
                                        if col not in df.columns:
                                            df[col] = 0
                                    
                                    df['activity'] = activity_name
                                    df['subject'] = subject_folder
                                    df['placement'] = 'smartphone'
                                    combined_data.append(df)
                                    files_loaded += 1
                                    activities_loaded.add(activity_name)
                                    
                            except Exception as e:
                                print(f"⚠️ Error loading {data_file}: {e}")
        
        if combined_data:
            result_df = pd.concat(combined_data, ignore_index=True)
            print(f"✅ {len(result_df)} samples from {files_loaded} files")
            print(f"   📊 Activities: {sorted(list(activities_loaded))}")
            return result_df
        else:
            print("❌ No MotionSense data found")
            return pd.DataFrame()

    def load_stm_placement_data(self, placement):
        """Load STM data for a specific placement - handles both structures including stairs"""
        print(f"📊 Loading STM {placement} multi-modal data...")
        
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
                                                
                                                # Check for accelerometer columns
                                                accel_cols = ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]']
                                                gyro_cols = ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']
                                                
                                                if all(col in df.columns for col in accel_cols):
                                                    # Add gyroscope columns with zeros if not present
                                                    for col in gyro_cols:
                                                        if col not in df.columns:
                                                            df[col] = 0
                                                    
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
                                            
                                            # Check for accelerometer columns
                                            accel_cols = ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]']
                                            gyro_cols = ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']
                                            
                                            if all(col in df.columns for col in accel_cols):
                                                # Add gyroscope columns with zeros if not present
                                                for col in gyro_cols:
                                                    if col not in df.columns:
                                                        df[col] = 0
                                                
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

    def create_dataset_from_dataframe(self, df, window_size=128, overlap=0.5):
        """Create windowed dataset with multimodal features"""
        print(f" 🔄 Creating multi-modal windowed dataset (window={window_size}, overlap={overlap:.1f})")
        
        features_list = []
        labels_list = []
        step_size = int(window_size * (1 - overlap))
        
        for activity in df['activity'].unique():
            activity_data = df[df['activity'] == activity]
            
            # Group by user/subject
            if 'subject' in activity_data.columns:
                user_col = 'subject'
            elif 'user_id' in activity_data.columns:
                user_col = 'user_id'
            elif 'user' in activity_data.columns:
                user_col = 'user'
            else:
                # If no user column, treat all data as one user
                user_col = None
            
            if user_col:
                for user in activity_data[user_col].unique():
                    user_data = activity_data[activity_data[user_col] == user]
                    # Sort by timestamp or index
                    if 'timestamp' in user_data.columns:
                        user_data = user_data.sort_values('timestamp')
                    else:
                        user_data = user_data.sort_index()
                    
                    # Create sliding windows
                    for start_idx in range(0, len(user_data) - window_size + 1, step_size):
                        window_data = user_data.iloc[start_idx:start_idx + window_size]
                        
                        # Extract multimodal features
                        features = self.extract_multimodal_features(window_data)
                        if features:  # Only add if features extracted successfully
                            features_list.append(features)
                            labels_list.append(self.activity_mapping[activity])
            else:
                # No user column, treat all activity data as one user
                user_data = activity_data
                # Sort by timestamp or index
                if 'timestamp' in user_data.columns:
                    user_data = user_data.sort_values('timestamp')
                else:
                    user_data = user_data.sort_index()
                
                # Create sliding windows
                for start_idx in range(0, len(user_data) - window_size + 1, step_size):
                    window_data = user_data.iloc[start_idx:start_idx + window_size]
                    
                    # Extract multimodal features
                    features = self.extract_multimodal_features(window_data)
                    if features:  # Only add if features extracted successfully
                        features_list.append(features)
                        labels_list.append(self.activity_mapping[activity])
        
        if features_list:
            features_df = pd.DataFrame(features_list)
            print(f" ✅ {len(features_df)} samples, {len(features_df.columns)} multimodal features")
            
            # Print feature breakdown
            accel_features = [col for col in features_df.columns if col.startswith('binary_')]
            gyro_features = [col for col in features_df.columns if col.startswith('gyro_')]
            print(f"    📊 Accelerometer: {len(accel_features)} binary features")
            print(f"    📊 Gyroscope: {len(gyro_features)} continuous features")
            
            return features_df, np.array(labels_list)
        else:
            print(" ❌ No features extracted")
            return None, None

    def train_and_evaluate_model(self, X_train, y_train, X_test, y_test, experiment_name):
        """Train Random Forest and evaluate performance with MI and KL metrics"""
        print(f" 🤖 Training Random Forest for {experiment_name}...")
        
        # Scale features (important for mixed binary + continuous features)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test_scaled)
        y_pred_proba = rf_model.predict_proba(X_test_scaled)
        
        # Standard metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Calculate information-theoretic metrics
        unique_labels = sorted(list(set(y_test)))
        le = LabelEncoder()
        y_test_encoded = le.fit_transform(y_test)
        y_pred_encoded = le.transform(y_pred)
        
        mi_value, nmi_percentage = calculate_mutual_information(y_test_encoded, y_pred_encoded)
        kl_avg, kl_normalized, reverse_kl_percentage = calculate_kl_divergence(
            y_test_encoded, y_pred_proba, len(unique_labels)
        )
        vulnerability = calculate_vulnerability(y_pred_proba)
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(
            f" ✅ {experiment_name}: Acc={accuracy:.3f}, F1={f1:.3f}, "
            f"NMI={nmi_percentage:.1f}%, RKL={reverse_kl_percentage:.1f}%, Vuln={vulnerability:.3f}"
        )

        return {
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
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': X_train.shape[1],
            'n_classes': len(unique_labels),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist(),
            'feature_importance': feature_importance.head(10).to_dict('records'),
            'all_feature_importance': rf_model.feature_importances_.tolist(),
            'feature_names': X_train.columns.tolist(),
            'model': rf_model,
            'scaler': scaler
        }

    def run_case1_stm_to_motionsense(self):
        """Case-1: Train on STM (all placements), test on MotionSense"""
        print("📊 CASE-1: STM (Multi-Placement) → MotionSense")
        print("=" * 50)
        
        # Load MotionSense test data
        ms_data = self.load_motionsense_data()
        if ms_data.empty:
            print("❌ No MotionSense data available")
            return []
        
        X_test, y_test = self.create_dataset_from_dataframe(ms_data)
        if X_test is None:
            print("❌ Could not create MotionSense test dataset")
            return []
        
        case1_results = []
        
        # Individual placements
        for placement in self.stm_placements:
            print(f"\n🎯 Training on STM {placement}...")
            
            stm_data = self.load_stm_placement_data(placement)
            if stm_data.empty:
                print(f"⚠️ No data for {placement}, skipping...")
                continue
            
            X_train, y_train = self.create_dataset_from_dataframe(stm_data)
            if X_train is None:
                print(f"⚠️ Could not create dataset for {placement}")
                continue
            
            # Ensure same features in train and test
            common_features = X_train.columns.intersection(X_test.columns)
            if len(common_features) == 0:
                print(f"⚠️ No common features between {placement} and MotionSense")
                continue
            
            X_train_common = X_train[common_features]
            X_test_common = X_test[common_features]
            
            # Train and evaluate
            result = self.train_and_evaluate_model(
                X_train_common, y_train, X_test_common, y_test,
                f"STM-{placement} → MotionSense"
            )
            
            result.update({
                'train_source': f'STM-{placement}',
                'test_source': 'MotionSense',
                'placement': placement,
                'case': 'case1'
            })
            
            case1_results.append(result)
        
        # Combined placements
        print(f"\n🎯 Training on STM Combined Placements...")
        
        all_stm_data = []
        for placement in self.stm_placements:
            stm_data = self.load_stm_placement_data(placement)
            if not stm_data.empty:
                all_stm_data.append(stm_data)
        
        if all_stm_data:
            combined_stm_data = pd.concat(all_stm_data, ignore_index=True)
            X_train, y_train = self.create_dataset_from_dataframe(combined_stm_data)
            
            if X_train is not None:
                # Ensure same features in train and test
                common_features = X_train.columns.intersection(X_test.columns)
                X_train_common = X_train[common_features]
                X_test_common = X_test[common_features]
                
                # Train and evaluate
                result = self.train_and_evaluate_model(
                    X_train_common, y_train, X_test_common, y_test,
                    "STM-Combined → MotionSense"
                )
                
                result.update({
                    'train_source': 'STM-Combined',
                    'test_source': 'MotionSense',
                    'placement': 'combined',
                    'case': 'case1'
                })
                
                case1_results.append(result)
        
        return case1_results

    def run_case2_motionsense_to_stm(self):
        """Case-2: Train on MotionSense, test on STM placements"""
        print("\n📊 CASE-2: MotionSense → STM (Individual Placements)")
        print("=" * 55)
        
        # Load MotionSense training data
        ms_data = self.load_motionsense_data()
        if ms_data.empty:
            print("❌ No MotionSense data available")
            return []
        
        X_train, y_train = self.create_dataset_from_dataframe(ms_data)
        if X_train is None:
            print("❌ Could not create MotionSense training dataset")
            return []
        
        case2_results = []
        
        # Test on each STM placement
        for placement in self.stm_placements:
            print(f"\n🎯 Testing on STM {placement}...")
            
            stm_data = self.load_stm_placement_data(placement)
            if stm_data.empty:
                print(f"⚠️ No data for {placement}, skipping...")
                continue
            
            X_test, y_test = self.create_dataset_from_dataframe(stm_data)
            if X_test is None:
                print(f"⚠️ Could not create dataset for {placement}")
                continue
            
            # Ensure same features in train and test
            common_features = X_train.columns.intersection(X_test.columns)
            if len(common_features) == 0:
                print(f"⚠️ No common features between MotionSense and {placement}")
                continue
            
            X_train_common = X_train[common_features]
            X_test_common = X_test[common_features]
            
            # Train and evaluate
            result = self.train_and_evaluate_model(
                X_train_common, y_train, X_test_common, y_test,
                f"MotionSense → STM-{placement}"
            )
            
            result.update({
                'train_source': 'MotionSense',
                'test_source': f'STM-{placement}',
                'placement': placement,
                'case': 'case2'
            })
            
            case2_results.append(result)
        
        return case2_results

    def save_results(self, case1_results, case2_results):
        """Save comprehensive results with MI and KL analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        all_results = case1_results + case2_results
        
        if not all_results:
            print("❌ No results to save")
            return {
                'case1_results': case1_results,
                'case2_results': case2_results,
                'status': 'failed'
            }
        
        # Calculate comprehensive summary statistics
        accuracies = [r['accuracy'] for r in all_results]
        f1_scores = [r['f1_score'] for r in all_results]
        nmi_percentages = [r['nmi_percentage'] for r in all_results]
        reverse_kl_percentages = [r['reverse_kl_percentage'] for r in all_results]
        vulnerabilities = [r.get('vulnerability', 0.0) for r in all_results]
        
        summary = {
            'timestamp': timestamp,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'mean_nmi': np.mean(nmi_percentages),
            'std_nmi': np.std(nmi_percentages),
            'mean_reverse_kl': np.mean(reverse_kl_percentages),
            'std_reverse_kl': np.std(reverse_kl_percentages),
            'mean_vulnerability': np.mean(vulnerabilities),
            'std_vulnerability': np.std(vulnerabilities),
            'total_experiments': len(all_results),
            'features_used': '20 multimodal features (15 accel binary + 5 gyro continuous)',
            'case1_results': case1_results,
            'case2_results': case2_results,
            'status': 'success'
        }
        
        # Save results
        results_file = os.path.join(self.results_dir, f"multimodal_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create comprehensive plots
        self.create_comprehensive_plots(all_results, timestamp)
        
        # Save detailed summary
        self.save_detailed_summary(summary, all_results, timestamp)
        
        # Print enhanced summary
        print("\n📊 MULTI-MODAL COMBINED Game-3 RESULTS SUMMARY")
        print("=" * 55)
        print(f"📈 Mean Accuracy: {summary['mean_accuracy']:.1%} ± {summary['std_accuracy']:.1%}")
        print(f"📊 Mean F1-Score: {summary['mean_f1']:.1%} ± {summary['std_f1']:.1%}")
        print(f"🧠 Mean NMI: {summary['mean_nmi']:.1f}% ± {summary['std_nmi']:.1f}%")
        print(f"🎯 Mean Reverse KL: {summary['mean_reverse_kl']:.1f}% ± {summary['std_reverse_kl']:.1f}%")
        print(f"🔐 Mean Vulnerability: {summary['mean_vulnerability']:.3f} ± {summary['std_vulnerability']:.3f}")
        print(f"📊 Range: {summary['min_accuracy']:.1%} - {summary['max_accuracy']:.1%}")
        print(f"🧪 Total Experiments: {summary['total_experiments']}")
        print(f"🔑 Features: 20 multimodal (15 accel + 5 gyro)")
        
        # Quality assessment
        avg_nmi = summary['mean_nmi']
        avg_kl = summary['mean_reverse_kl']
        
        print(f"\n🎯 Information-Theoretic Quality Assessment:")
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
        
        print(f"💾 Results saved: {results_file}")
        
        return summary

    def create_comprehensive_plots(self, all_results, timestamp):
        """Create comprehensive plots including MI and KL metrics"""
        print("\n📊 Creating comprehensive multimodal plots...")
        
        if not all_results:
            print("❌ No results to plot")
            return
        
        # Extract data for plotting
        experiment_names = [r.get('experiment_name', f'Exp_{i}') for i, r in enumerate(all_results)]
        accuracies = [r['accuracy'] for r in all_results]
        f1_scores = [r['f1_score'] for r in all_results]
        nmi_percentages = [r['nmi_percentage'] for r in all_results]
        reverse_kl_percentages = [r['reverse_kl_percentage'] for r in all_results]
        
        # 1. Comprehensive metrics comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Accuracy plot
        bars1 = ax1.bar(range(len(experiment_names)), accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('Multi-Modal Accuracy by Experiment')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.set_xticks(range(len(experiment_names)))
        ax1.set_xticklabels(experiment_names, rotation=45, ha='right')
        for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        
        # F1-Score plot
        bars2 = ax2.bar(range(len(experiment_names)), f1_scores, color='lightgreen', alpha=0.7)
        ax2.set_title('Multi-Modal F1-Score by Experiment')
        ax2.set_ylabel('F1-Score')
        ax2.set_ylim(0, 1)
        ax2.set_xticks(range(len(experiment_names)))
        ax2.set_xticklabels(experiment_names, rotation=45, ha='right')
        for i, (bar, f1) in enumerate(zip(bars2, f1_scores)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=8)
        
        # NMI Percentage plot
        bars3 = ax3.bar(range(len(experiment_names)), nmi_percentages, color='orange', alpha=0.7)
        ax3.set_title('Normalized Mutual Information by Experiment')
        ax3.set_ylabel('NMI (%)')
        ax3.set_ylim(0, 100)
        ax3.set_xticks(range(len(experiment_names)))
        ax3.set_xticklabels(experiment_names, rotation=45, ha='right')
        for i, (bar, nmi) in enumerate(zip(bars3, nmi_percentages)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{nmi:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Reverse KL Percentage plot
        bars4 = ax4.bar(range(len(experiment_names)), reverse_kl_percentages, color='pink', alpha=0.7)
        ax4.set_title('Prediction Quality (Reverse KL) by Experiment')
        ax4.set_ylabel('Reverse KL (%)')
        ax4.set_ylim(0, 100)
        ax4.set_xticks(range(len(experiment_names)))
        ax4.set_xticklabels(experiment_names, rotation=45, ha='right')
        for i, (bar, kl) in enumerate(zip(bars4, reverse_kl_percentages)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{kl:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plot_file = os.path.join(self.results_dir, f"multimodal_comprehensive_metrics_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Comprehensive metrics plot saved: {plot_file}")
        
        # 2. Information-theoretic correlation analysis
        plt.figure(figsize=(12, 10))
        
        # Scatter plot of NMI vs Accuracy
        plt.subplot(2, 2, 1)
        plt.scatter(nmi_percentages, accuracies, alpha=0.7, s=100, c='blue')
        plt.xlabel('NMI (%)')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs NMI')
        for i, name in enumerate(experiment_names):
            plt.annotate(name, (nmi_percentages[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Scatter plot of Reverse KL vs Accuracy
        plt.subplot(2, 2, 2)
        plt.scatter(reverse_kl_percentages, accuracies, alpha=0.7, s=100, c='red')
        plt.xlabel('Reverse KL (%)')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Reverse KL')
        for i, name in enumerate(experiment_names):
            plt.annotate(name, (reverse_kl_percentages[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # NMI vs Reverse KL correlation
        plt.subplot(2, 2, 3)
        plt.scatter(nmi_percentages, reverse_kl_percentages, alpha=0.7, s=100, c='green')
        plt.xlabel('NMI (%)')
        plt.ylabel('Reverse KL (%)')
        plt.title('NMI vs Reverse KL')
        for i, name in enumerate(experiment_names):
            plt.annotate(name, (nmi_percentages[i], reverse_kl_percentages[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Average metrics distribution
        plt.subplot(2, 2, 4)
        metrics = ['Accuracy', 'F1-Score', 'NMI%/100', 'RKL%/100']
        values = [np.mean(accuracies), np.mean(f1_scores), 
                 np.mean(nmi_percentages)/100, np.mean(reverse_kl_percentages)/100]
        bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'orange', 'pink'], alpha=0.7)
        plt.title('Average Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        for bar, val in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plot_file = os.path.join(self.results_dir, f"multimodal_information_theoretic_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Information-theoretic analysis plot saved: {plot_file}")
        
        # 3. Create confusion matrices for each experiment
        for i, result in enumerate(all_results):
            if 'confusion_matrix' in result:
                cm = np.array(result['confusion_matrix'])
                experiment_name = experiment_names[i]
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Multi-Modal Confusion Matrix - {experiment_name}')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.tight_layout()
                
                plot_file = os.path.join(self.results_dir, f"multimodal_confusion_matrix_{experiment_name}_{timestamp}.png")
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ Confusion matrix saved: {plot_file}")
        
        print("📊 All multimodal plots created successfully!")

    def save_detailed_summary(self, summary, all_results, timestamp):
        """Save detailed summary with MI and KL metrics"""
        summary_file = os.path.join(self.results_dir, f"multimodal_detailed_summary_{timestamp}.txt")
        
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GAME-3 MULTI-MODAL CROSS-SENSOR CROSS-DATASET RESULTS\n")
            f.write("Accelerometer (15 binary) + Gyroscope (5 continuous) = 20 features\n")
            f.write("With Information-Theoretic Analysis (MI & KL Divergence)\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total Experiments: {summary['total_experiments']}\n")
            f.write(f"Feature Configuration: {summary['features_used']}\n\n")
            
            f.write("COMPREHENSIVE PERFORMANCE METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy: {summary['mean_accuracy']:.3f} ± {summary['std_accuracy']:.3f}\n")
            f.write(f"F1-Score: {summary['mean_f1']:.3f} ± {summary['std_f1']:.3f}\n")
            f.write(f"NMI: {summary['mean_nmi']:.1f}% ± {summary['std_nmi']:.1f}%\n")
            f.write(f"Reverse KL: {summary['mean_reverse_kl']:.1f}% ± {summary['std_reverse_kl']:.1f}%\n\n")
            f.write(f"Vulnerability: {summary['mean_vulnerability']:.3f} ± {summary['std_vulnerability']:.3f}\n\n")
            
            f.write("DETAILED EXPERIMENT RESULTS:\n")
            f.write("-" * 35 + "\n")
            for i, result in enumerate(all_results):
                experiment_name = result.get('experiment_name', f'Experiment_{i+1}')
                f.write(f"{experiment_name}:\n")
                f.write(f"  Accuracy: {result['accuracy']:.3f}\n")
                f.write(f"  F1-Score: {result['f1_score']:.3f}\n")
                f.write(f"  Precision: {result['precision']:.3f}\n")
                f.write(f"  Recall: {result['recall']:.3f}\n")
                f.write(f"  NMI: {result['nmi_percentage']:.1f}%\n")
                f.write(f"  Reverse KL: {result['reverse_kl_percentage']:.1f}%\n")
                f.write(f"  Vulnerability: {result.get('vulnerability', 0.0):.3f}\n")
                f.write(f"  Train Samples: {result['n_train']}\n")
                f.write(f"  Test Samples: {result['n_test']}\n")
                f.write(f"  Features: {result['n_features']}\n")
                f.write(f"  Classes: {result['n_classes']}\n\n")
            
            # Information-theoretic quality assessment
            avg_nmi = summary['mean_nmi']
            avg_kl = summary['mean_reverse_kl']
            
            f.write("INFORMATION-THEORETIC QUALITY ASSESSMENT:\n")
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
            
            f.write(f"\nFEATURE ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Accelerometer Features: 15 binary scientifically validated features\n")
            f.write(f"Gyroscope Features: 5 placement-agnostic continuous features\n")
            f.write(f"Cross-sensor Transfer: STM (multi-placement) ↔ MotionSense (smartphone)\n")
            f.write(f"Multi-modal Fusion: Combined accelerometer + gyroscope data\n")
        
        print(f"✅ Detailed summary saved: {summary_file}")

def main():
    """Run multi-modal combined Game-3 experiments"""
    # Create analyzer
    analyzer = MultiModalAccelGyroCombinedHAR()
    
    # Run experiments
    print("🚀 Starting Multi-Modal Combined Game-3 Experiments")
    print("🔬 Using 15 accelerometer binary + 5 gyroscope features")
    print()
    
    try:
        # Case-1: STM → MotionSense
        case1_results = analyzer.run_case1_stm_to_motionsense()
        
        # Case-2: MotionSense → STM  
        case2_results = analyzer.run_case2_motionsense_to_stm()
        
        # Save and display results
        summary = analyzer.save_results(case1_results, case2_results)
        
        print("\n🎉 Multi-Modal Combined Game-3 Complete!")
        print("🔍 Check results directory for detailed analysis")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
