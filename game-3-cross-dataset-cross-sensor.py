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
from datetime import datetime
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from cross_sensor_path_utils import get_motionsense_root, get_stm_six_root
import warnings
warnings.filterwarnings('ignore')

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
        self.stm_path = str(get_stm_six_root())
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
        """Train Random Forest and evaluate performance"""
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
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f" ✅ {experiment_name}: Acc={accuracy:.3f}, F1={f1:.3f}")
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': feature_importance.head(10).to_dict('records'),
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
        """Save comprehensive results with analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        all_results = case1_results + case2_results
        
        if not all_results:
            print("❌ No results to save")
            return {
                'case1_results': case1_results,
                'case2_results': case2_results,
                'status': 'failed'
            }
        
        # Calculate summary statistics
        accuracies = [r['accuracy'] for r in all_results]
        summary = {
            'timestamp': timestamp,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
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
        
        # Print summary
        print("\n📊 MULTI-MODAL COMBINED Game-3 RESULTS SUMMARY")
        print("=" * 55)
        print(f"📈 Mean Accuracy: {summary['mean_accuracy']:.1%} ± {summary['std_accuracy']:.1%}")
        print(f"📊 Range: {summary['min_accuracy']:.1%} - {summary['max_accuracy']:.1%}")
        print(f"🧪 Total Experiments: {summary['total_experiments']}")
        print(f"🔑 Features: 20 multimodal (15 accel + 5 gyro)")
        print(f"💾 Results saved: {results_file}")
        
        return summary

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
