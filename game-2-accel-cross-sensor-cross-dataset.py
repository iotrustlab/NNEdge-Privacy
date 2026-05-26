#!/usr/bin/env python3
"""
Game-2 Binary Analysis-Optimized Cross-Sensor Adversarial HAR
============================================================

ENHANCED IMPLEMENTATION using scientifically validated binary features from comprehensive analysis:

🏆 TOP 15 MOST TRANSFERABLE BINARY FEATURES (0.262-0.389 transferability):
1. binary_dynamic_activity (0.389 ± 0.012) - Movement detection
2. binary_activity_detected (0.386 ± 0.010) - General activity detection  
3. binary_very_low_intensity (0.386 ± 0.010) - Energy-based classification
4. binary_z_energy_dominant (0.354 ± 0.080) - Axis energy analysis
5. binary_compact_distribution (0.352 ± 0.048) - Statistical distribution
6. binary_very_high_intensity (0.337 ± 0.057) - Energy intensity
7. binary_low_freq_dominant (0.306 ± 0.028) - Frequency domain
8. binary_high_crossing_rate (0.299 ± 0.038) - Temporal dynamics
9. binary_high_dominant_freq (0.282 ± 0.029) - Frequency analysis
10. binary_high_intensity (0.282 ± 0.021) - Energy analysis
11. binary_predictable_pattern (0.281 ± 0.025) - Pattern recognition
12. binary_magnitude_skewed (0.279 ± 0.035) - Statistical shape
13. binary_low_crossing_rate (0.272 ± 0.016) - Temporal pattern
14. binary_xz_positive_corr (0.269 ± 0.046) - Axis correlation
15. binary_strong_autocorr (0.262 ± 0.035) - Pattern consistency

EXPERIMENTAL DESIGN:
Case-1: STM (5 placements + combined) → MotionSense
Case-2: MotionSense → STM (5 placements + combined)

IMPROVEMENTS OVER BASELINE:
- Uses 15 scientifically validated binary features (vs basic temporal)
- Features selected based on cross-sensor transferability analysis
- Expected improvement: 8-15 percentage points over baseline Game-2
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
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from cross_dataset_utils import calculate_vulnerability, get_motionsense_root, get_stm_root

warnings.filterwarnings('ignore')

print("🎭 Game-2 BINARY ANALYSIS-OPTIMIZED ADVERSARIAL HAR")
print("=" * 55)
print("🔬 Using scientifically validated binary features")
print("🏆 Top 15 features with 0.262-0.389 transferability")
print("🎯 Case-1: STM → MotionSense | Case-2: MotionSense → STM")
print("📈 Expected: 8-15% improvement over baseline Game-2")
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

class BinaryAnalysisOptimizedHAR:
    """
    Enhanced Game-2 HAR using scientifically validated binary features
    from comprehensive transferability analysis
    """
    
    def __init__(self, results_dir="results/game-2-binary-optimized"):
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

    def extract_analysis_optimized_binary_features(self, accel_data):
        """
        Extract the top 15 scientifically validated binary features
        based on comprehensive transferability analysis
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
        from scipy import signal
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
        from scipy.stats import skew
        magnitude_skewness = abs(skew(magnitude))
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

    def load_motionsense_data(self):
        """Load MotionSense smartphone accelerometer data"""
        print("📊 Loading MotionSense smartphone accelerometer data...")
        
        base_path = self.motionsense_path
        combined_data = []
        files_loaded = 0
        activities_loaded = set()
        
        # Activity mapping for MotionSense
        ms_activity_mapping = {
            'activity_downstairs': 'downstairs',
            'activity_jogging': 'jogging', 
            'activity_sitting': 'sitting',
            'activity_standing': 'standing',
            'activity_upstairs': 'upstairs',
            'activity_walking': 'walking'
        }
        
        for subject_dir in sorted(os.listdir(base_path)):
            if not subject_dir.startswith('subject_'):
                continue
                
            subject_path = os.path.join(base_path, subject_dir)
            if not os.path.isdir(subject_path):
                continue
            
            for activity_dir in os.listdir(subject_path):
                if activity_dir not in ms_activity_mapping:
                    continue
                    
                activity_path = os.path.join(subject_path, activity_dir)
                if not os.path.isdir(activity_path):
                    continue
                
                # Load accelerometer data - use data.csv format
                data_file = os.path.join(activity_path, 'data.csv')
                if os.path.exists(data_file):
                    try:
                        df = pd.read_csv(data_file)
                        # Check for accelerometer columns
                        if all(col in df.columns for col in ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]']):
                            # Rename columns to match expected format
                            df = df.rename(columns={'acc_x[mg]': 'ax', 'acc_y[mg]': 'ay', 'acc_z[mg]': 'az'})
                            df['activity'] = ms_activity_mapping[activity_dir]
                            df['subject'] = subject_dir
                            combined_data.append(df)
                            files_loaded += 1
                            activities_loaded.add(ms_activity_mapping[activity_dir])
                    except Exception as e:
                        continue
        
        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            
            # Filter to 6 common activities
            common_activities = ['walking', 'sitting', 'standing', 'jogging', 'upstairs', 'downstairs']
            combined_df = combined_df[combined_df['activity'].isin(common_activities)]
            
            print(f" ✅ {len(combined_df)} samples from {files_loaded} files")
            activity_counts = combined_df['activity'].value_counts().to_dict()
            print(f"   📊 Activities: {activity_counts}")
            return combined_df
        else:
            print(" ❌ No MotionSense data found")
            return pd.DataFrame()

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
                                                if all(col in df.columns for col in ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]']):
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
                                            if all(col in df.columns for col in ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]']):
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
        """Create windowed dataset with analysis-optimized binary features"""
        print(f" 🔄 Creating windowed dataset (window={window_size}, overlap={overlap:.1f})")
        
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
                        
                        # Extract analysis-optimized binary features
                        features = self.extract_analysis_optimized_binary_features(window_data)
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
                    
                    # Extract analysis-optimized binary features
                    features = self.extract_analysis_optimized_binary_features(window_data)
                    if features:  # Only add if features extracted successfully
                        features_list.append(features)
                        labels_list.append(self.activity_mapping[activity])
        
        if features_list:
            features_df = pd.DataFrame(features_list)
            print(f" ✅ {len(features_df)} samples, {len(features_df.columns)} binary features")
            return features_df, np.array(labels_list)
        else:
            print(" ❌ No features extracted")
            return pd.DataFrame(), np.array([])

    def train_and_evaluate_model(self, X_train, y_train, X_test, y_test, model_name):
        """Train and evaluate RandomForest model with comprehensive metrics including MI and KL"""
        print(f" 🤖 Training {model_name}...")
        
        # Encode labels
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, y_train_encoded)
        
        # Predictions
        y_pred = rf_model.predict(X_test_scaled)
        y_pred_proba = rf_model.predict_proba(X_test_scaled)
        
        # Standard metrics
        accuracy = accuracy_score(y_test_encoded, y_pred)
        f1 = f1_score(y_test_encoded, y_pred, average='weighted')
        precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        
        # Calculate information-theoretic metrics
        mi_value, nmi_percentage = calculate_mutual_information(y_test_encoded, y_pred)
        kl_avg, kl_normalized, reverse_kl_percentage = calculate_kl_divergence(
            y_test_encoded, y_pred_proba, len(le.classes_)
        )
        vulnerability = calculate_vulnerability(y_pred_proba)
        
        results = {
            'model_name': model_name,
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
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'num_features': X_train.shape[1],
            'num_classes': len(le.classes_),
            'y_true': y_test_encoded.tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist(),
            'feature_importance': rf_model.feature_importances_.tolist(),
            'label_encoder_classes': le.classes_.tolist()
        }
        
        print(
            f" ✅ {model_name}: {accuracy:.1%} accuracy, NMI: {nmi_percentage:.1f}%, "
            f"RKL: {reverse_kl_percentage:.1f}%, Vuln: {vulnerability:.3f}"
        )
        return results, rf_model

    def run_case1_stm_to_motionsense(self):
        """Case-1: Train on STM placements → Test on MotionSense"""
        print("🎯 CASE-1: STM PLACEMENTS → MOTIONSENSE")
        print("=" * 45)
        
        # Load MotionSense test data
        ms_data = self.load_motionsense_data()
        if ms_data.empty:
            print("❌ No MotionSense data available")
            return []
        
        X_test_ms, y_test_ms = self.create_dataset_from_dataframe(ms_data)
        if X_test_ms.empty:
            print("❌ No MotionSense features extracted")
            return []
        
        case1_results = []
        
        # Train individual placement models
        all_stm_data = []
        for placement in self.stm_placements:
            print(f"\n📍 Training STM {placement} → MotionSense")
            
            # Load STM placement data
            stm_data = self.load_stm_placement_data(placement)
            if stm_data.empty:
                continue
            
            all_stm_data.append(stm_data)
            
            # Create training dataset
            X_train, y_train = self.create_dataset_from_dataframe(stm_data)
            if X_train.empty:
                continue
            
            # Align features between train and test
            common_features = list(set(X_train.columns) & set(X_test_ms.columns))
            if not common_features:
                print(f" ❌ No common features between STM {placement} and MotionSense")
                continue
            
            X_train_aligned = X_train[common_features]
            X_test_aligned = X_test_ms[common_features]
            
            # Train and evaluate
            results, model = self.train_and_evaluate_model(
                X_train_aligned, y_train, X_test_aligned, y_test_ms,
                f"STM_{placement}_to_MotionSense"
            )
            case1_results.append(results)
        
        # Train combined STM model
        if all_stm_data:
            print(f"\n📍 Training STM Combined → MotionSense")
            combined_stm = pd.concat(all_stm_data, ignore_index=True)
            X_train_combined, y_train_combined = self.create_dataset_from_dataframe(combined_stm)
            
            if not X_train_combined.empty:
                common_features = list(set(X_train_combined.columns) & set(X_test_ms.columns))
                if common_features:
                    X_train_aligned = X_train_combined[common_features]
                    X_test_aligned = X_test_ms[common_features]
                    
                    results, model = self.train_and_evaluate_model(
                        X_train_aligned, y_train_combined, X_test_aligned, y_test_ms,
                        "STM_Combined_to_MotionSense"
                    )
                    case1_results.append(results)
        
        return case1_results

    def run_case2_motionsense_to_stm(self):
        """Case-2: Train on MotionSense → Test on STM placements"""
        print("\n🎯 CASE-2: MOTIONSENSE → STM PLACEMENTS")
        print("=" * 45)
        
        # Load MotionSense training data
        ms_data = self.load_motionsense_data()
        if ms_data.empty:
            print("❌ No MotionSense data available")
            return []
        
        X_train_ms, y_train_ms = self.create_dataset_from_dataframe(ms_data)
        if X_train_ms.empty:
            print("❌ No MotionSense features extracted")
            return []
        
        case2_results = []
        
        # Test on individual STM placements
        all_stm_data = []
        for placement in self.stm_placements:
            print(f"\n📍 Testing MotionSense → STM {placement}")
            
            # Load STM placement data
            stm_data = self.load_stm_placement_data(placement)
            if stm_data.empty:
                continue
            
            all_stm_data.append(stm_data)
            
            # Create test dataset
            X_test, y_test = self.create_dataset_from_dataframe(stm_data)
            if X_test.empty:
                continue
            
            # Align features
            common_features = list(set(X_train_ms.columns) & set(X_test.columns))
            if not common_features:
                print(f" ❌ No common features between MotionSense and STM {placement}")
                continue
            
            X_train_aligned = X_train_ms[common_features]
            X_test_aligned = X_test[common_features]
            
            # Train and evaluate
            results, model = self.train_and_evaluate_model(
                X_train_aligned, y_train_ms, X_test_aligned, y_test,
                f"MotionSense_to_STM_{placement}"
            )
            case2_results.append(results)
        
        # Test on combined STM
        if all_stm_data:
            print(f"\n📍 Testing MotionSense → STM Combined")
            combined_stm = pd.concat(all_stm_data, ignore_index=True)
            X_test_combined, y_test_combined = self.create_dataset_from_dataframe(combined_stm)
            
            if not X_test_combined.empty:
                common_features = list(set(X_train_ms.columns) & set(X_test_combined.columns))
                if common_features:
                    X_train_aligned = X_train_ms[common_features]
                    X_test_aligned = X_test_combined[common_features]
                    
                    results, model = self.train_and_evaluate_model(
                        X_train_aligned, y_train_ms, X_test_aligned, y_test_combined,
                        "MotionSense_to_STM_Combined"
                    )
                    case2_results.append(results)
        
        return case2_results

    def create_comprehensive_plots(self, all_results, timestamp):
        """Create comprehensive visualization plots including MI and KL metrics"""
        if not all_results:
            print("⚠️ No results to plot")
            return
        
        try:
            # Convert results to DataFrame
            df = pd.DataFrame(all_results)
            
            # Create comprehensive plots
            self._create_performance_overview_plot(df, timestamp)
            self._create_information_theoretic_plots(df, timestamp)
            self._create_class_agnostic_comparison_plot(df, timestamp)
            self._create_metrics_heatmap(df, timestamp)
            
        except Exception as e:
            print(f"⚠️ Error creating plots: {e}")

    def _create_performance_overview_plot(self, df, timestamp):
        """Create overview plot with all key metrics"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Game-2 Binary Optimized: Performance Overview', fontsize=16)
            
            # Define colors for different model types
            colors = []
            for model_name in df['model_name']:
                if 'Combined' in model_name:
                    colors.append('green')
                elif 'MotionSense' in model_name:
                    colors.append('red')
                else:
                    colors.append('blue')
            
            # 1. Accuracy comparison
            bars1 = axes[0,0].bar(range(len(df)), df['accuracy'], color=colors, alpha=0.8, edgecolor='black')
            axes[0,0].set_xlabel('Experiment')
            axes[0,0].set_ylabel('Accuracy')
            axes[0,0].set_title('Accuracy by Experiment')
            axes[0,0].set_xticks(range(len(df)))
            axes[0,0].set_xticklabels(df['model_name'], rotation=45, ha='right')
            axes[0,0].grid(True, alpha=0.3)
            
            # Add values on bars
            for i, (bar, val) in enumerate(zip(bars1, df['accuracy'])):
                axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                              f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 2. F1-Score comparison
            bars2 = axes[0,1].bar(range(len(df)), df['f1_score'], color=colors, alpha=0.8, edgecolor='black')
            axes[0,1].set_xlabel('Experiment')
            axes[0,1].set_ylabel('F1-Score')
            axes[0,1].set_title('F1-Score by Experiment')
            axes[0,1].set_xticks(range(len(df)))
            axes[0,1].set_xticklabels(df['model_name'], rotation=45, ha='right')
            axes[0,1].grid(True, alpha=0.3)
            
            # Add values on bars
            for i, (bar, val) in enumerate(zip(bars2, df['f1_score'])):
                axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                              f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 3. Accuracy vs NMI scatter
            scatter1 = axes[1,0].scatter(df['accuracy'], df['nmi_percentage'], c=colors, s=100, alpha=0.7, edgecolors='black')
            axes[1,0].set_xlabel('Accuracy')
            axes[1,0].set_ylabel('NMI Percentage (%)')
            axes[1,0].set_title('Information Preservation vs Accuracy')
            axes[1,0].grid(True, alpha=0.3)
            
            # Add model name labels
            for i, row in df.iterrows():
                axes[1,0].annotate(row['model_name'], (row['accuracy'], row['nmi_percentage']), 
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # 4. Accuracy vs Reverse KL scatter
            scatter2 = axes[1,1].scatter(df['accuracy'], df['reverse_kl_percentage'], c=colors, s=100, alpha=0.7, edgecolors='black')
            axes[1,1].set_xlabel('Accuracy')
            axes[1,1].set_ylabel('Reverse KL Percentage (%)')
            axes[1,1].set_title('Model Calibration vs Accuracy')
            axes[1,1].grid(True, alpha=0.3)
            
            # Add model name labels
            for i, row in df.iterrows():
                axes[1,1].annotate(row['model_name'], (row['accuracy'], row['reverse_kl_percentage']), 
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.8, label='Combined Models'),
                Patch(facecolor='red', alpha=0.8, label='MotionSense Models'),
                Patch(facecolor='blue', alpha=0.8, label='STM Models')
            ]
            fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
            
            plt.tight_layout()
            plot_file = os.path.join(self.results_dir, f"performance_overview_{timestamp}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Performance overview plot saved: {plot_file}")
            
        except Exception as e:
            print(f"⚠️ Error creating performance overview plot: {e}")

    def _create_information_theoretic_plots(self, df, timestamp):
        """Create plots specifically for MI and KL metrics"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Information-Theoretic Metrics: Game-2 Binary Optimized Analysis', fontsize=16)
            
            # Define colors
            colors = []
            for model_name in df['model_name']:
                if 'Combined' in model_name:
                    colors.append('green')
                elif 'MotionSense' in model_name:
                    colors.append('red')
                else:
                    colors.append('blue')
            
            # 1. NMI Percentage comparison
            bars1 = axes[0,0].bar(range(len(df)), df['nmi_percentage'], color=colors, alpha=0.8, edgecolor='black')
            axes[0,0].set_xlabel('Experiment')
            axes[0,0].set_ylabel('NMI Percentage (%)')
            axes[0,0].set_title('Normalized Mutual Information\n(Information Preservation)')
            axes[0,0].set_xticks(range(len(df)))
            axes[0,0].set_xticklabels(df['model_name'], rotation=45, ha='right')
            axes[0,0].grid(True, alpha=0.3)
            
            # Add values on bars
            for i, (bar, val) in enumerate(zip(bars1, df['nmi_percentage'])):
                axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                              f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # 2. Reverse KL Percentage comparison  
            bars2 = axes[0,1].bar(range(len(df)), df['reverse_kl_percentage'], color=colors, alpha=0.8, edgecolor='black')
            axes[0,1].set_xlabel('Experiment')
            axes[0,1].set_ylabel('Reverse KL Percentage (%)')
            axes[0,1].set_title('Model Calibration Quality\n(Higher = Better Calibrated)')
            axes[0,1].set_xticks(range(len(df)))
            axes[0,1].set_xticklabels(df['model_name'], rotation=45, ha='right')
            axes[0,1].grid(True, alpha=0.3)
            
            # Add values on bars
            for i, (bar, val) in enumerate(zip(bars2, df['reverse_kl_percentage'])):
                axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                              f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # 3. Raw KL divergence (for debugging)
            bars3 = axes[1,0].bar(range(len(df)), df['kl_divergence'], color=colors, alpha=0.8, edgecolor='black')
            axes[1,0].set_xlabel('Experiment')
            axes[1,0].set_ylabel('KL Divergence (Raw)')
            axes[1,0].set_title('Raw KL Divergence\n(Lower = Better)')
            axes[1,0].set_xticks(range(len(df)))
            axes[1,0].set_xticklabels(df['model_name'], rotation=45, ha='right')
            axes[1,0].grid(True, alpha=0.3)
            
            # 4. Combined class-agnostic metrics
            x_pos = np.arange(len(df))
            width = 0.35
            
            bars4a = axes[1,1].bar(x_pos - width/2, df['nmi_percentage'], width, 
                                  label='NMI %', color='lightblue', alpha=0.8, edgecolor='black')
            bars4b = axes[1,1].bar(x_pos + width/2, df['reverse_kl_percentage'], width,
                                  label='Reverse KL %', color='lightcoral', alpha=0.8, edgecolor='black')
            
            axes[1,1].set_xlabel('Experiment')
            axes[1,1].set_ylabel('Percentage (%)')
            axes[1,1].set_title('Class-Agnostic Metrics Comparison')
            axes[1,1].set_xticks(x_pos)
            axes[1,1].set_xticklabels(df['model_name'], rotation=45, ha='right')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = os.path.join(self.results_dir, f"information_theoretic_metrics_{timestamp}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Information-theoretic metrics plot saved: {plot_file}")
            
        except Exception as e:
            print(f"⚠️ Error creating information-theoretic plots: {e}")

    def _create_class_agnostic_comparison_plot(self, df, timestamp):
        """Create plots showing class-agnostic metrics for direct comparison"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle('Class-Agnostic Metrics: Game-2 Binary Optimized (Comparable Across Classes)', fontsize=16)
            
            # Define colors
            colors = []
            for model_name in df['model_name']:
                if 'Combined' in model_name:
                    colors.append('green')
                elif 'MotionSense' in model_name:
                    colors.append('red')
                else:
                    colors.append('blue')
            
            # Sort by NMI for better visualization
            df_sorted = df.sort_values('nmi_percentage', ascending=True)
            colors_sorted = [colors[i] for i in df_sorted.index]
            
            # 1. Horizontal bar chart for NMI
            bars1 = ax1.barh(range(len(df_sorted)), df_sorted['nmi_percentage'], 
                            color=colors_sorted, alpha=0.8, edgecolor='black', linewidth=1)
            ax1.set_xlabel('NMI Percentage (%)', fontweight='bold')
            ax1.set_ylabel('Experiment', fontweight='bold')
            ax1.set_title('Normalized Mutual Information\n(Information Preservation)', fontweight='bold')
            ax1.set_yticks(range(len(df_sorted)))
            ax1.set_yticklabels(df_sorted['model_name'])
            ax1.grid(axis='x', alpha=0.3, linestyle='--')
            
            # Add values on bars
            for i, (bar, val) in enumerate(zip(bars1, df_sorted['nmi_percentage'])):
                ax1.text(val + 1, i, f'{val:.1f}%', va='center', fontweight='bold')
            
            # 2. Horizontal bar chart for Reverse KL
            df_sorted2 = df.sort_values('reverse_kl_percentage', ascending=True)
            colors_sorted2 = [colors[i] for i in df_sorted2.index]
            
            bars2 = ax2.barh(range(len(df_sorted2)), df_sorted2['reverse_kl_percentage'], 
                            color=colors_sorted2, alpha=0.8, edgecolor='black', linewidth=1)
            ax2.set_xlabel('Reverse KL Percentage (%)', fontweight='bold')
            ax2.set_ylabel('Experiment', fontweight='bold')
            ax2.set_title('Model Calibration Quality\n(Prediction Confidence)', fontweight='bold')
            ax2.set_yticks(range(len(df_sorted2)))
            ax2.set_yticklabels(df_sorted2['model_name'])
            ax2.grid(axis='x', alpha=0.3, linestyle='--')
            
            # Add values on bars
            for i, (bar, val) in enumerate(zip(bars2, df_sorted2['reverse_kl_percentage'])):
                ax2.text(val + 1, i, f'{val:.1f}%', va='center', fontweight='bold')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.8, label='Combined Models'),
                Patch(facecolor='red', alpha=0.8, label='MotionSense Models'),
                Patch(facecolor='blue', alpha=0.8, label='STM Models')
            ]
            fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)
            
            plt.tight_layout()
            plot_file = os.path.join(self.results_dir, f"class_agnostic_metrics_{timestamp}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Class-agnostic metrics plot saved: {plot_file}")
            
        except Exception as e:
            print(f"⚠️ Error creating class-agnostic comparison plot: {e}")

    def _create_metrics_heatmap(self, df, timestamp):
        """Create comprehensive metrics heatmap"""
        try:
            # Select metrics for heatmap
            metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'nmi_percentage', 'reverse_kl_percentage']
            
            # Create data matrix
            data_matrix = df[metrics].values
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(data_matrix.T, 
                       annot=True, 
                       fmt='.3f', 
                       cmap='RdYlBu_r',
                       xticklabels=df['model_name'],
                       yticklabels=metrics,
                       center=0.5, 
                       square=False, 
                       linewidths=0.5)
            
            plt.title('Comprehensive Metrics Heatmap: Game-2 Binary Optimized\nAll Metrics Including MI and KL')
            plt.xlabel('Experiments')
            plt.ylabel('Metrics')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plot_file = os.path.join(self.results_dir, f"metrics_heatmap_{timestamp}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"🔥 Metrics heatmap saved: {plot_file}")
            
        except Exception as e:
            print(f"⚠️ Error creating metrics heatmap: {e}")

    def save_results(self, case1_results, case2_results):
        """Save comprehensive results with enhanced MI and KL analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Combine all results
        all_results = case1_results + case2_results
        
        # Check if we have any results
        if not all_results:
            print("\n❌ No results to save - all experiments failed")
            return {
                'timestamp': timestamp,
                'mean_accuracy': 0.0,
                'std_accuracy': 0.0,
                'min_accuracy': 0.0,
                'max_accuracy': 0.0,
                'mean_vulnerability': 0.0,
                'std_vulnerability': 0.0,
                'total_experiments': 0,
                'features_used': 'Top 15 analysis-optimized binary features',
                'case1_results': case1_results,
                'case2_results': case2_results,
                'status': 'failed'
            }
        
        # Calculate comprehensive summary statistics
        accuracies = [r['accuracy'] for r in all_results]
        f1_scores = [r['f1_score'] for r in all_results]
        nmi_percentages = [r.get('nmi_percentage', 0) for r in all_results]
        reverse_kl_percentages = [r.get('reverse_kl_percentage', 0) for r in all_results]
        vulnerabilities = [r.get('vulnerability', 0.0) for r in all_results]
        
        summary = {
            'timestamp': timestamp,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'mean_f1_score': np.mean(f1_scores),
            'std_f1_score': np.std(f1_scores),
            'mean_nmi_percentage': np.mean(nmi_percentages),
            'std_nmi_percentage': np.std(nmi_percentages),
            'mean_reverse_kl_percentage': np.mean(reverse_kl_percentages),
            'std_reverse_kl_percentage': np.std(reverse_kl_percentages),
            'mean_vulnerability': np.mean(vulnerabilities),
            'std_vulnerability': np.std(vulnerabilities),
            'total_experiments': len(all_results),
            'features_used': 'Top 15 analysis-optimized binary features',
            'case1_results': case1_results,
            'case2_results': case2_results,
            'status': 'success'
        }
        
        # Save results
        results_file = os.path.join(self.results_dir, f"binary_optimized_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create comprehensive visualizations
        print("\n🎨 Creating comprehensive visualizations...")
        self.create_comprehensive_plots(all_results, timestamp)
        
        # Print enhanced summary with MI/KL analysis
        print("\n📊 BINARY ANALYSIS-OPTIMIZED Game-2 RESULTS SUMMARY")
        print("=" * 65)
        print(f"📈 Mean Accuracy:      {summary['mean_accuracy']:.1%} ± {summary['std_accuracy']:.1%}")
        print(f"📊 Mean F1-Score:      {summary['mean_f1_score']:.1%} ± {summary['std_f1_score']:.1%}")
        print(f"🧠 Mean NMI:           {summary['mean_nmi_percentage']:.1f}% ± {summary['std_nmi_percentage']:.1f}%")
        print(f"🎯 Mean Reverse KL:    {summary['mean_reverse_kl_percentage']:.1f}% ± {summary['std_reverse_kl_percentage']:.1f}%")
        print(f"🔐 Mean Vulnerability: {summary['mean_vulnerability']:.3f} ± {summary['std_vulnerability']:.3f}")
        print(f"📊 Accuracy Range:     {summary['min_accuracy']:.1%} - {summary['max_accuracy']:.1%}")
        print(f"🧪 Total Experiments:  {summary['total_experiments']}")
        print(f"🔑 Features:           15 scientifically validated binary features")
        
        # Information-theoretic analysis
        print(f"\n🧠 INFORMATION-THEORETIC ANALYSIS:")
        print("-" * 40)
        print("📊 Class-Agnostic Metrics (Comparable across datasets):")
        print(f"   • NMI (Information Preservation):  {summary['mean_nmi_percentage']:.1f}% ± {summary['std_nmi_percentage']:.1f}%")
        print(f"   • Reverse KL (Model Calibration):  {summary['mean_reverse_kl_percentage']:.1f}% ± {summary['std_reverse_kl_percentage']:.1f}%")
        print(f"   • Vulnerability (Top-Guess Confidence): {summary['mean_vulnerability']:.3f} ± {summary['std_vulnerability']:.3f}")
        
        # Quality assessment
        nmi_mean = summary['mean_nmi_percentage']
        rkl_mean = summary['mean_reverse_kl_percentage']
        
        if nmi_mean >= 60:
            nmi_quality = "🎉 EXCELLENT"
        elif nmi_mean >= 40:
            nmi_quality = "✅ GOOD"
        else:
            nmi_quality = "⚠️ POOR"
            
        if rkl_mean >= 50:
            rkl_quality = "🎉 EXCELLENT"
        elif rkl_mean >= 30:
            rkl_quality = "✅ GOOD"
        else:
            rkl_quality = "⚠️ POOR"
        
        print(f"\n🎭 QUALITY ASSESSMENT:")
        print(f"   • Information Preservation: {nmi_quality} ({nmi_mean:.1f}%)")
        print(f"   • Model Calibration:        {rkl_quality} ({rkl_mean:.1f}%)")
        
        print(f"\n💾 Results saved: {results_file}")
        print(f"📊 Visualizations created in: {self.results_dir}")
        
        return summary

def main():
    """Run binary analysis-optimized Game-2 experiments"""
    # Create analyzer
    analyzer = BinaryAnalysisOptimizedHAR()
    
    # Run experiments
    print("🚀 Starting Binary Analysis-Optimized Game-2 Experiments")
    print("🔬 Using top 15 scientifically validated binary features")
    print()
    
    try:
        # Case-1: STM → MotionSense
        case1_results = analyzer.run_case1_stm_to_motionsense()
        
        # Case-2: MotionSense → STM  
        case2_results = analyzer.run_case2_motionsense_to_stm()
        
        # Save and display results
        summary = analyzer.save_results(case1_results, case2_results)
        
        print("\n🎉 Binary Analysis-Optimized Game-2 Complete!")
        print("🔍 Check results directory for detailed analysis")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
