#!/usr/bin/env python3
"""
Game-2 Gyroscope-Based Cross-Sensor Adversarial HAR
===================================================

This script implements Game-2 cross-sensor cross-dataset HAR using basic statistical 
gyroscope features for STM-UC and UCI_HAR datasets.

🎯 FEATURES: 16 Basic Statistical Gyroscope Features
- X, Y, Z axes: mean, std, min, max (12 features)
- Magnitude: mean, std, min, max (4 features)

📊 DATASETS:
- STM-UC: 11 users, 5 sensor placements, 6 activities
- UCI_HAR: 30 subjects, smartphone placement, 6 activities  

🎭 EXPERIMENTAL DESIGN:
- Cross-sensor cross-dataset evaluation with information-theoretic privacy metrics
- Common activities: walking, sitting, standing, upstairs, downstairs, laying

📈 INFORMATION-THEORETIC METRICS:
- Normalized Mutual Information (NMI): Class-agnostic information preservation
- Normalized KL-Divergence: Class-agnostic model calibration assessment
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, normalized_mutual_info_score
from scipy.stats import entropy, skew
from scipy import signal, stats
from datetime import datetime
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from cross_sensor_path_utils import get_stm_uc_root, get_uci_root
import warnings
warnings.filterwarnings('ignore')

print("🎭 Game-2 GYROSCOPE-BASED CROSS-SENSOR HAR")
print("📊 STM-UC ↔ UCI_HAR Cross-Dataset Analysis (6 common activities)")
print("=" * 60)
print("🔬 Using 16 basic statistical gyroscope features")
print("🏆 Continuous features: mean, std, min, max for X, Y, Z, magnitude")
print("🎯 Common activities: walking, sitting, standing, upstairs, downstairs, laying")
print("📈 Enhanced with Information-Theoretic Privacy Metrics")
print()

class GyroscopeBasedHAR:
    """
    Enhanced Game-2 HAR using continuous gyroscope features
    with cross-dataset STM-UC and UCI_HAR compatibility and information-theoretic metrics
    Uses 16 basic statistical features: mean, std, min, max for X, Y, Z, and magnitude
    """
    
    def __init__(self, results_dir="results/game-2-gyro-stm-uci"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Dataset paths
        self.stm_path = str(get_stm_uc_root())
        self.uci_path = str(get_uci_root())
        
        # Activity mappings (6 common activities)
        self.activity_mapping = {
            'walking': 0, 'sitting': 1, 'standing': 2,
            'upstairs': 3, 'downstairs': 4, 'laying': 5
        }
        
        # STM placements
        self.stm_placements = ['left-ankle', 'left-wrist', 'right-ankle', 'right-pocket', 'right-wrist']
        
        print(f"📁 Results directory: {results_dir}")
        print(f"🎯 Target activities: {list(self.activity_mapping.keys())}")
        print(f"📍 STM placements: {self.stm_placements}")
        print(f"🔧 Using 16 continuous gyroscope features: mean, std, min, max")
        print()

    def _calculate_kl_divergence(self, y_true_encoded, y_pred_proba, label_encoder):
        """
        Calculate normalized KL-Divergence between true and predicted distributions
        """
        try:
            n_classes = len(label_encoder.classes_)
            n_samples = len(y_true_encoded)
            
            # Theoretical maximum KL (uniform vs one-hot)
            theoretical_max_kl = np.log(n_classes)
            
            # Ensure predicted probabilities are valid (avoid log(0))
            pred_dist = np.clip(y_pred_proba, 1e-15, 1.0)
            pred_dist = pred_dist / pred_dist.sum(axis=1, keepdims=True)
            
            # Calculate KL divergence for each sample
            # For one-hot true distribution, KL = -log(q[true_class])
            kl_divs = []
            for i in range(n_samples):
                true_class = y_true_encoded[i]
                predicted_prob_for_true_class = pred_dist[i, true_class]
                
                # KL divergence for one-hot vs predicted distribution
                kl_div = -np.log(predicted_prob_for_true_class)
                
                # Only include valid KL divergences
                if not (np.isnan(kl_div) or np.isinf(kl_div)) and kl_div >= 0:
                    kl_divs.append(kl_div)
            
            # Debug information
            avg_confidence = np.mean([pred_dist[i, y_true_encoded[i]] for i in range(n_samples)])
            print(f"    🔍 KL Debug: {len(kl_divs)}/{n_samples} valid samples, avg_confidence: {avg_confidence:.3f}")
            
            if len(kl_divs) > 0:
                avg_kl = np.mean(kl_divs)
                # Normalize by theoretical maximum
                normalized_kl = min(1.0, avg_kl / theoretical_max_kl)
                reverse_kl = 1.0 - normalized_kl  # Higher = better calibration
                print(f"    🔍 KL Values: avg_kl={avg_kl:.3f}, theoretical_max={theoretical_max_kl:.3f}, normalized={normalized_kl:.3f}, reverse={reverse_kl:.3f}")
                return normalized_kl, reverse_kl
            else:
                print(f"    ⚠️ No valid KL divergences found! All predictions likely have numerical issues.")
                # Return worst case: maximum divergence (poor calibration)
                return 1.0, 0.0
                
        except Exception as e:
            print(f"⚠️ Error calculating KL-Divergence: {e}")
            return 1.0, 0.0

    def extract_basic_gyro_features(self, gyro_data):
        """
        Extract basic statistical gyroscope features: mean, std, min, max
        These are fundamental, robust, and highly transferable across sensors
        """
        if len(gyro_data) == 0:
            return np.array([])
            
        # Handle different column naming conventions
        if 'gyro_x[mdps]' in gyro_data.columns:
            # STM dataset format
            gx = np.array(gyro_data['gyro_x[mdps]'])
            gy = np.array(gyro_data['gyro_y[mdps]']) 
            gz = np.array(gyro_data['gyro_z[mdps]'])
        elif 'gx' in gyro_data.columns:
            # UCI_HAR format
            gx = np.array(gyro_data['gx'])
            gy = np.array(gyro_data['gy']) 
            gz = np.array(gyro_data['gz'])
        else:
            return np.array([])
        
        # Calculate magnitude
        magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
        
        features = {}
        
        # BASIC STATISTICAL FEATURES for each axis
        
        # X-axis features
        features['gyro_x_mean'] = np.mean(gx)
        features['gyro_x_std'] = np.std(gx)
        features['gyro_x_min'] = np.min(gx)
        features['gyro_x_max'] = np.max(gx)
        
        # Y-axis features
        features['gyro_y_mean'] = np.mean(gy)
        features['gyro_y_std'] = np.std(gy)
        features['gyro_y_min'] = np.min(gy)
        features['gyro_y_max'] = np.max(gy)
        
        # Z-axis features
        features['gyro_z_mean'] = np.mean(gz)
        features['gyro_z_std'] = np.std(gz)
        features['gyro_z_min'] = np.min(gz)
        features['gyro_z_max'] = np.max(gz)
        
        # Magnitude features (orientation-independent)
        features['magnitude_mean'] = np.mean(magnitude)
        features['magnitude_std'] = np.std(magnitude)
        features['magnitude_min'] = np.min(magnitude)
        features['magnitude_max'] = np.max(magnitude)
        
        return np.array(list(features.values()))

    def load_stm_uc_data(self):
        """Load STM-UC dataset (6 common activities)"""
        print("📊 Loading STM-UC Dataset (6 common activities)...")
        all_data = []
        users_loaded = 0
        
        # Target activities (6 common activities)
        target_activities = ['Walking', 'Sitting', 'Standing', 'Upstairs', 'Downstairs', 'Laying']
        
        # Activity mapping
        activity_map = {
            'Walking': 'walking',
            'Sitting': 'sitting', 
            'Standing': 'standing',
            'Upstairs': 'upstairs',
            'Downstairs': 'downstairs',
            'Laying': 'laying'
        }
        
        for user_folder in sorted(os.listdir(self.stm_path)):
            if not user_folder.startswith('User '):
                continue
                
            user_id = user_folder
            user_path = os.path.join(self.stm_path, user_folder)
            processed_path = os.path.join(user_path, 'Processed')
            
            if not os.path.exists(processed_path):
                continue
            
            user_activities = 0
            
            for activity in target_activities:
                activity_dir = os.path.join(processed_path, activity)
                if not os.path.exists(activity_dir):
                    continue
                
                for placement in self.stm_placements:
                    # Handle stairs structure (multiple files in placement directory)
                    if activity in ['Upstairs', 'Downstairs']:
                        placement_dir = os.path.join(activity_dir, placement)
                        if os.path.exists(placement_dir):
                            csv_files = [f for f in os.listdir(placement_dir) if f.endswith('.csv')]
                            for csv_file in csv_files:
                                file_path = os.path.join(placement_dir, csv_file)
                                try:
                                    df = pd.read_csv(file_path)
                                    if all(col in df.columns for col in ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']):
                                        df['user_id'] = user_id
                                        df['activity'] = activity_map[activity]
                                        df['placement'] = placement
                                        all_data.append(df[['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]', 'user_id', 'activity', 'placement']])
                                        user_activities += 1
                                except Exception as e:
                                    print(f"⚠️ Error loading {file_path}: {e}")
                    
                    # Handle regular structure (single file per placement)
                    else:
                        file_path = os.path.join(activity_dir, f"{placement}.csv")
                        if os.path.exists(file_path):
                            try:
                                df = pd.read_csv(file_path)
                                if all(col in df.columns for col in ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']):
                                    df['user_id'] = user_id
                                    df['activity'] = activity_map[activity]
                                    df['placement'] = placement
                                    all_data.append(df[['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]', 'user_id', 'activity', 'placement']])
                                    user_activities += 1
                            except Exception as e:
                                print(f"⚠️ Error loading {file_path}: {e}")
            
            if user_activities > 0:
                users_loaded += 1
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"✅ STM-UC: {len(combined_df)} samples from {users_loaded} users")
            print(f"   📊 Activities: {dict(combined_df['activity'].value_counts())}")
            print(f"   📍 Placements: {dict(combined_df['placement'].value_counts())}")
            return combined_df
        else:
            print("❌ No STM-UC data loaded")
            return pd.DataFrame()

    def load_uci_har_data(self):
        """Load UCI_HAR dataset (6 common activities)"""
        print("📊 Loading UCI_HAR Dataset (6 common activities)...")
        all_data = []
        subjects_loaded = 0
        
        # Activity mapping for UCI_HAR dataset (6 common activities)
        uci_activity_mapping = {
            'activity_WALKING': 'walking',
            'activity_SITTING': 'sitting',
            'activity_STANDING': 'standing',
            'activity_WALKING_UPSTAIRS': 'upstairs',
            'activity_WALKING_DOWNSTAIRS': 'downstairs',
            'activity_LAYING': 'laying'
        }
        
        for subject_folder in sorted(os.listdir(self.uci_path)):
            if not subject_folder.startswith('subject_'):
                continue
                
            subject_id = subject_folder
            subject_path = os.path.join(self.uci_path, subject_folder)
            
            if not os.path.isdir(subject_path):
                continue
            
            subject_activities = 0
            
            for activity_folder in os.listdir(subject_path):
                if activity_folder not in uci_activity_mapping:
                    continue
                    
                activity_path = os.path.join(subject_path, activity_folder)
                if not os.path.isdir(activity_path):
                    continue
                
                # Load data.csv file
                data_file = os.path.join(activity_path, 'data.csv')
                if os.path.exists(data_file):
                    try:
                        df = pd.read_csv(data_file)
                        if all(col in df.columns for col in ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']):
                            df['user_id'] = subject_id
                            df['activity'] = uci_activity_mapping[activity_folder]
                            df['placement'] = 'smartphone'  # UCI_HAR uses smartphone placement
                            all_data.append(df[['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]', 'user_id', 'activity', 'placement']])
                            subject_activities += 1
                    except Exception as e:
                        print(f"⚠️ Error loading {data_file}: {e}")
            
            if subject_activities > 0:
                subjects_loaded += 1
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"✅ UCI_HAR: {len(combined_df)} samples from {subjects_loaded} subjects")
            print(f"   📊 Activities: {dict(combined_df['activity'].value_counts())}")
            return combined_df
        else:
            print("❌ No UCI_HAR data loaded")
            return pd.DataFrame()

    def create_dataset_from_dataframe(self, df, window_size=128, overlap=0.5):
        """Create windowed dataset with continuous gyroscope features"""
        print(f" 🔄 Creating windowed dataset (window={window_size}, overlap={overlap:.1f})")
        
        features_list = []
        labels_list = []
        users_list = []
        placements_list = []
        
        step_size = int(window_size * (1 - overlap))
        
        for activity in df['activity'].unique():
            activity_data = df[df['activity'] == activity]
            
            # Group by user and placement
            for user_id in activity_data['user_id'].unique():
                user_data = activity_data[activity_data['user_id'] == user_id]
                
                for placement in user_data['placement'].unique():
                    placement_data = user_data[user_data['placement'] == placement]
                    
                    # Create windows
                    for start_idx in range(0, max(1, len(placement_data) - window_size + 1), step_size):
                        end_idx = start_idx + window_size
                        if end_idx <= len(placement_data):
                            window_data = placement_data.iloc[start_idx:end_idx]
                            
                            # Extract continuous gyroscope features
                            gyro_features = self.extract_basic_gyro_features(window_data)
                            
                            if len(gyro_features) > 0:
                                features_list.append(gyro_features)
                                labels_list.append(activity)
                                users_list.append(user_id)
                                placements_list.append(placement)
        
        if features_list:
            # Convert to numpy array for continuous features
            features_array = np.array(features_list)
            
            results_df = pd.DataFrame({
                'features': list(features_array),  # Each row contains a feature vector
                'activity': labels_list,
                'user_id': users_list,
                'placement': placements_list
            })
            
            print(f" ✅ Created {len(results_df)} windowed samples")
            print(f"    📊 Activities: {dict(pd.Series(labels_list).value_counts())}")
            print(f"    📍 Placements: {dict(pd.Series(placements_list).value_counts())}")
            print(f"    🔧 Features per window: {features_array.shape[1]} continuous features")
            return results_df
        else:
            print(" ❌ No windowed samples created")
            return pd.DataFrame()

    def train_and_evaluate_model(self, train_df, test_df, train_name, test_name, placement=None):
        """Train model and evaluate with information-theoretic metrics"""
        model_name = f"{train_name}_to_{test_name}"
        if placement:
            model_name += f"_{placement}"
        
        print(f"  🔬 Training: {model_name}")
        
        # Prepare training data - extract feature vectors
        X_train = np.array([feat for feat in train_df['features'].values])
        y_train = train_df['activity'].values
        
        # Prepare test data - extract feature vectors  
        X_test = np.array([feat for feat in test_df['features'].values])
        y_test = test_df['activity'].values
        
        # Standardize features for better transferability
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Label encoding for activities
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Train model - use continuous features directly
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train_encoded)
        
        # Predictions
        y_pred_encoded = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # Convert back to original labels
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Calculate NMI
        nmi = normalized_mutual_info_score(y_test, y_pred)
        nmi_percentage = nmi * 100
        
        # Calculate KL-Divergence
        kl_divergence, reverse_kl = self._calculate_kl_divergence(y_test_encoded, y_pred_proba, label_encoder)
        reverse_kl_percentage = reverse_kl * 100
        
        # Detailed classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        results = {
            'model_name': model_name,
            'train_dataset': train_name,
            'test_dataset': test_name,
            'placement': placement,
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'accuracy': accuracy,
            'f1_score': f1,
            'nmi': nmi,
            'nmi_percentage': nmi_percentage,
            'kl_divergence': kl_divergence,
            'reverse_kl': reverse_kl,
            'reverse_kl_percentage': reverse_kl_percentage,
            'classification_report': report,
            'feature_type': '16_continuous_gyro_features',
            'feature_count': X_test.shape[1]  # Number of continuous features used
        }
        
        print(f"    📊 Accuracy: {accuracy:.3f}, F1: {f1:.3f}, NMI: {nmi_percentage:.1f}%, Reverse KL: {reverse_kl_percentage:.1f}%")
        print(f"    🔧 Used {X_test.shape[1]} continuous gyroscope features")
        
        return results

    def _create_information_theoretic_plots(self, df, title_prefix, save_dir):
        """Create comprehensive information-theoretic visualization plots"""
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(df)))
        
        # 1. NMI Percentage Bar Chart
        bars1 = ax1.bar(range(len(df)), df['nmi_percentage'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Placement Configuration', fontweight='bold')
        ax1.set_ylabel('NMI Percentage (%)', fontweight='bold')
        ax1.set_title('Normalized Mutual Information\\n(Higher = Better Information Preservation)', fontweight='bold', pad=15)
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels(df['placement'], rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Add NMI values on bars
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 2. Reverse KL Percentage Bar Chart
        bars2 = ax2.bar(range(len(df)), df['reverse_kl_percentage'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Placement Configuration', fontweight='bold')
        ax2.set_ylabel('Reverse KL-Divergence (%)', fontweight='bold')
        ax2.set_title('Reverse KL: Model Calibration\\n(Higher = Better Calibration)', fontweight='bold', pad=15)
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels(df['placement'], rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Add Reverse KL values on bars
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 3. NMI vs Accuracy Scatter Plot
        scatter1 = ax3.scatter(df['accuracy'], df['nmi_percentage'], 
                              c=range(len(df)), cmap='Set3', s=100, alpha=0.8, edgecolors='black')
        ax3.set_xlabel('Accuracy', fontweight='bold')
        ax3.set_ylabel('NMI (%)', fontweight='bold')
        ax3.set_title('NMI vs Accuracy\\nInformation-Performance Trade-off', fontweight='bold', pad=15)
        ax3.grid(True, alpha=0.3)
        
        # Add placement labels to scatter points
        for i, placement in enumerate(df['placement']):
            ax3.annotate(placement, (df['accuracy'].iloc[i], df['nmi_percentage'].iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
        
        # 4. Reverse KL vs Accuracy Scatter Plot
        scatter2 = ax4.scatter(df['accuracy'], df['reverse_kl_percentage'], 
                              c=range(len(df)), cmap='Set3', s=100, alpha=0.8, edgecolors='black')
        ax4.set_xlabel('Accuracy', fontweight='bold')
        ax4.set_ylabel('Reverse KL (%)', fontweight='bold')
        ax4.set_title('Reverse KL vs Accuracy\\nCalibration-Performance Trade-off', fontweight='bold', pad=15)
        ax4.grid(True, alpha=0.3)
        
        # Add placement labels to scatter points
        for i, placement in enumerate(df['placement']):
            ax4.annotate(placement, (df['accuracy'].iloc[i], df['reverse_kl_percentage'].iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(save_dir, f"{title_prefix.lower().replace(' ', '_')}_information_theoretic_analysis.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_file

    def run_case1_stm_to_uci(self):
        """Case-1: STM-UC (5 placements + combined) → UCI_HAR"""
        print("🎯 Case-1: STM-UC → UCI_HAR")
        print("=" * 50)
        
        # Load datasets
        stm_data = self.load_stm_uc_data()
        uci_data = self.load_uci_har_data()
        
        if stm_data.empty or uci_data.empty:
            print("❌ Failed to load datasets for Case-1")
            return []
        
        case1_results = []
        case1_dir = os.path.join(self.results_dir, "STM_UC_to_UCI_HAR")
        os.makedirs(case1_dir, exist_ok=True)
        
        # Create UCI test dataset
        print("📊 Creating UCI_HAR test dataset...")
        uci_test_df = self.create_dataset_from_dataframe(uci_data)
        
        if uci_test_df.empty:
            print("❌ Failed to create UCI test dataset")
            return []
        
        # Test each STM placement
        for placement in self.stm_placements:
            print(f"\n📍 Testing STM placement: {placement}")
            
            # Create STM training dataset for this placement
            stm_placement_data = stm_data[stm_data['placement'] == placement]
            if stm_placement_data.empty:
                print(f"⚠️ No data for placement: {placement}")
                continue
            
            stm_train_df = self.create_dataset_from_dataframe(stm_placement_data)
            
            if stm_train_df.empty:
                print(f"⚠️ No training data created for placement: {placement}")
                continue
            
            # Train and evaluate
            result = self.train_and_evaluate_model(
                stm_train_df, uci_test_df, 
                f"STM_UC_{placement}", "UCI_HAR", placement
            )
            
            case1_results.append(result)
            
            # Save individual result
            result_file = os.path.join(case1_dir, f"results_{placement}.json")
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        
        # Combined STM → UCI
        print(f"\n📍 Testing STM combined (all placements)")
        stm_combined_df = self.create_dataset_from_dataframe(stm_data)
        
        if not stm_combined_df.empty:
            result = self.train_and_evaluate_model(
                stm_combined_df, uci_test_df,
                "STM_UC_Combined", "UCI_HAR", "combined"
            )
            case1_results.append(result)
            
            # Save combined result
            result_file = os.path.join(case1_dir, "results_combined.json")
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        
        # Create visualization
        if case1_results:
            df = pd.DataFrame(case1_results)
            plot_file = self._create_information_theoretic_plots(df, "STM-UC to UCI_HAR", case1_dir)
            print(f"📊 Case-1 visualization saved: {plot_file}")
        
        return case1_results

    def run_case2_uci_to_stm(self):
        """Case-2: UCI_HAR → STM-UC (5 placements + combined)"""
        print("\n🎯 Case-2: UCI_HAR → STM-UC")
        print("=" * 50)
        
        # Load datasets
        stm_data = self.load_stm_uc_data()
        uci_data = self.load_uci_har_data()
        
        if stm_data.empty or uci_data.empty:
            print("❌ Failed to load datasets for Case-2")
            return []
        
        case2_results = []
        case2_dir = os.path.join(self.results_dir, "UCI_HAR_to_STM_UC")
        os.makedirs(case2_dir, exist_ok=True)
        
        # Create UCI training dataset
        print("📊 Creating UCI_HAR training dataset...")
        uci_train_df = self.create_dataset_from_dataframe(uci_data)
        
        if uci_train_df.empty:
            print("❌ Failed to create UCI training dataset")
            return []
        
        # Test each STM placement
        for placement in self.stm_placements:
            print(f"\n📍 Testing STM placement: {placement}")
            
            # Create STM test dataset for this placement
            stm_placement_data = stm_data[stm_data['placement'] == placement]
            if stm_placement_data.empty:
                print(f"⚠️ No data for placement: {placement}")
                continue
            
            stm_test_df = self.create_dataset_from_dataframe(stm_placement_data)
            
            if stm_test_df.empty:
                print(f"⚠️ No test data created for placement: {placement}")
                continue
            
            # Train and evaluate
            result = self.train_and_evaluate_model(
                uci_train_df, stm_test_df,
                "UCI_HAR", f"STM_UC_{placement}", placement
            )
            
            case2_results.append(result)
            
            # Save individual result
            result_file = os.path.join(case2_dir, f"results_{placement}.json")
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        
        # UCI → STM Combined
        print(f"\n📍 Testing STM combined (all placements)")
        stm_combined_df = self.create_dataset_from_dataframe(stm_data)
        
        if not stm_combined_df.empty:
            result = self.train_and_evaluate_model(
                uci_train_df, stm_combined_df,
                "UCI_HAR", "STM_UC_Combined", "combined"
            )
            case2_results.append(result)
            
            # Save combined result
            result_file = os.path.join(case2_dir, "results_combined.json")
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        
        # Create visualization
        if case2_results:
            df = pd.DataFrame(case2_results)
            plot_file = self._create_information_theoretic_plots(df, "UCI_HAR to STM-UC", case2_dir)
            print(f"📊 Case-2 visualization saved: {plot_file}")
        
        return case2_results

    def save_results(self, case1_results, case2_results):
        """Save comprehensive results with information-theoretic analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Combine all results
        all_results = case1_results + case2_results
        
        if not all_results:
            print("❌ No results to save")
            return {
                'timestamp': timestamp,
                'case1_results': case1_results,
                'case2_results': case2_results,
                'status': 'failed'
            }
        
        # Calculate summary statistics
        df = pd.DataFrame(all_results)
        
        summary = {
            'overall': {
                'mean_accuracy': df['accuracy'].mean(),
                'std_accuracy': df['accuracy'].std(),
                'mean_f1': df['f1_score'].mean(),
                'std_f1': df['f1_score'].std(),
                'mean_nmi': df['nmi_percentage'].mean(),
                'std_nmi': df['nmi_percentage'].std(),
                'mean_reverse_kl': df['reverse_kl_percentage'].mean(),
                'std_reverse_kl': df['reverse_kl_percentage'].std()
            }
        }
        
        # Split by case direction
        case1_df = df[df['train_dataset'].str.contains('STM_UC')]
        case2_df = df[df['train_dataset'].str.contains('UCI_HAR')]
        
        if len(case1_df) > 0:
            summary['case1_stm_to_uci'] = {
                'count': len(case1_df),
                'mean_accuracy': case1_df['accuracy'].mean(),
                'std_accuracy': case1_df['accuracy'].std(),
                'mean_f1': case1_df['f1_score'].mean(),
                'std_f1': case1_df['f1_score'].std(),
                'mean_nmi': case1_df['nmi_percentage'].mean(),
                'std_nmi': case1_df['nmi_percentage'].std(),
                'mean_reverse_kl': case1_df['reverse_kl_percentage'].mean(),
                'std_reverse_kl': case1_df['reverse_kl_percentage'].std()
            }
        
        if len(case2_df) > 0:
            summary['case2_uci_to_stm'] = {
                'count': len(case2_df),
                'mean_accuracy': case2_df['accuracy'].mean(),
                'std_accuracy': case2_df['accuracy'].std(),
                'mean_f1': case2_df['f1_score'].mean(),
                'std_f1': case2_df['f1_score'].std(),
                'mean_nmi': case2_df['nmi_percentage'].mean(),
                'std_nmi': case2_df['nmi_percentage'].std(),
                'mean_reverse_kl': case2_df['reverse_kl_percentage'].mean(),
                'std_reverse_kl': case2_df['reverse_kl_percentage'].std()
            }
        
        # Save comprehensive results
        results = {
            'timestamp': timestamp,
            'summary_statistics': summary,
            'case1_results': case1_results,
            'case2_results': case2_results,
            'total_experiments': len(all_results),
            'features_used': '16 continuous gyroscope features (mean, std, min, max)',
            'datasets': 'STM-UC ↔ UCI_HAR',
            'activities': list(self.activity_mapping.keys()),
            'status': 'success'
        }
        
        # Save results
        results_file = os.path.join(self.results_dir, f"game2_gyro_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print detailed summary
        print("\n📊 GYROSCOPE-BASED Game-2 RESULTS SUMMARY")
        print("=" * 60)
        print(f"📈 Overall Performance:")
        print(f"   🎯 Mean Accuracy: {summary['overall']['mean_accuracy']:.3f} ± {summary['overall']['std_accuracy']:.3f}")
        print(f"   🎯 Mean F1-Score: {summary['overall']['mean_f1']:.3f} ± {summary['overall']['std_f1']:.3f}")
        print(f"   🎯 Mean NMI: {summary['overall']['mean_nmi']:.1f}% ± {summary['overall']['std_nmi']:.1f}%")
        print(f"   🎯 Mean Reverse KL: {summary['overall']['mean_reverse_kl']:.1f}% ± {summary['overall']['std_reverse_kl']:.1f}%")
        
        if 'case1_stm_to_uci' in summary:
            print(f"\n📊 Case-1 (STM-UC → UCI_HAR): {summary['case1_stm_to_uci']['count']} experiments")
            print(f"   Accuracy: {summary['case1_stm_to_uci']['mean_accuracy']:.3f} ± {summary['case1_stm_to_uci']['std_accuracy']:.3f}")
            print(f"   NMI: {summary['case1_stm_to_uci']['mean_nmi']:.1f}% ± {summary['case1_stm_to_uci']['std_nmi']:.1f}%")
            print(f"   Reverse KL: {summary['case1_stm_to_uci']['mean_reverse_kl']:.1f}% ± {summary['case1_stm_to_uci']['std_reverse_kl']:.1f}%")
        
        if 'case2_uci_to_stm' in summary:
            print(f"\n📊 Case-2 (UCI_HAR → STM-UC): {summary['case2_uci_to_stm']['count']} experiments")
            print(f"   Accuracy: {summary['case2_uci_to_stm']['mean_accuracy']:.3f} ± {summary['case2_uci_to_stm']['std_accuracy']:.3f}")
            print(f"   NMI: {summary['case2_uci_to_stm']['mean_nmi']:.1f}% ± {summary['case2_uci_to_stm']['std_nmi']:.1f}%")
            print(f"   Reverse KL: {summary['case2_uci_to_stm']['mean_reverse_kl']:.1f}% ± {summary['case2_uci_to_stm']['std_reverse_kl']:.1f}%")
        
        print(f"\n🧪 Total Experiments: {len(all_results)}")
        print(f"🔑 Features: 16 continuous gyroscope features (mean, std, min, max)")
        print(f"📊 Datasets: STM-UC (5 placements) ↔ UCI_HAR (smartphone)")
        print(f"💾 Results saved: {results_file}")
        
        # Best performing models
        best_by_accuracy = df.loc[df['accuracy'].idxmax()]
        best_by_nmi = df.loc[df['nmi_percentage'].idxmax()]
        best_by_rkl = df.loc[df['reverse_kl_percentage'].idxmax()]
        
        print(f"\n🏆 Best Performance:")
        print(f"   Accuracy: {best_by_accuracy['placement']} = {best_by_accuracy['accuracy']:.3f}")
        print(f"   NMI: {best_by_nmi['placement']} = {best_by_nmi['nmi_percentage']:.1f}%")
        print(f"   Reverse KL: {best_by_rkl['placement']} = {best_by_rkl['reverse_kl_percentage']:.1f}%")
        
        return results

def main():
    """Run gyroscope-based Game-2 experiments with information-theoretic metrics"""
    # Create analyzer
    analyzer = GyroscopeBasedHAR()
    
    # Run experiments
    print("🚀 Starting Gyroscope-Based Game-2 Experiments")
    print("🔬 Using 16 continuous statistical gyroscope features")
    print("📊 STM-UC ↔ UCI_HAR Cross-Dataset Analysis")
    print("📈 Enhanced with Information-Theoretic Privacy Metrics")
    print()
    
    try:
        # Case-1: STM-UC → UCI_HAR
        case1_results = analyzer.run_case1_stm_to_uci()
        
        # Case-2: UCI_HAR → STM-UC
        case2_results = analyzer.run_case2_uci_to_stm()
        
        # Save and display results
        summary = analyzer.save_results(case1_results, case2_results)
        
        print("\n🎉 Gyroscope-Based Game-2 Complete!")
        print("🔍 Check results directory for detailed analysis and visualizations")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
