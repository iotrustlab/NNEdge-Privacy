#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
from pathlib import Path
import glob
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class IncrementalBeliefUpdater:
    """
    Incremental belief updating system for prior-based adversary.
    Updates population priors iteratively as new users are observed.
    """
    
    def __init__(self, data_dir="../Data", results_dir="incremental_belief_results"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.sample_rate = 50
        self.binary_column = 'dec_tree_out_1'
        self.placement_mapping = {
            'left-ankle': 'left_ankle', 'left-wrist': 'left_wrist',
            'right-ankle': 'right_ankle', 'right-pocket': 'right_pocket', 
            'right-wrist': 'right_wrist'
        }
        self.activities = ['Downstairs', 'Jogging', 'Laying', 'Sitting', 'Standing', 'Upstairs', 'Walking']
        
        # Model components (will be updated incrementally)
        self.clusterer = None
        self.scaler = None
        self.pca = None
        
        # Belief tracking
        self.belief_history = []
        self.user_order = []
        self.P_A_given_C_history = []
        self.P_C_given_A_history = []
        
        print("🔄 INCREMENTAL BELIEF UPDATER INITIALIZED")
        print("=" * 60)
        print("📊 Sequential population prior learning from binary streams")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"💾 Results directory: {self.results_dir}")
        
    def debug_data_structure(self, max_subjects=3):
        """Debug function to explore the actual data structure"""
        print(f"\n🔍 DEBUGGING DATA STRUCTURE")
        print(f"=" * 50)
        print(f"📂 Base directory: {self.data_dir}")
        
        # Check if base directory exists
        if not self.data_dir.exists():
            print(f"❌ Base directory does not exist!")
            return
        
        # List all subdirectories
        subdirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        print(f"📁 Found {len(subdirs)} subdirectories:")
        
        for i, subdir in enumerate(subdirs[:10]):  # Show first 10
            print(f"  {i+1}. {subdir.name}")
            
            # Look for user-like patterns
            if any(pattern in subdir.name.lower() for pattern in ['user', 'subject']):
                print(f"    🎯 User directory detected")
                
                # Check for Processed subdirectory
                processed_dir = subdir / "Processed"
                if processed_dir.exists():
                    print(f"      � Has Processed/ subdirectory")
                    
                    # Check activity folders
                    activity_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
                    print(f"      🎭 Activity folders: {len(activity_dirs)}")
                    
                    for activity_dir in activity_dirs[:3]:  # Show first 3
                        csv_files = list(activity_dir.glob("*.csv"))
                        print(f"        📁 {activity_dir.name}/: {len(csv_files)} CSV files")
                        if csv_files:
                            for csv_file in csv_files[:3]:  # Show first 3 files
                                print(f"          📄 {csv_file.name}")
                
                # Check other contents
                other_items = [item for item in subdir.iterdir() if item.name != "Processed"]
                if other_items:
                    print(f"      📄 Other items: {[item.name for item in other_items[:3]]}")
        
        print(f"\n🔍 Testing file finding for first {max_subjects} subjects:")
        for subject_id in range(1, max_subjects + 1):
            print(f"\n--- Subject {subject_id} ---")
            files = self._find_csv_files(subject_id)
            if files:
                print(f"✅ Found {len(files)} files")
                
                # Test metadata parsing for first few files
                print(f"    🧪 Testing metadata parsing:")
                for i, file_path in enumerate(files[:3]):
                    activity, placement = self._parse_file_metadata(file_path)
                    try:
                        rel_path = file_path.relative_to(self.data_dir)
                    except ValueError:
                        rel_path = file_path
                    print(f"      {i+1}. {rel_path}")
                    print(f"         Activity: {activity}, Placement: {placement}")
            else:
                print(f"❌ No files found")
        
        print(f"\n📊 STRUCTURE SUMMARY:")
        print(f"Expected structure: User X/Processed/Activity/placement.csv")
        print(f"Activities: {self.activities}")
        print(f"Placements: {list(self.placement_mapping.values())}")
        
    def _find_csv_files(self, subject_id):
        """Find CSV files for a subject, handling nested activity/placement structure"""
        files_found = []
        
        print(f"  🔍 Searching for Subject {subject_id} files...")
        
        # Base directory patterns
        base_patterns = [
            f"User {subject_id}",
            f"User{subject_id}",
            f"user {subject_id}",
            f"user{subject_id}",
            f"subject{subject_id}",
            f"Subject {subject_id}"
        ]
        
        # Search in multiple locations
        search_dirs = [
            self.data_dir,
            self.data_dir.parent,
            self.data_dir / "Data" if (self.data_dir / "Data").exists() else self.data_dir
        ]
        
        for search_dir in search_dirs:
            for pattern in base_patterns:
                user_dir = search_dir / pattern
                if user_dir.exists():
                    print(f"    📁 Found user directory: {user_dir}")
                    
                    # Check for the standard structure: User X/Processed/Activity/placement.csv
                    processed_dir = user_dir / "Processed"
                    if processed_dir.exists():
                        print(f"      📂 Found Processed directory")
                        
                        # Look through each activity folder
                        for activity in self.activities:
                            activity_dir = processed_dir / activity
                            if activity_dir.exists():
                                csv_files = list(activity_dir.glob("*.csv"))
                                files_found.extend(csv_files)
                                if csv_files:
                                    print(f"        🎯 {activity}: {len(csv_files)} files")
                    
                    # Also check other possible subdirectories
                    subdirs_to_check = ["Raw", "data", "Data", "csv", "CSV"]
                    for subdir_name in subdirs_to_check:
                        subdir = user_dir / subdir_name
                        if subdir.exists():
                            # Look for activity folders within this subdirectory
                            for activity in self.activities:
                                activity_dir = subdir / activity
                                if activity_dir.exists():
                                    activity_files = list(activity_dir.glob("*.csv"))
                                    files_found.extend(activity_files)
                                    if activity_files:
                                        print(f"        🎯 {subdir_name}/{activity}: {len(activity_files)} files")
                            
                            # Also get any direct CSV files in subdirectory
                            direct_files = list(subdir.glob("*.csv"))
                            files_found.extend(direct_files)
                    
                    # Get any CSV files directly in user directory
                    direct_user_files = list(user_dir.glob("*.csv"))
                    files_found.extend(direct_user_files)
                    
                    # Recursive search as fallback
                    recursive_files = list(user_dir.rglob("*.csv"))
                    files_found.extend(recursive_files)
        
        # Also try more aggressive search patterns if needed
        if not files_found:
            print(f"    🔄 Trying aggressive search patterns...")
            aggressive_patterns = [
                f"**/*{subject_id}*/**/*.csv",           # Any folder containing subject ID
                f"**/*user*{subject_id}*/**/*.csv",     # Nested user folders
                f"**/User*{subject_id}*/**/*.csv",      # Nested User folders
                f"**/*subject*{subject_id}*/**/*.csv",  # Subject folders
            ]
            
            for search_dir in search_dirs:
                for pattern in aggressive_patterns:
                    matches = list(search_dir.glob(pattern))
                    files_found.extend(matches)
                    if matches:
                        print(f"      🎯 Aggressive pattern found {len(matches)} files")
        
        # Remove duplicates
        files_found = list(set(files_found))
        
        print(f"  📊 Found {len(files_found)} CSV files for Subject {subject_id}")
        if files_found:
            print(f"    📂 Sample paths:")
            for i, f in enumerate(files_found[:5]):  # Show first 5 files
                try:
                    # Try to get relative path for cleaner display
                    rel_path = f.relative_to(self.data_dir)
                except ValueError:
                    # If not relative to data_dir, just show the full path
                    rel_path = f
                print(f"      {i+1}. {rel_path}")
            if len(files_found) > 5:
                print(f"      ... and {len(files_found)-5} more")
        
        return files_found
    
    def _parse_file_metadata(self, file_path):
        """Extract activity and placement from file path"""
        filename = file_path.stem.lower()  # e.g., "left-ankle"
        parent_dirs = [p.name for p in file_path.parents]
        path_str = str(file_path).lower()
        
        # Extract activity - for the standard structure, it's the parent directory
        activity = None
        
        # First check if parent directory is an activity
        for parent in parent_dirs:
            for act in self.activities:
                if parent == act:  # Exact match
                    activity = act
                    break
            if activity:
                break
        
        # If not found, check in path string
        if activity is None:
            for act in self.activities:
                if act.lower() in path_str:
                    activity = act
                    break
        
        # Extract placement from filename - this is very specific now
        placement = None
        
        # Direct mapping from common filename patterns
        filename_to_placement = {
            'left-ankle': 'left_ankle',
            'left_ankle': 'left_ankle',
            'leftankle': 'left_ankle',
            'right-ankle': 'right_ankle', 
            'right_ankle': 'right_ankle',
            'rightankle': 'right_ankle',
            'left-wrist': 'left_wrist',
            'left_wrist': 'left_wrist',
            'leftwrist': 'left_wrist',
            'right-wrist': 'right_wrist',
            'right_wrist': 'right_wrist', 
            'rightwrist': 'right_wrist',
            'right-pocket': 'right_pocket',
            'right_pocket': 'right_pocket',
            'rightpocket': 'right_pocket',
            'pocket': 'right_pocket'
        }
        
        # Check for exact filename match
        if filename in filename_to_placement:
            placement = filename_to_placement[filename]
        
        # If not found, check for partial matches
        if placement is None:
            placement_variations = {
                'left_ankle': ['left-ankle', 'left_ankle', 'leftankle', 'ankle_left', 'ankle-left', 'la'],
                'right_ankle': ['right-ankle', 'right_ankle', 'rightankle', 'ankle_right', 'ankle-right', 'ra'],
                'left_wrist': ['left-wrist', 'left_wrist', 'leftwrist', 'wrist_left', 'wrist-left', 'lw'],
                'right_wrist': ['right-wrist', 'right_wrist', 'rightwrist', 'wrist_right', 'wrist-right', 'rw'],
                'right_pocket': ['right-pocket', 'right_pocket', 'rightpocket', 'pocket_right', 'pocket-right', 'pocket', 'rp', 'p']
            }
            
            for standard_name, variations in placement_variations.items():
                for variation in variations:
                    if (variation in filename or 
                        variation in path_str or 
                        any(variation in parent.lower() for parent in parent_dirs)):
                        placement = standard_name
                        break
                if placement:
                    break
        
        return activity, placement
    
    def _load_subject_data(self, subject_id):
        """Load all data for a specific subject"""
        print(f"  👤 Loading Subject {subject_id}...")
        files = self._find_csv_files(subject_id)
        
        if not files:
            print(f"  ⚠️  No files found for Subject {subject_id}")
            return pd.DataFrame()
        
        all_data = []
        loaded_combinations = set()
        
        for file_path in files:
            try:
                # Extract metadata from file path
                activity, placement = self._parse_file_metadata(file_path)
                
                if activity is None:
                    print(f"    ❓ Cannot determine activity from: {file_path.name}")
                    continue
                    
                if placement is None:
                    print(f"    ❓ Cannot determine placement from: {file_path.name}")
                    continue
                
                # Check if we already loaded this combination
                combo = (activity, placement)
                if combo in loaded_combinations:
                    continue
                
                # Load data
                df = pd.read_csv(file_path)
                
                # Check for binary column
                binary_columns = [col for col in df.columns if 'dec_tree' in col.lower() and 'out' in col.lower()]
                if not binary_columns:
                    print(f"    ❌ No binary column in: {file_path.name}")
                    continue
                
                binary_col = binary_columns[0]  # Use first matching column
                
                # Clean and prepare data
                df_clean = df.copy()
                df_clean['subject_id'] = subject_id
                df_clean['activity'] = activity
                df_clean['placement'] = placement
                df_clean['timestamp'] = df_clean.index / self.sample_rate
                df_clean['y'] = df_clean[binary_col].astype(int)
                
                # Keep only essential columns
                essential_cols = ['subject_id', 'activity', 'placement', 'timestamp', 'y']
                df_final = df_clean[essential_cols].copy()
                
                all_data.append(df_final)
                loaded_combinations.add(combo)
                
                print(f"    ✅ {activity}/{placement}: {len(df_final)} samples")
                
            except Exception as e:
                print(f"    ❌ Error loading {file_path.name}: {e}")
                continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"  ✅ Subject {subject_id}: {len(combined_df)} total samples, {len(loaded_combinations)} activity/placement combinations")
            return combined_df
        else:
            print(f"  ❌ No data loaded for Subject {subject_id}")
            return pd.DataFrame()
    
    def _create_windows(self, data, window_size_sec=10, stride_sec=1.0):
        """Create sliding windows from continuous data"""
        print(f"  🔄 Creating {window_size_sec}s windows with {stride_sec}s stride...")
        
        windows = []
        window_size_samples = int(window_size_sec * self.sample_rate)
        stride_samples = int(stride_sec * self.sample_rate)
        
        for subject_id in data['subject_id'].unique():
            subject_data = data[data['subject_id'] == subject_id]
            
            for activity in subject_data['activity'].unique():
                activity_data = subject_data[subject_data['activity'] == activity]
                
                for placement in activity_data['placement'].unique():
                    placement_data = activity_data[activity_data['placement'] == placement]
                    placement_data = placement_data.sort_values('timestamp').reset_index(drop=True)
                    
                    # Create windows
                    for start_idx in range(0, len(placement_data) - window_size_samples + 1, stride_samples):
                        end_idx = start_idx + window_size_samples
                        window_data = placement_data.iloc[start_idx:end_idx]
                        
                        if len(window_data) == window_size_samples:
                            windows.append({
                                'subject_id': subject_id,
                                'activity': activity,
                                'placement': placement,
                                'start_time': window_data['timestamp'].iloc[0],
                                'end_time': window_data['timestamp'].iloc[-1],
                                'y_sequence': window_data['y'].values
                            })
        
        print(f"  ✅ Created {len(windows)} windows")
        return pd.DataFrame(windows)
    
    def _extract_binary_features(self, windows_df):
        """Extract comprehensive binary features from windows"""
        print(f"  🔧 Extracting binary features from {len(windows_df)} windows...")
        
        features = []
        
        for idx, row in windows_df.iterrows():
            y = row['y_sequence']
            placement = row['placement']
            
            # Basic statistics
            duty_cycle = np.mean(y)
            
            # Edge statistics  
            edges = np.diff(y.astype(int))
            rises = np.sum(edges == 1)
            falls = np.sum(edges == -1)
            edge_rate = (rises + falls) / len(y) * self.sample_rate if len(y) > 0 else 0
            
            # Run length statistics
            if len(y) > 0:
                runs = []
                current_run = 1
                for i in range(1, len(y)):
                    if y[i] == y[i-1]:
                        current_run += 1
                    else:
                        runs.append(current_run / self.sample_rate)  # Convert to seconds
                        current_run = 1
                runs.append(current_run / self.sample_rate)
                
                mean_run_length = np.mean(runs) if runs else 0
                var_run_length = np.var(runs) if len(runs) > 1 else 0
                max_run_length = np.max(runs) if runs else 0
            else:
                mean_run_length = var_run_length = max_run_length = 0
            
            # Entropy measures
            if 0 < duty_cycle < 1:
                bernoulli_entropy = -duty_cycle * np.log2(duty_cycle) - (1-duty_cycle) * np.log2(1-duty_cycle)
            else:
                bernoulli_entropy = 0
            
            # Autocorrelation (short lags)
            autocorrs = []
            for lag in range(1, min(26, len(y))):
                if len(y) > lag:
                    y_lag = y[lag:]
                    y_base = y[:-lag]
                    if len(set(y_base)) > 1 and len(set(y_lag)) > 1:
                        autocorr = np.corrcoef(y_base, y_lag)[0,1]
                        autocorr = autocorr if not np.isnan(autocorr) else 0
                    else:
                        autocorr = 0
                    autocorrs.append(autocorr)
                else:
                    autocorrs.append(0)
            
            # Pad autocorrelations to fixed length
            while len(autocorrs) < 25:
                autocorrs.append(0)
            
            # Placement encoding (one-hot)
            placement_encoding = [0] * 5
            placement_names = ['left_ankle', 'left_wrist', 'right_ankle', 'right_pocket', 'right_wrist']
            if placement in placement_names:
                placement_encoding[placement_names.index(placement)] = 1
            
            # Combine all features
            feature_vector = [
                duty_cycle, edge_rate, rises/len(y)*self.sample_rate if len(y) > 0 else 0, 
                falls/len(y)*self.sample_rate if len(y) > 0 else 0,
                mean_run_length, var_run_length, max_run_length, bernoulli_entropy
            ] + autocorrs + placement_encoding
            
            features.append(feature_vector)
        
        feature_names = [
            'duty_cycle', 'edge_rate', 'rise_rate', 'fall_rate',
            'mean_run_length', 'var_run_length', 'max_run_length', 'bernoulli_entropy'
        ] + [f'autocorr_lag_{i}' for i in range(1, 26)] + [
            'placement_left_ankle', 'placement_left_wrist', 'placement_right_ankle', 
            'placement_right_pocket', 'placement_right_wrist'
        ]
        
        print(f"  ✅ Extracted {len(feature_names)}-dimensional features")
        return np.array(features), feature_names
    
    def _initialize_model(self, X_init, y_init, k_clusters=16):
        """Initialize the model with first batch of data"""
        print(f"  🔧 Initializing model with {len(X_init)} samples...")
        
        # Initialize preprocessors
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_init)
        
        self.pca = PCA(n_components=0.95, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Initialize clusterer
        self.clusterer = GaussianMixture(
            n_components=k_clusters, 
            covariance_type='diag',
            random_state=42,
            max_iter=100
        )
        self.clusterer.fit(X_pca)
        
        # Compute initial beliefs
        P_C_train = self.clusterer.predict_proba(X_pca)
        initial_beliefs = self._compute_beliefs(P_C_train, y_init)
        
        print(f"  ✅ Model initialized: K={k_clusters}, PCA dims={X_pca.shape[1]}")
        return initial_beliefs
    
    def _compute_beliefs(self, P_C, y, smooth_alpha=1.0):
        """Compute P(A|C) and P(C|A) from cluster responsibilities and labels"""
        activities = sorted(set(y))
        n_clusters = P_C.shape[1]
        
        # Compute P(C|A) with smoothing
        P_C_given_A = np.zeros((len(activities), n_clusters))
        
        for i, activity in enumerate(activities):
            activity_mask = (y == activity)
            if np.sum(activity_mask) > 0:
                P_C_given_A[i] = np.mean(P_C[activity_mask], axis=0)
            
            # Add smoothing
            P_C_given_A[i] += smooth_alpha
            P_C_given_A[i] /= np.sum(P_C_given_A[i])
        
        # Compute priors P(A)
        activity_counts = pd.Series(y).value_counts()
        P_A = np.array([activity_counts.get(act, 0) for act in activities])
        P_A = P_A / np.sum(P_A) if np.sum(P_A) > 0 else P_A
        
        # Compute P(A|C) via Bayes
        P_A_given_C = np.zeros((len(activities), n_clusters))
        for c in range(n_clusters):
            for i, activity in enumerate(activities):
                P_A_given_C[i, c] = P_C_given_A[i, c] * P_A[i]
            
            # Normalize
            col_sum = np.sum(P_A_given_C[:, c])
            if col_sum > 0:
                P_A_given_C[:, c] /= col_sum
        
        return {
            'activities': activities,
            'P_A': P_A,
            'P_C_given_A': P_C_given_A,
            'P_A_given_C': P_A_given_C,
            'n_samples': len(y)
        }
    
    def _update_beliefs(self, prev_beliefs, new_P_C, new_y, update_weight=0.1):
        """Update beliefs incrementally with new user data"""
        # Compute beliefs for new user
        new_beliefs = self._compute_beliefs(new_P_C, new_y)
        
        # Weighted update of P(C|A)
        updated_P_C_given_A = (1 - update_weight) * prev_beliefs['P_C_given_A'] + \
                              update_weight * new_beliefs['P_C_given_A']
        
        # Update P(A) based on combined sample counts
        total_samples = prev_beliefs['n_samples'] + new_beliefs['n_samples']
        updated_P_A = (prev_beliefs['n_samples'] * prev_beliefs['P_A'] + \
                       new_beliefs['n_samples'] * new_beliefs['P_A']) / total_samples
        
        # Recompute P(A|C)
        updated_P_A_given_C = np.zeros_like(prev_beliefs['P_A_given_C'])
        for c in range(updated_P_C_given_A.shape[1]):
            for i in range(len(prev_beliefs['activities'])):
                updated_P_A_given_C[i, c] = updated_P_C_given_A[i, c] * updated_P_A[i]
            
            # Normalize
            col_sum = np.sum(updated_P_A_given_C[:, c])
            if col_sum > 0:
                updated_P_A_given_C[:, c] /= col_sum
        
        return {
            'activities': prev_beliefs['activities'],
            'P_A': updated_P_A,
            'P_C_given_A': updated_P_C_given_A,
            'P_A_given_C': updated_P_A_given_C,
            'n_samples': total_samples
        }
    
    def _predict_with_beliefs(self, X_test, beliefs):
        """Make predictions using current beliefs"""
        if self.scaler is None or self.pca is None or self.clusterer is None:
            raise ValueError("Model not initialized")
        
        # Preprocess test data
        X_scaled = self.scaler.transform(X_test)
        X_pca = self.pca.transform(X_scaled)
        
        # Get cluster probabilities
        P_C_test = self.clusterer.predict_proba(X_pca)
        
        # Compute activity probabilities
        P_A_test = P_C_test @ beliefs['P_A_given_C'].T
        
        # Predict
        y_pred = np.argmax(P_A_test, axis=1)
        y_pred_labels = [beliefs['activities'][i] for i in y_pred]
        
        return y_pred_labels, P_A_test
    
    def run_incremental_experiment(self, train_subjects, test_subjects, window_size=10):
        """Run the incremental belief updating experiment"""
        print(f"\n🚀 Starting Incremental Belief Updating Experiment")
        print(f"=" * 60)
        print(f"  📊 Train subjects: {train_subjects}")
        print(f"  🎯 Test subjects: {test_subjects}")
        print(f"  ⏱️  Window size: {window_size}s")
        
        # Initialize results tracking
        results = {
            'user_order': [],
            'belief_evolution': [],
            'test_performance': [],
            'metadata': {
                'train_subjects': train_subjects,
                'test_subjects': test_subjects,
                'window_size': window_size
            }
        }
        
        # Load and prepare test data (will be evaluated after each belief update)
        print(f"\n📂 Loading test data...")
        test_data = []
        for test_subject in test_subjects:
            subject_data = self._load_subject_data(test_subject)
            if not subject_data.empty:
                test_data.append(subject_data)
        
        if not test_data:
            print("❌ No test data found!")
            return results
        
        test_combined = pd.concat(test_data, ignore_index=True)
        test_windows = self._create_windows(test_combined, window_size)
        
        if len(test_windows) == 0:
            print("❌ No test windows created!")
            return results
            
        X_test, _ = self._extract_binary_features(test_windows)
        y_test = test_windows['activity'].values
        
        print(f"✅ Test data prepared: {len(test_windows)} windows")
        
        # Process training subjects incrementally
        print(f"\n🔄 Processing training subjects incrementally...")
        
        current_beliefs = None
        
        for i, train_subject in enumerate(train_subjects):
            print(f"\n{'='*50}")
            print(f"👤 Processing Subject {train_subject} (Step {i+1}/{len(train_subjects)})")
            print(f"{'='*50}")
            
            # Load subject data
            subject_data = self._load_subject_data(train_subject)
            if subject_data.empty:
                print(f"  ⚠️  Skipping subject {train_subject} - no data")
                continue
            
            # Create windows and extract features
            windows = self._create_windows(subject_data, window_size)
            if len(windows) == 0:
                print(f"  ⚠️  Skipping subject {train_subject} - no windows")
                continue
                
            X_subject, _ = self._extract_binary_features(windows)
            y_subject = windows['activity'].values
            
            print(f"  📊 Subject {train_subject}: {len(windows)} windows, {len(set(y_subject))} activities")
            
            if i == 0:
                # Initialize model with first subject
                current_beliefs = self._initialize_model(X_subject, y_subject)
            else:
                # Update beliefs with new subject
                X_scaled = self.scaler.transform(X_subject)
                X_pca = self.pca.transform(X_scaled)
                P_C_subject = self.clusterer.predict_proba(X_pca)
                
                prev_beliefs = current_beliefs.copy()
                current_beliefs = self._update_beliefs(
                    current_beliefs, P_C_subject, y_subject, 
                    update_weight=0.1  # 10% weight for new user
                )
                
                # Compute belief change
                belief_change = np.mean(np.abs(current_beliefs['P_A_given_C'] - prev_beliefs['P_A_given_C']))
                print(f"  🔄 Beliefs updated (change: {belief_change:.4f})")
            
            # Evaluate on test data with current beliefs
            y_pred, P_A_test = self._predict_with_beliefs(X_test, current_beliefs)
            
            # Compute metrics
            accuracy = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            print(f"  📈 Test Performance:")
            print(f"    • Accuracy: {accuracy:.3f}")
            print(f"    • Macro-F1: {macro_f1:.3f}")
            
            # Save results
            results['user_order'].append(train_subject)
            results['belief_evolution'].append({
                'user_id': train_subject,
                'step': i + 1,
                'P_A': current_beliefs['P_A'].tolist(),
                'n_samples': current_beliefs['n_samples'],
                'activities': current_beliefs['activities']
            })
            results['test_performance'].append({
                'user_id': train_subject,
                'step': i + 1,
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'n_train_samples': current_beliefs['n_samples']
            })
        
        # Generate final evaluation
        if current_beliefs:
            y_pred_final, _ = self._predict_with_beliefs(X_test, current_beliefs)
            final_report = classification_report(y_test, y_pred_final, output_dict=True, zero_division=0)
            
            # Save results
            self._save_results(results, final_report, y_test, y_pred_final, window_size)
        
        return results
    
    def _save_results(self, results, final_report, y_test, y_pred, window_size):
        """Save experimental results"""
        output_dir = self.results_dir / f"W{window_size}s"
        output_dir.mkdir(exist_ok=True)
        
        # Save results JSON
        with open(output_dir / "incremental_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save final classification report
        report_text = "INCREMENTAL BELIEF UPDATING - FINAL RESULTS\n"
        report_text += "=" * 60 + "\n"
        report_text += f"Window Size: {window_size}s\n"
        report_text += f"Training Users: {results['metadata']['train_subjects']}\n"
        report_text += f"Test Users: {results['metadata']['test_subjects']}\n"
        report_text += f"Processing Order: {results['user_order']}\n"
        report_text += "\n" + "=" * 60 + "\n"
        
        # Add classification metrics
        for activity in self.activities:
            if activity in final_report:
                metrics = final_report[activity]
                report_text += f"{activity:>12}: precision={metrics['precision']:.3f}, "
                report_text += f"recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}, "
                report_text += f"support={metrics['support']}\n"
        
        report_text += f"\nAccuracy: {final_report['accuracy']:.3f}\n"
        report_text += f"Macro-F1: {final_report['macro avg']['f1-score']:.3f}\n"
        
        with open(output_dir / "final_classification_report.txt", 'w') as f:
            f.write(report_text)
        
        # Create performance evolution plots
        if len(results['test_performance']) > 1:
            perf_df = pd.DataFrame(results['test_performance'])
            
            plt.figure(figsize=(15, 10))
            
            # Performance evolution
            plt.subplot(2, 2, 1)
            plt.plot(perf_df['step'], perf_df['accuracy'], 'bo-', linewidth=2, markersize=8, label='Accuracy')
            plt.xlabel('Training Users Added')
            plt.ylabel('Test Accuracy')
            plt.title('Incremental Learning: Accuracy Evolution')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.subplot(2, 2, 2)
            plt.plot(perf_df['step'], perf_df['macro_f1'], 'ro-', linewidth=2, markersize=8, label='Macro-F1')
            plt.xlabel('Training Users Added')
            plt.ylabel('Test Macro-F1')
            plt.title('Incremental Learning: Macro-F1 Evolution')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Sample accumulation
            plt.subplot(2, 2, 3)
            plt.plot(perf_df['step'], perf_df['n_train_samples'], 'go-', linewidth=2, markersize=8)
            plt.xlabel('Training Users Added')
            plt.ylabel('Total Training Samples')
            plt.title('Training Data Accumulation')
            plt.grid(True, alpha=0.3)
            
            # Performance vs samples
            plt.subplot(2, 2, 4)
            plt.scatter(perf_df['n_train_samples'], perf_df['macro_f1'], c=perf_df['step'], 
                       cmap='viridis', s=100, alpha=0.7)
            plt.colorbar(label='Step')
            plt.xlabel('Total Training Samples')
            plt.ylabel('Test Macro-F1')
            plt.title('Performance vs Training Data Size')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "incremental_learning_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=self.activities)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.activities, yticklabels=self.activities)
        plt.title(f'Final Confusion Matrix (W={window_size}s)')
        plt.ylabel('True Activity')
        plt.xlabel('Predicted Activity')
        plt.tight_layout()
        plt.savefig(output_dir / "final_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n💾 Results saved to: {output_dir}")
        print(f"📊 Final performance: Accuracy={final_report['accuracy']:.3f}, Macro-F1={final_report['macro avg']['f1-score']:.3f}")

def main():
    """Main execution function"""
    print("🔄 INCREMENTAL BELIEF UPDATER")
    print("=" * 60)
    
    # Initialize system
    believer = IncrementalBeliefUpdater()
    
    # Debug data structure first
    print("\n🛠️  DEBUGGING DATA STRUCTURE...")
    believer.debug_data_structure(max_subjects=3)
    
    # Ask user if they want to continue
    print(f"\n{'='*60}")
    response = input("🤔 Continue with full experiment? (y/n): ").lower().strip()
    if response != 'y':
        print("👋 Exiting...")
        return
    
    # Configuration
    train_subjects = [1, 2, 3, 4, 6, 7, 8, 9]  # 8 training subjects
    test_subjects = [5, 10, 11]                 # 3 test subjects
    window_sizes = [10, 5]                      # Multiple window sizes
    
    # Run experiments
    for window_size in window_sizes:
        print(f"\n{'='*70}")
        print(f"🔄 EXPERIMENT: {window_size}s WINDOWS")
        print(f"{'='*70}")
        
        results = believer.run_incremental_experiment(
            train_subjects=train_subjects,
            test_subjects=test_subjects, 
            window_size=window_size
        )
        
        if results and results['test_performance']:
            final_perf = results['test_performance'][-1]
            print(f"\n🎯 FINAL RESULTS (W={window_size}s):")
            print(f"  📊 Final Accuracy: {final_perf['accuracy']:.3f}")
            print(f"  📊 Final Macro-F1: {final_perf['macro_f1']:.3f}")
            print(f"  👥 Users processed: {len(results['user_order'])}")
            print(f"  📈 Total training samples: {final_perf['n_train_samples']}")
        else:
            print(f"❌ Experiment failed for window size {window_size}s")

if __name__ == "__main__":
    main()
