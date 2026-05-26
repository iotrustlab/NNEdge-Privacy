#!/usr/bin/env python3
"""
Prior-Based Adversary (A₂) for Binary Wearable Streams
=====================================================

Implementation of A₂ (Template-Mixture) and A₃ (Unlabeled Adaptation)
following the specification for binary 50Hz wearable streams.

Key Features:
- Binary-only features (no raw IMU)
- 8-train / 3-test subject splits
- Template-Mixture clustering approach
- Optional A₃ adaptation with prior-shift reweighting
- Rigorous anti-leakage protocols
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import glob
from collections import defaultdict, Counter
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

class PriorBasedAdversary:
    """
    Implementation of A₂ (Prior-Based) and A₃ (Adaptation) adversaries
    for binary wearable stream analysis
    """
    
    def __init__(self, data_dir="../Data", results_dir="prior_adversary_results"):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.binary_column = 'dec_tree_out_1'
        
        # Configuration (defaults from spec)
        self.config = {
            'WINDOWS': [10, 5, 20],  # Start with 10s
            'STRIDE_SEC': 1.0,
            'SAMPLE_RATE': 50,
            'CLUSTERER': "GMM_DIAG",
            'K_GRID': [8, 12, 16, 24],
            'SMOOTH_ALPHA': 1.0,
            'CALIBRATION': "temperature",
            'EM_ITERS': 2,
            'UNLABELED_BUDGETS': [0, 30, 60, 120],
            'SEED': 42
        }
        
        # Dataset characteristics
        self.subjects = list(range(1, 12))  # 11 subjects
        self.placements = ['left-ankle', 'right-ankle', 'left-wrist', 'right-wrist', 'right-pocket']
        self.activities = ['Downstairs', 'Jogging', 'Laying', 'Sitting', 'Standing', 'Upstairs', 'Walking']
        
        # Create organized directory structure
        self._setup_directory_structure()
        
        np.random.seed(self.config['SEED'])
        
        print("🎯 PRIOR-BASED ADVERSARY (A₂) IMPLEMENTATION")
        print("=" * 60)
        print("📊 Template-Mixture approach with binary features only")
        print(f"👥 Subjects: {len(self.subjects)} (8-train / 3-test splits)")
        print(f"📍 Placements: {len(self.placements)}")
        print(f"🏃 Activities: {len(self.activities)}")
        
    def _setup_directory_structure(self):
        """Create organized directory structure for results"""
        # Main results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Adversary-specific directories
        self.a0_dir = os.path.join(self.results_dir, "A0_Baseline")
        self.a2_dir = os.path.join(self.results_dir, "A2_Template_Mixture")
        self.a3_dir = os.path.join(self.results_dir, "A3_Adaptation")
        
        # Subdirectories for each adversary
        for adversary_dir in [self.a0_dir, self.a2_dir, self.a3_dir]:
            os.makedirs(adversary_dir, exist_ok=True)
            os.makedirs(os.path.join(adversary_dir, "predictions"), exist_ok=True)
            os.makedirs(os.path.join(adversary_dir, "reports"), exist_ok=True)
            os.makedirs(os.path.join(adversary_dir, "confusion_matrices"), exist_ok=True)
            os.makedirs(os.path.join(adversary_dir, "plots"), exist_ok=True)
            os.makedirs(os.path.join(adversary_dir, "models"), exist_ok=True)
        
        # General directories
        os.makedirs(os.path.join(self.results_dir, "splits"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "features"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "summary_plots"), exist_ok=True)
        
        print(f"📁 Directory structure created in: {self.results_dir}")
        
    def generate_splits(self, n_splits=20):
        """Generate 8-train / 3-test subject splits"""
        print(f"\n📋 Generating {n_splits} subject splits...")
        
        splits = []
        for split_id in range(n_splits):
            # Shuffle subjects with fixed seed for reproducibility
            np.random.seed(self.config['SEED'] + split_id)
            shuffled_subjects = np.random.permutation(self.subjects)
            
            train_subjects = shuffled_subjects[:8].tolist()
            test_subjects = shuffled_subjects[8:11].tolist()
            
            splits.append({
                'split_id': split_id,
                'train_subjects': train_subjects,
                'test_subjects': test_subjects
            })
        
        # Save splits
        splits_df = []
        for split in splits:
            for subj in split['train_subjects']:
                splits_df.append({'split_id': split['split_id'], 'subject_id': subj, 'split_type': 'train'})
            for subj in split['test_subjects']:
                splits_df.append({'split_id': split['split_id'], 'subject_id': subj, 'split_type': 'test'})
        
        splits_file = os.path.join(self.results_dir, 'splits_8x3.csv')
        pd.DataFrame(splits_df).to_csv(splits_file, index=False)
        print(f"  💾 Splits saved: {splits_file}")
        
        return splits
    
    def load_unified_data(self):
        """Load and unify binary data into long table format"""
        print("\n📂 Loading unified binary data...")
        
        unified_data = []
        load_stats = defaultdict(int)
        
        for subject_id in self.subjects:
            subject_dir = os.path.join(self.data_dir, f"User {subject_id}/Processed")
            if not os.path.exists(subject_dir):
                continue
                
            print(f"  👤 Loading Subject {subject_id}...")
            
            for activity in self.activities:
                activity_dir = os.path.join(subject_dir, activity)
                if not os.path.exists(activity_dir):
                    continue
                    
                for placement in self.placements:
                    # Try multiple file locations
                    possible_files = [
                        os.path.join(activity_dir, f"{placement}.csv"),
                        os.path.join(activity_dir, placement, f"{placement}.csv"),
                    ]
                    
                    # Also try glob pattern
                    placement_subdir = os.path.join(activity_dir, placement)
                    if os.path.exists(placement_subdir):
                        csv_files = glob.glob(os.path.join(placement_subdir, "*.csv"))
                        possible_files.extend(csv_files)
                    
                    file_found = False
                    for csv_file in possible_files:
                        if os.path.exists(csv_file):
                            try:
                                df = pd.read_csv(csv_file)
                                if self.binary_column in df.columns:
                                    binary_stream = df[self.binary_column].values
                                    
                                    # Create timestamps
                                    timestamps = np.arange(len(binary_stream)) / self.config['SAMPLE_RATE']
                                    
                                    # Add to unified data
                                    for i, (timestamp, y_val) in enumerate(zip(timestamps, binary_stream)):
                                        unified_data.append({
                                            'subject_id': subject_id,
                                            'activity': activity,
                                            'placement': placement,
                                            'sensor_id': f"{subject_id}_{placement}",
                                            'timestamp': timestamp,
                                            'y': int(y_val)
                                        })
                                    
                                    load_stats[f"{activity}_{placement}"] += 1
                                    file_found = True
                                    break
                            except Exception as e:
                                continue
                    
                    if not file_found:
                        print(f"    ❌ Missing: {activity} - {placement}")
        
        unified_df = pd.DataFrame(unified_data)
        print(f"✅ Unified data loaded: {len(unified_df)} samples from {len(load_stats)} combinations")
        return unified_df
    
    def create_windows(self, unified_df, window_sec=10):
        """Create sliding windows with majority labeling"""
        print(f"\n🔄 Creating {window_sec}s windows with {self.config['STRIDE_SEC']}s stride...")
        
        windows = []
        window_id = 0
        
        for subject_id in self.subjects:
            subject_data = unified_df[unified_df['subject_id'] == subject_id]
            
            if len(subject_data) == 0:
                continue
                
            for activity in self.activities:
                activity_data = subject_data[subject_data['activity'] == activity]
                
                if len(activity_data) == 0:
                    continue
                
                # Group by placement for simultaneous windowing
                placement_groups = {}
                for placement in self.placements:
                    placement_data = activity_data[activity_data['placement'] == placement]
                    if len(placement_data) > 0:
                        placement_groups[placement] = placement_data.sort_values('timestamp')
                
                if not placement_groups:
                    continue
                
                # Find common time range
                min_start = max(group['timestamp'].min() for group in placement_groups.values())
                max_end = min(group['timestamp'].max() for group in placement_groups.values())
                
                # Create windows
                window_length_samples = int(window_sec * self.config['SAMPLE_RATE'])
                stride_samples = int(self.config['STRIDE_SEC'] * self.config['SAMPLE_RATE'])
                
                start_time = min_start
                while start_time + window_sec <= max_end:
                    end_time = start_time + window_sec
                    
                    # Extract data for this window from all placements
                    window_data = {}
                    valid_window = True
                    
                    for placement, group in placement_groups.items():
                        window_mask = (group['timestamp'] >= start_time) & (group['timestamp'] < end_time)
                        window_samples_data = group[window_mask]
                        
                        if len(window_samples_data) < window_length_samples * 0.8:  # Require 80% coverage
                            valid_window = False
                            break
                            
                        window_data[placement] = window_samples_data['y'].values
                    
                    if valid_window and len(window_data) >= 3:  # Require at least 3 placements
                        windows.append({
                            'window_id': window_id,
                            'subject_id': subject_id,
                            'start_time': start_time,
                            'end_time': end_time,
                            'activity_window': activity,
                            'placement_data': window_data
                        })
                        window_id += 1
                    
                    start_time += self.config['STRIDE_SEC']
        
        print(f"✅ Created {len(windows)} windows")
        return windows
    
    def extract_binary_features(self, windows):
        """Extract binary-only features following the specification"""
        print("\n🔧 Extracting binary-only features...")
        
        feature_data = []
        
        for window in windows:
            window_features = {
                'window_id': window['window_id'],
                'subject_id': window['subject_id'],
                'activity': window['activity_window']
            }
            
            placement_data = window['placement_data']
            placements_present = list(placement_data.keys())
            
            # Extract features per placement
            all_placement_features = {}
            
            for placement in self.placements:
                if placement in placement_data:
                    y = placement_data[placement]
                    features = self._extract_univariate_features(y)
                    all_placement_features[placement] = features
                else:
                    # Missing placement - fill with zeros
                    all_placement_features[placement] = self._get_zero_features()
            
            # Extract pairwise cross-sensor features
            cross_features = self._extract_cross_sensor_features(placement_data)
            
            # Combine all features in fixed order
            feature_vector = []
            
            # Per-placement features in fixed order
            for placement in self.placements:
                placement_features = all_placement_features[placement]
                for feature_name in sorted(placement_features.keys()):
                    feature_vector.append(placement_features[feature_name])
                    window_features[f"{placement}_{feature_name}"] = placement_features[feature_name]
            
            # Cross-sensor features
            for feature_name in sorted(cross_features.keys()):
                feature_vector.append(cross_features[feature_name])
                window_features[f"cross_{feature_name}"] = cross_features[feature_name]
            
            # Placement one-hot encoding
            for placement in self.placements:
                window_features[f"placement_{placement}"] = int(placement in placements_present)
                feature_vector.append(int(placement in placements_present))
            
            window_features['feature_vector'] = np.array(feature_vector)
            feature_data.append(window_features)
        
        print(f"✅ Extracted features for {len(feature_data)} windows")
        print(f"  📊 Feature vector length: {len(feature_data[0]['feature_vector']) if feature_data else 0}")
        
        return feature_data
    
    def _extract_univariate_features(self, y):
        """Extract univariate features from binary signal"""
        if len(y) == 0:
            return self._get_zero_features()
        
        y = np.array(y, dtype=int)
        window_sec = len(y) / self.config['SAMPLE_RATE']
        
        features = {}
        
        # Basic statistics
        features['duty_cycle'] = np.mean(y)
        
        # Transition analysis
        transitions = np.abs(np.diff(y))
        features['edges_per_sec'] = np.sum(transitions) / window_sec
        
        rises = np.sum((np.diff(y) == 1))
        falls = np.sum((np.diff(y) == -1))
        features['rises_per_sec'] = rises / window_sec
        features['falls_per_sec'] = falls / window_sec
        
        # Run length statistics
        run_stats = self._compute_run_stats(y, window_sec)
        features.update(run_stats)
        
        # Entropies
        p1 = features['duty_cycle']
        p0 = 1 - p1
        if p1 > 0 and p0 > 0:
            features['bernoulli_entropy'] = -p1 * np.log2(p1) - p0 * np.log2(p0)
        else:
            features['bernoulli_entropy'] = 0
        
        # Autocorrelation (short lags)
        features.update(self._compute_autocorr_features(y))
        
        return features
    
    def _compute_run_stats(self, y, window_sec):
        """Compute run length statistics"""
        if len(y) <= 1:
            return {'on_run_mean': 0, 'on_run_var': 0, 'on_run_max': 0,
                   'off_run_mean': 0, 'off_run_var': 0, 'off_run_max': 0}
        
        # Find runs
        on_runs = []
        off_runs = []
        
        current_run = 1
        current_state = y[0]
        
        for i in range(1, len(y)):
            if y[i] == current_state:
                current_run += 1
            else:
                # End of run
                run_duration_sec = current_run / self.config['SAMPLE_RATE']
                if current_state == 1:
                    on_runs.append(run_duration_sec)
                else:
                    off_runs.append(run_duration_sec)
                
                current_run = 1
                current_state = y[i]
        
        # Handle final run
        run_duration_sec = current_run / self.config['SAMPLE_RATE']
        if current_state == 1:
            on_runs.append(run_duration_sec)
        else:
            off_runs.append(run_duration_sec)
        
        # Compute statistics
        stats = {}
        
        if on_runs:
            stats['on_run_mean'] = np.mean(on_runs)
            stats['on_run_var'] = np.var(on_runs)
            stats['on_run_max'] = np.max(on_runs)
        else:
            stats['on_run_mean'] = 0
            stats['on_run_var'] = 0
            stats['on_run_max'] = 0
        
        if off_runs:
            stats['off_run_mean'] = np.mean(off_runs)
            stats['off_run_var'] = np.var(off_runs)
            stats['off_run_max'] = np.max(off_runs)
        else:
            stats['off_run_mean'] = 0
            stats['off_run_var'] = 0
            stats['off_run_max'] = 0
        
        return stats
    
    def _compute_autocorr_features(self, y):
        """Compute short-lag autocorrelation features"""
        features = {}
        
        if len(y) <= 25:
            for lag in range(1, 26):
                features[f'autocorr_lag_{lag}'] = 0
            return features
        
        # Compute autocorrelations for lags 1-25
        for lag in range(1, min(26, len(y))):
            if lag >= len(y):
                features[f'autocorr_lag_{lag}'] = 0
            else:
                if np.std(y[:-lag]) > 0 and np.std(y[lag:]) > 0:
                    corr = np.corrcoef(y[:-lag], y[lag:])[0, 1]
                    features[f'autocorr_lag_{lag}'] = corr if not np.isnan(corr) else 0
                else:
                    features[f'autocorr_lag_{lag}'] = 0
        
        return features
    
    def _extract_cross_sensor_features(self, placement_data):
        """Extract pairwise cross-sensor features"""
        features = {}
        
        placements = list(placement_data.keys())
        
        for i, placement_i in enumerate(placements):
            for j, placement_j in enumerate(placements):
                if i < j:  # Only upper triangle
                    yi = placement_data[placement_i]
                    yj = placement_data[placement_j]
                    
                    # Align lengths
                    min_len = min(len(yi), len(yj))
                    yi_aligned = yi[:min_len]
                    yj_aligned = yj[:min_len]
                    
                    if min_len == 0:
                        continue
                    
                    # AND rate
                    and_rate = np.mean(yi_aligned & yj_aligned)
                    features[f'and_rate_{placement_i}_{placement_j}'] = and_rate
                    
                    # XOR rate
                    xor_rate = np.mean(yi_aligned ^ yj_aligned)
                    features[f'xor_rate_{placement_i}_{placement_j}'] = xor_rate
                    
                    # Jaccard of ON sets
                    on_i = set(np.where(yi_aligned == 1)[0])
                    on_j = set(np.where(yj_aligned == 1)[0])
                    
                    if len(on_i | on_j) > 0:
                        jaccard = len(on_i & on_j) / len(on_i | on_j)
                    else:
                        jaccard = 0
                    features[f'jaccard_{placement_i}_{placement_j}'] = jaccard
                    
                    # Edge cross-correlation peak lag
                    edges_i = np.abs(np.diff(yi_aligned.astype(int)))
                    edges_j = np.abs(np.diff(yj_aligned.astype(int)))
                    
                    if len(edges_i) > 25 and np.sum(edges_i) > 0 and np.sum(edges_j) > 0:
                        cross_corr = np.correlate(edges_i, edges_j, mode='full')
                        lags = np.arange(-25, 26)
                        valid_range = slice(len(cross_corr)//2 - 25, len(cross_corr)//2 + 26)
                        peak_lag = lags[np.argmax(cross_corr[valid_range])]
                        features[f'edge_peak_lag_{placement_i}_{placement_j}'] = peak_lag
                    else:
                        features[f'edge_peak_lag_{placement_i}_{placement_j}'] = 0
        
        return features
    
    def _get_zero_features(self):
        """Get zero-filled features for missing placements"""
        features = {
            'duty_cycle': 0, 'edges_per_sec': 0, 'rises_per_sec': 0, 'falls_per_sec': 0,
            'on_run_mean': 0, 'on_run_var': 0, 'on_run_max': 0,
            'off_run_mean': 0, 'off_run_var': 0, 'off_run_max': 0,
            'bernoulli_entropy': 0
        }
        
        # Add autocorr features
        for lag in range(1, 26):
            features[f'autocorr_lag_{lag}'] = 0
        
        return features
    
    def preprocess_features(self, feature_data, train_subjects):
        """Preprocess features with standardization and optional PCA"""
        print("\n🔧 Preprocessing features...")
        
        # Separate train and test data
        train_data = [f for f in feature_data if f['subject_id'] in train_subjects]
        test_data = [f for f in feature_data if f['subject_id'] not in train_subjects]
        
        print(f"  📊 Train windows: {len(train_data)}")
        print(f"  📊 Test windows: {len(test_data)}")
        
        # Extract feature matrices
        X_train = np.array([f['feature_vector'] for f in train_data])
        y_train = np.array([f['activity'] for f in train_data])
        X_test = np.array([f['feature_vector'] for f in test_data])
        y_test = np.array([f['activity'] for f in test_data])
        
        # Standardization (fit on train only)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Optional PCA (95% variance retention)
        pca = PCA(n_components=0.95, random_state=self.config['SEED'])
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        print(f"  📊 Original features: {X_train.shape[1]}")
        print(f"  📊 PCA features: {X_train_pca.shape[1]}")
        print(f"  📊 Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        return {
            'X_train': X_train_pca, 'y_train': y_train,
            'X_test': X_test_pca, 'y_test': y_test,
            'scaler': scaler, 'pca': pca,
            'train_data': train_data, 'test_data': test_data
        }
    
    def implement_a2_template_mixture(self, processed_data):
        """Implement A₂ Template-Mixture approach"""
        print("\n🎯 Implementing A₂ Template-Mixture...")
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # 1. Clustering of train windows
        print("  🔗 Clustering train windows...")
        best_clusterer, best_k, bic_scores = self._select_best_clusterer(X_train)
        
        # 2. Soft responsibilities
        P_C_train = best_clusterer.predict_proba(X_train)  # (N_train × K)
        
        # 3. Population mapping: P(C|A) with Laplace smoothing
        print("  📊 Computing population mapping...")
        P_A_train = self._compute_class_priors(y_train_encoded)
        P_C_given_A = self._compute_conditional_probs(P_C_train, y_train_encoded, 
                                                     self.config['SMOOTH_ALPHA'])
        P_A_given_C = self._compute_bayes_inversion(P_C_given_A, P_A_train)
        
        # 4. Predict on test data
        print("  🎯 Predicting on test data...")
        P_C_test = best_clusterer.predict_proba(X_test)  # (N_test × K)
        P_A_test = (P_A_given_C @ P_C_test.T).T  # (N_test × |A|)
        
        y_pred = np.argmax(P_A_test, axis=1)
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        
        # Compute metrics
        macro_f1 = f1_score(y_test, y_pred_labels, average='macro')
        accuracy = accuracy_score(y_test, y_pred_labels)
        
        print(f"  ✅ A₂ Results: Macro-F1 = {macro_f1:.3f}, Accuracy = {accuracy:.3f}")
        
        return {
            'clusterer': best_clusterer,
            'best_k': best_k,
            'bic_scores': bic_scores,
            'label_encoder': label_encoder,
            'P_A_given_C': P_A_given_C,
            'P_A_train': P_A_train,
            'P_C_train': P_C_train,
            'y_pred': y_pred_labels,
            'y_true': y_test,
            'macro_f1': macro_f1,
            'accuracy': accuracy,
            'probabilities': P_A_test
        }
    
    def _select_best_clusterer(self, X_train):
        """Select best clusterer using BIC"""
        print("    🔍 Selecting optimal K using BIC...")
        
        bic_scores = []
        clusterers = []
        
        for k in self.config['K_GRID']:
            if self.config['CLUSTERER'] == "GMM_DIAG":
                clusterer = GaussianMixture(
                    n_components=k, 
                    covariance_type='diag',
                    random_state=self.config['SEED'],
                    max_iter=100
                )
            else:  # KMEANS
                clusterer = KMeans(
                    n_clusters=k,
                    random_state=self.config['SEED'],
                    n_init=10
                )
            
            clusterer.fit(X_train)
            
            if hasattr(clusterer, 'bic'):
                bic = clusterer.bic(X_train)
            else:
                # For K-means, use inertia as proxy
                bic = clusterer.inertia_
            
            bic_scores.append(bic)
            clusterers.append(clusterer)
        
        # Select best K (lowest BIC for GMM, lowest inertia for K-means)
        best_idx = np.argmin(bic_scores)
        best_k = self.config['K_GRID'][best_idx]
        best_clusterer = clusterers[best_idx]
        
        print(f"    ✅ Selected K = {best_k} (BIC/Inertia = {bic_scores[best_idx]:.2f})")
        
        return best_clusterer, best_k, bic_scores
    
    def _compute_class_priors(self, y_encoded):
        """Compute class priors"""
        classes, counts = np.unique(y_encoded, return_counts=True)
        priors = counts / len(y_encoded)
        return dict(zip(classes, priors))
    
    def _compute_conditional_probs(self, P_C_train, y_train_encoded, alpha):
        """Compute P(C|A) with Laplace smoothing"""
        n_classes = len(np.unique(y_train_encoded))
        n_clusters = P_C_train.shape[1]
        
        # Initialize with smoothing
        P_C_given_A = np.full((n_clusters, n_classes), alpha)
        
        # Accumulate soft counts
        for i, y in enumerate(y_train_encoded):
            P_C_given_A[:, y] += P_C_train[i, :]
        
        # Normalize to get probabilities
        P_C_given_A = P_C_given_A / P_C_given_A.sum(axis=0, keepdims=True)
        
        return P_C_given_A
    
    def _compute_bayes_inversion(self, P_C_given_A, P_A):
        """Compute P(A|C) using Bayes rule"""
        n_clusters, n_classes = P_C_given_A.shape
        P_A_given_C = np.zeros((n_classes, n_clusters))
        
        for c in range(n_clusters):
            for a in range(n_classes):
                if a in P_A:
                    P_A_given_C[a, c] = P_C_given_A[c, a] * P_A[a]
        
        # Normalize
        P_A_given_C = P_A_given_C / (P_A_given_C.sum(axis=0, keepdims=True) + 1e-12)
        
        return P_A_given_C
    
    def implement_a3_adaptation(self, a2_results, processed_data, unlabeled_budget_sec=60):
        """Implement A₃ prior-shift reweighting adaptation"""
        print(f"\n🔄 Implementing A₃ adaptation (U={unlabeled_budget_sec}s)...")
        
        test_data = processed_data['test_data']
        scaler = processed_data['scaler']
        pca = processed_data['pca']
        
        adapted_results = {}
        
        # Process each test subject separately
        test_subjects = list(set(f['subject_id'] for f in test_data))
        
        for test_subject in test_subjects:
            print(f"  👤 Adapting for Subject {test_subject}...")
            
            # Get unlabeled windows for this subject
            subject_windows = [f for f in test_data if f['subject_id'] == test_subject]
            
            # Limit to unlabeled budget
            max_windows = int(unlabeled_budget_sec / self.config['STRIDE_SEC'])
            unlabeled_windows = subject_windows[:max_windows]
            
            if len(unlabeled_windows) == 0:
                print(f"    ⚠️  No unlabeled data for Subject {test_subject}")
                continue
            
            # Extract and preprocess features
            X_unlabeled = np.array([f['feature_vector'] for f in unlabeled_windows])
            X_unlabeled_scaled = scaler.transform(X_unlabeled)
            X_unlabeled_pca = pca.transform(X_unlabeled_scaled)
            
            # Prior-shift reweighting
            clusterer = a2_results['clusterer']
            P_C_unlabeled = clusterer.predict_proba(X_unlabeled_pca)
            P_C_target = P_C_unlabeled.mean(axis=0)
            P_C_train_mean = a2_results['P_C_train'].mean(axis=0)
            
            # Compute reweighting ratios
            eps = 1e-12
            r = (P_C_target + eps) / (P_C_train_mean + eps)
            
            # Apply adaptation to all test windows for this subject
            all_subject_windows = [f for f in test_data if f['subject_id'] == test_subject]
            X_subject = np.array([f['feature_vector'] for f in all_subject_windows])
            X_subject_scaled = scaler.transform(X_subject)
            X_subject_pca = pca.transform(X_subject_scaled)
            
            # Adapted prediction
            P_C_subject = clusterer.predict_proba(X_subject_pca)
            P_A_given_C_adapted = a2_results['P_A_given_C'] * r[None, :]
            P_A_given_C_adapted = P_A_given_C_adapted / (P_A_given_C_adapted.sum(axis=0, keepdims=True) + eps)
            
            P_A_subject_adapted = (P_A_given_C_adapted @ P_C_subject.T).T
            
            y_pred_adapted = np.argmax(P_A_subject_adapted, axis=1)
            y_pred_adapted_labels = a2_results['label_encoder'].inverse_transform(y_pred_adapted)
            
            y_true_subject = [f['activity'] for f in all_subject_windows]
            
            # Compute metrics
            macro_f1_adapted = f1_score(y_true_subject, y_pred_adapted_labels, average='macro')
            accuracy_adapted = accuracy_score(y_true_subject, y_pred_adapted_labels)
            
            adapted_results[test_subject] = {
                'y_pred': y_pred_adapted_labels,
                'y_true': y_true_subject,
                'macro_f1': macro_f1_adapted,
                'accuracy': accuracy_adapted,
                'unlabeled_budget': unlabeled_budget_sec,
                'n_unlabeled_windows': len(unlabeled_windows),
                'reweighting_ratios': r
            }
            
            print(f"    ✅ Adapted: Macro-F1 = {macro_f1_adapted:.3f}, Accuracy = {accuracy_adapted:.3f}")
        
        return adapted_results
    
    def evaluate_split(self, split, window_sec=10):
        """Evaluate a single train/test split with detailed saving"""
        print(f"\n🔬 Evaluating Split {split['split_id']} (W={window_sec}s)...")
        print(f"  👥 Train: {split['train_subjects']} (8 subjects)")
        print(f"  🧪 Test: {split['test_subjects']} (3 subjects)")
        
        # Load and prepare data
        unified_df = self.load_unified_data()
        windows = self.create_windows(unified_df, window_sec)
        feature_data = self.extract_binary_features(windows)
        
        # Preprocess
        processed_data = self.preprocess_features(feature_data, split['train_subjects'])
        
        # A₀ Baseline
        print("  📊 Evaluating A₀ Baseline...")
        a0_results = self._evaluate_a0_baseline(processed_data)
        
        # A₂ Template-Mixture
        print("  🎯 Evaluating A₂ Template-Mixture...")
        a2_results = self.implement_a2_template_mixture(processed_data)
        
        # A₃ Adaptation for different budgets
        print("  🔄 Evaluating A₃ Adaptation...")
        a3_results = {}
        for budget in self.config['UNLABELED_BUDGETS']:
            if budget > 0:
                print(f"    💡 Budget: {budget}s")
                a3_results[budget] = self.implement_a3_adaptation(
                    a2_results, processed_data, budget
                )
        
        # Prepare complete results
        complete_results = {
            'split_id': split['split_id'],
            'window_sec': window_sec,
            'a0_results': a0_results,
            'a2_results': a2_results,
            'a3_results': a3_results,
            'train_subjects': split['train_subjects'],
            'test_subjects': split['test_subjects']
        }
        
        # Save detailed results for each adversary
        print("  💾 Saving detailed results...")
        try:
            self._save_detailed_results(complete_results, method="A0")
            self._save_detailed_results(complete_results, method="A2")
            self._save_detailed_results(complete_results, method="A3")
        except Exception as e:
            print(f"    ⚠️ Warning: Could not save detailed results - {e}")
        
        return complete_results
    
    def _evaluate_a0_baseline(self, processed_data):
        """Evaluate A₀ baseline (discriminative model)"""
        print("  📊 A₀ Baseline (Logistic Regression)...")
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        # Logistic Regression baseline
        clf = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=self.config['SEED']
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"    ✅ A₀ Results: Macro-F1 = {macro_f1:.3f}, Accuracy = {accuracy:.3f}")
        
        return {
            'y_pred': y_pred,
            'y_true': y_test,
            'macro_f1': macro_f1,
            'accuracy': accuracy
        }
    
    def run_complete_evaluation(self, n_splits=5):
        """Run complete evaluation across multiple splits with detailed outputs"""
        print("\n🚀 Starting Complete Prior-Based Adversary Evaluation")
        print("="*60)
        print(f"📁 Results will be organized in: {self.results_dir}")
        print("📂 Directory structure:")
        print("   ├── A0_Baseline/")
        print("   ├── A2_Template_Mixture/")
        print("   ├── A3_Adaptation/")
        print("   ├── splits/")
        print("   ├── features/")
        print("   └── summary_plots/")
        
        # Generate splits
        splits = self.generate_splits(n_splits)
        
        all_results = []
        
        # Evaluate each split for each window size
        for window_sec in self.config['WINDOWS']:
            print(f"\n📊 Evaluating Window Size: {window_sec}s")
            print("-" * 40)
            
            for split in splits[:n_splits]:  # Limit to requested number of splits
                try:
                    results = self.evaluate_split(split, window_sec)
                    all_results.append(results)
                    
                    # Save intermediate aggregated results
                    self._save_split_results(results)
                    
                except Exception as e:
                    print(f"❌ Split {split['split_id']} failed: {e}")
                    continue
        
        # Generate final analysis
        self._generate_final_report(all_results)
        
        print("\n" + "="*60)
        print("🎉 PRIOR-BASED ADVERSARY EVALUATION COMPLETED!")
        print("="*60)
        print(f"📊 All results saved to: {self.results_dir}/")
        print("📋 Deliverables organized by adversary:")
        print("   • Raw predictions (CSV)")
        print("   • Classification reports (JSON + TXT)")
        print("   • Confusion matrices (CSV + PNG)")
        print("   • Model components (PKL)")
        print("   • Performance plots (PNG)")
        print("   • Comparison analysis")
        print("="*60)
        
        return all_results
    
    def _save_split_results(self, results):
        """Save results for individual split"""
        split_id = results['split_id']
        window_sec = results['window_sec']
        
        # Prepare results row
        result_rows = []
        
        # A₂ results
        a2 = results['a2_results']
        result_rows.append({
            'split_id': split_id,
            'method': 'A2_Template_Mixture',
            'window_sec': window_sec,
            'macro_f1': a2['macro_f1'],
            'accuracy': a2['accuracy'],
            'K': a2['best_k'],
            'U': None
        })
        
        # A₃ results
        for budget, a3_subjects in results['a3_results'].items():
            # Average across test subjects
            macro_f1_scores = [subj['macro_f1'] for subj in a3_subjects.values()]
            accuracy_scores = [subj['accuracy'] for subj in a3_subjects.values()]
            
            if macro_f1_scores:
                result_rows.append({
                    'split_id': split_id,
                    'method': f'A3_Adaptation',
                    'window_sec': window_sec,
                    'macro_f1': np.mean(macro_f1_scores),
                    'accuracy': np.mean(accuracy_scores),
                    'K': a2['best_k'],
                    'U': budget
                })
        
        # Save to CSV
        results_file = os.path.join(self.results_dir, f'results_W{window_sec}.csv')
        
        if os.path.exists(results_file):
            existing_df = pd.read_csv(results_file)
            new_df = pd.DataFrame(result_rows)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = pd.DataFrame(result_rows)
        
        combined_df.to_csv(results_file, index=False)
        print(f"  💾 Split results saved: {results_file}")
    
    def _generate_final_report(self, all_results):
        """Generate comprehensive final report"""
        print("\n📝 Generating final report...")
        
        # Aggregate results by method and window size
        summary_stats = defaultdict(list)
        
        for result in all_results:
            window_sec = result['window_sec']
            key_prefix = f"W{window_sec}"
            
            # A₂
            summary_stats[f"{key_prefix}_A2_macro_f1"].append(result['a2_results']['macro_f1'])
            summary_stats[f"{key_prefix}_A2_accuracy"].append(result['a2_results']['accuracy'])
            
            # A₃ (best budget)
            if result['a3_results']:
                best_budget = max(result['a3_results'].keys())
                a3_subjects = result['a3_results'][best_budget]
                macro_f1_scores = [subj['macro_f1'] for subj in a3_subjects.values()]
                if macro_f1_scores:
                    summary_stats[f"{key_prefix}_A3_macro_f1"].append(np.mean(macro_f1_scores))
        
        # Generate summary report
        report_file = os.path.join(self.results_dir, 'prior_adversary_summary_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("PRIOR-BASED ADVERSARY (A₂ & A₃) EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("METHODOLOGY:\n")
            f.write("- A₂: Template-Mixture with diagonal-covariance GMM\n")
            f.write("- A₃: Prior-shift reweighting adaptation\n")
            f.write("- Binary-only features from 5 sensor placements\n")
            f.write("- 8-train / 3-test subject splits\n\n")
            
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 30 + "\n")
            
            for metric, values in summary_stats.items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    f.write(f"{metric}: {mean_val:.3f} ± {std_val:.3f} (n={len(values)})\n")
            
            f.write("\nKEY FINDINGS:\n")
            f.write("-" * 15 + "\n")
            
            # Calculate improvements
            for window_sec in self.config['WINDOWS']:
                key_prefix = f"W{window_sec}"
                a2_key = f"{key_prefix}_A2_macro_f1"
                a3_key = f"{key_prefix}_A3_macro_f1"
                
                if a2_key in summary_stats:
                    a2_mean = np.mean(summary_stats[a2_key])
                    f.write(f"Window {window_sec}s: A₂ Template-Mixture = {a2_mean:.3f}\n")
                    
                    if a3_key in summary_stats:
                        a3_mean = np.mean(summary_stats[a3_key])
                        delta_a3 = a3_mean - a2_mean
                        f.write(f"Window {window_sec}s: A₃ vs A₂ = Δ{delta_a3:+.3f}\n")
        
        print(f"  📝 Final report saved: {report_file}")
    
    def _save_detailed_results(self, results, method="A2"):
        """Save detailed results including predictions, reports, and plots"""
        split_id = results['split_id']
        window_sec = results['window_sec']
        
        # Determine output directory
        if method == "A0":
            base_dir = self.a0_dir
        elif method == "A2":
            base_dir = self.a2_dir
        elif method == "A3":
            base_dir = self.a3_dir
        else:
            base_dir = self.results_dir
        
        # File naming convention
        base_name = f"split{split_id:02d}_W{window_sec}s"
        
        if method == "A2":
            self._save_a2_detailed_results(results, base_dir, base_name)
        elif method == "A3":
            self._save_a3_detailed_results(results, base_dir, base_name)
        elif method == "A0":
            self._save_a0_detailed_results(results, base_dir, base_name)
    
    def _save_a2_detailed_results(self, results, base_dir, base_name):
        """Save detailed A₂ results"""
        a2 = results['a2_results']
        
        # 1. Raw Predictions
        predictions_df = pd.DataFrame({
            'y_true': a2['y_true'],
            'y_pred': a2['y_pred'],
            'split_id': results['split_id'],
            'window_sec': results['window_sec'],
            'method': 'A2_Template_Mixture'
        })
        
        # Add probability columns
        prob_cols = {}
        for i, activity in enumerate(self.activities):
            if i < a2['probabilities'].shape[1]:
                prob_cols[f'prob_{activity}'] = a2['probabilities'][:, i]
        predictions_df = pd.concat([predictions_df, pd.DataFrame(prob_cols)], axis=1)
        
        pred_file = os.path.join(base_dir, "predictions", f"{base_name}_predictions.csv")
        predictions_df.to_csv(pred_file, index=False)
        
        # 2. Classification Report
        report = classification_report(a2['y_true'], a2['y_pred'], 
                                     target_names=self.activities, 
                                     output_dict=True)
        
        report_file = os.path.join(base_dir, "reports", f"{base_name}_classification_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Text version
        report_txt = classification_report(a2['y_true'], a2['y_pred'], 
                                         target_names=self.activities)
        report_txt_file = os.path.join(base_dir, "reports", f"{base_name}_classification_report.txt")
        with open(report_txt_file, 'w') as f:
            f.write(f"A₂ Template-Mixture Classification Report\n")
            f.write(f"Split: {results['split_id']}, Window: {results['window_sec']}s\n")
            f.write(f"Train Subjects: {results['train_subjects']}\n")
            f.write(f"Test Subjects: {results['test_subjects']}\n")
            f.write(f"Optimal K: {a2['best_k']}\n")
            f.write("=" * 50 + "\n")
            f.write(report_txt)
        
        # 3. Confusion Matrix
        cm = confusion_matrix(a2['y_true'], a2['y_pred'], labels=self.activities)
        
        # Save as CSV
        cm_df = pd.DataFrame(cm, index=self.activities, columns=self.activities)
        cm_file = os.path.join(base_dir, "confusion_matrices", f"{base_name}_confusion_matrix.csv")
        cm_df.to_csv(cm_file)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', 
                   cbar_kws={'label': 'Count'})
        plt.title(f'A₂ Confusion Matrix\nSplit {results["split_id"]}, Window {results["window_sec"]}s, K={a2["best_k"]}')
        plt.ylabel('True Activity')
        plt.xlabel('Predicted Activity')
        plt.tight_layout()
        
        cm_plot_file = os.path.join(base_dir, "confusion_matrices", f"{base_name}_confusion_matrix.png")
        plt.savefig(cm_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. BIC/Model Selection Plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.config['K_GRID'], a2['bic_scores'], 'bo-', linewidth=2, markersize=8)
        plt.axvline(x=a2['best_k'], color='red', linestyle='--', alpha=0.7, 
                   label=f'Selected K={a2["best_k"]}')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('BIC Score')
        plt.title(f'A₂ Model Selection (BIC)\nSplit {results["split_id"]}, Window {results["window_sec"]}s')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        bic_plot_file = os.path.join(base_dir, "plots", f"{base_name}_bic_curve.png")
        plt.savefig(bic_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. P(A|C) Heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(a2['P_A_given_C'], annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=[f'C{i}' for i in range(a2['best_k'])],
                   yticklabels=self.activities,
                   cbar_kws={'label': 'P(A|C)'})
        plt.title(f'A₂ Activity-Cluster Mapping P(A|C)\nSplit {results["split_id"]}, Window {results["window_sec"]}s')
        plt.ylabel('Activity')
        plt.xlabel('Cluster')
        plt.tight_layout()
        
        mapping_plot_file = os.path.join(base_dir, "plots", f"{base_name}_activity_cluster_mapping.png")
        plt.savefig(mapping_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Save model components
        model_data = {
            'clusterer': a2['clusterer'],
            'label_encoder': a2['label_encoder'],
            'P_A_given_C': a2['P_A_given_C'],
            'P_A_train': a2['P_A_train'],
            'best_k': a2['best_k'],
            'bic_scores': a2['bic_scores']
        }
        
        model_file = os.path.join(base_dir, "models", f"{base_name}_model.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"  💾 A₂ detailed results saved to: {base_dir}")
    
    def _save_a3_detailed_results(self, results, base_dir, base_name):
        """Save detailed A₃ results"""
        a3_results = results['a3_results']
        
        for budget, a3_subjects in a3_results.items():
            budget_name = f"{base_name}_U{budget}s"
            
            # Combine all test subjects for this budget
            all_predictions = []
            
            for subject_id, subject_results in a3_subjects.items():
                subject_df = pd.DataFrame({
                    'y_true': subject_results['y_true'],
                    'y_pred': subject_results['y_pred'],
                    'subject_id': subject_id,
                    'split_id': results['split_id'],
                    'window_sec': results['window_sec'],
                    'unlabeled_budget': budget,
                    'method': 'A3_Adaptation'
                })
                all_predictions.append(subject_df)
            
            if all_predictions:
                # 1. Raw Predictions
                combined_predictions = pd.concat(all_predictions, ignore_index=True)
                pred_file = os.path.join(base_dir, "predictions", f"{budget_name}_predictions.csv")
                combined_predictions.to_csv(pred_file, index=False)
                
                # 2. Classification Report (combined)
                y_true_all = combined_predictions['y_true'].tolist()
                y_pred_all = combined_predictions['y_pred'].tolist()
                
                report = classification_report(y_true_all, y_pred_all, 
                                             target_names=self.activities, 
                                             output_dict=True)
                
                report_file = os.path.join(base_dir, "reports", f"{budget_name}_classification_report.json")
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2)
                
                # Text version
                report_txt = classification_report(y_true_all, y_pred_all, 
                                                 target_names=self.activities)
                report_txt_file = os.path.join(base_dir, "reports", f"{budget_name}_classification_report.txt")
                with open(report_txt_file, 'w') as f:
                    f.write(f"A₃ Adaptation Classification Report\n")
                    f.write(f"Split: {results['split_id']}, Window: {results['window_sec']}s, Budget: {budget}s\n")
                    f.write(f"Test Subjects: {results['test_subjects']}\n")
                    f.write("=" * 50 + "\n")
                    f.write(report_txt)
                
                # 3. Confusion Matrix
                cm = confusion_matrix(y_true_all, y_pred_all, labels=self.activities)
                
                cm_df = pd.DataFrame(cm, index=self.activities, columns=self.activities)
                cm_file = os.path.join(base_dir, "confusion_matrices", f"{budget_name}_confusion_matrix.csv")
                cm_df.to_csv(cm_file)
                
                # Plot confusion matrix
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm_df, annot=True, fmt='d', cmap='Greens', 
                           cbar_kws={'label': 'Count'})
                plt.title(f'A₃ Confusion Matrix\nSplit {results["split_id"]}, Window {results["window_sec"]}s, Budget {budget}s')
                plt.ylabel('True Activity')
                plt.xlabel('Predicted Activity')
                plt.tight_layout()
                
                cm_plot_file = os.path.join(base_dir, "confusion_matrices", f"{budget_name}_confusion_matrix.png")
                plt.savefig(cm_plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                # 4. Per-subject performance plot
                subject_metrics = []
                for subject_id, subject_results in a3_subjects.items():
                    subject_metrics.append({
                        'subject_id': subject_id,
                        'macro_f1': subject_results['macro_f1'],
                        'accuracy': subject_results['accuracy'],
                        'n_unlabeled': subject_results['n_unlabeled_windows']
                    })
                
                subject_df = pd.DataFrame(subject_metrics)
                
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.bar(subject_df['subject_id'], subject_df['macro_f1'], color='skyblue', alpha=0.7)
                plt.xlabel('Test Subject')
                plt.ylabel('Macro-F1')
                plt.title(f'A₃ Per-Subject Performance\nBudget: {budget}s')
                plt.xticks(rotation=45)
                
                plt.subplot(1, 2, 2)
                plt.bar(subject_df['subject_id'], subject_df['accuracy'], color='lightcoral', alpha=0.7)
                plt.xlabel('Test Subject')
                plt.ylabel('Accuracy')
                plt.title(f'A₃ Per-Subject Accuracy\nBudget: {budget}s')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                
                subject_plot_file = os.path.join(base_dir, "plots", f"{budget_name}_per_subject_performance.png")
                plt.savefig(subject_plot_file, dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"  💾 A₃ detailed results saved to: {base_dir}")
    
    def _save_a0_detailed_results(self, results, base_dir, base_name):
        """Save detailed A₀ baseline results"""
        a0_results = results['a0_results']
        
        # 1. Raw Predictions
        predictions_df = pd.DataFrame({
            'y_true': a0_results['y_true'],
            'y_pred': a0_results['y_pred'],
            'split_id': results['split_id'],
            'window_sec': results['window_sec'],
            'method': 'A0_Baseline'
        })
        
        pred_file = os.path.join(base_dir, "predictions", f"{base_name}_predictions.csv")
        predictions_df.to_csv(pred_file, index=False)
        
        # 2. Classification Report
        report = classification_report(a0_results['y_true'], a0_results['y_pred'], 
                                     target_names=self.activities, 
                                     output_dict=True)
        
        report_file = os.path.join(base_dir, "reports", f"{base_name}_classification_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Text version
        report_txt = classification_report(a0_results['y_true'], a0_results['y_pred'], 
                                         target_names=self.activities)
        report_txt_file = os.path.join(base_dir, "reports", f"{base_name}_classification_report.txt")
        with open(report_txt_file, 'w') as f:
            f.write(f"A₀ Baseline Classification Report\n")
            f.write(f"Split: {results['split_id']}, Window: {results['window_sec']}s\n")
            f.write(f"Train Subjects: {results['train_subjects']}\n")
            f.write(f"Test Subjects: {results['test_subjects']}\n")
            f.write("=" * 50 + "\n")
            f.write(report_txt)
        
        # 3. Confusion Matrix
        cm = confusion_matrix(a0_results['y_true'], a0_results['y_pred'], labels=self.activities)
        
        cm_df = pd.DataFrame(cm, index=self.activities, columns=self.activities)
        cm_file = os.path.join(base_dir, "confusion_matrices", f"{base_name}_confusion_matrix.csv")
        cm_df.to_csv(cm_file)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Oranges', 
                   cbar_kws={'label': 'Count'})
        plt.title(f'A₀ Baseline Confusion Matrix\nSplit {results["split_id"]}, Window {results["window_sec"]}s')
        plt.ylabel('True Activity')
        plt.xlabel('Predicted Activity')
        plt.tight_layout()
        
        cm_plot_file = os.path.join(base_dir, "confusion_matrices", f"{base_name}_confusion_matrix.png")
        plt.savefig(cm_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 A₀ detailed results saved to: {base_dir}")

if __name__ == "__main__":
    # Run the Prior-Based Adversary evaluation
    adversary = PriorBasedAdversary(
        data_dir="../Data",
        results_dir="prior_adversary_results"
    )
    
    results = adversary.run_complete_evaluation(n_splits=3)  # Start with 3 splits for testing
