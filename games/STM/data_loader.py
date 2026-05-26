"""
Data loading utilities for STM games.

This module handles loading and preprocessing of IMU-derived activity data
for privacy leakage experiments.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from games.STM.config import get_game_config, VALID_ACTIVITIES, WINDOW_SIZE, PLACEMENT_FILES, BINARY_FEATURE_NAME, USE_TIMESTAMP_ALIGNMENT
from games.STM.features.binary_features import compute_multi_scale_features, TOTAL_MICRO_FEATURES
from games.STM.features.cross_features import compute_cross_features, TOTAL_CROSS_FEATURES


def load_user_metadata(data_root):
    """
    Load and preprocess user metadata for FiLM conditioning.
    
    Args:
        data_root: Root directory containing user data
        
    Returns:
        Dictionary mapping user_id to metadata vector
    """
    metadata_path = os.path.join(os.path.dirname(__file__), 'data', 'user_meta.csv')
    
    if not os.path.exists(metadata_path):
        print(f"⚠️ User metadata file not found: {metadata_path}")
        return {}
    
    try:
        # Load metadata
        df = pd.read_csv(metadata_path)
        print(f"📊 Loaded user metadata for {len(df)} users")
        
        # Preprocess metadata
        user_metadata = {}
        
        for _, row in df.iterrows():
            user_id = int(row['user_id'])
            
            # Extract numeric features and z-score normalize
            numeric_features = [
                row['height_in'],
                row['leg_len_in'], 
                row['arm_len_in'],
                row['torso_len_in'],
                row['shoe_size']
            ]
            
            # One-hot encode dominant hand and foot
            dom_hand = [1, 0] if row['dom_hand'] == 1 else [0, 1]  # Right=1, Left=0
            dom_foot = [1, 0] if row['dom_foot'] == 1 else [0, 1]  # Right=1, Left=0
            
            # Combine all features
            metadata_vector = numeric_features + dom_hand + dom_foot
            
            user_metadata[user_id] = np.array(metadata_vector, dtype=np.float32)
        
        # Z-score normalize numeric features across all users
        numeric_features_matrix = np.array([user_metadata[uid][:5] for uid in sorted(user_metadata.keys())])
        scaler = StandardScaler()
        normalized_numeric = scaler.fit_transform(numeric_features_matrix)
        
        # Update user metadata with normalized features
        for i, user_id in enumerate(sorted(user_metadata.keys())):
            user_metadata[user_id] = np.concatenate([
                normalized_numeric[i],  # 5 normalized numeric features
                user_metadata[user_id][5:]  # 4 one-hot encoded features
            ])
        
        print(f"✅ Processed user metadata: {len(user_metadata)} users, {len(user_metadata[list(user_metadata.keys())[0]])} features per user")
        return user_metadata
        
    except Exception as e:
        print(f"❌ Error loading user metadata: {e}")
        return {}


def load_placement_dataframe(user_dir, placement_name, target_activity=None, required_features=None):
    """Load dataframe for a specific placement with required columns"""
    
    # Default to binary features if not specified
    if required_features is None:
        required_features = [BINARY_FEATURE_NAME]
    
    # Get possible filenames for this placement
    possible_files = PLACEMENT_FILES.get(placement_name, [])
    
    # If target_activity is specified, only search in that activity
    activities_to_search = [target_activity] if target_activity else VALID_ACTIVITIES
    
    for filename in possible_files:
        # Search for the file in specified activity directories
        for activity in activities_to_search:
            if activity is None:
                continue
                
            activity_path = os.path.join(user_dir, "Processed", activity)
            if not os.path.exists(activity_path):
                continue
            
            # Try flat structure first (e.g., Sitting/right-wrist.csv)
            file_path = os.path.join(activity_path, filename)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    
                    # Check if required columns exist
                    missing_features = [f for f in required_features if f not in df.columns]
                    if not missing_features:
                        # If no timestamp column, synthesize one
                        if "timestamp" not in df.columns:
                            df["timestamp"] = np.arange(len(df))
                        return df, activity
                        
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
            
            # Try nested structure (e.g., Downstairs/right-wrist/right-wrist_downstairs1.csv)
            placement_dir = os.path.join(activity_path, placement_name)
            if os.path.exists(placement_dir) and os.path.isdir(placement_dir):
                # Find all CSV files in the placement directory
                csv_files = glob.glob(os.path.join(placement_dir, "*.csv"))
                if csv_files:
                    # Sort files to ensure consistent ordering
                    csv_files.sort()
                    
                    # Load and concatenate all files for this placement
                    dfs = []
                    for csv_file in csv_files:
                        try:
                            temp_df = pd.read_csv(csv_file)
                            missing_features = [f for f in required_features if f not in temp_df.columns]
                            if not missing_features:
                                dfs.append(temp_df)
                        except Exception as e:
                            print(f"Error reading {csv_file}: {e}")
                            continue
                    
                    if dfs:
                        # Concatenate all dataframes
                        df = pd.concat(dfs, ignore_index=True)
                        
                        # If no timestamp column, synthesize one
                        if "timestamp" not in df.columns:
                            df["timestamp"] = np.arange(len(df))
                        
                        return df, activity
    
    return None, None


def align_and_stack(placement_dfs, window_size, features=None):
    """Align and stack multiple placement dataframes with continuous features"""
    
    if not placement_dfs:
        return None
    
    # Default to binary features if not specified
    if features is None:
        features = [BINARY_FEATURE_NAME]
    
    # Check if all dataframes have timestamps
    has_timestamps = all("timestamp" in df.columns for df, _ in placement_dfs.values())
    
    if has_timestamps and USE_TIMESTAMP_ALIGNMENT:
        # Align by timestamp using inner join
        aligned_data = {}
        min_length = float('inf')
        
        # Find the minimum length across all placements
        for placement, (df, _) in placement_dfs.items():
            min_length = min(min_length, len(df))
        
        # Truncate all dataframes to minimum length and align by index
        for placement, (df, _) in placement_dfs.items():
            # Check if all required features exist
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                print(f"    ⚠️ Missing features {missing_features} for placement {placement}")
                return None
            aligned_data[placement] = df.head(min_length)[features].values
    else:
        # Align by index (truncate to minimum length)
        min_length = min(len(df) for df, _ in placement_dfs.values())
        aligned_data = {}
        
        for placement, (df, _) in placement_dfs.items():
            # Check if all required features exist
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                print(f"    ⚠️ Missing features {missing_features} for placement {placement}")
                return None
            aligned_data[placement] = df.head(min_length)[features].values
    
    # Stack into (T, P, C) array where P = number of placements, C = number of features
    if len(aligned_data) > 0:
        # Get the number of features from the first placement
        num_features = aligned_data[list(aligned_data.keys())[0]].shape[1]
        
        # Stack all placements: (T, P, C)
        stacked = np.stack(list(aligned_data.values()), axis=1)
        
        # Verify shape: should be (T, P, C) where P=5 placements, C=7 features
        if stacked.shape[1] != len(placement_dfs):
            print(f"    ⚠️ Expected {len(placement_dfs)} placements, got {stacked.shape[1]}")
            return None
        
        if stacked.shape[2] != num_features:
            print(f"    ⚠️ Expected {num_features} features, got {stacked.shape[2]}")
            return None
        
        return stacked
    
    return None


def is_multi_placement_game(game_name, features):
    """Check if this is a multi-placement game"""
    # Game-1 is multi-placement when used with --grid multi
    # Other games are multi-placement if they have multiple dec_tree_out features
    return (game_name == "Game-1" or 
            any("dec_tree_out_" in f and "_" in f and f != "dec_tree_out_1" for f in features))


def load_data_for_game(game_name, data_root, num_classes=None):
    """Load and preprocess data for specific game"""
    
    print(f"📂 Loading data for {game_name}...")
    
    # Get game configuration
    game_config = get_game_config(game_name)
    features = game_config["input_features"]
    
    # Check if this is a multi-placement game
    is_multi = is_multi_placement_game(game_name, features)
    
    # Get activities list - filter by num_classes if specified
    if num_classes and num_classes < len(VALID_ACTIVITIES):
        # Use first N activities for consistent ordering
        activities_to_process = list(VALID_ACTIVITIES)[:num_classes]
        print(f"🎯 Filtering to {num_classes} classes: {activities_to_process}")
    else:
        activities_to_process = VALID_ACTIVITIES
        print(f"🎯 Using all {len(VALID_ACTIVITIES)} classes: {activities_to_process}")
    
    print(f"🎯 Features: {features}")
    print(f"🎯 Activities: {activities_to_process}")
    print(f"🔗 Multi-placement mode: {is_multi}")
    
    # Load user metadata for FiLM conditioning
    user_metadata = load_user_metadata(data_root)
    
    # Load user data
    user_data = {}
    all_labels = []
    
    # Get user directories
    user_dirs = sorted([d for d in os.listdir(data_root) if re.match(r"User \d+", d)],
                      key=lambda x: int(re.findall(r"\d+", x)[0]))
    
    for user in user_dirs:
        print(f"  📁 Processing {user}...")
        X_user, y_user = [], []
        user_path = os.path.join(data_root, user)
        
        # Extract user ID for metadata lookup
        user_id = int(re.findall(r"\d+", user)[0])
        user_meta = user_metadata.get(user_id, None)
        
        if is_multi:
            # Multi-placement path
            X_user, y_user = _load_multi_placement_data(user_path, activities_to_process, features)
        else:
            # Single-placement path (existing behavior)
            X_user, y_user = _load_single_placement_data(user_path, activities_to_process, features)
        
        if X_user:
            # Include user metadata in the data tuple
            user_data[user] = (np.array(X_user), np.array(y_user), user_meta)
            all_labels.extend(y_user)
            print(f"    ✅ {user}: {len(X_user)} samples")
            if user_meta is not None:
                print(f"    📊 User metadata: {len(user_meta)} features")
        else:
            print(f"    ⚠️ No data for {user}")
    
    # Add hard assertions and logging for channel detection
    if user_data:
        # Get a sample to check dimensions
        sample_user = list(user_data.keys())[0]
        sample_data = user_data[sample_user][0]  # X data
        
        if is_multi:
            # For multi-placement: should be (num_samples, T, P, C)
            if len(sample_data.shape) == 4:
                num_samples, T, P, C = sample_data.shape
                print(f"🔍 Multi-placement data shape: {sample_data.shape}")
                print(f"   📊 Time steps (T): {T}")
                print(f"   📊 Placements (P): {P}")
                print(f"   📊 Features per placement (C): {C}")
                
                # Check continuous vs binary features
                if features:
                    binary_indices, continuous_indices = get_feature_indices(features)
                    cont_dim = len(continuous_indices)
                    bin_dim = len(binary_indices)
                    
                    print(f"   📊 Continuous features: {cont_dim}")
                    print(f"   📊 Binary features: {bin_dim}")
                    
                    # Check if this is a binary-only game (Game-1)
                    if game_name == "Game-1":
                        # For binary-only games, we expect only binary features
                        assert bin_dim > 0, f"No binary features detected for binary-only game! Expected > 0, got {bin_dim}"
                        print(f"✅ Binary-only game: {bin_dim} binary features")
                    else:
                        # For other games, we expect continuous features
                        assert cont_dim > 0, f"No continuous features detected! Expected > 0, got {cont_dim}"
                        print(f"✅ Continuous features present: {cont_dim} features")
                else:
                    print(f"⚠️ No feature list provided for dimension analysis")
            else:
                print(f"⚠️ Unexpected multi-placement data shape: {sample_data.shape}")
        else:
            # For single-placement: should be (num_samples, T, C)
            if len(sample_data.shape) == 3:
                num_samples, T, C = sample_data.shape
                print(f"🔍 Single-placement data shape: {sample_data.shape}")
                print(f"   📊 Time steps (T): {T}")
                print(f"   📊 Features (C): {C}")
            else:
                print(f"⚠️ Unexpected single-placement data shape: {sample_data.shape}")
    
    return user_data, all_labels


def _load_single_placement_data(user_path, activities_to_process, features):
    """Load data for single-placement games (existing behavior)"""
    
    X_user, y_user = [], []
    
    # Process each activity
    for activity in activities_to_process:
        activity_path = os.path.join(user_path, "Processed", activity)
        if not os.path.exists(activity_path):
            continue
            
        # Find CSV files (including subdirectories)
        csv_files = glob.glob(os.path.join(activity_path, "**/*.csv"), recursive=True)
        if not csv_files:
            continue
            
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Check if required features exist
                missing_features = [f for f in features if f not in df.columns]
                if missing_features:
                    continue
                
                # Segment data into windows
                for i in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
                    window = df[features].iloc[i:i + WINDOW_SIZE].values
                    if window.shape[0] == WINDOW_SIZE:
                        X_user.append(window)
                        y_user.append(activity)
                        
            except Exception as e:
                continue
    
    return X_user, y_user


def _load_multi_placement_data(user_path, activities_to_process, features=None):
    """Load data for multi-placement games with continuous features"""
    
    from games.STM.config import PLACEMENT_ORDER
    
    X_user, y_user = [], []
    
    # Process each activity
    for activity in activities_to_process:
        activity_path = os.path.join(user_path, "Processed", activity)
        if not os.path.exists(activity_path):
            continue
        
        # Load dataframes for all placements in the correct order
        placement_dfs = {}
        for placement_name in PLACEMENT_ORDER:
            df, df_activity = load_placement_dataframe(user_path, placement_name, target_activity=activity, required_features=features)
            if df is not None and df_activity == activity:
                placement_dfs[placement_name] = (df, df_activity)
            else:
                print(f"    ⚠️ Missing placement {placement_name} for {activity}")
                break
        
        # Skip if we don't have data for all placements
        if len(placement_dfs) < len(PLACEMENT_ORDER):
            print(f"    ⚠️ Missing placements for {activity}, skipping")
            continue
        
        # Align and stack the data in the correct order
        ordered_dfs = {name: placement_dfs[name] for name in PLACEMENT_ORDER}
        stacked_data = align_and_stack(ordered_dfs, WINDOW_SIZE, features)
        if stacked_data is None:
            continue
        
        # Verify data shape: should be (T, P, C) where P=5 placements, C=7 features
        if len(stacked_data.shape) != 3:
            print(f"    ⚠️ Expected 3D data, got shape {stacked_data.shape}")
            continue
        
        T, P, C = stacked_data.shape
        if P != len(PLACEMENT_ORDER):
            print(f"    ⚠️ Expected {len(PLACEMENT_ORDER)} placements, got {P}")
            continue
        
        if features and C != len(features):
            print(f"    ⚠️ Expected {len(features)} features, got {C}")
            continue
        
        # Window the stacked data
        for i in range(0, T - WINDOW_SIZE + 1, WINDOW_SIZE):
            window = stacked_data[i:i + WINDOW_SIZE]
            if window.shape[0] == WINDOW_SIZE:
                X_user.append(window)
                y_user.append(activity)
    
    return X_user, y_user


def get_feature_indices(features):
    """Get binary and continuous feature indices based on feature names"""
    
    binary_indices = []
    continuous_indices = []
    
    for i, feature in enumerate(features):
        if (feature.startswith("dec_tree_out") or 
            feature == BINARY_FEATURE_NAME or
            "binary" in feature.lower()):
            binary_indices.append(i)
        else:
            continuous_indices.append(i)
    
    return binary_indices, continuous_indices


def compute_semantic_features_for_window(window_data, placement_names, featureset='base', 
                                       normalizer=None, use_attention=False):
    """
    Compute semantic features for a single window with optional enhancements.
    
    Args:
        window_data: Array of shape (window_size, num_placements) with binary data
        placement_names: List of placement names in order
        featureset: Feature set to compute ('base', 'micro', 'micro+cross', 'micro+cross+semantics', 'micro+cross+semantics+film')
        normalizer: Optional SemanticFeatureNormalizer for feature normalization
        use_attention: Whether to use attention-based fusion
    
    Returns:
        Dictionary with semantic features and summary token
    """
    if featureset == 'base':
        return {'summary_token': None, 'semantic_features': {}, 'enhanced': False}
    
    # Use enhanced semantic features if normalizer or attention is requested
    if normalizer is not None or use_attention:
        from .features.enhanced_semantic_features import compute_enhanced_semantic_features
        return compute_enhanced_semantic_features(
            window_data, placement_names, featureset, normalizer, use_attention
        )
    
    # Extract binary sequences for each placement
    bits_by_placement = {}
    for i, placement in enumerate(placement_names):
        if i < window_data.shape[1]:
            # For continuous features, extract the binary feature (last feature)
            if len(window_data.shape) == 3:
                # window_data shape: (window_size, num_placements, num_features)
                # Extract the binary feature (last feature) from each placement
                bits_by_placement[placement] = window_data[:, i, -1]  # Last feature is binary
            else:
                # window_data shape: (window_size, num_placements) - already binary
                bits_by_placement[placement] = window_data[:, i]
    
    semantic_features = {}
    
    # Compute micro-features
    if 'micro' in featureset:
        for placement, bits in bits_by_placement.items():
            micro_features = compute_multi_scale_features(bits)
            for key, value in micro_features.items():
                semantic_features[f"{placement}_{key}"] = value
    
    # Compute cross-placement features
    if 'cross' in featureset:
        cross_features = compute_cross_features(bits_by_placement)
        semantic_features.update(cross_features)
    
    # Create summary token with configurable dimension
    if semantic_features:
        feature_values = list(semantic_features.values())
        full_summary = np.array(feature_values, dtype=np.float32)
        
        # Determine target dimension based on featureset
        if featureset == 'semantic-14':
            target_dim = 14
        elif featureset == 'semantic-94':
            target_dim = 94
        else:
            # Default to 14 for backward compatibility
            target_dim = 14
        
        if len(full_summary) > target_dim:
            # Method 1: Take first N features (most important micro-features)
            summary_token = full_summary[:target_dim]
        else:
            # Pad with zeros if we have fewer than target_dim features
            summary_token = np.zeros(target_dim, dtype=np.float32)
            summary_token[:len(full_summary)] = full_summary
    else:
        summary_token = None
    
    return {
        'summary_token': summary_token,
        'semantic_features': semantic_features,
        'enhanced': False
    }


def add_semantic_features_to_data(X, y, placement_names, featureset='base', use_summary_token=True,
                                normalizer=None, use_attention=False, apply_augmentation=False):
    """
    Add semantic features to the dataset with optional enhancements.
    
    Args:
        X: Input data of shape (num_samples, window_size, num_placements)
        y: Labels
        placement_names: List of placement names
        featureset: Feature set to use
        use_summary_token: Whether to add summary token (True) or broadcast features (False)
        normalizer: Optional SemanticFeatureNormalizer for feature normalization
        use_attention: Whether to use attention-based fusion
        apply_augmentation: Whether to apply data augmentation
    
    Returns:
        Tuple of (X_enhanced, y, semantic_info)
    """
    if featureset == 'base':
        return X, y, {}
    
    # Apply data augmentation if requested
    if apply_augmentation:
        from .features.enhanced_semantic_features import create_class_rebalancing_augmentation
        X, y = create_class_rebalancing_augmentation(X, y, target_samples_per_class=1000)
        print(f"🔧 Applied data augmentation: {X.shape[0]} samples")
    
    num_samples, window_size, num_placements = X.shape
    semantic_info = {
        'featureset': featureset,
        'use_summary_token': use_summary_token,
        'total_semantic_features': 0,
        'enhanced': normalizer is not None or use_attention,
        'augmented': apply_augmentation
    }
    
    # Fit normalizer if provided and not already fitted
    if normalizer is not None and not normalizer.is_fitted:
        print("🔧 Fitting semantic feature normalizer...")
        semantic_features_list = []
        for i in range(min(1000, num_samples)):  # Use subset for fitting
            window_data = X[i]
            result = compute_semantic_features_for_window(window_data, placement_names, featureset)
            if result['summary_token'] is not None:
                semantic_features_list.append(result['summary_token'])
        if semantic_features_list:
            normalizer.fit(semantic_features_list)
            print(f"✅ Normalizer fitted on {len(semantic_features_list)} samples")
    
    if use_summary_token:
        # Add summary token mode: append 1 token, then models expect (T+1, P+extras)
        summary_tokens = []
        
        for i in range(num_samples):
            window_data = X[i]  # (window_size, num_placements) or (window_size, num_placements, num_features)
            result = compute_semantic_features_for_window(window_data, placement_names, featureset, 
                                                        normalizer, use_attention)
            
            if result['summary_token'] is not None:
                summary_tokens.append(result['summary_token'])
                semantic_info['total_semantic_features'] = len(result['summary_token'])
            else:
                # Fallback: zero vector with appropriate dimension
                target_dim = 14 if featureset == 'semantic-14' else 94 if featureset == 'semantic-94' else 14
                summary_tokens.append(np.zeros(target_dim, dtype=np.float32))
        
        # Stack summary tokens
        summary_tokens = np.array(summary_tokens)  # (num_samples, num_semantic_features)
        
        # Reshape to add as extra dimension: (num_samples, 1, num_semantic_features)
        summary_tokens = summary_tokens.reshape(num_samples, 1, -1)
        
        # Create enhanced features: (num_samples, window_size + 1, num_placements + num_semantic_features)
        # First, pad original data to accommodate summary token
        X_padded = np.zeros((num_samples, window_size + 1, num_placements), dtype=np.float32)
        X_padded[:, 1:, :] = X  # Summary token will be at index 0
        
        # Create enhanced features by concatenating along feature dimension
        # summary_tokens: (num_samples, 1, num_semantic_features)
        # X_padded: (num_samples, window_size + 1, num_placements)
        # We need to pad summary_tokens to match the time dimension
        summary_tokens_padded = np.zeros((num_samples, window_size + 1, summary_tokens.shape[2]), dtype=np.float32)
        summary_tokens_padded[:, 0, :] = summary_tokens[:, 0, :]  # Only first time step has summary token
        
        # Concatenate along feature dimension
        X_enhanced = np.concatenate([summary_tokens_padded, X_padded], axis=2).astype(np.float32)
        
        semantic_info['summary_token_dim'] = summary_tokens.shape[2]
        semantic_info['enhanced_shape'] = X_enhanced.shape
        
    else:
        # Broadcast mode: tile semantic features across time
        enhanced_windows = []
        
        for i in range(num_samples):
            window_data = X[i]  # (window_size, num_placements)
            result = compute_semantic_features_for_window(window_data, placement_names, featureset,
                                                        normalizer, use_attention)
            
            if result['summary_token'] is not None:
                # Tile semantic features across time
                semantic_features_tiled = np.tile(result['summary_token'], (window_size, 1))  # (window_size, num_semantic_features)
                enhanced_window = np.concatenate([window_data, semantic_features_tiled], axis=1)
                enhanced_windows.append(enhanced_window)
                semantic_info['total_semantic_features'] = len(result['summary_token'])
            else:
                # Fallback: original data
                enhanced_windows.append(window_data)
        
        X_enhanced = np.array(enhanced_windows, dtype=np.float32)
        semantic_info['enhanced_shape'] = X_enhanced.shape
    
    return X_enhanced, y, semantic_info


def apply_per_user_normalization(user_data, game_name):
    """Apply per-user normalization to improve cross-user generalization"""
    
    print(f"🔧 Applying per-user normalization for {game_name}...")
    
    # Get game configuration for feature names
    game_config = get_game_config(game_name)
    features = game_config.get("input_features", [])
    
    # Get feature indices
    binary_indices, continuous_indices = get_feature_indices(features)
    
    print(f"  📊 Binary features: {[features[i] for i in binary_indices]}")
    print(f"  📊 Continuous features: {[features[i] for i in continuous_indices]}")
    
    normalized_user_data = {}
    
    for user, user_data_tuple in user_data.items():
        print(f"  📊 Normalizing {user}...")
        
        # Handle new data structure with user metadata
        if len(user_data_tuple) == 3:
            X_user, y_user, user_meta = user_data_tuple
        else:
            X_user, y_user = user_data_tuple
            user_meta = None
        
        if len(continuous_indices) == 0:
            # No continuous features to normalize
            if user_meta is not None:
                normalized_user_data[user] = (X_user, y_user, user_meta)
            else:
                normalized_user_data[user] = (X_user, y_user)
            continue
        
        # Extract continuous and binary data
        # Handle 4D data: (samples, time_steps, placements, features)
        if len(X_user.shape) == 4:
            # For 4D data, we need to handle placements separately
            continuous_data = X_user[:, :, :, continuous_indices]
            binary_data = X_user[:, :, :, binary_indices] if binary_indices else None
        else:
            # For 3D data: (samples, time_steps, features)
            continuous_data = X_user[:, :, continuous_indices]
            binary_data = X_user[:, :, binary_indices] if binary_indices else None
        
        # Apply per-user normalization to continuous data only
        scaler = StandardScaler()
        
        if len(X_user.shape) == 4:
            # For 4D data: (samples, time_steps, placements, features)
            continuous_reshaped = continuous_data.reshape(-1, continuous_data.shape[-1])
            continuous_normalized = scaler.fit_transform(continuous_reshaped)
            continuous_normalized = continuous_normalized.reshape(continuous_data.shape)
            
            # Reconstruct data with normalized continuous + original binary
            if binary_data is not None:
                # Combine continuous and binary features in original order
                all_features = []
                continuous_idx = 0
                binary_idx = 0
                
                for i in range(len(features)):
                    if i in continuous_indices:
                        all_features.append(continuous_normalized[:, :, :, continuous_idx:continuous_idx+1])
                        continuous_idx += 1
                    elif i in binary_indices:
                        all_features.append(binary_data[:, :, :, binary_idx:binary_idx+1])
                        binary_idx += 1
                
                normalized_data = np.concatenate(all_features, axis=3)
            else:
                normalized_data = continuous_normalized
        else:
            # For 3D data: (samples, time_steps, features)
            continuous_reshaped = continuous_data.reshape(-1, continuous_data.shape[-1])
            continuous_normalized = scaler.fit_transform(continuous_reshaped)
            continuous_normalized = continuous_normalized.reshape(continuous_data.shape)
            
            # Reconstruct data with normalized continuous + original binary
            if binary_data is not None:
                # Combine continuous and binary features in original order
                all_features = []
                continuous_idx = 0
                binary_idx = 0
                
                for i in range(len(features)):
                    if i in continuous_indices:
                        all_features.append(continuous_normalized[:, :, continuous_idx:continuous_idx+1])
                        continuous_idx += 1
                    elif i in binary_indices:
                        all_features.append(binary_data[:, :, binary_idx:binary_idx+1])
                        binary_idx += 1
                
                normalized_data = np.concatenate(all_features, axis=2)
            else:
                normalized_data = continuous_normalized
        
        # Return with user metadata if available
        if user_meta is not None:
            normalized_user_data[user] = (normalized_data, y_user, user_meta)
        else:
            normalized_user_data[user] = (normalized_data, y_user)
        
        print(f"    ✅ {user}: Normalized {len(continuous_indices)} continuous features")
    
    return normalized_user_data


def prepare_cross_user_data(user_data, train_users, test_users, use_per_user_norm=False, game_name=None, 
                           featureset='base', use_summary_token=True, use_normalization=False, 
                           use_attention=False, use_augmentation=False):
    """Prepare data for cross-user experiments with optional per-user normalization and semantic features"""
    
    if use_per_user_norm and game_name:
        print(f"🔧 Using per-user normalization for cross-user experiment...")
        user_data = apply_per_user_normalization(user_data, game_name)
    
    # Prepare training data
    X_train = np.vstack([user_data[u][0] for u in train_users])
    y_train = np.hstack([user_data[u][1] for u in train_users])
    
    # Prepare test data
    X_test = np.vstack([user_data[u][0] for u in test_users])
    y_test = np.hstack([user_data[u][1] for u in test_users])
    
    # Ensure data types are consistent (convert to float32 to avoid casting issues)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    # Collect user metadata for FiLM conditioning
    train_user_metadata = []
    test_user_metadata = []
    
    for user in train_users:
        if len(user_data[user]) > 2 and user_data[user][2] is not None:
            # Repeat metadata for each sample from this user
            num_samples = len(user_data[user][0])
            user_meta = user_data[user][2]
            train_user_metadata.extend([user_meta] * num_samples)
        else:
            # No metadata available, use zeros
            num_samples = len(user_data[user][0])
            dummy_meta = np.zeros(9, dtype=np.float32)  # 5 numeric + 2 dom_hand + 2 dom_foot
            train_user_metadata.extend([dummy_meta] * num_samples)
    
    for user in test_users:
        if len(user_data[user]) > 2 and user_data[user][2] is not None:
            # Repeat metadata for each sample from this user
            num_samples = len(user_data[user][0])
            user_meta = user_data[user][2]
            test_user_metadata.extend([user_meta] * num_samples)
        else:
            # No metadata available, use zeros
            num_samples = len(user_data[user][0])
            dummy_meta = np.zeros(9, dtype=np.float32)  # 5 numeric + 2 dom_hand + 2 dom_foot
            test_user_metadata.extend([dummy_meta] * num_samples)
    
    # Convert to numpy arrays
    train_user_metadata = np.array(train_user_metadata)
    test_user_metadata = np.array(test_user_metadata)
    
    # Add semantic features if requested
    if featureset != 'base' and featureset != 'concatenated':
        print(f"🔧 Adding semantic features: {featureset}")
        
        # Get placement names from game config
        placement_names = []
        if game_name:
            game_config = get_game_config(game_name)
            if 'input_features' in game_config:
                # Extract placement names from feature names
                for feature in game_config['input_features']:
                    if 'dec_tree_out_' in feature:
                        placement = feature.replace('dec_tree_out_', '')
                        placement_names.append(placement)
        
        if not placement_names:
            # Fallback: use default placement order
            from games.STM.config import PLACEMENT_ORDER
            placement_names = PLACEMENT_ORDER
        
        # Create normalizer if requested
        normalizer = None
        if use_normalization:
            from .features.enhanced_semantic_features import SemanticFeatureNormalizer
            normalizer = SemanticFeatureNormalizer(method='standard')
        
        # Handle continuous features: extract binary features for semantic processing
        if len(X_train.shape) == 4:  # (num_samples, T, P, C) - continuous features
            print(f"  🔧 Detected continuous features: {X_train.shape}")
            # Extract binary features (last feature) for semantic processing
            X_train_binary = X_train[:, :, :, -1]  # (num_samples, T, P)
            X_test_binary = X_test[:, :, :, -1]    # (num_samples, T, P)
            
            # Add semantic features using binary data
            X_train_enhanced, y_train, train_semantic_info = add_semantic_features_to_data(
                X_train_binary, y_train, placement_names, featureset, use_summary_token,
                normalizer=normalizer, use_attention=use_attention, apply_augmentation=use_augmentation
            )
            
            X_test_enhanced, y_test, test_semantic_info = add_semantic_features_to_data(
                X_test_binary, y_test, placement_names, featureset, use_summary_token,
                normalizer=normalizer, use_attention=use_attention, apply_augmentation=False
            )
            
            # Use the enhanced features with semantic information
            X_train = X_train_enhanced
            X_test = X_test_enhanced
            print(f"  ✅ Semantic features integrated successfully")
            print(f"  📊 Training data shape: {X_train.shape} (enhanced with semantic features)")
            print(f"  📊 Test data shape: {X_test.shape} (enhanced with semantic features)")
            print(f"  📊 Semantic features: {train_semantic_info.get('total_semantic_features', 0)}")
        else:
            # Binary features: (num_samples, T, P)
            X_train, y_train, train_semantic_info = add_semantic_features_to_data(
                X_train, y_train, placement_names, featureset, use_summary_token,
                normalizer=normalizer, use_attention=use_attention, apply_augmentation=use_augmentation
            )
            
            X_test, y_test, test_semantic_info = add_semantic_features_to_data(
                X_test, y_test, placement_names, featureset, use_summary_token,
                normalizer=normalizer, use_attention=use_attention, apply_augmentation=False
            )
            
            print(f"  📊 Training data shape: {X_train.shape}")
            print(f"  📊 Test data shape: {X_test.shape}")
            print(f"  📊 Semantic features: {train_semantic_info.get('total_semantic_features', 0)}")
    elif featureset == 'concatenated':
        print(f"🔧 Using simple concatenation: {featureset}")
        # Simple concatenation: flatten placement dimension (100, 5, 1) -> (100, 5)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)
        print(f"  📊 Training data shape: {X_train.shape} (concatenated)")
        print(f"  📊 Test data shape: {X_test.shape} (concatenated)")
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    return X_train, y_train_encoded, X_test, y_test_encoded, le, train_user_metadata, test_user_metadata


def prepare_intra_user_data(user_data, user_id, use_per_user_norm=False, game_name=None, 
                           featureset='base', use_summary_token=True):
    """Prepare data for intra-user experiments with optional per-user normalization and semantic features"""
    
    if user_id not in user_data:
        raise ValueError(f"User {user_id} not found in data")
    
    if use_per_user_norm and game_name:
        print(f"🔧 Using per-user normalization for intra-user experiment...")
        user_data = apply_per_user_normalization(user_data, game_name)
    
    X_user, y_user = user_data[user_id]
    
    # Split data for this user
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_user, y_user, test_size=0.3, random_state=42, stratify=y_user
    )
    
    # Add semantic features if requested
    if featureset != 'base':
        print(f"🔧 Adding semantic features: {featureset}")
        
        # Get placement names from game config
        placement_names = []
        if game_name:
            game_config = get_game_config(game_name)
            if 'input_features' in game_config:
                # Extract placement names from feature names
                for feature in game_config['input_features']:
                    if 'dec_tree_out_' in feature:
                        placement = feature.replace('dec_tree_out_', '')
                        placement_names.append(placement)
        
        if not placement_names:
            # Fallback: use default placement order
            from games.STM.config import PLACEMENT_ORDER
            placement_names = PLACEMENT_ORDER
        
        # Add semantic features to training data
        X_train, y_train, train_semantic_info = add_semantic_features_to_data(
            X_train, y_train, placement_names, featureset, use_summary_token
        )
        
        # Add semantic features to test data
        X_test, y_test, test_semantic_info = add_semantic_features_to_data(
            X_test, y_test, placement_names, featureset, use_summary_token
        )
        
        print(f"  📊 Training data shape: {X_train.shape}")
        print(f"  📊 Test data shape: {X_test.shape}")
        print(f"  📊 Semantic features: {train_semantic_info.get('total_semantic_features', 0)}")
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    return X_train, y_train_encoded, X_test, y_test_encoded, le


def prepare_leave_one_user_out_data(user_data, test_user, use_per_user_norm=False, game_name=None, 
                                   featureset='base', use_summary_token=True):
    """Prepare data for leave-one-user-out experiments with optional per-user normalization and semantic features"""
    
    if test_user not in user_data:
        raise ValueError(f"Test user {test_user} not found in data")
    
    if use_per_user_norm and game_name:
        print(f"🔧 Using per-user normalization for leave-one-user-out experiment...")
        user_data = apply_per_user_normalization(user_data, game_name)
    
    # All users except test_user for training
    train_users = [u for u in user_data.keys() if u != test_user]
    
    # Prepare training data
    X_train = np.vstack([user_data[u][0] for u in train_users])
    y_train = np.hstack([user_data[u][1] for u in train_users])
    
    # Prepare test data
    X_test, y_test = user_data[test_user]
    
    # Add semantic features if requested
    if featureset != 'base':
        print(f"🔧 Adding semantic features: {featureset}")
        
        # Get placement names from game config
        placement_names = []
        if game_name:
            game_config = get_game_config(game_name)
            if 'input_features' in game_config:
                # Extract placement names from feature names
                for feature in game_config['input_features']:
                    if 'dec_tree_out_' in feature:
                        placement = feature.replace('dec_tree_out_', '')
                        placement_names.append(placement)
        
        if not placement_names:
            # Fallback: use default placement order
            from games.STM.config import PLACEMENT_ORDER
            placement_names = PLACEMENT_ORDER
        
        # Add semantic features to training data
        X_train, y_train, train_semantic_info = add_semantic_features_to_data(
            X_train, y_train, placement_names, featureset, use_summary_token
        )
        
        # Add semantic features to test data
        X_test, y_test, test_semantic_info = add_semantic_features_to_data(
            X_test, y_test, placement_names, featureset, use_summary_token
        )
        
        print(f"  📊 Training data shape: {X_train.shape}")
        print(f"  📊 Test data shape: {X_test.shape}")
        print(f"  📊 Semantic features: {train_semantic_info.get('total_semantic_features', 0)}")
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    return X_train, y_train_encoded, X_test, y_test_encoded, le 