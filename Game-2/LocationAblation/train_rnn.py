import os
import re
import glob
import json
import pickle
import time
import uuid
import gc
import shutil
from datetime import datetime
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    mutual_info_score, precision_recall_fscore_support,
    accuracy_score
)
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Bidirectional, Dense, Dropout, 
    BatchNormalization, Concatenate, Lambda, GlobalAveragePooling1D,
    Conv1D, MaxPooling1D, Flatten, TimeDistributed, Attention
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from scipy.special import rel_entr
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message=".*Some weights of the model checkpoint.*")
warnings.filterwarnings("ignore", message=".*softmax over axis.*")  # Suppress the softmax warning

# -------- GPU CONFIGURATION --------
# Configure GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🚀 GPU(s) detected and configured: {len(gpus)} GPU(s)")
        print(f"📋 GPU details: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")
else:
    print("⚠️ No GPU detected, using CPU")

# Verify GPU availability
print(f"🔍 TensorFlow GPU available: {tf.config.list_physical_devices('GPU')}")
print(f"🔍 TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")

# Set mixed precision if GPU is available for better performance
if tf.config.list_physical_devices('GPU'):
    try:
        # Enable mixed precision for better GPU utilization
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("🚀 Mixed precision enabled for improved GPU performance")
    except:
        print("⚠️ Mixed precision not available, using default precision")

def build_enhanced_rnn(input_shape, num_classes):
    """
    Enhanced RNN architecture for HAR with dual-pathway processing and motion-aware design.
    Optimized for speed and performance with advanced features.
    
    Features:
    - Dual-pathway processing (sensor vs motion features)
    - Bidirectional LSTM/GRU layers for temporal modeling
    - Motion-aware attention mechanisms
    - Efficient architecture for fast training and inference
    - Mixed precision compatibility
    """
    # Clear any existing sessions to avoid naming conflicts
    tf.keras.backend.clear_session()
    
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        
        # Input layer: (batch_size, time_steps, 26_features)
        inputs = Input(shape=input_shape, name='input_sequences')
        
        # Feature separation: 19 sensor + 7 motion
        sensor_features = Lambda(
            lambda x: x[:, :, :19], 
            name='extract_sensor_features'
        )(inputs)
        
        motion_features = Lambda(
            lambda x: x[:, :, 19:], 
            name='extract_motion_features'
        )(inputs)
        
        # -------- DUAL-PATHWAY PROCESSING --------
        
        # === SENSOR PATHWAY ===
        # Bidirectional LSTM for temporal modeling
        sensor_lstm1 = Bidirectional(
            LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            name='sensor_bilstm_1'
        )(sensor_features)
        sensor_lstm1 = BatchNormalization(name='sensor_bn_1')(sensor_lstm1)
        
        # Second LSTM layer with attention
        sensor_lstm2 = Bidirectional(
            LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            name='sensor_bilstm_2'
        )(sensor_lstm1)
        sensor_lstm2 = BatchNormalization(name='sensor_bn_2')(sensor_lstm2)
        
        # === MOTION PATHWAY ===
        # Simpler pathway for binary motion features
        motion_gru = Bidirectional(
            GRU(16, return_sequences=True, dropout=0.1, recurrent_dropout=0.1),
            name='motion_bigru'
        )(motion_features)
        motion_gru = BatchNormalization(name='motion_bn')(motion_gru)
        
        # -------- MOTION-AWARE ATTENTION --------
        # Compute attention weights based on motion context (fixed softmax issue)
        motion_attention_weights = Dense(32, activation='tanh', name='motion_attention_dense')(motion_gru)
        motion_attention_weights = Dense(1, activation='sigmoid', name='motion_attention_final')(motion_attention_weights)
        
        # Apply motion-guided attention to sensor features using Multiply layer
        sensor_attended = tf.keras.layers.Multiply(name='motion_guided_attention')([sensor_lstm2, motion_attention_weights])
        
        # -------- TEMPORAL AGGREGATION --------
        # Multiple pooling strategies for comprehensive feature extraction
        
        # Global average pooling
        sensor_global_avg = GlobalAveragePooling1D(name='sensor_global_avg')(sensor_attended)
        motion_global_avg = GlobalAveragePooling1D(name='motion_global_avg')(motion_gru)
        
        # Attention-based pooling using Keras layers (fixed softmax over time dimension)
        sensor_attn_dense = Dense(32, activation='tanh', name='sensor_attn_dense')(sensor_attended)
        sensor_attn_weights = Dense(1, activation='sigmoid', name='sensor_attn_weights')(sensor_attn_dense)
        sensor_attn_pool = Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1), name='sensor_attn_pool')([sensor_attended, sensor_attn_weights])
        
        motion_attn_dense = Dense(16, activation='tanh', name='motion_attn_dense')(motion_gru)
        motion_attn_weights = Dense(1, activation='sigmoid', name='motion_attn_weights')(motion_attn_dense)
        motion_attn_pool = Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1), name='motion_attn_pool')([motion_gru, motion_attn_weights])
        
        print(f"🔧 Enhanced RNN: Fixed Keras functional API compatibility")
        print(f"   ✅ Using Lambda layers for TensorFlow operations")
        print(f"   ✅ Attention pooling implemented correctly")
        
        # Last timestep features (for sequential patterns)
        sensor_last = Lambda(lambda x: x[:, -1, :], name='sensor_last')(sensor_attended)
        motion_last = Lambda(lambda x: x[:, -1, :], name='motion_last')(motion_gru)
        
        # -------- FEATURE FUSION --------
        # Combine all extracted features
        combined_features = Concatenate(name='combined_features')([
            sensor_global_avg, motion_global_avg,
            sensor_attn_pool, motion_attn_pool,
            sensor_last, motion_last
        ])
        
        # -------- CLASSIFICATION HEAD --------
        # Efficient dense layers for fast inference
        dense1 = Dense(256, activation='relu', kernel_initializer='he_normal', name='dense_1')(combined_features)
        dense1 = BatchNormalization(name='dense_bn_1')(dense1)
        dense1 = Dropout(0.3, name='dense_dropout_1')(dense1)
        
        dense2 = Dense(128, activation='relu', kernel_initializer='he_normal', name='dense_2')(dense1)
        dense2 = BatchNormalization(name='dense_bn_2')(dense2)
        dense2 = Dropout(0.3, name='dense_dropout_2')(dense2)
        
        # Motion context branch for enhanced classification
        motion_context = Dense(64, activation='relu', kernel_initializer='he_normal', name='motion_context')(combined_features)
        motion_context = BatchNormalization(name='motion_context_bn')(motion_context)
        motion_context = Dropout(0.2, name='motion_context_dropout')(motion_context)
        
        # Final feature combination
        final_features = Concatenate(name='final_features')([dense2, motion_context])
        final_dense = Dense(128, activation='relu', kernel_initializer='he_normal', name='final_dense')(final_features)
        final_dense = BatchNormalization(name='final_bn')(final_dense)
        final_dense = Dropout(0.3, name='final_dropout')(final_dense)
        
        # Output layer
        outputs = Dense(num_classes, activation='softmax', kernel_initializer='glorot_uniform', name='activity_output')(final_dense)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='EnhancedMotionAwareRNN')
        
        # Optimized optimizer for RNN training
        optimizer = Adam(
            learning_rate=0.001,  # Higher learning rate for RNNs
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0  # Gradient clipping for RNN stability
        )
        
        # Compile with comprehensive metrics
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.F1Score(name='f1_score')
            ]
        )
        
        print(f"🏗️ Built Enhanced RNN:")
        print(f"   📊 Total parameters: {model.count_params():,}")
        print(f"   🎯 Input shape: {input_shape}")
        print(f"   🎯 Output classes: {num_classes}")
        print(f"   🔧 Architecture: Dual-pathway BiLSTM/BiGRU with motion-aware attention")
        print(f"   ⚙️ Optimizer: Adam (lr=0.001)")
        
        return model

# Legacy function name for compatibility
def build_rnn(input_shape, num_classes):
    """Legacy wrapper for enhanced RNN - now uses the improved architecture."""
    return build_enhanced_rnn(input_shape, num_classes)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -------- CONFIG --------
data_root = os.environ.get("NNEDGE_PRIVACY_STM_ROOT", os.path.join(REPO_ROOT, "STMDATASET", "Data"))
game = "Game-2"
model_name = "Enhanced_RNN"
locations = ["left-ankle", "left-wrist", "right-ankle", "right-pocket", "right-wrist"]
window_size = 20  # Reduced for faster processing while maintaining effectiveness
ratios = [(0.8, 0.2), (0.7, 0.3), (0.5, 0.5)]
EPOCHS = 20  # Reduced from 100 for faster training with checkpointing
BATCH_SIZE = 32  # Optimized batch size for RNN training
LEARNING_RATE = 0.001  # Higher learning rate for RNNs
MIN_DELTA = 0.001  # Minimum change in monitored quantity to qualify as an improvement
PATIENCE = 10  # Reduced patience for faster convergence
DROPOUT_RATE = 0.3  # Dropout rate for regularization
VALID_ACTIVITIES = {"jogging", "laying", "sitting", "standing", "upstairs", "downstairs", "walking"}

# -------- CHECKPOINT SYSTEM --------
# Generate consistent script identifier or reuse existing one
CHECKPOINT_DIR = os.environ.get("NNEDGE_PRIVACY_CHECKPOINT_ROOT", os.path.join(REPO_ROOT, "checkpoints", "Game-2_RNN_LocationAblation_Checkpoints"))

# Try to find existing script ID from checkpoint directory, otherwise create new one
def get_or_create_script_id():
    if os.path.exists(CHECKPOINT_DIR):
        # Look for ALL existing experiment directories 
        existing_dirs = [d for d in os.listdir(CHECKPOINT_DIR) 
                        if d.startswith("Game-2_Enhanced_RNN_LocationAblation_") and len(d.split('_')) >= 5]
        
        # Find script IDs and their experiment counts
        script_id_stats = {}
        for dir_name in existing_dirs:
            if dir_name == "data_loading":  # Skip data loading directory
                continue
                
            progress_file = os.path.join(CHECKPOINT_DIR, dir_name, "training_progress.json")
            script_id = "_".join(dir_name.split('_')[:5])  # Extract script ID
            
            if script_id not in script_id_stats:
                script_id_stats[script_id] = {'total': 0, 'completed': 0, 'in_progress': 0}
            
            script_id_stats[script_id]['total'] += 1
            
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r') as f:
                        progress = json.load(f)
                    
                    current_epoch = progress.get('current_epoch', 0)
                    status = progress.get('status', 'unknown')
                    
                    if status == 'completed':
                        script_id_stats[script_id]['completed'] += 1
                    elif current_epoch > 0:
                        script_id_stats[script_id]['in_progress'] += 1
                        
                    print(f"🔍 Found experiment: {dir_name} (Epoch {current_epoch}/{progress.get('total_epochs', 20)}, Status: {status})")
                except Exception as e:
                    print(f"⚠️ Error reading checkpoint {dir_name}: {e}")
                    continue
        
        # Print summary of existing script IDs
        for script_id, stats in script_id_stats.items():
            print(f"📊 Script ID {script_id}: {stats['total']} experiments ({stats['completed']} completed, {stats['in_progress']} in progress)")
        
        # If we found script IDs with any experiments, use the one with the most experiments
        if script_id_stats:
            # Choose script ID with most total experiments (prioritizing existing work)
            best_script_id = max(script_id_stats.keys(), key=lambda x: script_id_stats[x]['total'])
            print(f"🔄 Reusing existing Script ID: {best_script_id}")
            return best_script_id
    
    # Create new script ID if none found
    new_script_id = f"Game-2_Enhanced_RNN_LocationAblation_{uuid.uuid4().hex[:8]}"
    print(f"🆕 Created new Script ID: {new_script_id}")
    return new_script_id

SCRIPT_ID = get_or_create_script_id()

def cleanup_empty_checkpoints():
    """Remove checkpoint directories with no training progress."""
    if not os.path.exists(CHECKPOINT_DIR):
        return
    
    dirs_to_remove = []
    for dir_name in os.listdir(CHECKPOINT_DIR):
        if not dir_name.startswith("Game-2_Enhanced_RNN_LocationAblation_"):
            continue
        
        dir_path = os.path.join(CHECKPOINT_DIR, dir_name)
        if not os.path.isdir(dir_path):
            continue
            
        progress_file = os.path.join(dir_path, "training_progress.json")
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                # Mark for removal if no progress
                if progress.get('current_epoch', 0) == 0 and len(progress.get('results', {}).get('epochs_completed', [])) == 0:
                    dirs_to_remove.append(dir_path)
            except Exception as e:
                print(f"⚠️ Error reading {progress_file}: {e}")
    
    # Remove empty directories
    for dir_path in dirs_to_remove:
        try:
            shutil.rmtree(dir_path)
            print(f"🗑️ Removed empty checkpoint: {os.path.basename(dir_path)}")
        except Exception as e:
            print(f"⚠️ Failed to remove {dir_path}: {e}")

# Clean up empty checkpoints on startup
cleanup_empty_checkpoints()

print(f"🔒 Script ID: {SCRIPT_ID}")
print(f"📁 Checkpoint Directory: {CHECKPOINT_DIR}")

# Check for existing data cache
global_cache_dir = os.path.join(CHECKPOINT_DIR, "data_loading")
if os.path.exists(global_cache_dir):
    cache_files = [f for f in os.listdir(global_cache_dir) if f.endswith("_cache.pkl")]
    if cache_files:
        print(f"📦 Found {len(cache_files)} cached datasets in data_loading:")
        for cache_file in cache_files:
            cache_path = os.path.join(global_cache_dir, cache_file)
            cache_size = os.path.getsize(cache_path) / (1024*1024)  # MB
            print(f"   - {cache_file} ({cache_size:.1f} MB)")
    else:
        print("📦 No cached datasets found in data_loading")
else:
    print("📦 No data_loading directory found")




MODES = {
    "accel": ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]', 'dec_tree_out_1'],
    "gyro": ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]', 'dec_tree_out_1']
}

# Validate MODES configuration
print("🔧 Validating MODES configuration:")
for modality, features in MODES.items():
    print(f"   ✅ {modality}: {features}")
    sensor_features = [f for f in features if 'acc_' in f or 'gyro_' in f]
    motion_features = [f for f in features if 'dec_tree_out' in f]
    print(f"      📊 Sensor features: {len(sensor_features)} (should be 3)")
    print(f"      🎯 Motion features: {len(motion_features)} (should be 1)")
    if len(sensor_features) != 3 or len(motion_features) != 1:
        raise ValueError(f"Invalid MODES configuration for {modality}")
print("✅ MODES configuration validated successfully")

# -------- CHECKPOINT SYSTEM FUNCTIONS --------
def save_checkpoint(experiment_id, progress_data):
    """Save training progress to isolated checkpoint."""
    exp_dir = os.path.join(CHECKPOINT_DIR, experiment_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    checkpoint_file = os.path.join(exp_dir, "training_progress.json")
    progress_data['last_updated'] = datetime.now().isoformat()
    progress_data['script_id'] = SCRIPT_ID
    
    with open(checkpoint_file, 'w') as f:
        json.dump(progress_data, f, indent=2)
    
    print(f"💾 Checkpoint saved: {checkpoint_file}")

def load_checkpoint(experiment_id):
    """Load training progress from checkpoint."""
    exp_dir = os.path.join(CHECKPOINT_DIR, experiment_id)
    checkpoint_file = os.path.join(exp_dir, "training_progress.json")
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                progress = json.load(f)
            print(f"📖 Checkpoint loaded: {checkpoint_file}")
            return progress
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}")
    
    return None

def make_serializable(obj):
    """Convert TensorFlow tensors and numpy arrays to serializable Python types."""
    # Handle TensorFlow tensors 
    if hasattr(obj, 'numpy'):  # EagerTensor
        obj = obj.numpy()
    
    # Handle numpy types
    if hasattr(obj, 'dtype'):
        if obj.ndim == 0:  # Scalar array
            return obj.item()  # Convert to Python scalar
        else:  # Multi-dimensional array
            return obj.tolist()  # Convert to Python list
    
    # Handle standard Python collections recursively
    if isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    
    # Handle numpy scalars that might not have been caught above
    if hasattr(obj, 'item'):
        try:
            return obj.item()
        except (ValueError, TypeError):
            pass
    
    # Return as-is for standard Python types (int, float, str, bool, None)
    return obj

class EpochProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback to save epoch-level progress."""
    
    def __init__(self, experiment_id, total_epochs):
        super().__init__()
        self.experiment_id = experiment_id
        self.total_epochs = total_epochs
        self.epoch_start_time = None
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(f"🚀 Starting epoch {epoch + 1}/{self.total_epochs}")
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        
        # Load existing progress
        progress = load_checkpoint(self.experiment_id) or {
            'experiment_id': self.experiment_id,
            'script_id': SCRIPT_ID,
            'current_epoch': 0,
            'total_epochs': self.total_epochs,
            'status': 'running',
            'results': {'epochs_completed': [], 'best_accuracy': 0, 'best_loss': float('inf')}
        }
        
        # Update progress with current epoch
        progress['current_epoch'] = epoch + 1
        
        epoch_data = {
            'epoch': epoch + 1,
            'epoch_time': epoch_time,
            'timestamp': datetime.now().isoformat()
        }
        
        if logs:
            # Make logs serializable before adding to epoch_data
            serializable_logs = make_serializable(logs)
            epoch_data.update(serializable_logs)
            
            # Update best metrics safely
            if 'accuracy' in serializable_logs:
                current_acc = serializable_logs['accuracy']
                progress['results']['best_accuracy'] = max(
                    progress['results']['best_accuracy'], 
                    current_acc
                )
            if 'loss' in serializable_logs:
                current_loss = serializable_logs['loss'] 
                progress['results']['best_loss'] = min(
                    progress['results']['best_loss'], 
                    current_loss
                )
        
        progress['results']['epochs_completed'].append(epoch_data)
        
        # Save checkpoint
        save_checkpoint(self.experiment_id, progress)
        
        print(f"✅ Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        if logs:
            print(f"   📊 Metrics: {make_serializable(logs)}")

def create_experiment_id(stage, modality, location=None, ratio=None):
    """Create unique experiment identifier."""
    parts = [SCRIPT_ID, stage, modality]
    if location:
        parts.append(str(location))  # Convert to string to handle various types
    if ratio:
        if isinstance(ratio, (tuple, list)) and len(ratio) == 2:
            parts.append(f"ratio_{ratio[0]:.1f}_{ratio[1]:.1f}")
        else:
            parts.append(str(ratio))  # Handle non-tuple ratios
    return "_".join(parts)


# -------- ENHANCED DATA LOADING AND FEATURE EXTRACTION --------
def extract_sequences_with_motion_context(user_data, time_window, overlap=0.5):
    """
    Extract sequences with enhanced 26-feature set including binary motion context.
    Optimized for RNN processing with efficient memory usage.
    
    Returns:
    - X: shape (n_sequences, time_window, 26) = 19 sensor features + 7 motion features
    - y: shape (n_sequences,) = activity labels
    - metadata: dict with sequence information
    """
    step_size = max(1, int(time_window * (1 - overlap)))
    sequences = []
    labels = []
    metadata = []
    
    print(f"📊 Extracting sequences with {time_window}-sample windows, {step_size}-sample step...")
    
    for activity, group in user_data.groupby('activity'):
        if len(group) < time_window:
            print(f"⚠️ Skipping {activity}: only {len(group)} samples < {time_window}")
            continue
            
        activity_sequences = 0
        
        # Process each contiguous segment to avoid boundary mixing
        segments = []
        current_segment = []
        last_index = None
        
        for idx, (_, row) in enumerate(group.iterrows()):
            if last_index is None or idx == last_index + 1:
                current_segment.append(row)
            else:
                if len(current_segment) >= time_window:
                    segments.append(pd.DataFrame(current_segment))
                current_segment = [row]
            last_index = idx
            
        if len(current_segment) >= time_window:
            segments.append(pd.DataFrame(current_segment))
        
        # Extract sequences from segments
        for segment in segments:
            segment_data = segment.reset_index(drop=True)
            
            for start_idx in range(0, len(segment_data) - time_window + 1, step_size):
                end_idx = start_idx + time_window
                sequence_data = segment_data.iloc[start_idx:end_idx]
                
                # Verify we have 26 features: 19 sensor + 7 motion context
                sensor_features = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z',
                                 'accel_mag', 'gyro_mag', 'accel_energy', 'gyro_energy',
                                 'accel_spectral_centroid', 'gyro_spectral_centroid',
                                 'accel_mean', 'accel_std', 'gyro_mean', 'gyro_std',
                                 'accel_rms', 'gyro_rms', 'movement_intensity']
                
                motion_features = ['is_stationary', 'is_walking', 'is_jogging', 'is_transitioning',
                                 'vertical_motion', 'lateral_motion', 'rotational_motion']
                
                if not all(col in sequence_data.columns for col in sensor_features + motion_features):
                    missing = [col for col in sensor_features + motion_features if col not in sequence_data.columns]
                    print(f"⚠️ Missing features: {missing}")
                    continue
                
                # Combine sensor and motion features (26 total)
                sequence_features = sequence_data[sensor_features + motion_features].values
                
                if sequence_features.shape != (time_window, 26):
                    print(f"⚠️ Unexpected sequence shape: {sequence_features.shape}, expected ({time_window}, 26)")
                    continue
                
                sequences.append(sequence_features)
                labels.append(activity)
                
                metadata.append({
                    'activity': activity,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'segment_length': len(segment_data),
                    'user': sequence_data.iloc[0].get('user', 'unknown'),
                    'location': sequence_data.iloc[0].get('location', 'unknown')
                })
                
                activity_sequences += 1
        
        print(f"✅ {activity}: {activity_sequences} sequences extracted")
    
    if not sequences:
        raise ValueError("No valid sequences extracted!")
    
    X = np.array(sequences)
    y = np.array(labels)
    
    print(f"📊 Final dataset shape: X={X.shape}, y={y.shape}")
    print(f"📊 Feature composition: 19 sensor + 7 motion = 26 total features")
    print(f"📊 Activities: {np.unique(y)}")
    
    return X, y, metadata

def extract_sequences(df, features):
    """
    Enhanced sequence extraction with comprehensive feature engineering for HAR data.
    Optimized for RNN processing with motion context integration.
    """
    if not all(f in df.columns for f in features):
        return np.empty((0, window_size, 1))  # Return empty with proper shape
    
    sequences = []
    expected_feature_count = None  # Will be set by first successful window
    
    for i in range(0, len(df) - window_size + 1, window_size):
        window = df[features].iloc[i:i + window_size]
        if window.shape[0] == window_size:
            try:
                # Enhanced feature engineering for HAR
                enhanced_features = []
                
                # Separate sensor features from decision tree output
                sensor_features = [f for f in features if 'acc_' in f or 'gyro_' in f]
                decision_features = [f for f in features if 'dec_tree_out' in f]
                
                # Process 3D sensor data (accelerometer or gyroscope)
                if sensor_features:
                    sensor_data = window[sensor_features].values  # Shape: (window_size, 3)
                    
                    # 1. Original 3D sensor signals (3 features)
                    enhanced_features.append(sensor_data)
                    
                    # 2. Magnitude of 3D vector (1 feature)
                    magnitude = np.sqrt(np.sum(sensor_data**2, axis=1)).reshape(-1, 1)
                    enhanced_features.append(magnitude)
                    
                    # 3. Statistical features for each axis (6 features: 3 axes × 2 stats)
                    for axis in range(3):  # Always process exactly 3 axes
                        if axis < sensor_data.shape[1]:
                            signal = sensor_data[:, axis]
                        else:
                            signal = np.zeros(window_size)  # Pad with zeros if missing
                        
                        # Rolling statistics
                        rolling_mean = np.convolve(signal, np.ones(5)/5, mode='same').reshape(-1, 1)
                        rolling_std = np.array([np.std(signal[max(0, j-2):j+3]) for j in range(len(signal))]).reshape(-1, 1)
                        
                        enhanced_features.append(rolling_mean)
                        enhanced_features.append(rolling_std)
                    
                    # 4. Inter-axis differences (3 features: xy, xz, yz)
                    # Ensure we have at least 3 axes by padding
                    padded_sensor = np.zeros((window_size, 3))
                    padded_sensor[:, :min(3, sensor_data.shape[1])] = sensor_data[:, :min(3, sensor_data.shape[1])]
                    
                    diff_xy = (padded_sensor[:, 0] - padded_sensor[:, 1]).reshape(-1, 1)
                    diff_xz = (padded_sensor[:, 0] - padded_sensor[:, 2]).reshape(-1, 1)
                    diff_yz = (padded_sensor[:, 1] - padded_sensor[:, 2]).reshape(-1, 1)
                    
                    enhanced_features.extend([diff_xy, diff_xz, diff_yz])
                    
                    # 5. Velocity features for accelerometer (3 features)
                    if 'acc_' in sensor_features[0]:
                        velocity = np.cumsum(padded_sensor - np.mean(padded_sensor, axis=0), axis=0)
                        enhanced_features.append(velocity)
                    else:
                        # Add zeros for gyroscope to maintain consistency
                        enhanced_features.append(np.zeros((window_size, 3)))
                    
                    # 6. Signal derivatives (3 features)
                    signal_diff = np.diff(padded_sensor, axis=0, prepend=padded_sensor[0:1])
                    enhanced_features.append(signal_diff)
                else:
                    # If no sensor features, add zeros to maintain structure (19 features total)
                    enhanced_features.extend([np.zeros((window_size, 3))] * 6)  # 6 groups × 3 features each
                    enhanced_features.append(np.zeros((window_size, 1)))  # magnitude
                
                # Process decision tree output (7 features always)
                if decision_features:
                    decision_data = window[decision_features].values  # Shape: (window_size, 1)
                    
                    # Binary motion features based on decision tree predictions
                    # These 7 features provide rich motion context for activity recognition
                    is_stationary = (decision_data == 0).astype(float)  # No movement
                    is_walking = (decision_data == 1).astype(float)     # Walking motion
                    is_jogging = (decision_data == 2).astype(float)     # Jogging motion
                    is_transitioning = (decision_data == 3).astype(float) # Transition states
                    
                    enhanced_features.extend([is_stationary, is_walking, is_jogging, is_transitioning])
                    
                    # Motion direction features (advanced binary context)
                    if sensor_features:
                        sensor_data = window[sensor_features].values
                        padded_sensor = np.zeros((window_size, 3))
                        padded_sensor[:, :min(3, sensor_data.shape[1])] = sensor_data[:, :min(3, sensor_data.shape[1])]
                        
                        vertical_motion = ((np.abs(padded_sensor[:, 2]) > np.std(padded_sensor[:, 2])) & 
                                         (decision_data[:, 0] > 0)).astype(float).reshape(-1, 1)
                        lateral_motion = ((np.max(np.abs(padded_sensor[:, :2]), axis=1) > 
                                          np.max(np.std(padded_sensor[:, :2], axis=0))) & 
                                         (decision_data[:, 0] > 0)).astype(float).reshape(-1, 1)
                        rotational_motion = ((np.max(np.abs(padded_sensor), axis=1) > 
                                             np.max(np.std(padded_sensor, axis=0))) & 
                                            (decision_data[:, 0] > 0)).astype(float).reshape(-1, 1)
                        
                        enhanced_features.extend([vertical_motion, lateral_motion, rotational_motion])
                    else:
                        # Add zeros if no sensor features (3 features)
                        enhanced_features.extend([np.zeros((window_size, 1))] * 3)
                else:
                    # If no decision features, add zeros (7 features total)
                    enhanced_features.extend([np.zeros((window_size, 1))] * 7)
                
                # Concatenate all features
                if enhanced_features:
                    feature_matrix = np.concatenate(enhanced_features, axis=1)
                    
                    # Validate consistent feature count
                    if expected_feature_count is None:
                        expected_feature_count = feature_matrix.shape[1]
                        print(f"🔧 Feature engineering complete - {expected_feature_count} features per window:")
                        print(f"   📊 Sensor features (19): 3D signals + magnitude + stats + differences + velocity + derivatives")
                        print(f"   🎯 Motion features (7): binary signal + transitions + percentage + stability + motion-aware")
                        print(f"   ✅ Total: {expected_feature_count} features (should be 26)")
                        
                        if expected_feature_count != 26:
                            print(f"⚠️ Warning: Expected 26 features but got {expected_feature_count}")
                    elif feature_matrix.shape[1] != expected_feature_count:
                        print(f"❌ Feature count mismatch: expected {expected_feature_count}, got {feature_matrix.shape[1]}. Skipping window.")
                        continue
                    
                    sequences.append(feature_matrix)
                
            except Exception as e:
                print(f"⚠️ Error processing window {i}: {e}")
                continue
    
    if sequences:
        try:
            result = np.array(sequences)
            print(f"✅ Extracted {len(sequences)} sequences with shape {result.shape}")
            return result
        except Exception as e:
            print(f"❌ Failed to stack sequences: {e}")
            return np.empty((0, window_size, 1))
    else:
        return np.empty((0, window_size, 1))

def train_model_with_checkpoints(model, X_train, y_train_cat, X_test, experiment_id, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Enhanced training function with comprehensive checkpointing for RNN models."""
    
    # Check for existing checkpoint
    progress = load_checkpoint(experiment_id)
    start_epoch = 0
    
    # Check if experiment is already completed
    if progress and progress.get('status') == 'completed':
        print(f"✅ Experiment {experiment_id} already completed, skipping")
        # Load model and return predictions without retraining
        exp_dir = os.path.join(CHECKPOINT_DIR, experiment_id)
        model_file = os.path.join(exp_dir, f"model_epoch_{progress['current_epoch']}.weights.h5")
        if os.path.exists(model_file):
            model.load_weights(model_file)
            print(f"✅ Loaded completed model from epoch {progress['current_epoch']}")
            return model.predict(X_test)
        else:
            print(f"⚠️ Warning: Completed experiment {experiment_id} missing model file, retraining...")
    
    if progress and progress['current_epoch'] > 0:
        print(f"🔄 Resuming from epoch {progress['current_epoch']}")
        start_epoch = progress['current_epoch']
        
        # Load saved model if available
        exp_dir = os.path.join(CHECKPOINT_DIR, experiment_id)
        model_file = os.path.join(exp_dir, f"model_epoch_{start_epoch}.weights.h5")
        if os.path.exists(model_file):
            model.load_weights(model_file)
            print(f"🔄 Model weights loaded from epoch {start_epoch}")
    else:
        # Initialize new progress tracking
        progress = {
            'experiment_id': experiment_id,
            'script_id': SCRIPT_ID,
            'current_epoch': 0,
            'total_epochs': epochs,
            'status': 'starting',
            'start_time': datetime.now().isoformat(),
            'results': {'epochs_completed': [], 'best_accuracy': 0, 'best_loss': float('inf')}
        }
        save_checkpoint(experiment_id, progress)
    
    # Setup callbacks with checkpointing
    exp_dir = os.path.join(CHECKPOINT_DIR, experiment_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    callbacks = [
        EpochProgressCallback(experiment_id, epochs),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(exp_dir, "model_epoch_{epoch}.weights.h5"),
            monitor='loss',
            save_best_only=False,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,  # More aggressive LR reduction for RNNs
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    device_name = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    
    with tf.device(device_name):
        print(f"🔥 Training Enhanced RNN on device: {'GPU' if 'GPU' in device_name else 'CPU'}")
        
        # Train only remaining epochs
        if start_epoch < epochs:
            remaining_epochs = epochs - start_epoch
            print(f"🚀 Training {remaining_epochs} remaining epochs (from {start_epoch} to {epochs})")
            
            history = model.fit(
                X_train, y_train_cat, 
                epochs=remaining_epochs,
                batch_size=batch_size, 
                verbose=1, 
                callbacks=callbacks, 
                validation_split=0.1, 
                shuffle=True,
                initial_epoch=start_epoch
            )
        else:
            print(f"✅ Training already completed ({start_epoch}/{epochs} epochs)")
        
        # Make predictions
        predictions = model.predict(X_test, batch_size=batch_size)
        
        # Clear GPU memory
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Mark as completed
        final_progress = load_checkpoint(experiment_id) or progress
        final_progress['status'] = 'completed'
        final_progress['completion_time'] = datetime.now().isoformat()
        save_checkpoint(experiment_id, final_progress)
        
        print(f"🎉 Experiment {experiment_id} completed successfully!")
        print(f"🧹 GPU memory cleared for next experiment")
    
    return predictions

def get_user_dirs(base):
    user_dirs = [d for d in os.listdir(base) if re.match(r"User \d+$", d)]
    return sorted(user_dirs, key=lambda x: int(re.findall(r"\d+", x)[0]))

def compute_metrics(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    mi = mutual_info_score(y_true, y_pred)
    p_true = np.bincount([labels.index(y) for y in y_true], minlength=len(labels)) / len(y_true)
    p_pred = np.bincount([labels.index(y) for y in y_pred], minlength=len(labels)) / len(y_pred)
    kl = np.sum(rel_entr(p_true + 1e-12, p_pred + 1e-12))
    return {
        "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "mutual_information": float(mi),
        "kl_divergence": float(kl),
        "confusion_matrix": cm.tolist(),
        "labels": labels
    }

def save_results(y_true, y_pred, labels, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"🔎 Saving results to: {out_dir}")
    print(f"🧪 Ground truth sample: {y_true[:10]}")
    print(f"🎯 Predicted sample:   {y_pred[:10]}")
    print(f"📊 Labels used:        {labels}")
    metrics = compute_metrics(y_true, y_pred, labels)
    report = classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(report)

    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(f"Macro Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}\n")
        f.write(f"Macro Recall: {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}\n")
        f.write(f"Macro F1-score: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"KL Divergence: {metrics['kl_divergence']:.4f}\n")
        f.write(f"Mutual Information: {metrics['mutual_information']:.4f}\n")

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    cm_df = pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels)
    cm_df.to_csv(os.path.join(out_dir, "confusion_matrix.csv"))


def load_data_by_location(features):
    data = {loc: {} for loc in locations}
    users = get_user_dirs(data_root)
    print("🗂️ Starting data loading...")

    for user in users:
        user_path = os.path.join(data_root, user, "Processed")
        print(f"\n📁 Processing user: {user}")
        if not os.path.isdir(user_path):
            print(f"❌ Missing user path: {user_path}")
            continue

        for activity in os.listdir(user_path):
            activity_path = os.path.join(user_path, activity)
            if not os.path.isdir(activity_path) or activity.lower() not in VALID_ACTIVITIES:
                print(f"⚠️ Skipping activity: {activity}")
                continue

            print(f"✅ Found activity: {activity.lower()}")

            for loc in locations:
                loc_path = os.path.join(activity_path, loc)
                files = (
                    glob.glob(os.path.join(loc_path, "*.csv"))
                    if os.path.isdir(loc_path)
                    else glob.glob(os.path.join(activity_path, f"*{loc}*.csv"))
                )

                feats_list, labels_list = [], []
                for file in files:
                    print(f"   📄 Reading file: {file}")
                    try:
                        df = pd.read_csv(file)
                        if 'dec_tree_out_1' not in df.columns or df['dec_tree_out_1'].isnull().any():
                            print(f"   ⚠️ Invalid or missing 'dec_tree_out_1' in: {file}")
                            continue
                        sequences = extract_sequences(df, features)
                        if sequences.size == 0:
                            print(f"   ⚠️ No sequences extracted from: {file}")
                            continue
                        feats_list.append(sequences)
                        labels_list.extend([activity.lower()] * sequences.shape[0])
                        print(f"   ✅ Extracted {sequences.shape[0]} sequences")
                    except Exception as e:
                        print(f"   ❌ Error reading file {file}: {e}")
                        continue

                if feats_list:
                    feats = np.vstack(feats_list)
                    labels = np.array(labels_list)

                    if user in data[loc]:
                        prev_feats, prev_labels = data[loc][user]
                        feats = np.vstack([prev_feats, feats])
                        labels = np.concatenate([prev_labels, labels])

                    data[loc][user] = (feats, labels)
                    print(f"   📊 Accumulated {len(labels)} samples for user {user} at {loc}")

    # Summary
    for loc in data:
        print(f"\n📦 Summary for location '{loc}': {len(data[loc])} users loaded.")
    return data



def run_lo_location():
    """Enhanced Leave-One-Location-Out with checkpointing and data caching."""
    print("🚀 Starting Enhanced Leave-One-Location-Out evaluation...")
    
    for modality, features in MODES.items():
        print(f"\n� Running LOLO for modality: {modality}")
        
        # Check for cached data first
        cache_file = os.path.join(global_cache_dir, f"{modality}_location_cache.pkl")
        if os.path.exists(cache_file):
            print(f"📦 Loading cached data for {modality} from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    data_by_loc = pickle.load(f)
                print(f"✅ Loaded cached data with {sum(len(data_by_loc[loc]) for loc in data_by_loc)} location datasets")
            except Exception as e:
                print(f"⚠️ Error loading cache: {e}, rebuilding...")
                data_by_loc = load_data_by_location(features)
                os.makedirs(global_cache_dir, exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(data_by_loc, f)
        else:
            print(f"📂 Building data cache for {modality}...")
            data_by_loc = load_data_by_location(features)
            os.makedirs(global_cache_dir, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(data_by_loc, f)
            print(f"💾 Cached data saved to {cache_file}")
        
        output_root = os.path.join(data_root, "Results", game, model_name, modality, "leave-one-location-out")
        os.makedirs(output_root, exist_ok=True)

        for test_loc in locations:
            print(f"\n🎯 LOLO: Testing on {test_loc}")

            for train_ratio, test_ratio in ratios:
                subdir = f"cross-user/train{int(train_ratio*100)}_test{int(test_ratio*100)}"
                out_path = os.path.join(output_root, subdir, f"test_on_{test_loc}")
                
                # Check if experiment already completed
                experiment_id = create_experiment_id("lolo", modality, test_loc, f"{train_ratio}_{test_ratio}")
                existing_progress = load_checkpoint(experiment_id)
                if existing_progress and existing_progress.get('status') == 'completed':
                    print(f"✅ LOLO experiment {experiment_id} already completed, skipping")
                    continue
                
                train_users, test_users = [], []

                for loc in locations:
                    users = list(data_by_loc[loc].keys())
                    np.random.shuffle(users)
                    n_train = int(train_ratio * len(users))
                    if loc != test_loc:
                        train_users += [(loc, u) for u in users[:n_train]]
                    else:
                        test_users += [(loc, u) for u in users[n_train:]]

                X_train, y_train, X_test, y_test = [], [], [], []
                for loc, user in train_users:
                    X, y = data_by_loc[loc][user]
                    X_train.append(X)
                    y_train.extend(y)
                for loc, user in test_users:
                    X, y = data_by_loc[loc][user]
                    X_test.append(X)
                    y_test.extend(y)

                if X_train and X_test:
                    print(f"📊 Training samples: {len(y_train)}, Testing samples: {len(y_test)}")
                    X_train = np.vstack(X_train)
                    X_test = np.vstack(X_test)
                    
                    # Data validation
                    if np.array_equal(X_train, X_test):
                        print("❌ [LEAKAGE] X_train and X_test are identical!")
                        continue
                    
                    # Encode labels
                    le = LabelEncoder().fit(y_train + y_test)
                    print("🧾 LabelEncoder classes:", dict(zip(le.classes_, le.transform(le.classes_))))
                    y_train_cat = to_categorical(le.transform(y_train))
                    
                    # Build model
                    print("🏗️ Building Enhanced RNN model...")
                    model = build_enhanced_rnn(X_train.shape[1:], y_train_cat.shape[1])
                    
                    print("🔍 y_train distribution:", Counter(y_train))
                    print("🔍 y_test  distribution:", Counter(y_test))
                    
                    # Train with checkpoints
                    print(f"🚀 Starting training with experiment ID: {experiment_id}")
                    y_pred_probs = train_model_with_checkpoints(model, X_train, y_train_cat, X_test, experiment_id)
                    y_pred = le.inverse_transform(np.argmax(y_pred_probs, axis=1))
                    
                    # Save results
                    save_results(y_test, y_pred.tolist(), list(le.classes_), out_path)
                    print(f"✅ LOLO completed for {test_loc} with ratio {train_ratio}_{test_ratio}")

            # Intra-User Evaluation
            print(f"🔄 Running intra-user evaluation for {test_loc}...")
            intra_preds, intra_trues = [], []
            for user in data_by_loc.get(test_loc, {}):
                X, y = data_by_loc[test_loc][user]
                if len(X) < 2 or len(np.unique(y)) < 2:
                    print(f"⚠️ Skipping user {user} due to insufficient data")
                    continue
                
                try:
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
                    le = LabelEncoder().fit(y_tr.tolist() + y_te.tolist())
                    y_tr_cat = to_categorical(le.transform(y_tr))
                    
                    experiment_id = create_experiment_id("intra", modality, test_loc, user)
                    model = build_enhanced_rnn(X_tr.shape[1:], y_tr_cat.shape[1])
                    
                    y_pred_probs = train_model_with_checkpoints(model, X_tr, y_tr_cat, X_te, experiment_id)
                    y_pred = le.inverse_transform(np.argmax(y_pred_probs, axis=1))
                    
                    intra_preds.extend(y_pred)
                    intra_trues.extend(y_te)
                except Exception as e:
                    print(f"⚠️ Error in intra-user for {user}: {e}")
                    continue

            if intra_preds:
                intra_path = os.path.join(output_root, "intra-user", f"test_on_{test_loc}")
                save_results(intra_trues, intra_preds, sorted(set(intra_trues)), intra_path)
                print(f"✅ Intra-user completed for {test_loc}")

            # Leave-One-User-Out
            print(f"🔄 Running LOUO evaluation for {test_loc}...")
            louo_preds, louo_trues = [], []
            all_users = list(data_by_loc.get(test_loc, {}).keys())
            
            for test_user in all_users[:3]:  # Limit to first 3 users for efficiency
                train_users = [u for u in all_users if u != test_user]
                X_train, y_train = [], []
                for u in train_users:
                    X, y = data_by_loc[test_loc][u]
                    X_train.append(X)
                    y_train.extend(y)
                    
                X_test, y_test = data_by_loc[test_loc][test_user]
                
                if X_train and len(X_test) > 0:
                    try:
                        X_train = np.vstack(X_train)
                        le = LabelEncoder().fit(y_train + y_test.tolist())
                        y_train_cat = to_categorical(le.transform(y_train))
                        
                        experiment_id = create_experiment_id("louo", modality, test_loc, test_user)
                        model = build_enhanced_rnn(X_train.shape[1:], y_train_cat.shape[1])
                        
                        y_pred_probs = train_model_with_checkpoints(model, X_train, y_train_cat, X_test, experiment_id)
                        y_pred = le.inverse_transform(np.argmax(y_pred_probs, axis=1))
                        
                        louo_preds.extend(y_pred)
                        louo_trues.extend(y_test)
                    except Exception as e:
                        print(f"⚠️ Error in LOUO for {test_user}: {e}")
                        continue

            if louo_preds:
                louo_path = os.path.join(output_root, "leave-one-user-out", f"test_on_{test_loc}")
                save_results(louo_trues, louo_preds, sorted(set(louo_trues)), louo_path)
                print(f"✅ LOUO completed for {test_loc}")

    print("🎉 Enhanced Leave-One-Location-Out evaluation completed!")




def run_train_one_test_others():
    """Enhanced Train-One-Test-Others with checkpointing and data caching."""
    print("🚀 Starting Enhanced Train-One-Test-Others evaluation...")
    
    for modality, features in MODES.items():
        print(f"\n� Running Train-One-Test-Others for modality: {modality}")
        
        # Check for cached data first
        cache_file = os.path.join(global_cache_dir, f"{modality}_location_cache.pkl")
        if os.path.exists(cache_file):
            print(f"📦 Loading cached data for {modality}")
            with open(cache_file, 'rb') as f:
                data_by_loc = pickle.load(f)
        else:
            data_by_loc = load_data_by_location(features)
            os.makedirs(global_cache_dir, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(data_by_loc, f)
        
        output_root = os.path.join(data_root, "Results", game, model_name, modality, "train-one-test-others")
        os.makedirs(output_root, exist_ok=True)

        for train_loc in locations:
            test_locs = [loc for loc in locations if loc != train_loc]
            print(f"\n🎯 Training on {train_loc}, testing on: {test_locs}")

            for train_ratio, test_ratio in ratios:
                subdir = f"cross-user/train{int(train_ratio*100)}_test{int(test_ratio*100)}"
                out_path = os.path.join(output_root, subdir, f"train_on_{train_loc}")
                
                # Check if experiment already completed
                experiment_id = create_experiment_id("toto", modality, train_loc, f"{train_ratio}_{test_ratio}")
                existing_progress = load_checkpoint(experiment_id)
                if existing_progress and existing_progress.get('status') == 'completed':
                    print(f"✅ TOTO experiment {experiment_id} already completed, skipping")
                    continue
                
                users = list(data_by_loc[train_loc].keys())
                np.random.shuffle(users)
                train_users = users[:int(train_ratio * len(users))]
                test_users = [(loc, u) for loc in test_locs for u in data_by_loc[loc].keys()]

                X_train, y_train, X_test, y_test = [], [], [], []
                for u in train_users:
                    X, y = data_by_loc[train_loc][u]
                    X_train.append(X)
                    y_train.extend(y)
                for loc, u in test_users:
                    X, y = data_by_loc[loc][u]
                    X_test.append(X)
                    y_test.extend(y)

                if X_train and X_test:
                    print(f"📊 Training samples: {len(y_train)}, Testing samples: {len(y_test)}")
                    X_train = np.vstack(X_train)
                    X_test = np.vstack(X_test)
                    
                    if np.array_equal(X_train, X_test):
                        print("❌ [LEAKAGE] X_train and X_test are identical!")
                        continue
                        
                    le = LabelEncoder().fit(y_train + y_test)
                    y_train_cat = to_categorical(le.transform(y_train))
                    
                    print("🏗️ Building Enhanced RNN model...")
                    model = build_enhanced_rnn(X_train.shape[1:], y_train_cat.shape[1])
                    
                    print("🔍 y_train distribution:", Counter(y_train))
                    print("🔍 y_test  distribution:", Counter(y_test))
                    
                    # Train with checkpoints
                    y_pred_probs = train_model_with_checkpoints(model, X_train, y_train_cat, X_test, experiment_id)
                    y_pred = le.inverse_transform(np.argmax(y_pred_probs, axis=1))
                    save_results(y_test, y_pred.tolist(), list(le.classes_), out_path)
                    print(f"✅ TOTO completed for {train_loc}")

    print("🎉 Enhanced Train-One-Test-Others evaluation completed!")






# -------- ENHANCED MAIN TRAINING FUNCTION --------
def load_and_process_data_efficiently(file_path, max_samples=None):
    """
    Load and process data with enhanced feature engineering and memory efficiency.
    Optimized for RNN training with comprehensive feature extraction.
    """
    print(f"📂 Loading data from: {file_path}")
    
    try:
        data = pd.read_csv(file_path)
        print(f"📊 Loaded {len(data)} total samples with {data.shape[1]} columns")
        
        if max_samples and len(data) > max_samples:
            # Sample uniformly across activities to maintain balance
            data = data.groupby('activity').apply(
                lambda x: x.sample(min(len(x), max_samples // len(data['activity'].unique())))
            ).reset_index(drop=True)
            print(f"📊 Sampled down to {len(data)} samples for efficiency")
        
        # Enhanced feature engineering for Human Activity Recognition
        print("🔧 Applying enhanced feature engineering...")
        
        # Basic sensor features validation
        required_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        if not all(col in data.columns for col in required_cols):
            print(f"❌ Missing required sensor columns. Available: {list(data.columns)}")
            return None
        
        # 1. Enhanced accelerometer features (9 features)
        data['accel_x'] = data['acc_x']
        data['accel_y'] = data['acc_y']
        data['accel_z'] = data['acc_z']
        data['accel_mag'] = np.sqrt(data['acc_x']**2 + data['acc_y']**2 + data['acc_z']**2)
        data['accel_energy'] = data['accel_mag']**2
        data['accel_spectral_centroid'] = data['accel_mag'].rolling(window=5, center=True).mean()
        data['accel_mean'] = data[['acc_x', 'acc_y', 'acc_z']].mean(axis=1)
        data['accel_std'] = data[['acc_x', 'acc_y', 'acc_z']].std(axis=1)
        data['accel_rms'] = np.sqrt((data['acc_x']**2 + data['acc_y']**2 + data['acc_z']**2) / 3)
        
        # 2. Enhanced gyroscope features (9 features)
        data['gyro_x'] = data['gyro_x']
        data['gyro_y'] = data['gyro_y']
        data['gyro_z'] = data['gyro_z']
        data['gyro_mag'] = np.sqrt(data['gyro_x']**2 + data['gyro_y']**2 + data['gyro_z']**2)
        data['gyro_energy'] = data['gyro_mag']**2
        data['gyro_spectral_centroid'] = data['gyro_mag'].rolling(window=5, center=True).mean()
        data['gyro_mean'] = data[['gyro_x', 'gyro_y', 'gyro_z']].mean(axis=1)
        data['gyro_std'] = data[['gyro_x', 'gyro_y', 'gyro_z']].std(axis=1)
        data['gyro_rms'] = np.sqrt((data['gyro_x']**2 + data['gyro_y']**2 + data['gyro_z']**2) / 3)
        
        # 3. Movement intensity feature (1 feature)
        data['movement_intensity'] = (data['accel_mag'] + data['gyro_mag']) / 2
        
        # 4. Binary motion context features from decision tree output (7 features)
        if 'dec_tree_out_1' in data.columns:
            print("🎯 Adding binary motion context features from decision tree...")
            # Binary activity states
            data['is_stationary'] = (data['dec_tree_out_1'] == 0).astype(float)
            data['is_walking'] = (data['dec_tree_out_1'] == 1).astype(float)
            data['is_jogging'] = (data['dec_tree_out_1'] == 2).astype(float)
            data['is_transitioning'] = (data['dec_tree_out_1'] == 3).astype(float)
            
            # Advanced motion direction features
            data['vertical_motion'] = ((np.abs(data['acc_z']) > data['acc_z'].std()) & 
                                     (data['dec_tree_out_1'] > 0)).astype(float)
            data['lateral_motion'] = ((np.maximum(np.abs(data['acc_x']), np.abs(data['acc_y'])) > 
                                     np.maximum(data['acc_x'].std(), data['acc_y'].std())) & 
                                    (data['dec_tree_out_1'] > 0)).astype(float)
            data['rotational_motion'] = ((data['gyro_mag'] > data['gyro_mag'].std()) & 
                                        (data['dec_tree_out_1'] > 0)).astype(float)
        else:
            print("⚠️ No decision tree output found, using derived motion features...")
            # Create motion features from sensor data
            accel_threshold = data['accel_mag'].quantile(0.3)
            gyro_threshold = data['gyro_mag'].quantile(0.3)
            
            data['is_stationary'] = (data['accel_mag'] < accel_threshold).astype(float)
            data['is_walking'] = ((data['accel_mag'] >= accel_threshold) & 
                                (data['accel_mag'] < data['accel_mag'].quantile(0.7))).astype(float)
            data['is_jogging'] = (data['accel_mag'] >= data['accel_mag'].quantile(0.7)).astype(float)
            data['is_transitioning'] = (data['gyro_mag'] > gyro_threshold).astype(float)
            data['vertical_motion'] = (np.abs(data['acc_z']) > data['acc_z'].std()).astype(float)
            data['lateral_motion'] = (np.maximum(np.abs(data['acc_x']), np.abs(data['acc_y'])) > 
                                    np.maximum(data['acc_x'].std(), data['acc_y'].std())).astype(float)
            data['rotational_motion'] = (data['gyro_mag'] > data['gyro_mag'].std()).astype(float)
        
        # Handle NaN values from rolling operations
        data = data.fillna(method='bfill').fillna(method='ffill')
        
        print(f"✅ Feature engineering complete - {len(data)} samples with enhanced features")
        print(f"📊 Activity distribution: {data['activity'].value_counts().to_dict()}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
        X_train, X_test, y_train, y_test = train_test_split(
            X_normalized, y_categorical, test_size=0.2, random_state=42, stratify=all_y
        )
        
        print(f"📊 Training set: X={X_train.shape}, y={y_train.shape}")
        print(f"📊 Test set: X={X_test.shape}, y={y_test.shape}")
        
        # Build and train enhanced RNN model
        print("\n🧠 Building Enhanced RNN Model...")
        model = build_enhanced_rnn(
            input_shape=(window_size, n_features),
            num_classes=n_classes,
            dropout_rate=DROPOUT_RATE
        )
        
        # Print model summary
        model.summary()
        
        # Train model with checkpointing
        experiment_id = f"{experiment_base_id}_main"
        print(f"\n🚀 Training model with experiment ID: {experiment_id}")
        
        predictions = train_model_with_checkpoints(
            model, X_train, y_train, X_test, experiment_id, 
            epochs=EPOCHS, batch_size=BATCH_SIZE
        )
        
        # Evaluate model
        print("\n📊 Evaluating Enhanced RNN Model...")
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"🎯 Enhanced RNN Results:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        
        # Detailed classification report
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
        
        # Save final results
        results = {
            'experiment_id': experiment_id,
            'model_type': 'Enhanced_RNN',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'n_classes': n_classes,
            'classes': label_encoder.classes_.tolist(),
            'data_shape': all_X.shape,
            'window_size': window_size,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'features': n_features,
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = f"results/enhanced_rnn_results_{experiment_id}.json"
        os.makedirs("results", exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to {results_file}")
        print("🎉 Enhanced RNN training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up GPU memory
        tf.keras.backend.clear_session()
        gc.collect()
        print("🧹 GPU memory cleared")

# -------- MAIN EXECUTION --------
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎯 ENHANCED RNN TRAINING PIPELINE")
    print("="*80)
    
    print(f"🔧 Configuration:")
    print(f"   📊 Window size: {window_size}")
    print(f"   🔄 Epochs: {EPOCHS}")
    print(f"   📦 Batch size: {BATCH_SIZE}")
    print(f"   📍 Locations: {locations}")
    print(f"   📈 Ratios: {ratios}")
    print(f"   🎯 Script ID: {SCRIPT_ID}")
    
    try:
        # Run Leave-One-Location-Out evaluation with enhanced features
        print("\n🚀 Starting Leave-One-Location-Out evaluation...")
        run_lo_location()
        
        # Run Train-One-Test-Others evaluation with enhanced features
        print("\n🚀 Starting Train-One-Test-Others evaluation...")
        # run_train_one_test_others()
        
        print("\n🎉 All evaluations completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up GPU memory
        tf.keras.backend.clear_session()
        gc.collect()
        print("🧹 GPU memory cleared")
