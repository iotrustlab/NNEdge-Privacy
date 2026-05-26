import os
import re
import glob
import json
import pickle
import time
import uuid
from datetime import datetime
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    mutual_info_score, precision_recall_fscore_support
)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, Add,
    MultiHeadAttention, GlobalAveragePooling1D, 
    Conv1D, BatchNormalization, Embedding,
    Concatenate, Lambda, Reshape
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from scipy.special import rel_entr
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import gc

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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -------- CONFIG --------
data_root = os.environ.get("NNEDGE_PRIVACY_STM_ROOT", os.path.join(REPO_ROOT, "STMDATASET", "Data"))
game = "Game-2"
model_name = "Transformer"
locations = ["left-ankle", "left-wrist", "right-ankle", "right-pocket", "right-wrist"]
window_size = 20
ratios = [(0.8, 0.2), (0.7, 0.3), (0.5, 0.5)]
EPOCHS = 20  # Reduced from 100 for faster training with checkpointing
BATCH_SIZE = 16  # Reduced from 32 to prevent memory overflow with large datasets
LEARNING_RATE = 0.001
MIN_DELTA = 0.001  # Minimum change in monitored quantity to qualify as an improvement
PATIENCE = 15  # Number of epochs with no improvement after which training will be stopped
VALID_ACTIVITIES = {"jogging", "laying", "sitting", "standing", "upstairs", "downstairs", "walking"}

# -------- CHECKPOINT SYSTEM --------
# Generate consistent script identifier or reuse existing one
CHECKPOINT_DIR = os.environ.get("NNEDGE_PRIVACY_CHECKPOINT_ROOT", os.path.join(REPO_ROOT, "checkpoints", "Game-2_Transformer_LocationAblation_Checkpoints"))

# Try to find existing script ID from checkpoint directory, otherwise create new one
def get_or_create_script_id():
    if os.path.exists(CHECKPOINT_DIR):
        # Look for ALL existing experiment directories 
        existing_dirs = [d for d in os.listdir(CHECKPOINT_DIR) 
                        if d.startswith("Game-2_Transformer_LocationAblation_") and len(d.split('_')) >= 4]
        
        # Find script IDs and their experiment counts
        script_id_stats = {}
        for dir_name in existing_dirs:
            if dir_name == "data_loading":  # Skip data loading directory
                continue
                
            progress_file = os.path.join(CHECKPOINT_DIR, dir_name, "training_progress.json")
            script_id = "_".join(dir_name.split('_')[:4])  # Extract script ID
            
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
    new_script_id = f"Game-2_Transformer_LocationAblation_{uuid.uuid4().hex[:8]}"
    print(f"🆕 Created new Script ID: {new_script_id}")
    return new_script_id

SCRIPT_ID = get_or_create_script_id()

def cleanup_empty_checkpoints():
    """Remove checkpoint directories with no training progress."""
    if not os.path.exists(CHECKPOINT_DIR):
        return
    
    dirs_to_remove = []
    for dir_name in os.listdir(CHECKPOINT_DIR):
        if not dir_name.startswith("Game-2_Transformer_LocationAblation_"):
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
            import shutil
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


MODES = {
#    "accel": ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]', 'dec_tree_out_1'],
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


# -------- HELPERS --------
def extract_sequences(df, features):
    """
    Robust sequence extraction with consistent feature engineering for HAR data.
    Ensures all windows produce exactly the same number of features.
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
                
                # Process decision tree output (4 features always)
                if decision_features:
                    decision_data = window[decision_features].values  # Shape: (window_size, 1)
                    
                    # Original binary motion signal (1 feature)
                    enhanced_features.append(decision_data)
                    
                    # Motion state transitions (1 feature)
                    motion_transitions = np.diff(decision_data, axis=0, prepend=decision_data[0:1])
                    enhanced_features.append(motion_transitions)
                    
                    # Motion percentage (1 feature)
                    motion_percentage = np.mean(decision_data)
                    motion_percentage_expanded = np.full((window_size, 1), motion_percentage)
                    enhanced_features.append(motion_percentage_expanded)
                    
                    # Motion stability (1 feature)
                    motion_stability = 1.0 - np.mean(np.abs(motion_transitions))
                    motion_stability_expanded = np.full((window_size, 1), motion_stability)
                    enhanced_features.append(motion_stability_expanded)
                    
                    # Motion-aware features (3 additional features)
                    if sensor_features:
                        # Motion-aware magnitude (1 feature)
                        motion_aware_magnitude = magnitude * (0.5 + 0.5 * decision_data)
                        enhanced_features.append(motion_aware_magnitude)
                        
                        # Static vs dynamic analysis (2 features)
                        static_mask = (decision_data == 0).astype(float)
                        dynamic_mask = (decision_data == 1).astype(float)
                        
                        # Always add both features even if all zeros
                        padded_sensor = np.zeros((window_size, 3))
                        if len(sensor_features) > 0:
                            sensor_data = window[sensor_features].values
                            padded_sensor[:, :min(3, sensor_data.shape[1])] = sensor_data[:, :min(3, sensor_data.shape[1])]
                        
                        static_sensor_variance = np.var(padded_sensor, axis=1, keepdims=True) * static_mask
                        dynamic_sensor_range = (np.max(padded_sensor, axis=1, keepdims=True) - 
                                              np.min(padded_sensor, axis=1, keepdims=True)) * dynamic_mask
                        
                        enhanced_features.extend([static_sensor_variance, dynamic_sensor_range])
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
                        print(f"   🔧 This indicates inconsistent feature engineering - debugging needed")
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

def get_user_dirs(base):
    user_dirs = [d for d in os.listdir(base) if re.match(r"User \d+$", d)]
    return sorted(user_dirs, key=lambda x: int(re.findall(r"\d+", x)[0]))

def build_transformer(input_shape, num_classes):
    """Enhanced Transformer architecture for HAR with motion-aware processing."""
    # Clear any existing sessions to avoid naming conflicts
    tf.keras.backend.clear_session()
    
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        
        # Input layer
        inputs = Input(shape=input_shape, name='input_sequences')
        
        # Feature projection for optimal attention computation
        d_model = 128  # Model dimension for transformer
        
        # Project input features to model dimension
        x = Dense(d_model, activation='relu', name='feature_projection')(inputs)
        x = LayerNormalization(epsilon=1e-6, name='input_norm')(x)
        
        # Positional encoding for temporal awareness
        seq_len = input_shape[0]  # window_size
        position_embedding = tf.keras.layers.Embedding(
            input_dim=seq_len, 
            output_dim=d_model, 
            name='positional_embedding'
        )
        
        # Create position indices and expand for batch dimension
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_encoded = position_embedding(positions)
        
        # Expand positional encoding to match batch dimension
        pos_encoded = tf.expand_dims(pos_encoded, axis=0)  # Shape: (1, seq_len, d_model)
        
        # Debug print shapes for verification
        print(f"🔧 Positional encoding fix applied:")
        print(f"   Input x shape will be: (batch_size, {seq_len}, {d_model})")
        print(f"   Positional encoding shape: (1, {seq_len}, {d_model})")
        print(f"   Broadcasting will handle batch dimension automatically")
        
        # Add positional encoding (broadcasting will handle batch dimension)
        x = Add(name='add_position_encoding')([x, pos_encoded])
        x = Dropout(0.1, name='pos_dropout')(x)
        
        # Multi-layer Transformer encoder
        num_transformer_layers = 3
        num_attention_heads = 8
        ff_dim = 256
        
        for layer_idx in range(num_transformer_layers):
            # Multi-head self-attention
            attention_output = MultiHeadAttention(
                num_heads=num_attention_heads,
                key_dim=d_model // num_attention_heads,
                name=f'attention_layer_{layer_idx}'
            )(x, x)
            
            # Add & Norm
            x = Add(name=f'add_attention_{layer_idx}')([x, attention_output])
            x = LayerNormalization(epsilon=1e-6, name=f'norm_attention_{layer_idx}')(x)
            
            # Feed-forward network
            ff_output = Dense(ff_dim, activation='relu', name=f'ff1_layer_{layer_idx}')(x)
            ff_output = Dropout(0.1, name=f'ff_dropout_{layer_idx}')(ff_output)
            ff_output = Dense(d_model, name=f'ff2_layer_{layer_idx}')(ff_output)
            
            # Add & Norm
            x = Add(name=f'add_ff_{layer_idx}')([x, ff_output])
            x = LayerNormalization(epsilon=1e-6, name=f'norm_ff_{layer_idx}')(x)
        
        # Motion-aware attention pooling
        # Extract motion features for attention weighting
        motion_features = Lambda(
            lambda seq: seq[:, :, -7:],  # Last 7 features are motion-related
            name='extract_motion_features'
        )(x)
        
        # Compute attention weights based on motion context
        motion_attention = Dense(1, activation='sigmoid', name='motion_attention')(motion_features)
        motion_attention = tf.keras.layers.Softmax(axis=1, name='motion_softmax')(motion_attention)
        
        # Apply motion-aware weighted pooling
        motion_weighted = tf.keras.layers.Multiply(name='motion_weighting')([x, motion_attention])
        
        # Combine global average pooling with motion-aware pooling
        global_pooled = GlobalAveragePooling1D(name='global_avg_pool')(x)
        motion_pooled = GlobalAveragePooling1D(name='motion_avg_pool')(motion_weighted)
        
        # Concatenate different pooling strategies
        combined_features = Concatenate(name='combine_features')([global_pooled, motion_pooled])
        
        # Activity-specific feature processing
        activity_features = Dense(512, activation='relu', name='activity_dense1')(combined_features)
        activity_features = BatchNormalization(name='activity_bn1')(activity_features)
        activity_features = Dropout(0.3, name='activity_dropout1')(activity_features)
        
        activity_features = Dense(256, activation='relu', name='activity_dense2')(activity_features)
        activity_features = BatchNormalization(name='activity_bn2')(activity_features)
        activity_features = Dropout(0.3, name='activity_dropout2')(activity_features)
        
        # Motion-context classification head
        motion_context = Dense(128, activation='relu', name='motion_context')(combined_features)
        motion_context = BatchNormalization(name='motion_bn')(motion_context)
        motion_context = Dropout(0.2, name='motion_dropout')(motion_context)
        
        # Final feature combination
        final_features = Concatenate(name='final_combine')([activity_features, motion_context])
        final_features = Dense(256, activation='relu', name='final_dense')(final_features)
        final_features = BatchNormalization(name='final_bn')(final_features)
        final_features = Dropout(0.3, name='final_dropout')(final_features)
        
        # Output layer
        outputs = Dense(num_classes, activation='softmax', name='activity_output')(final_features)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='MotionAwareTransformer')
        
        # Advanced optimizer with motion-aware settings
        optimizer = Adam(
            learning_rate=LEARNING_RATE,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            clipnorm=1.0  # Gradient clipping for stability
        )
        
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
        
        print(f"🏗️ Motion-aware Transformer model built on device: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
        print(f"📊 Model parameters: {model.count_params():,}")
        print(f"🎯 Input shape: {input_shape}")
        print(f"🏷️ Output classes: {num_classes}")
        print(f"🎮 Motion context: Utilizes binary motion indicator and enhanced attention mechanisms")
        print(f"🔄 Architecture: {num_transformer_layers} transformer layers, {num_attention_heads} attention heads")
    
    return model

def train_model_with_checkpoints(model, X_train, y_train_cat, X_test, experiment_id, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Enhanced training function with comprehensive checkpointing."""
    
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
            factor=0.7,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Memory-efficient data augmentation for HAR
    def augment_har_sequences(X, y, noise_factor=0.01, max_augmentation_ratio=1.5):
        """
        Memory-efficient augmentation that limits total augmented samples.
        Max ratio of 1.5 means we'll have at most 1.5x the original data.
        """
        original_size = len(X)
        max_total_size = int(original_size * max_augmentation_ratio)
        max_new_samples = max_total_size - original_size
        
        print(f"🔧 Memory-efficient augmentation: {original_size} -> max {max_total_size} samples")
        
        X_aug, y_aug = [], []
        new_samples_added = 0
        
        # Always include original data
        for i in range(len(X)):
            X_aug.append(X[i])
            y_aug.append(y[i])
        
        # Add augmented samples up to the limit
        for i in range(len(X)):
            if new_samples_added >= max_new_samples:
                break
                
            # Extract motion state from features (assuming dec_tree_out_1 is in the features)
            motion_states = X[i][:, -4]  # Motion indicator should be among the last features
            motion_percentage = np.mean(motion_states)
            
            # Selective augmentation based on motion patterns
            augmentation_probability = 0.2  # Only augment 20% of samples
            
            if np.random.random() < augmentation_probability and new_samples_added < max_new_samples:
                # Time shifting (limited)
                if motion_percentage > 0.5:  # Only for dynamic activities
                    shift = np.random.randint(1, min(3, X[i].shape[0]))
                    if shift < X[i].shape[0]:
                        X_shifted = np.roll(X[i], shift, axis=0)
                        X_aug.append(X_shifted)
                        y_aug.append(y[i])
                        new_samples_added += 1
                        
                        if new_samples_added >= max_new_samples:
                            break
            
            if np.random.random() < 0.1 and new_samples_added < max_new_samples:
                # Motion-aware noise injection (selective)
                if motion_percentage > 0.7:  # High movement
                    X_noisy = X[i].copy()
                    sensor_noise = np.random.normal(0, noise_factor * 1.2, X_noisy[:, :6].shape)
                    X_noisy[:, :6] += sensor_noise
                    X_aug.append(X_noisy)
                    y_aug.append(y[i])
                    new_samples_added += 1
                elif motion_percentage < 0.3:  # Low movement
                    X_noisy = X[i].copy()
                    sensor_noise = np.random.normal(0, noise_factor * 0.3, X_noisy[:, :6].shape)
                    X_noisy[:, :6] += sensor_noise
                    X_aug.append(X_noisy)
                    y_aug.append(y[i])
                    new_samples_added += 1
        
        final_X = np.array(X_aug)
        final_y = np.array(y_aug)
        
        print(f"✅ Augmentation complete: {original_size} -> {len(final_X)} samples (added {new_samples_added})")
        return final_X, final_y

    # Augment training data with memory limits
    X_train_aug, y_train_aug = augment_har_sequences(X_train, np.argmax(y_train_cat, axis=1))
    y_train_cat_aug = to_categorical(y_train_aug)

    # Memory management: if dataset is too large, use sampling
    max_training_samples = 500000  # Limit to 500K samples to prevent memory issues
    if len(X_train_aug) > max_training_samples:
        print(f"⚠️ Dataset too large ({len(X_train_aug)} samples), sampling to {max_training_samples}")
        
        # Simple random sampling to reduce memory usage
        indices = np.random.choice(len(X_train_aug), size=max_training_samples, replace=False)
        X_train_aug = X_train_aug[indices]
        y_train_cat_aug = y_train_cat_aug[indices]
        
        print(f"✅ Sampled dataset: {len(X_train_aug)} samples")

    device_name = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    
    with tf.device(device_name):
        print(f"🔥 Training on device: {'GPU' if 'GPU' in device_name else 'CPU'} with augmented data: {X_train_aug.shape[0]} samples")
        
        # Train only remaining epochs
        if start_epoch < epochs:
            remaining_epochs = epochs - start_epoch
            print(f"🚀 Training {remaining_epochs} remaining epochs (from {start_epoch} to {epochs})")
            
            # Memory-efficient training with smaller batch size and steps per epoch
            steps_per_epoch = min(1000, len(X_train_aug) // batch_size)  # Limit steps to prevent memory issues
            
            history = model.fit(
                X_train_aug, y_train_cat_aug, 
                epochs=remaining_epochs,
                batch_size=batch_size, 
                steps_per_epoch=steps_per_epoch,  # Limit training steps
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
    """Load data with caching support for faster restarts."""
    # Create deterministic cache name based on modality (no hashing)
    features_str = str(features).lower()
    if 'acc_' in features_str:
        cache_name = "data_by_location_accelerometer"
    elif 'gyro_' in features_str:
        cache_name = "data_by_location_gyroscope"
    else:
        # Fallback: use sorted feature names for consistency
        sorted_features = sorted([str(f) for f in features])
        cache_name = f"data_by_location_{'_'.join(sorted_features[:2])}"  # Use first 2 features
    
    # Use data_loading directory as global cache
    global_cache_dir = os.path.join(CHECKPOINT_DIR, "data_loading")
    os.makedirs(global_cache_dir, exist_ok=True)
    
    cache_file = os.path.join(global_cache_dir, f"{cache_name}_cache.pkl")
    
    # Try to load from cache first
    if os.path.exists(cache_file):
        try:
            print(f"🔍 Attempting to load cache: {cache_file}")
            cache_size = os.path.getsize(cache_file) / (1024*1024)  # MB
            print(f"📦 Cache file size: {cache_size:.1f} MB")
            
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            print(f"✅ Successfully loaded cached data from: {cache_file}")
            print(f"📊 Cache contains data for {len(cached_data)} locations")
            for loc, users in cached_data.items():
                total_samples = sum(len(data[1]) for data in users.values()) if users else 0
                print(f"   - {loc}: {len(users)} users, {total_samples} total samples")
            
            print("🚀 Cache loaded successfully - skipping data loading!")
            return cached_data
        except Exception as e:
            print(f"❌ Error loading cache: {e}")
            print("📥 Cache corrupted or incompatible, reloading data from files...")
    else:
        print(f"📥 No cache found at: {cache_file}")
        print("📥 Loading data from files...")
        
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
                        print(f"   ✅ Extracted {sequences.shape[0]} sequences with enhanced features")
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
                    print(f"   📊 Accumulated {len(labels)} samples for user {user} at {loc} with shape {feats.shape}")

    # Summary
    for loc in data:
        print(f"\n📦 Summary for location '{loc}': {len(data[loc])} users loaded.")
    
    # Save to global cache for future runs with validation
    try:
        print(f"💾 Saving cache to: {cache_file}")
        
        # Validate data before saving
        total_locations = len(data)
        total_users = sum(len(users) for users in data.values())
        total_samples = 0
        
        for loc, users in data.items():
            for user, (feats, labels) in users.items():
                total_samples += len(labels)
                # Validate feature consistency
                if len(feats) > 0:
                    expected_features = feats.shape[-1]  # Last dimension is feature count
                    print(f"🔍 Location {loc}, User {user}: {feats.shape} features, {len(labels)} labels")
        
        print(f"📊 Final data summary: {total_locations} locations, {total_users} users, {total_samples} total samples")
        
        # Save with error handling
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Verify the saved cache immediately
        cache_size = os.path.getsize(cache_file) / (1024*1024)  # MB
        print(f"✅ Data cached successfully at: {cache_file} ({cache_size:.1f} MB)")
        
        # Quick validation load
        with open(cache_file, 'rb') as f:
            test_load = pickle.load(f)
        print(f"✅ Cache validation successful - can be loaded properly")
        
    except Exception as e:
        print(f"❌ Error saving cache: {e}")
        # Don't fail the entire process if cache saving fails
        print("⚠️ Continuing without cache - data will be reloaded next time")
    
    return data


def run_lo_location():
    for modality, features in MODES.items():
        print(f"\n🚀 Running Leave-One-Location-Out for modality: {modality}")
        data_by_loc = load_data_by_location(features)
        output_root = os.path.join(data_root, "Results", game, model_name, modality, "leave-one-location-out")
        os.makedirs(output_root, exist_ok=True)

        for test_loc in locations:
            print(f"\n🚨 LOLO: Testing on {test_loc}")

            for train_ratio, test_ratio in ratios:
                subdir = f"cross-user/train{int(train_ratio*100)}_test{int(test_ratio*100)}"
                out_path = os.path.join(output_root, subdir, f"test_on_{test_loc}")
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
                    if np.array_equal(X_train, X_test):
                        print("❌ [LEAKAGE] X_train and X_test are identical!")
                    le = LabelEncoder().fit(y_train + y_test)
                    print("🧾 LabelEncoder classes:", dict(zip(le.classes_, le.transform(le.classes_))))
                    y_train_cat = to_categorical(le.transform(y_train))
                    model = build_transformer(X_train.shape[1:], y_train_cat.shape[1])
                    print("🔍 y_train distribution:", Counter(y_train))
                    print("🔍 y_test  distribution:", Counter(y_test))
                    
                    # Create experiment ID for checkpointing
                    experiment_id = create_experiment_id("LOLO_cross_user", modality, test_loc, (train_ratio, test_ratio))
                    predictions = train_model_with_checkpoints(model, X_train, y_train_cat, X_test, experiment_id)
                    y_pred = le.inverse_transform(np.argmax(predictions, axis=1))
                    save_results(y_test, y_pred.tolist(), list(le.classes_), out_path)

            # Intra-User
            intra_preds, intra_trues = [], []
            for user in data_by_loc.get(test_loc, {}):
                X, y = data_by_loc[test_loc][user]
                if len(X) < 2 or len(np.unique(y)) < 2:
                    print(f"Skipping user {user} due to insufficient data or class imbalance.")
                    continue
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
                le = LabelEncoder().fit(y_tr.tolist() + y_te.tolist())
                print("🧾 LabelEncoder classes:", dict(zip(le.classes_, le.transform(le.classes_))))
                y_tr_cat = to_categorical(le.transform(y_tr))
                model = build_transformer(X_tr.shape[1:], y_tr_cat.shape[1])
                print("🔍 Intra-user y_train:", Counter(y_tr))
                print("🔍 Intra-user y_test:", Counter(y_te))
                
                # Create experiment ID for checkpointing
                experiment_id = create_experiment_id("LOLO_intra_user", modality, f"{test_loc}_{user}")
                predictions = train_model_with_checkpoints(model, X_tr, y_tr_cat, X_te, experiment_id)
                y_pred = le.inverse_transform(np.argmax(predictions, axis=1))
                intra_preds.extend(y_pred)
                intra_trues.extend(y_te)

            if intra_preds:
                intra_path = os.path.join(output_root, "intra-user", f"test_on_{test_loc}")
                save_results(intra_trues, intra_preds, sorted(set(intra_trues)), intra_path)

            # Leave-One-User-Out
            louo_preds, louo_trues = [], []
            all_users = list(data_by_loc.get(test_loc, {}).keys())
            for test_user in all_users:
                train_users = [u for u in all_users if u != test_user]
                X_train, y_train = [], []
                for u in train_users:
                    X, y = data_by_loc[test_loc][u]
                    X_train.append(X)
                    y_train.extend(y)
                X_test, y_test = data_by_loc[test_loc][test_user]
                if X_train and len(X_test) > 0:
                    X_train = np.vstack(X_train)
                    le = LabelEncoder().fit(y_train + y_test.tolist())
                    print("🧾 LabelEncoder classes:", dict(zip(le.classes_, le.transform(le.classes_))))
                    y_train_cat = to_categorical(le.transform(y_train))
                    model = build_transformer(X_train.shape[1:], y_train_cat.shape[1])
                    print("🔍 LOUO y_train:", Counter(y_train))
                    print("🔍 LOUO y_test:", Counter(y_test))
                    
                    # Create experiment ID for checkpointing
                    experiment_id = create_experiment_id("LOLO_LOUO", modality, f"{test_loc}_{test_user}")
                    predictions = train_model_with_checkpoints(model, X_train, y_train_cat, X_test, experiment_id)
                    y_pred = le.inverse_transform(np.argmax(predictions, axis=1))
                    louo_preds.extend(y_pred)
                    louo_trues.extend(y_test)

            if louo_preds:
                louo_path = os.path.join(output_root, "leave-one-user-out", f"test_on_{test_loc}")
                save_results(louo_trues, louo_preds, sorted(set(louo_trues)), louo_path)


def run_train_one_test_others():
    for modality, features in MODES.items():
        print(f"\n🚀 Running Train-One-Test-Others for modality: {modality}")
        data_by_loc = load_data_by_location(features)
        output_root = os.path.join(data_root, "Results", game, model_name, modality, "train-one-test-others")
        os.makedirs(output_root, exist_ok=True)

        for train_loc in locations:
            test_locs = [loc for loc in locations if loc != train_loc]
            print(f"\n🚨 Training on {train_loc}, testing on: {test_locs}")

            for train_ratio, test_ratio in ratios:
                subdir = f"cross-user/train{int(train_ratio*100)}_test{int(test_ratio*100)}"
                out_path = os.path.join(output_root, subdir, f"train_on_{train_loc}")
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
                    le = LabelEncoder().fit(y_train + y_test)
                    print("🧾 LabelEncoder classes:", dict(zip(le.classes_, le.transform(le.classes_))))
                    y_train_cat = to_categorical(le.transform(y_train))
                    model = build_transformer(X_train.shape[1:], y_train_cat.shape[1])
                    print("🔍 y_train distribution:", Counter(y_train))
                    print("🔍 y_test  distribution:", Counter(y_test))
                    
                    # Create experiment ID for checkpointing
                    experiment_id = create_experiment_id("TOTO_cross_user", modality, train_loc, (train_ratio, test_ratio))
                    predictions = train_model_with_checkpoints(model, X_train, y_train_cat, X_test, experiment_id)
                    y_pred = le.inverse_transform(np.argmax(predictions, axis=1))
                    save_results(y_test, y_pred.tolist(), list(le.classes_), out_path)

            # Intra-User Evaluation
            intra_preds, intra_trues = [], []
            for loc in test_locs:
                for user in data_by_loc[loc]:
                    X, y = data_by_loc[loc][user]
                    if len(X) < 2 or len(np.unique(y)) < 2:
                        print(f"Skipping user {user} due to insufficient data or class imbalance.")
                        continue
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
                    X_train, y_train = [], []
                    for u in data_by_loc[train_loc]:
                        X_tmp, y_tmp = data_by_loc[train_loc][u]
                        X_train.append(X_tmp)
                        y_train.extend(y_tmp)
                    if not X_train:
                        continue
                    X_train = np.vstack(X_train)
                    le = LabelEncoder().fit(y_train + y_te.tolist())
                    print("🧾 LabelEncoder classes:", dict(zip(le.classes_, le.transform(le.classes_))))
                    y_train_cat = to_categorical(le.transform(y_train))
                    model = build_transformer(X_train.shape[1:], y_train_cat.shape[1])
                    print("🔍 Intra y_train:", Counter(y_train))
                    print("🔍 Intra y_test:", Counter(y_te))
                    
                    # Create experiment ID for checkpointing
                    experiment_id = create_experiment_id("TOTO_intra_user", modality, f"{train_loc}_{loc}_{user}")
                    predictions = train_model_with_checkpoints(model, X_train, y_train_cat, X_te, experiment_id)
                    y_pred = le.inverse_transform(np.argmax(predictions, axis=1))
                    intra_preds.extend(y_pred)
                    intra_trues.extend(y_te)

            if intra_preds:
                intra_path = os.path.join(output_root, "intra-user", f"train_on_{train_loc}")
                save_results(intra_trues, intra_preds, sorted(set(intra_trues)), intra_path)

            # Leave-One-User-Out
            louo_preds, louo_trues = [], []
            for loc in test_locs:
                for test_user in data_by_loc[loc]:
                    X_test, y_test = data_by_loc[loc][test_user]
                    X_train, y_train = [], []
                    for u in data_by_loc[train_loc]:
                        X_tmp, y_tmp = data_by_loc[train_loc][u]
                        X_train.append(X_tmp)
                        y_train.extend(y_tmp)
                    if not X_train or len(X_test) == 0:
                        continue
                    X_train = np.vstack(X_train)
                    le = LabelEncoder().fit(y_train + y_test.tolist())
                    print("🧾 LabelEncoder classes:", dict(zip(le.classes_, le.transform(le.classes_))))
                    y_train_cat = to_categorical(le.transform(y_train))
                    model = build_transformer(X_train.shape[1:], y_train_cat.shape[1])
                    print("🔍 LOUO y_train:", Counter(y_train))
                    print("🔍 LOUO y_test:", Counter(y_test))
                    
                    # Create experiment ID for checkpointing
                    experiment_id = create_experiment_id("TOTO_LOUO", modality, f"{train_loc}_{loc}_{test_user}")
                    predictions = train_model_with_checkpoints(model, X_train, y_train_cat, X_test, experiment_id)
                    y_pred = le.inverse_transform(np.argmax(predictions, axis=1))
                    louo_preds.extend(y_pred)
                    louo_trues.extend(y_test)

            if louo_preds:
                louo_path = os.path.join(output_root, "leave-one-user-out", f"train_on_{train_loc}")
                save_results(louo_trues, louo_preds, sorted(set(louo_trues)), louo_path)


# Call both experiment runners
# run_lo_location()
run_train_one_test_others()
