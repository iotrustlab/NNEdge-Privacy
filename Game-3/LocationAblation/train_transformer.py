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
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, 
    GlobalAveragePooling1D, BatchNormalization, Activation, Embedding,
    Conv1D, MaxPooling1D, Concatenate, Reshape
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from scipy.special import rel_entr
import warnings
from sklearn.exceptions import UndefinedMetricWarning

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

# Disable XLA compilation to avoid slow compile issues
tf.config.optimizer.set_jit(False)
print("🔧 XLA JIT compilation disabled to prevent slow compilation")

# Disable experimental MLIR bridge which can cause compilation issues
try:
    tf.config.experimental.enable_mlir_bridge(False)
    print("🔧 MLIR bridge disabled")
except TypeError:
    # Newer TensorFlow versions don't support this parameter
    print("🔧 MLIR bridge configuration not available in this TensorFlow version")

# Set mixed precision policy for better performance and memory usage
try:
    mixed_precision = tf.keras.mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("🔧 Mixed precision (float16) enabled for better performance")
except Exception as e:
    print(f"⚠️ Mixed precision setup failed: {e}")

# Verify GPU availability
print(f"🔍 TensorFlow GPU available: {tf.config.list_physical_devices('GPU')}")
print(f"🔍 TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -------- CONFIG --------
data_root = os.environ.get("NNEDGE_PRIVACY_STM_ROOT", os.path.join(REPO_ROOT, "STMDATASET", "Data"))
game = "Game-3"
model_name = "Transformer"
locations = ["right-ankle", "right-pocket",  "right-wrist" , "left-ankle", "left-wrist"]
window_size = 50  # Reduced from 100 for better memory efficiency and performance
ratios = [(0.8, 0.2), (0.7, 0.3), (0.5, 0.5)]
EPOCHS = 20
BATCH_SIZE = 8  # Further reduced for memory efficiency and faster compilation
LEARNING_RATE = 0.001
MIN_DELTA = 0.001
PATIENCE = 8
VALID_ACTIVITIES = {"jogging", "laying", "sitting", "standing", "upstairs", "downstairs", "walking"}

# -------- CHECKPOINT SYSTEM --------
CHECKPOINT_DIR = os.environ.get("NNEDGE_PRIVACY_CHECKPOINT_ROOT", os.path.join(REPO_ROOT, "checkpoints", "Game-3_Transformer_LocationAblation_Checkpoints"))

def get_or_create_script_id():
    if os.path.exists(CHECKPOINT_DIR):
        # Look for ALL existing experiment directories 
        existing_dirs = [d for d in os.listdir(CHECKPOINT_DIR) 
                        if d.startswith("Game-3_Transformer_LocationAblation_") and len(d.split('_')) >= 4]
        
        # Find script IDs and their experiment counts
        script_id_stats = {}
        for dir_name in existing_dirs:
            if dir_name == "data_loading":  # Skip data loading directory
                continue
                
            progress_file = os.path.join(CHECKPOINT_DIR, dir_name, "training_progress.json")
            script_id = "_".join(dir_name.split('_')[:5])  # Extract script ID (updated for transformer)
            
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
    new_script_id = f"Game-3_Transformer_LocationAblation_{uuid.uuid4().hex[:8]}"
    print(f"🆕 Created new Script ID: {new_script_id}")
    return new_script_id

SCRIPT_ID = get_or_create_script_id()

def cleanup_empty_checkpoints():
    """Remove checkpoint directories with no training progress."""
    if not os.path.exists(CHECKPOINT_DIR):
        return
    
    dirs_to_remove = []
    for dir_name in os.listdir(CHECKPOINT_DIR):
        if not dir_name.startswith("Game-3_Transformer_LocationAblation_"):
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

def cleanup_incompatible_checkpoints():
    """Remove checkpoint directories with incompatible model architectures."""
    if not os.path.exists(CHECKPOINT_DIR):
        return
    
    dirs_to_remove = []
    for dir_name in os.listdir(CHECKPOINT_DIR):
        if not dir_name.startswith("Game-3_Transformer_LocationAblation_"):
            continue
        
        dir_path = os.path.join(CHECKPOINT_DIR, dir_name)
        if not os.path.isdir(dir_path):
            continue
        
        # Check if there's a model architecture mismatch
        arch_file = os.path.join(dir_path, "model_architecture.json")
        if os.path.exists(arch_file):
            try:
                with open(arch_file, 'r') as f:
                    saved_arch = json.load(f)
                
                # Check if architecture is from old version (different input features)
                if saved_arch.get('input_shape', [0, 0])[1] != 16:  # Current architecture expects 16 features
                    print(f"🔄 Found incompatible checkpoint with {saved_arch.get('input_shape', [0, 0])[1]} features: {dir_name}")
                    dirs_to_remove.append(dir_path)
                    
            except Exception as e:
                print(f"⚠️ Error reading architecture {arch_file}: {e}")
    
    # Remove incompatible directories
    for dir_path in dirs_to_remove:
        try:
            import shutil
            shutil.rmtree(dir_path)
            print(f"🗑️ Removed incompatible checkpoint: {os.path.basename(dir_path)}")
        except Exception as e:
            print(f"⚠️ Failed to remove {dir_path}: {e}")

# Clean up empty and incompatible checkpoints on startup
cleanup_empty_checkpoints()
cleanup_incompatible_checkpoints()

print(f"🔒 Script ID: {SCRIPT_ID}")
print(f"📁 Checkpoint Directory: {CHECKPOINT_DIR}")

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

def save_data_cache(experiment_id, data, cache_name):
    """Save preprocessed data to avoid reloading."""
    exp_dir = os.path.join(CHECKPOINT_DIR, experiment_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    cache_file = os.path.join(exp_dir, f"{cache_name}_cache.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"🗂️ Data cached: {cache_file}")

def load_data_cache(experiment_id, cache_name):
    """Load preprocessed data from cache."""
    exp_dir = os.path.join(CHECKPOINT_DIR, experiment_id)
    cache_file = os.path.join(exp_dir, f"{cache_name}_cache.pkl")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"🗂️ Data loaded from cache: {cache_file}")
            return data
        except Exception as e:
            print(f"⚠️ Error loading cache: {e}")
    
    return None

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

def create_experiment_id(stage, location=None, ratio=None):
    """Create unique experiment identifier."""
    parts = [SCRIPT_ID, stage]
    if location:
        parts.append(str(location))  # Convert to string to handle various types
    if ratio:
        if isinstance(ratio, (tuple, list)) and len(ratio) == 2:
            parts.append(f"ratio_{ratio[0]:.1f}_{ratio[1]:.1f}")
        else:
            parts.append(str(ratio))  # Handle non-tuple ratios
    return "_".join(parts)



ALL_FEATURES = ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]',
                'gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]',
                'dec_tree_out_1']

# -------- HELPERS --------
def extract_sequences(df):
    """
    Enhanced sequence extraction with feature engineering for HAR with binary shake detection.
    Combines accelerometer, gyroscope, and binary shake data for better activity recognition.
    Optimized for Transformer architecture with richer temporal features.
    """
    if not all(f in df.columns for f in ALL_FEATURES):
        return np.empty((0, window_size, 16))  # 16 features total after enhancement
    
    sequences = []
    
    for i in range(0, len(df) - window_size + 1, window_size):
        window_data = df[ALL_FEATURES].iloc[i:i + window_size]
        
        if window_data.shape[0] == window_size:
            # Original sensor data (6 features)
            acc_data = window_data[['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]']].values
            gyro_data = window_data[['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']].values
            binary_data = window_data['dec_tree_out_1'].values
            
            # Feature engineering optimized for Transformer attention
            # 1. Magnitude features (global movement intensity)
            acc_magnitude = np.sqrt(np.sum(acc_data**2, axis=1)).reshape(-1, 1)
            gyro_magnitude = np.sqrt(np.sum(gyro_data**2, axis=1)).reshape(-1, 1)
            
            # 2. Statistical features for temporal context
            acc_mean = np.mean(acc_data, axis=1).reshape(-1, 1)
            gyro_mean = np.mean(gyro_data, axis=1).reshape(-1, 1)
            
            # 3. Temporal derivative features (rate of change)
            acc_diff = np.diff(acc_data, axis=0, prepend=acc_data[0:1])
            gyro_diff = np.diff(gyro_data, axis=0, prepend=gyro_data[0:1])
            acc_derivative_mag = np.sqrt(np.sum(acc_diff**2, axis=1)).reshape(-1, 1)
            gyro_derivative_mag = np.sqrt(np.sum(gyro_diff**2, axis=1)).reshape(-1, 1)
            
            # 4. Binary shake pattern features
            shake_transitions = np.diff(binary_data.astype(int))
            shake_transitions = np.pad(shake_transitions, (1, 0), 'constant').reshape(-1, 1)
            
            # 5. Movement intensity features
            movement_intensity = np.var(acc_data, axis=1).reshape(-1, 1)
            rotation_intensity = np.var(gyro_data, axis=1).reshape(-1, 1)
            
            # 6. Cross-correlation between accelerometer and gyroscope
            cross_corr = np.sum(acc_data * gyro_data, axis=1).reshape(-1, 1)
            
            # Combine all features: 6 (original) + 2 (magnitude) + 2 (mean) + 2 (derivatives) + 1 (transitions) + 2 (intensity) + 1 (cross-corr) + 1 (binary) = 17 features
            # But let's use 16 for power-of-2 efficiency
            enhanced_features = np.column_stack([
                acc_data,                    # 3 features: acc_x, acc_y, acc_z
                gyro_data,                   # 3 features: gyro_x, gyro_y, gyro_z
                acc_magnitude,               # 1 feature: acceleration magnitude
                gyro_magnitude,              # 1 feature: gyroscope magnitude
                acc_mean,                    # 1 feature: acceleration mean
                gyro_mean,                   # 1 feature: gyroscope mean
                acc_derivative_mag,          # 1 feature: acceleration change rate
                gyro_derivative_mag,         # 1 feature: gyroscope change rate
                shake_transitions,           # 1 feature: shake state transitions
                movement_intensity,          # 1 feature: acceleration variance
                rotation_intensity,          # 1 feature: gyroscope variance
                cross_corr,                  # 1 feature: acc-gyro correlation
                binary_data.reshape(-1, 1)  # 1 feature: binary shake detection
            ])
            
            sequences.append(enhanced_features)
    
    return np.array(sequences)

def get_user_dirs(base):
    user_dirs = [d for d in os.listdir(base) if re.match(r"User \d+$", d)]
    return sorted(user_dirs, key=lambda x: int(re.findall(r"\d+", x)[0]))

# ---------- MODEL ----------
def build_transformer(input_shape, num_classes, d_model=32, num_heads=4, ff_dim=64, 
                     num_transformer_blocks=1, dropout_rate=0.1):
    """
    Simplified Transformer model for multivariate time-series HAR classification.
    Reduced complexity to avoid XLA compilation issues while maintaining effectiveness.
    
    Args:
        input_shape (tuple): (seq_len, num_features)
        num_classes (int): Number of target classes
        d_model (int): Dimension to project input and use throughout transformer (reduced)
        num_heads (int): Number of attention heads (reduced)
        ff_dim (int): Dimension of the feedforward layer (reduced)
        num_transformer_blocks (int): Number of transformer blocks (reduced)
        dropout_rate (float): Dropout rate

    Returns:
        tf.keras.Model: Compiled transformer model
    """
    seq_len, num_features = input_shape
    
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        inputs = Input(shape=input_shape, name='sensor_input')  # (seq_len, num_features)

        # Simplified feature preprocessing
        x = Dense(d_model, activation='relu', name='feature_projection')(inputs)
        x = LayerNormalization()(x)
        x = Dropout(dropout_rate)(x)

        # Simplified transformer blocks
        for i in range(num_transformer_blocks):
            # Multi-head self-attention
            attn_output = MultiHeadAttention(
                num_heads=num_heads, 
                key_dim=d_model//num_heads,
                name=f'attention_block_{i}'
            )(x, x)
            attn_output = Dropout(dropout_rate)(attn_output)
            x = Add(name=f'add_attention_{i}')([x, attn_output])
            x = LayerNormalization(name=f'norm_attention_{i}')(x)

            # Simplified feed-forward block
            ff_output = Dense(ff_dim, activation='relu', name=f'ff_dense1_{i}')(x)
            ff_output = Dropout(dropout_rate)(ff_output)
            ff_output = Dense(d_model, name=f'ff_dense2_{i}')(ff_output)
            ff_output = Dropout(dropout_rate)(ff_output)
            x = Add(name=f'add_ff_{i}')([x, ff_output])
            x = LayerNormalization(name=f'norm_ff_{i}')(x)

        # Simplified global representation extraction
        global_avg = GlobalAveragePooling1D(name='global_avg_pool')(x)
        global_avg = Dropout(dropout_rate)(global_avg)
        
        # Simplified classification head
        dense1 = Dense(ff_dim, activation='relu', name='dense1')(global_avg)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(dropout_rate)(dense1)
        
        # Output layer with float32 to avoid mixed precision issues
        outputs = Dense(num_classes, activation='softmax', name='classification_output', dtype='float32')(dense1)

        model = Model(inputs, outputs, name='Simplified_Transformer_HAR')
        
        # Enhanced optimizer with learning rate scheduling
        optimizer = Adam(
            learning_rate=LEARNING_RATE,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer, 
            loss='categorical_crossentropy', 
            metrics=['accuracy'],
            run_eagerly=False  # Keep graph compilation but simplified architecture should work
        )
        
        print(f"🏗️ Simplified Transformer model built on device: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
        print(f"📊 Model parameters: {model.count_params():,}")
        print(f"🎯 Input shape: {input_shape}, Output classes: {num_classes}")
        print(f"🔧 Architecture: {num_transformer_blocks} transformer blocks, {num_heads} attention heads, {d_model} model dim")
    
    return model


def train_model_with_checkpoints(model, X_train, y_train_cat, X_test, experiment_id, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Enhanced training function with comprehensive checkpointing and data augmentation."""
    
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
            try:
                model.load_weights(model_file)
                print(f"✅ Loaded completed model from epoch {progress['current_epoch']}")
                return model.predict(X_test)
            except ValueError as e:
                print(f"⚠️ Model architecture mismatch: {e}")
                print("🔄 Model architecture has changed, retraining from scratch...")
                # Clear the checkpoint and start fresh
                progress = None
        else:
            print(f"⚠️ Warning: Completed experiment {experiment_id} missing model file, retraining...")
    
    if progress and progress['current_epoch'] > 0:
        print(f"🔄 Resuming from epoch {progress['current_epoch']}")
        start_epoch = progress['current_epoch']
        
        # Load saved model if available
        exp_dir = os.path.join(CHECKPOINT_DIR, experiment_id)
        model_file = os.path.join(exp_dir, f"model_epoch_{start_epoch}.weights.h5")
        if os.path.exists(model_file):
            try:
                model.load_weights(model_file)
                print(f"🔄 Model weights loaded from epoch {start_epoch}")
            except ValueError as e:
                print(f"⚠️ Cannot load weights due to architecture mismatch: {e}")
                print("🔄 Starting training from scratch due to model architecture change...")
                start_epoch = 0
                progress = None
    
    if progress is None:
        # Initialize new progress tracking
        progress = {
            'experiment_id': experiment_id,
            'script_id': SCRIPT_ID,
            'current_epoch': 0,
            'total_epochs': epochs,
            'status': 'starting',
            'start_time': datetime.now().isoformat(),
            'model_architecture': {
                'input_shape': X_train.shape[1:],
                'num_classes': y_train_cat.shape[1],
                'model_params': model.count_params()
            },
            'results': {'epochs_completed': [], 'best_accuracy': 0, 'best_loss': float('inf')}
        }
        save_checkpoint(experiment_id, progress)
        start_epoch = 0
    
    # Setup callbacks with checkpointing
    exp_dir = os.path.join(CHECKPOINT_DIR, experiment_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save model architecture info for validation
    arch_info_file = os.path.join(exp_dir, "model_architecture.json")
    arch_info = {
        'input_shape': X_train.shape[1:],
        'num_classes': y_train_cat.shape[1],
        'model_params': model.count_params(),
        'timestamp': datetime.now().isoformat()
    }
    with open(arch_info_file, 'w') as f:
        json.dump(arch_info, f, indent=2)
    
    callbacks = [
        EpochProgressCallback(experiment_id, epochs),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(exp_dir, "model_epoch_{epoch}.weights.h5"),
            monitor='val_loss',
            save_best_only=False,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Simplified data augmentation to reduce complexity
    def augment_har_sequences(X, y, shift_range=1, noise_std=0.005):
        X_aug, y_aug = [], []
        for i in range(len(X)):
            X_aug.append(X[i])
            y_aug.append(y[i])
            
            # Minimal augmentation to reduce training time
            if np.random.random() < 0.2:  # Reduced to 20% chance for augmentation
                shift = np.random.randint(1, shift_range + 1)
                if shift < X[i].shape[0]:
                    X_shifted = np.roll(X[i], shift, axis=0)
                    X_aug.append(X_shifted)
                    y_aug.append(y[i])
        
        return np.array(X_aug), np.array(y_aug)

    # Minimal augmentation for faster training
    X_train_aug, y_train_aug = augment_har_sequences(X_train, np.argmax(y_train_cat, axis=1))
    y_train_cat_aug = to_categorical(y_train_aug, num_classes=y_train_cat.shape[1])

    device_name = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    
    with tf.device(device_name):
        print(f"🔥 Training on device: {'GPU' if 'GPU' in device_name else 'CPU'} with augmented data: {X_train_aug.shape[0]} samples")
        
        # Train only remaining epochs
        if start_epoch < epochs:
            remaining_epochs = epochs - start_epoch
            print(f"🚀 Training {remaining_epochs} remaining epochs (from {start_epoch} to {epochs})")
            
            try:
                # First attempt with graph compilation
                history = model.fit(
                    X_train_aug, y_train_cat_aug, 
                    epochs=remaining_epochs,
                    batch_size=batch_size, 
                    verbose=1, 
                    callbacks=callbacks, 
                    validation_split=0.15,  # Slightly larger validation split for transformer
                    shuffle=True,
                    initial_epoch=start_epoch
                )
            except Exception as e:
                print(f"⚠️ Graph compilation failed with error: {e}")
                print("🔄 Retrying with eager execution...")
                
                # Recompile model with eager execution
                model.compile(
                    optimizer=model.optimizer,
                    loss=model.loss,
                    metrics=model.metrics,
                    run_eagerly=True
                )
                
                history = model.fit(
                    X_train_aug, y_train_cat_aug, 
                    epochs=remaining_epochs,
                    batch_size=batch_size, 
                    verbose=1, 
                    callbacks=callbacks, 
                    validation_split=0.15,
                    shuffle=True,
                    initial_epoch=start_epoch
                )
        else:
            print(f"✅ Training already completed ({start_epoch}/{epochs} epochs)")
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Explicit memory cleanup
        del X_train_aug, y_train_cat_aug
        import gc
        gc.collect()
        
        # Clear GPU memory if available
        if tf.config.list_physical_devices('GPU'):
            tf.keras.backend.clear_session()
        
        # Mark as completed
        final_progress = load_checkpoint(experiment_id) or progress
        final_progress['status'] = 'completed'
        final_progress['completion_time'] = datetime.now().isoformat()
        save_checkpoint(experiment_id, final_progress)
        
        print(f"🎉 Experiment {experiment_id} completed successfully!")
    
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


def load_data_by_location():
    """Load data with caching support for faster restarts."""
    cache_name = "data_by_location"
    
    # Try to load from cache first
    cached_data = load_data_cache("data_loading", cache_name)
    if cached_data is not None:
        print("🚀 Using cached data - loading complete!")
        return cached_data
    
    print("📥 Cache not found, loading data from files...")
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
                        sequences = extract_sequences(df)
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
    
    # Cache the loaded data for future runs
    save_data_cache("data_loading", data, cache_name)
    
    return data



def run_lo_location():
    data_by_loc = load_data_by_location()
    output_root = os.path.join(data_root, "Results", game, model_name, "leave-one-location-out")
    os.makedirs(output_root, exist_ok=True)

    for test_loc in locations:
        print(f"\n🚨 LOLO: Testing on {test_loc}")

        for train_ratio, test_ratio in ratios:
            subdir = f"cross-user/train{int(train_ratio*100)}_test{int(test_ratio*100)}"
            out_path = os.path.join(output_root, subdir, f"test_on_{test_loc}")
            train_users, test_users = [], []

            for loc in locations:
                users = list(data_by_loc[loc].keys())
                np.random.shuffle(users)  # shuffle to avoid ordering bias
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
                experiment_id = create_experiment_id("LOLO_cross_user", test_loc, (train_ratio, test_ratio))
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
            experiment_id = create_experiment_id("LOLO_intra_user", f"{test_loc}_{user}")
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
                experiment_id = create_experiment_id("LOLO_LOUO", f"{test_loc}_{test_user}")
                predictions = train_model_with_checkpoints(model, X_train, y_train_cat, X_test, experiment_id)
                y_pred = le.inverse_transform(np.argmax(predictions, axis=1))
                louo_preds.extend(y_pred)
                louo_trues.extend(y_test)

        if louo_preds:
            louo_path = os.path.join(output_root, "leave-one-user-out", f"test_on_{test_loc}")
            save_results(louo_trues, louo_preds, sorted(set(louo_trues)), louo_path)



def run_train_one_test_others():
    data_by_loc = load_data_by_location()
    output_root = os.path.join(data_root, "Results", game, model_name, "train-one-test-others")
    os.makedirs(output_root, exist_ok=True)

    for train_loc in locations:
        test_locs = [loc for loc in locations if loc != train_loc]
        print(f"\n🚨 Training on {train_loc}, testing on: {test_locs}")

        # for train_ratio, test_ratio in ratios:
        #     subdir = f"cross-user/train{int(train_ratio*100)}_test{int(test_ratio*100)}"
        #     out_path = os.path.join(output_root, subdir, f"train_on_{train_loc}")
        #     users = list(data_by_loc[train_loc].keys())
        #     np.random.shuffle(users)
        #     train_users = users[:int(train_ratio * len(users))]
        #     test_users = [(loc, u) for loc in test_locs for u in data_by_loc[loc].keys()]

        #     X_train, y_train, X_test, y_test = [], [], [], []
        #     for u in train_users:
        #         X, y = data_by_loc[train_loc][u]
        #         X_train.append(X)
        #         y_train.extend(y)
        #     for loc, u in test_users:
        #         X, y = data_by_loc[loc][u]
        #         X_test.append(X)
        #         y_test.extend(y)

        #     if X_train and X_test:
        #         print(f"📊 Training samples: {len(y_train)}, Testing samples: {len(y_test)}")
        #         X_train = np.vstack(X_train)
        #         X_test = np.vstack(X_test)
        #         if np.array_equal(X_train, X_test):
        #             print("❌ [LEAKAGE] X_train and X_test are identical!")
        #         le = LabelEncoder().fit(y_train + y_test)
        #         print("🧾 LabelEncoder classes:", dict(zip(le.classes_, le.transform(le.classes_))))
        #         y_train_cat = to_categorical(le.transform(y_train))
        #         model = build_transformer(X_train.shape[1:], y_train_cat.shape[1])
        #         print("🔍 y_train distribution:", Counter(y_train))
        #         print("🔍 y_test  distribution:", Counter(y_test))
                
        #         # Create experiment ID for checkpointing
        #         experiment_id = create_experiment_id("TOTO_cross_user", train_loc, (train_ratio, test_ratio))
        #         predictions = train_model_with_checkpoints(model, X_train, y_train_cat, X_test, experiment_id)
        #         y_pred = le.inverse_transform(np.argmax(predictions, axis=1))
        #         save_results(y_test, y_pred.tolist(), list(le.classes_), out_path)

        # Intra-User
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
                experiment_id = create_experiment_id("TOTO_intra_user", f"{train_loc}_{loc}_{user}")
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
                experiment_id = create_experiment_id("TOTO_LOUO", f"{train_loc}_{loc}_{test_user}")
                predictions = train_model_with_checkpoints(model, X_train, y_train_cat, X_test, experiment_id)
                y_pred = le.inverse_transform(np.argmax(predictions, axis=1))
                louo_preds.extend(y_pred)
                louo_trues.extend(y_test)

        if louo_preds:
            louo_path = os.path.join(output_root, "leave-one-user-out", f"train_on_{train_loc}")
            save_results(louo_trues, louo_preds, sorted(set(louo_trues)), louo_path)


# Call both experiment runners
print("🚀 Starting Game-3 Transformer Location Ablation Experiments")
print("=" * 70)
# run_lo_location()
run_train_one_test_others()
print("🎉 All experiments completed!")
