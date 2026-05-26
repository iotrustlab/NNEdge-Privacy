import os
import re
import glob
import json
import pickle
import time
import uuid
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mutual_info_score, precision_score, recall_score, f1_score
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, 
    GlobalAveragePooling1D, Add, Conv1D, BatchNormalization, 
    Concatenate, GlobalMaxPooling1D, Embedding, Reshape
)
from tensorflow.keras.utils import to_categorical
from scipy.special import rel_entr
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from collections import Counter

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
game = "Game-1"
model_name = "Transformer"
locations = ["left-ankle", "left-wrist", "right-ankle", "right-pocket", "right-wrist"]
window_size = 100
ratios = [(0.8, 0.2), (0.7, 0.3), (0.5, 0.5)]
EPOCHS = 20
BATCH_SIZE = 32
VALID_ACTIVITIES = {"jogging", "laying", "sitting", "standing", "upstairs", "downstairs", "walking"}

# -------- CHECKPOINT SYSTEM --------
# Generate consistent script identifier or reuse existing one
CHECKPOINT_DIR = os.environ.get("NNEDGE_PRIVACY_CHECKPOINT_ROOT", os.path.join(REPO_ROOT, "checkpoints", "Game-1_Transformer_LocationAblation_Checkpoints"))

# Try to find existing script ID from checkpoint directory, otherwise create new one
def get_or_create_script_id():
    if os.path.exists(CHECKPOINT_DIR):
        # Look for existing experiment directories with any progress
        existing_dirs = [d for d in os.listdir(CHECKPOINT_DIR) 
                        if d.startswith("Game-1_Transformer_LocationAblation_") and len(d.split('_')) >= 4]
        
        # Find script IDs that have actual experiments (completed or in progress)
        script_ids_with_progress = set()
        for dir_name in existing_dirs:
            progress_file = os.path.join(CHECKPOINT_DIR, dir_name, "training_progress.json")
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r') as f:
                        progress = json.load(f)
                    # Extract script ID from directory name (first 4 parts)
                    script_id = "_".join(dir_name.split('_')[:4])
                    # Count any experiment that has been started (has epochs_completed data)
                    if (progress.get('current_epoch', 0) > 0 or 
                        len(progress.get('results', {}).get('epochs_completed', [])) > 0 or
                        progress.get('status') == 'completed'):
                        script_ids_with_progress.add(script_id)
                        print(f"� Found existing experiment: {dir_name} (Epoch {progress.get('current_epoch', 0)}/{progress.get('total_epochs', 20)}, Status: {progress.get('status', 'unknown')})")
                except Exception as e:
                    print(f"⚠️ Error reading checkpoint {dir_name}: {e}")
                    continue
        
        # If we found script IDs with progress, use the most recent one
        if script_ids_with_progress:
            # Sort by creation time (newest first) - assuming timestamp in script ID
            sorted_script_ids = sorted(script_ids_with_progress)
            script_id = sorted_script_ids[-1]  # Use the most recent one
            print(f"🔄 Reusing existing Script ID with experiments: {script_id}")
            return script_id
    
    # Create new script ID if none found with progress
    new_script_id = f"Game-1_Transformer_LocationAblation_{uuid.uuid4().hex[:8]}"
    print(f"🆕 Created new Script ID: {new_script_id}")
    return new_script_id

SCRIPT_ID = get_or_create_script_id()

def cleanup_empty_checkpoints():
    """Remove checkpoint directories with no training progress."""
    if not os.path.exists(CHECKPOINT_DIR):
        return
    
    dirs_to_remove = []
    for dir_name in os.listdir(CHECKPOINT_DIR):
        if not dir_name.startswith("Game-1_Transformer_LocationAblation_"):
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

def save_checkpoint(experiment_id, progress_data):
    """Save training progress to isolated checkpoint."""
    exp_dir = os.path.join(CHECKPOINT_DIR, experiment_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    checkpoint_file = os.path.join(exp_dir, "training_progress.json")
    progress_data['last_updated'] = datetime.now().isoformat()
    progress_data['script_id'] = SCRIPT_ID
    
    # Ensure all data is JSON-serializable
    def make_serializable(obj):
        """Recursively convert TensorFlow tensors and numpy arrays to Python types."""
        if hasattr(obj, 'numpy'):  # TensorFlow tensor
            numpy_val = obj.numpy()
            if numpy_val.ndim == 0:  # Scalar tensor
                return float(numpy_val)
            else:  # Array tensor - take mean
                return float(numpy_val.mean()) if numpy_val.size > 0 else 0.0
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            if obj.ndim == 0:  # numpy scalar
                return obj.item()
            else:  # numpy array - convert to list or take mean for large arrays
                if obj.size <= 10:  # Small arrays can be converted to list
                    return obj.tolist()
                else:  # Large arrays - just take mean
                    return float(obj.mean())
        elif isinstance(obj, dict):
            return {key: make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            try:
                return float(obj)
            except (TypeError, ValueError):
                return str(obj)
    
    serializable_data = make_serializable(progress_data)
    
    with open(checkpoint_file, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    print(f"💾 Checkpoint saved: {checkpoint_file}")

def load_checkpoint(experiment_id):
    """Load training progress from checkpoint."""
    exp_dir = os.path.join(CHECKPOINT_DIR, experiment_id)
    checkpoint_file = os.path.join(exp_dir, "training_progress.json")
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                progress = json.load(f)
            
            # Validate checkpoint data
            current_epoch = progress.get('current_epoch', 0)
            total_epochs = progress.get('total_epochs', EPOCHS)
            
            if current_epoch > 0:
                print(f"📖 Checkpoint loaded: {checkpoint_file}")
                print(f"   📊 Progress: {current_epoch}/{total_epochs} epochs completed")
                return progress
            else:
                print(f"⚠️ Checkpoint found but no progress made: {checkpoint_file}")
                return None
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"⚠️ Error loading checkpoint (corrupted file): {e}")
            print(f"🗑️ Removing corrupted checkpoint: {checkpoint_file}")
            try:
                os.remove(checkpoint_file)
            except OSError:
                pass
            return None
        except Exception as e:
            print(f"⚠️ Unexpected error loading checkpoint: {e}")
            return None
    else:
        print(f"📝 No checkpoint found for: {experiment_id}")
    
    return None

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
            # Convert all TensorFlow tensors to Python native types
            serializable_logs = {}
            for key, value in logs.items():
                if hasattr(value, 'numpy'):  # TensorFlow tensor
                    numpy_val = value.numpy()
                    if numpy_val.ndim == 0:  # Scalar tensor
                        serializable_logs[key] = float(numpy_val)
                    else:  # Array tensor - take mean or first element
                        serializable_logs[key] = float(numpy_val.mean()) if numpy_val.size > 0 else 0.0
                elif hasattr(value, 'item'):  # numpy scalar
                    serializable_logs[key] = value.item()
                elif isinstance(value, (int, float, str, bool)):
                    serializable_logs[key] = value
                else:
                    try:
                        serializable_logs[key] = float(value)
                    except (TypeError, ValueError):
                        serializable_logs[key] = str(value)
            
            epoch_data.update(serializable_logs)
            
            # Update best metrics with converted values
            if 'accuracy' in serializable_logs:
                progress['results']['best_accuracy'] = max(
                    progress['results']['best_accuracy'], 
                    serializable_logs['accuracy']
                )
            if 'loss' in serializable_logs:
                progress['results']['best_loss'] = min(
                    progress['results']['best_loss'], 
                    serializable_logs['loss']
                )
        
        progress['results']['epochs_completed'].append(epoch_data)
        
        # Save checkpoint
        save_checkpoint(self.experiment_id, progress)
        
        print(f"✅ Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        if logs:
            print(f"   📊 Metrics: {logs}")

def create_experiment_id(stage, location=None, ratio=None, user=None):
    """Create unique experiment identifier."""
    parts = [SCRIPT_ID, stage]
    if location:
        parts.append(location)
    if ratio and isinstance(ratio, (tuple, list)) and len(ratio) == 2:
        parts.append(f"ratio_{ratio[0]:.1f}_{ratio[1]:.1f}")
    elif user:
        # Clean user string for safe filename usage
        clean_user = str(user).replace(" ", "_")
        parts.append(clean_user)
    elif ratio:  # If ratio is actually a user string passed as ratio parameter
        clean_user = str(ratio).replace(" ", "_")
        parts.append(clean_user)
    return "_".join(parts)

# -------- HELPERS --------

def extract_sequences(df):
    """
    Enhanced sequence extraction with feature engineering for binary HAR data.
    Extracts temporal patterns and statistical features from binary shake/stationary signals.
    """
    sequences = []
    
    for i in range(0, len(df) - window_size + 1, window_size):
        window = df[['dec_tree_out_1']].iloc[i:i + window_size].values
        if window.shape[0] == window_size:
            # Original binary sequence
            binary_seq = window.flatten()
            
            # Extract temporal features from binary data
            # 1. Transition patterns (state changes)
            transitions = np.diff(binary_seq.astype(int))  # -1, 0, 1 for transitions
            
            # 2. Run-length encoding features
            shake_runs = []  # lengths of consecutive shake periods
            still_runs = []  # lengths of consecutive still periods
            current_run = 1
            for j in range(1, len(binary_seq)):
                if binary_seq[j] == binary_seq[j-1]:
                    current_run += 1
                else:
                    if binary_seq[j-1] == 1:
                        shake_runs.append(current_run)
                    else:
                        still_runs.append(current_run)
                    current_run = 1
            
            # Handle last run
            if binary_seq[-1] == 1:
                shake_runs.append(current_run)
            else:
                still_runs.append(current_run)
            
            # 3. Statistical features
            shake_ratio = np.mean(binary_seq)  # proportion of shake vs still
            num_transitions = np.sum(np.abs(transitions))  # total state changes
            avg_shake_length = np.mean(shake_runs) if shake_runs else 0
            avg_still_length = np.mean(still_runs) if still_runs else 0
            max_shake_length = max(shake_runs) if shake_runs else 0
            max_still_length = max(still_runs) if still_runs else 0
            
            # 4. Pattern features
            # Sliding window variance (movement intensity)
            window_size_var = min(10, len(binary_seq))
            variances = []
            for k in range(len(binary_seq) - window_size_var + 1):
                sub_window = binary_seq[k:k + window_size_var]
                variances.append(np.var(sub_window))
            avg_variance = np.mean(variances) if variances else 0
            
            # 5. Frequency domain features
            # Simple frequency analysis of transitions
            transition_density = num_transitions / window_size if window_size > 0 else 0
            
            # Create enhanced feature vector for each timestep
            enhanced_features = np.column_stack([
                binary_seq.reshape(-1, 1),  # original binary signal
                np.pad(transitions, (1, 0), 'constant').reshape(-1, 1),  # transition signal
                np.full((window_size, 1), shake_ratio),  # shake ratio feature
                np.full((window_size, 1), transition_density),  # transition density
                np.full((window_size, 1), avg_shake_length),  # avg shake length
                np.full((window_size, 1), avg_still_length),  # avg still length
                np.full((window_size, 1), max_shake_length),  # max shake length  
                np.full((window_size, 1), max_still_length),  # max still length
                np.full((window_size, 1), avg_variance),  # movement variance
            ])
            
            sequences.append(enhanced_features)
    
    return np.array(sequences)

def segment_sequences(df, seq_len=100):
    """Legacy function for backward compatibility - delegates to extract_sequences"""
    return extract_sequences(df)

def get_user_dirs(base):
    user_dirs = [d for d in os.listdir(base) if re.match(r"User \d+$", d)]
    return sorted(user_dirs, key=lambda x: int(re.findall(r"\d+", x)[0]))

# ---------- ENHANCED TRANSFORMER MODEL ----------
def positional_encoding(seq_len, d_model):
    """Create positional encoding for transformer"""
    pos_enc = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return tf.constant(pos_enc, dtype=tf.float32)

def build_transformer(input_shape, num_classes, num_heads=8, ff_dim=256, num_layers=4, dropout_rate=0.1):
    """Enhanced Transformer model for Human Activity Recognition with improved architecture"""
    
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        inputs = Input(shape=input_shape, name='sequence_input')
        
        # Feature expansion layer for richer representation
        x = Dense(ff_dim, activation='relu', name='feature_expansion')(inputs)
        x = Dropout(dropout_rate)(x)
        
        # Add positional encoding
        seq_len, d_model = input_shape[0], ff_dim
        pos_enc = positional_encoding(seq_len, d_model)
        x = x + pos_enc
        
        # Multi-layer transformer encoder
        for layer_idx in range(num_layers):
            # Multi-head self-attention
            x_norm = LayerNormalization(epsilon=1e-6, name=f'layer_norm_1_{layer_idx}')(x)
            attention_output = MultiHeadAttention(
                num_heads=num_heads, 
                key_dim=d_model // num_heads,
                dropout=dropout_rate,
                name=f'multi_head_attention_{layer_idx}'
            )(x_norm, x_norm)
            attention_output = Dropout(dropout_rate)(attention_output)
            x = Add(name=f'add_attention_{layer_idx}')([x, attention_output])
            
            # Feed-forward network
            x_norm = LayerNormalization(epsilon=1e-6, name=f'layer_norm_2_{layer_idx}')(x)
            ff_output = Dense(ff_dim * 2, activation='relu', name=f'ff_dense_1_{layer_idx}')(x_norm)
            ff_output = Dropout(dropout_rate)(ff_output)
            ff_output = Dense(ff_dim, name=f'ff_dense_2_{layer_idx}')(ff_output)
            ff_output = Dropout(dropout_rate)(ff_output)
            x = Add(name=f'add_ff_{layer_idx}')([x, ff_output])
        
        # Final layer normalization
        x = LayerNormalization(epsilon=1e-6, name='final_layer_norm')(x)
        
        # Global pooling with both average and max pooling
        avg_pool = GlobalAveragePooling1D(name='global_avg_pool')(x)
        max_pool = GlobalMaxPooling1D(name='global_max_pool')(x)
        
        # Combine pooled features
        combined = Concatenate(name='combine_pools')([avg_pool, max_pool])
        
        # Dense layers for classification
        x = Dense(ff_dim, activation='relu', name='dense_1')(combined)
        x = BatchNormalization(name='batch_norm_1')(x)
        x = Dropout(dropout_rate * 2)(x)
        
        x = Dense(ff_dim // 2, activation='relu', name='dense_2')(x)
        x = BatchNormalization(name='batch_norm_2')(x)
        x = Dropout(dropout_rate)(x)
        
        x = Dense(ff_dim // 4, activation='relu', name='dense_3')(x)
        x = Dropout(dropout_rate)(x)
        
        # Output layer
        outputs = Dense(num_classes, activation='softmax', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='enhanced_transformer')
        
        # Advanced optimizer configuration
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
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
        
        print(f"🏗️ Enhanced Transformer model built on device: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
        print(f"📊 Model parameters: {model.count_params():,}")
        
    return model

def train_model_with_checkpoints(model, X_train, y_train_cat, X_test, experiment_id, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Enhanced training function with comprehensive checkpointing and data augmentation."""
    
    # Check for existing checkpoint
    progress = load_checkpoint(experiment_id)
    start_epoch = 0
    
    if progress and progress.get('current_epoch', 0) > 0:
        start_epoch = progress['current_epoch']
        print(f"🔄 Resuming training from epoch {start_epoch}")
        
        # Load saved model if available
        exp_dir = os.path.join(CHECKPOINT_DIR, experiment_id)
        model_file = os.path.join(exp_dir, f"model_epoch_{start_epoch}.weights.h5")
        if os.path.exists(model_file):
            try:
                model.load_weights(model_file)
                print(f"✅ Model weights loaded from epoch {start_epoch}")
            except Exception as e:
                print(f"⚠️ Failed to load model weights: {e}")
                print(f"🔄 Starting fresh training")
                start_epoch = 0
        else:
            print(f"⚠️ Model weights file not found: {model_file}")
            print(f"🔄 Starting fresh training")
            start_epoch = 0
    else:
        print(f"🆕 Starting new training for: {experiment_id}")
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
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=8,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Data augmentation: enhanced techniques for binary sequences
    def augment_binary_sequences(X, y, shift_range=5, noise_level=0.05):
        X_aug, y_aug = [], []
        for i in range(len(X)):
            # Original sample
            X_aug.append(X[i])
            y_aug.append(y[i])
            
            # Time shifts
            for shift in range(1, shift_range + 1):
                if shift < X[i].shape[0]:
                    X_shifted = np.roll(X[i], shift, axis=0)
                    X_aug.append(X_shifted)
                    y_aug.append(y[i])
                    X_shifted_back = np.roll(X[i], -shift, axis=0)
                    X_aug.append(X_shifted_back)
                    y_aug.append(y[i])
            
            # Add small amount of noise to feature columns (not binary column)
            X_noisy = X[i].copy()
            # Only add noise to non-binary features (columns 2 onwards)
            if X_noisy.shape[1] > 1:
                noise = np.random.normal(0, noise_level, X_noisy[:, 1:].shape)
                X_noisy[:, 1:] += noise
                X_aug.append(X_noisy)
                y_aug.append(y[i])
                
        return np.array(X_aug), np.array(y_aug)

    # Augment training data
    X_train_aug, y_train_aug = augment_binary_sequences(X_train, np.argmax(y_train_cat, axis=1))
    y_train_cat_aug = to_categorical(y_train_aug)

    device_name = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    
    with tf.device(device_name):
        print(f"🔥 Training on device: {'GPU' if 'GPU' in device_name else 'CPU'} with augmented data: {X_train_aug.shape[0]} samples")
        
        # Train only remaining epochs
        if start_epoch < epochs:
            remaining_epochs = epochs - start_epoch
            print(f"🚀 Training {remaining_epochs} remaining epochs (from {start_epoch} to {epochs})")
            
            history = model.fit(
                X_train_aug, y_train_cat_aug, 
                epochs=epochs,
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
    """Enhanced data loading with caching and improved feature extraction"""
    
    # Try to load from cache first
    cache_key = "data_by_location_enhanced"
    cached_data = load_data_cache("data_loading", cache_key)
    if cached_data is not None:
        print("🚀 Using cached data for faster loading")
        return cached_data
    
    data = {loc: {} for loc in locations}
    users = get_user_dirs(data_root)
    print("🗂️ Starting enhanced data loading...")

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
                        
                        # Use enhanced sequence extraction
                        sequences = extract_sequences(df)
                        if sequences.size == 0:
                            print(f"   ⚠️ No sequences extracted from: {file}")
                            continue
                        feats_list.append(sequences)
                        labels_list.extend([activity.lower()] * sequences.shape[0])
                        print(f"   ✅ Extracted {sequences.shape[0]} enhanced sequences with {sequences.shape[2]} features")
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

    # Summary and cache the data
    total_samples = 0
    for loc in data:
        loc_samples = sum(len(data[loc][user][1]) for user in data[loc])
        total_samples += loc_samples
        print(f"\n📦 Summary for location '{loc}': {len(data[loc])} users, {loc_samples} samples loaded.")
    
    print(f"\n🎯 Total samples loaded: {total_samples}")
    
    # Save to cache for faster future loading
    save_data_cache("data_loading", data, cache_key)
    
    return data



def run_lo_location():
    """Enhanced Leave-One-Location-Out experiment with checkpointing"""
    data_by_loc = load_data_by_location()
    output_root = os.path.join(data_root, "Results", game, model_name, "leave-one-location-out")
    os.makedirs(output_root, exist_ok=True)

    for test_loc in locations:
        print(f"\n🚨 LOLO: Testing on {test_loc}")

        for train_ratio, test_ratio in ratios:
            subdir = f"cross-user/train{int(train_ratio*100)}_test{int(test_ratio*100)}"
            out_path = os.path.join(output_root, subdir, f"test_on_{test_loc}")
            
            # Create experiment ID for checkpointing
            experiment_id = create_experiment_id("LOLO_cross_user", test_loc, (train_ratio, test_ratio))
            
            # Check if already completed
            progress = load_checkpoint(experiment_id)
            if progress and progress.get('status') == 'completed':
                print(f"✅ Experiment {experiment_id} already completed, skipping")
                continue
            
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
                
                # Use enhanced training with checkpoints
                y_pred_probs = train_model_with_checkpoints(
                    model, X_train, y_train_cat, X_test, experiment_id
                )
                y_pred = le.inverse_transform(np.argmax(y_pred_probs, axis=1))
                save_results(y_test, y_pred.tolist(), list(le.classes_), out_path)

        # Intra-User experiments
        intra_preds, intra_trues = [], []
        for user in data_by_loc.get(test_loc, {}):
            experiment_id = create_experiment_id("LOLO_intra_user", test_loc, user=user)
            
            # Check if already completed
            progress = load_checkpoint(experiment_id)
            if progress and progress.get('status') == 'completed':
                print(f"✅ Intra-user experiment for {user} already completed, skipping")
                continue
                
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
            
            y_pred_probs = train_model_with_checkpoints(
                model, X_tr, y_tr_cat, X_te, experiment_id
            )
            y_pred = le.inverse_transform(np.argmax(y_pred_probs, axis=1))
            intra_preds.extend(y_pred)
            intra_trues.extend(y_te)

        if intra_preds:
            intra_path = os.path.join(output_root, "intra-user", f"test_on_{test_loc}")
            save_results(intra_trues, intra_preds, sorted(set(intra_trues)), intra_path)

        # Leave-One-User-Out experiments
        louo_preds, louo_trues = [], []
        all_users = list(data_by_loc.get(test_loc, {}).keys())
        for test_user in all_users:
            experiment_id = create_experiment_id("LOUO", test_loc, user=test_user)
            
            # Check if already completed
            progress = load_checkpoint(experiment_id)
            if progress and progress.get('status') == 'completed':
                print(f"✅ LOUO experiment for {test_user} already completed, skipping")
                continue
                
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
                
                y_pred_probs = train_model_with_checkpoints(
                    model, X_train, y_train_cat, X_test, experiment_id
                )
                y_pred = le.inverse_transform(np.argmax(y_pred_probs, axis=1))
                louo_preds.extend(y_pred)
                louo_trues.extend(y_test)

        if louo_preds:
            louo_path = os.path.join(output_root, "leave-one-user-out", f"test_on_{test_loc}")
            save_results(louo_trues, louo_preds, sorted(set(louo_trues)), louo_path)



def run_train_one_test_others():
    """Enhanced Train-One-Test-Others experiment with checkpointing"""
    data_by_loc = load_data_by_location()
    output_root = os.path.join(data_root, "Results", game, model_name, "train-one-test-others")
    os.makedirs(output_root, exist_ok=True)

    for train_loc in locations:
        test_locs = [loc for loc in locations if loc != train_loc]
        print(f"\n🚨 Training on {train_loc}, testing on: {test_locs}")

        for train_ratio, test_ratio in ratios:
            subdir = f"cross-user/train{int(train_ratio*100)}_test{int(test_ratio*100)}"
            out_path = os.path.join(output_root, subdir, f"train_on_{train_loc}")
            
            # Create experiment ID for checkpointing
            experiment_id = create_experiment_id("TOTO_cross_user", train_loc, (train_ratio, test_ratio))
            
            # Check if already completed
            progress = load_checkpoint(experiment_id)
            if progress and progress.get('status') == 'completed':
                print(f"✅ Experiment {experiment_id} already completed, skipping")
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
                le = LabelEncoder().fit(y_train + y_test)
                print("🧾 LabelEncoder classes:", dict(zip(le.classes_, le.transform(le.classes_))))
                y_train_cat = to_categorical(le.transform(y_train))
                
                model = build_transformer(X_train.shape[1:], y_train_cat.shape[1])
                print("🔍 y_train distribution:", Counter(y_train))
                print("🔍 y_test  distribution:", Counter(y_test))
                
                # Use enhanced training with checkpoints
                y_pred_probs = train_model_with_checkpoints(
                    model, X_train, y_train_cat, X_test, experiment_id
                )
                y_pred = le.inverse_transform(np.argmax(y_pred_probs, axis=1))
                save_results(y_test, y_pred.tolist(), list(le.classes_), out_path)

        # Intra-User experiments
        intra_preds, intra_trues = [], []
        for loc in test_locs:
            for user in data_by_loc[loc]:
                experiment_id = create_experiment_id("TOTO_intra_user", f"{train_loc}_to_{loc}", user=user)
                
                # Check if already completed
                progress = load_checkpoint(experiment_id)
                if progress and progress.get('status') == 'completed':
                    print(f"✅ Intra-user experiment for {user} already completed, skipping")
                    continue
                    
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
                
                y_pred_probs = train_model_with_checkpoints(
                    model, X_train, y_train_cat, X_te, experiment_id
                )
                y_pred = le.inverse_transform(np.argmax(y_pred_probs, axis=1))
                intra_preds.extend(y_pred)
                intra_trues.extend(y_te)

        if intra_preds:
            intra_path = os.path.join(output_root, "intra-user", f"train_on_{train_loc}")
            save_results(intra_trues, intra_preds, sorted(set(intra_trues)), intra_path)

        # Leave-One-User-Out experiments
        louo_preds, louo_trues = [], []
        for loc in test_locs:
            for test_user in data_by_loc[loc]:
                experiment_id = create_experiment_id("TOTO_LOUO", f"{train_loc}_to_{loc}", user=test_user)
                
                # Check if already completed
                progress = load_checkpoint(experiment_id)
                if progress and progress.get('status') == 'completed':
                    print(f"✅ LOUO experiment for {test_user} already completed, skipping")
                    continue
                    
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
                
                y_pred_probs = train_model_with_checkpoints(
                    model, X_train, y_train_cat, X_test, experiment_id
                )
                y_pred = le.inverse_transform(np.argmax(y_pred_probs, axis=1))
                louo_preds.extend(y_pred)
                louo_trues.extend(y_test)

        if louo_preds:
            louo_path = os.path.join(output_root, "leave-one-user-out", f"train_on_{train_loc}")
            save_results(louo_trues, louo_preds, sorted(set(louo_trues)), louo_path)





def print_experiment_summary():
    """Print a summary of all experiments and their status"""
    print("\n" + "="*80)
    print("🎯 ENHANCED TRANSFORMER EXPERIMENT SUMMARY")
    print("="*80)
    
    if not os.path.exists(CHECKPOINT_DIR):
        print("📝 No experiments have been run yet")
        return
    
    total_experiments = 0
    completed_experiments = 0
    in_progress_experiments = 0
    
    for dir_name in os.listdir(CHECKPOINT_DIR):
        if not dir_name.startswith(SCRIPT_ID):
            continue
            
        progress_file = os.path.join(CHECKPOINT_DIR, dir_name, "training_progress.json")
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                
                total_experiments += 1
                status = progress.get('status', 'unknown')
                current_epoch = progress.get('current_epoch', 0)
                total_epochs = progress.get('total_epochs', EPOCHS)
                
                if status == 'completed':
                    completed_experiments += 1
                    print(f"✅ {dir_name}: COMPLETED ({current_epoch}/{total_epochs} epochs)")
                elif current_epoch > 0:
                    in_progress_experiments += 1
                    print(f"🔄 {dir_name}: IN PROGRESS ({current_epoch}/{total_epochs} epochs)")
                else:
                    print(f"📝 {dir_name}: NOT STARTED")
                    
            except Exception as e:
                print(f"⚠️ Error reading {dir_name}: {e}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total experiments: {total_experiments}")
    print(f"   Completed: {completed_experiments}")
    print(f"   In progress: {in_progress_experiments}")
    print(f"   Remaining: {total_experiments - completed_experiments}")
    print("="*80)

if __name__ == "__main__":
    print("🚀 Starting Enhanced Transformer Location Ablation Study")
    print(f"🔧 Configuration:")
    print(f"   - Data root: {data_root}")
    print(f"   - Locations: {locations}")
    print(f"   - Ratios: {ratios}")
    print(f"   - Epochs: {EPOCHS}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print(f"   - Mixed precision: {'Enabled' if tf.config.list_physical_devices('GPU') else 'Disabled'}")
    
    try:
        # Print current experiment status
        print_experiment_summary()
        
        print("\n🔥 Starting Leave-One-Location-Out experiments...")
        # run_lo_location()
        
        print("\n🔥 Starting Train-One-Test-Others experiments...")
        run_train_one_test_others()
        
        # Final summary
        print_experiment_summary()
        print("\n🎉 All experiments completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user")
        print_experiment_summary()
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        print_experiment_summary()
    finally:
        print(f"\n📁 Checkpoints saved in: {CHECKPOINT_DIR}")
        print("🔄 You can resume interrupted experiments by running the script again")
