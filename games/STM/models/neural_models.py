#!/usr/bin/env python3
"""
Improved models for all games with advanced architectures and preprocessing.
"""

import os
import sys
import re
import glob
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, 
    GlobalAveragePooling1D, Add, Embedding, Conv1D, MaxPooling1D,
    Flatten, Concatenate, BatchNormalization, LSTM, Bidirectional,
    Attention, Reshape, TimeDistributed, GlobalMaxPooling1D
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from games.STM.config import get_data_paths, get_game_config, VALID_ACTIVITIES, WINDOW_SIZE
from games.STM.models.placement_semantics import GraphSemantics
from games.STM.models.film import AnatomyConditionedFiLM
from games.STM.models.training_utils import (
    FocalLoss, LabelSmoothingLoss, BalancedSoftmaxLoss, LogitAdjustedLoss,
    compute_class_weights, compute_class_priors, compute_class_counts,
    create_training_callbacks, log_class_distribution, ClassUniformSampler
)

# Wrapper functions for loss classes
def balanced_softmax_loss(y_train):
    """Wrapper function for BalancedSoftmaxLoss"""
    class_priors = compute_class_priors(y_train)
    return BalancedSoftmaxLoss(class_priors=class_priors)

def logit_adjust_loss(y_train):
    """Wrapper function for LogitAdjustedLoss"""
    class_counts = compute_class_counts(y_train)
    return LogitAdjustedLoss(class_counts=class_counts)

warnings.filterwarnings("ignore")


def create_hierarchical_loss(hier_lambda):
    """Create hierarchical loss function that safely handles tensor unpacking."""
    def hierarchical_loss(y_true, y_pred):
        # Handle tensor unpacking safely
        if isinstance(y_pred, list):
            fine_pred, coarse_pred = y_pred[0], y_pred[1]
        else:
            # If not a list, assume it's the fine prediction
            fine_pred, coarse_pred = y_pred, y_pred
        
        if isinstance(y_true, list):
            fine_true, coarse_true = y_true[0], y_true[1]
        else:
            # If not a list, assume it's the fine target
            fine_true, coarse_true = y_true, y_true
        
        # Fine loss
        fine_loss = tf.keras.losses.categorical_crossentropy(fine_true, fine_pred)
        
        # Coarse loss
        coarse_loss = tf.keras.losses.categorical_crossentropy(coarse_true, coarse_pred)
        
        # Combined loss - use tf.constant to avoid symbolic tensor issues
        total_loss = fine_loss + tf.constant(hier_lambda, dtype=tf.float32) * coarse_loss
        return total_loss
    
    return hierarchical_loss

def create_loss_function(loss_type, y_train, label_smoothing=0.0):
    """
    Create loss function based on type and training data.
    
    Args:
        loss_type: 'ce', 'focal', 'balanced_softmax', 'logit_adjust'
        y_train: Training labels for computing class statistics
        label_smoothing: Label smoothing factor (ignored for focal/balanced_softmax/logit_adjust)
    
    Returns:
        Loss function
    """
    if loss_type == 'ce':
        if label_smoothing > 0:
            return LabelSmoothingLoss(smoothing=label_smoothing, from_logits=True)
        else:
            return 'categorical_crossentropy'
    elif loss_type == 'focal':
        return FocalLoss(gamma=2.0, from_logits=True)
    elif loss_type == 'balanced_softmax':
        class_priors = compute_class_priors(y_train)
        return BalancedSoftmaxLoss(class_priors=class_priors, from_logits=True)
    elif loss_type == 'logit_adjust':
        class_counts = compute_class_counts(y_train)
        return LogitAdjustedLoss(class_counts=class_counts, from_logits=True)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def create_hierarchical_head(x, num_classes, hier_lambda=0.3):
    """
    Create hierarchical head with coarse and fine classifiers.
    
    Args:
        x: Input features
        num_classes: Number of fine classes
        hier_lambda: Weight for coarse loss
    
    Returns:
        Tuple of (fine_outputs, coarse_outputs, hier_lambda)
    """
    # Coarse head: 2-way classifier {static, locomotion}
    coarse_head = Dense(32, activation='relu')(x)
    coarse_head = Dropout(0.2)(coarse_head)
    coarse_outputs = Dense(2, activation='softmax', name='coarse_output')(coarse_head)
    
    # Fine head: num_classes classifier
    fine_head = Dense(64, activation='relu')(x)
    fine_head = Dropout(0.3)(fine_head)
    fine_head = Dense(32, activation='relu')(fine_head)
    fine_head = Dropout(0.2)(fine_head)
    fine_outputs = Dense(num_classes, activation='softmax', name='fine_output')(fine_head)
    
    return fine_outputs, coarse_outputs, hier_lambda


def create_coarse_targets(y_fine, activity_names):
    """
    Create coarse targets from fine targets.
    
    Args:
        y_fine: Fine-grained labels (integer encoded)
        activity_names: List of activity names
    
    Returns:
        Coarse labels (0=static, 1=locomotion)
    """
    # Define coarse categories
    static_activities = {"standing", "sitting", "laying"}
    locomotion_activities = {"walking", "jogging", "upstairs", "downstairs"}
    
    # Create mapping
    coarse_mapping = {}
    for i, name in enumerate(activity_names):
        name_lower = name.lower()
        if any(static in name_lower for static in static_activities):
            coarse_mapping[i] = 0  # static
        elif any(locomotion in name_lower for locomotion in locomotion_activities):
            coarse_mapping[i] = 1  # locomotion
        else:
            # Default to locomotion for unknown activities
            coarse_mapping[i] = 1
            print(f"⚠️ Unknown activity '{name}' mapped to locomotion")
    
    # Convert fine labels to coarse
    y_coarse = np.array([coarse_mapping[label] for label in y_fine])
    
    print(f"Coarse mapping: {coarse_mapping}")
    print(f"Static activities: {[name for i, name in enumerate(activity_names) if coarse_mapping[i] == 0]}")
    print(f"Locomotion activities: {[name for i, name in enumerate(activity_names) if coarse_mapping[i] == 1]}")
    
    return y_coarse


def plot_confusion_matrix(y_true, y_pred, model_type, game_name, split_name):
    """Plot and save confusion matrix"""
    try:
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Get class names from config
        class_names = VALID_ACTIVITIES
        
        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_type} ({game_name}, {split_name})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        plot_dir = "confusion_matrices"
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{model_type}_{game_name}_{split_name}_cm.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confusion matrix saved to: {plot_path}")
        
    except Exception as e:
        print(f"⚠️ Error plotting confusion matrix: {e}")

class ImprovedPositionalEncoding(tf.keras.layers.Layer):
    """Improved positional encoding for multimodal inputs"""
    def __init__(self, sequence_length, d_model, **kwargs):
        super(ImprovedPositionalEncoding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)
    
    def positional_encoding(self, position, d_model):
        angles = self.get_angles(np.arange(position)[:, np.newaxis],
                               np.arange(d_model)[np.newaxis, :],
                               d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        pos_encoding = angles[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

def build_improved_transformer(input_shape, num_classes, game_name):
    """Improved Transformer model for binary time series classification"""
    
    inputs = Input(shape=input_shape)
    
    # For binary input, use embedding layer
    if input_shape[1] == 1:  # Binary data (Game-1)
        x = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.int32))(inputs)
        x = Embedding(2, 64)(x)  # Embed binary values
        x = Reshape((input_shape[0], 64))(x)
    else:
        # For multi-feature data, use dense layer
        x = Dense(64, activation='relu')(inputs)
    
    # Add positional encoding
    x = ImprovedPositionalEncoding(input_shape[0], 64)(x)
    
    # Transformer blocks
    for _ in range(3):
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=4, key_dim=16, dropout=0.1
        )(x, x)
        x = Add()([x, attention_output])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Feed forward network
        ffn = Dense(128, activation='relu')(x)
        ffn = Dropout(0.1)(ffn)
        ffn = Dense(64)(ffn)
        x = Add()([x, ffn])
        x = LayerNormalization(epsilon=1e-6)(x)
    
    # Global average pooling
    x = GlobalAveragePooling1D()(x)
    
    # Classification head
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


def build_advanced_cnn(input_shape, num_classes, game_name, featureset='base', use_film=False,
                      loss_type='ce', label_smoothing=0.0, y_train=None, 
                      adjacency_mode='fixed', gate_alpha_init=0.1,
                      hierarchical=False, hier_lambda=0.3, activity_names=None):
    """Advanced CNN with residual connections and attention"""
    
    print(f"🔧 Building advanced_cnn with input_shape: {input_shape}, featureset: {featureset}")
    
    # Handle different input shapes based on featureset
    if featureset == 'concatenated':
        # Input shape: (100, 5) - concatenated binary features
        if len(input_shape) != 2 or input_shape[1] != 5:
            raise ValueError(f"Expected concatenated input shape (100, 5), got {input_shape}")
        print(f"   📊 Using concatenated input: {input_shape[0]} time steps, {input_shape[1]} features")
    else:
        # Standard input shape: (100, 5, 1) or (100, 5, 4) etc.
        print(f"   📊 Using standard input: {input_shape}")
    
    inputs = Input(shape=input_shape, name='main_input')
    
    # User metadata input for FiLM conditioning
    if use_film:
        user_meta_input = Input(shape=(9,), name='user_metadata')
        model_inputs = [inputs, user_meta_input]
    else:
        user_meta_input = None
        model_inputs = [inputs]
    
    # Initial convolution
    x = Conv1D(64, 7, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    
    # Residual blocks
    for i in range(3):
        residual = x
        x = Conv1D(64 * (2**i), 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv1D(64 * (2**i), 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        
        # Add residual connection if dimensions match
        if residual.shape[-1] == x.shape[-1]:
            x = Add()([x, residual])
        
        x = MaxPooling1D(2)(x)
        x = Dropout(0.2)(x)
    
    # Global attention pooling
    attention_weights = Dense(1, activation='tanh')(x)
    attention_weights = tf.keras.layers.Softmax(axis=1)(attention_weights)
    # Apply attention weights - ensure proper broadcasting
    x = tf.keras.layers.Lambda(lambda inputs: inputs[0] * inputs[1])([x, attention_weights])
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Add FiLM conditioning if requested
    if use_film and user_meta_input is not None:
        # Apply FiLM conditioning
        film_layer = FiLMLayer(128)
        x = film_layer([x, user_meta_input])
    
    # Output layer
    if loss_type == 'ce':
        outputs = Dense(num_classes, activation='softmax')(x)
    elif loss_type == 'focal':
        outputs = Dense(num_classes, activation='sigmoid')(x)
    else:
        outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=model_inputs, outputs=outputs, name='advanced_cnn')
    
    # Compile model
    if loss_type == 'ce':
        loss = 'categorical_crossentropy'
    elif loss_type == 'focal':
        loss = focal_loss(alpha=0.25, gamma=2.0)
    elif loss_type == 'balanced_softmax':
        loss = balanced_softmax_loss(y_train)
    elif loss_type == 'logit_adjust':
        loss = logit_adjust_loss(y_train)
    else:
        loss = 'categorical_crossentropy'
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy']
    )
    
    print(f"✅ Advanced CNN model built successfully")
    print(f"   📊 Input shape: {input_shape}")
    print(f"   📊 Output classes: {num_classes}")
    print(f"   📊 Loss function: {loss_type}")
    
    return model

def build_lstm_attention(input_shape, num_classes, game_name, featureset='base', use_film=False,
                        loss_type='ce', label_smoothing=0.0, y_train=None, 
                        adjacency_mode='fixed', gate_alpha_init=0.1,
                        hierarchical=False, hier_lambda=0.3, activity_names=None):
    """LSTM with attention mechanism"""
    
    print(f"🔧 Building lstm_attention with input_shape: {input_shape}, featureset: {featureset}")
    
    # Handle different input shapes based on featureset
    if featureset == 'concatenated':
        # Input shape: (100, 5) - concatenated binary features
        if len(input_shape) != 2 or input_shape[1] != 5:
            raise ValueError(f"Expected concatenated input shape (100, 5), got {input_shape}")
        print(f"   📊 Using concatenated input: {input_shape[0]} time steps, {input_shape[1]} features")
    else:
        # Standard input shape: (100, 5, 1) or (100, 5, 4) etc.
        print(f"   📊 Using standard input: {input_shape}")
    
    inputs = Input(shape=input_shape, name='main_input')
    
    # User metadata input for FiLM conditioning
    if use_film:
        user_meta_input = Input(shape=(9,), name='user_metadata')
        model_inputs = [inputs, user_meta_input]
    else:
        user_meta_input = None
        model_inputs = [inputs]
    
    # LSTM layers with bidirectional processing
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    
    # Attention mechanism
    attention_weights = Dense(1, activation='tanh')(x)
    attention_weights = tf.keras.layers.Softmax(axis=1)(attention_weights)
    # Apply attention weights - ensure proper broadcasting
    x = tf.keras.layers.Lambda(lambda inputs: inputs[0] * inputs[1])([x, attention_weights])
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Add FiLM conditioning if requested
    if use_film and user_meta_input is not None:
        # Apply FiLM conditioning
        film_layer = FiLMLayer(64)
        x = film_layer([x, user_meta_input])
    
    # Output layer
    if loss_type == 'ce':
        outputs = Dense(num_classes, activation='softmax')(x)
    elif loss_type == 'focal':
        outputs = Dense(num_classes, activation='sigmoid')(x)
    else:
        outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=model_inputs, outputs=outputs, name='lstm_attention')
    
    # Compile model
    if loss_type == 'ce':
        loss = 'categorical_crossentropy'
    elif loss_type == 'focal':
        loss = focal_loss(alpha=0.25, gamma=2.0)
    elif loss_type == 'balanced_softmax':
        loss = balanced_softmax_loss(y_train)
    elif loss_type == 'logit_adjust':
        loss = logit_adjust_loss(y_train)
    else:
        loss = 'categorical_crossentropy'
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy']
    )
    
    print(f"✅ LSTM Attention model built successfully")
    print(f"   📊 Input shape: {input_shape}")
    print(f"   📊 Output classes: {num_classes}")
    print(f"   📊 Loss function: {loss_type}")
    
    return model

def build_multimodal_transformer(input_shape, num_classes, game_name, featureset='base', use_film=False,
                                loss_type='ce', label_smoothing=0.0, y_train=None, 
                                adjacency_mode='fixed', gate_alpha_init=0.1,
                                hierarchical=False, hier_lambda=0.3, activity_names=None):
    """Advanced multimodal Transformer for Game-2 and Game-3"""
    
    print(f"🔧 Building multimodal_transformer with input_shape: {input_shape}, featureset: {featureset}")
    
    # Handle different input shapes based on featureset
    if featureset == 'concatenated':
        # Input shape: (100, 5) - concatenated binary features
        if len(input_shape) != 2 or input_shape[1] != 5:
            raise ValueError(f"Expected concatenated input shape (100, 5), got {input_shape}")
        print(f"   📊 Using concatenated input: {input_shape[0]} time steps, {input_shape[1]} features")
    else:
        # Standard input shape: (100, 5, 1) or (100, 5, 4) etc.
        print(f"   📊 Using standard input: {input_shape}")
    
    inputs = Input(shape=input_shape, name='main_input')
    
    # User metadata input for FiLM conditioning
    if use_film:
        user_meta_input = Input(shape=(9,), name='user_metadata')
        model_inputs = [inputs, user_meta_input]
    else:
        user_meta_input = None
        model_inputs = [inputs]
    
    # Feature embedding for multimodal data
    if game_name == "Game-2":
        # Game-2 has sensor + binary features (4 features: 3 sensor + 1 binary)
        # Simplified approach: treat all features as continuous
        x = Dense(64, activation='relu')(inputs)
        x = Dense(48, activation='relu')(x)
        d_model = 48
        
    elif game_name == "Game-3":
        # Game-3 has full IMU + binary features (7 features: 6 IMU + 1 binary)
        # Simplified approach: treat all features as continuous
        x = Dense(96, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        d_model = 64
    else:
        # Game-1: Binary data - Simplified approach
        x = Dense(32, activation='relu')(inputs)
        d_model = 32
    
    # Positional encoding
    x = ImprovedPositionalEncoding(input_shape[0], d_model)(x)
    
    # Transformer blocks
    for i in range(2):
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=4, key_dim=d_model//4, dropout=0.1
        )(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        # Feed-forward network
        ffn = Dense(d_model * 2, activation='relu')(x)
        ffn = Dropout(0.1)(ffn)
        ffn = Dense(d_model)(ffn)
        x = LayerNormalization(epsilon=1e-6)(x + ffn)
    
    # Global pooling and classification
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_hybrid_cnn_transformer(input_shape, num_classes, game_name, featureset='base', use_film=False,
                                loss_type='ce', label_smoothing=0.0, y_train=None, 
                                adjacency_mode='fixed', gate_alpha_init=0.1,
                                hierarchical=False, hier_lambda=0.3, activity_names=None):
    """Hybrid CNN-Transformer model"""
    
    print(f"🔧 Building hybrid_cnn_transformer with input_shape: {input_shape}, featureset: {featureset}")
    
    # Handle different input shapes based on featureset
    if featureset == 'concatenated':
        # Input shape: (100, 5) - concatenated binary features
        if len(input_shape) != 2 or input_shape[1] != 5:
            raise ValueError(f"Expected concatenated input shape (100, 5), got {input_shape}")
        print(f"   📊 Using concatenated input: {input_shape[0]} time steps, {input_shape[1]} features")
    else:
        # Standard input shape: (100, 5, 1) or (100, 5, 4) etc.
        print(f"   📊 Using standard input: {input_shape}")
    
    inputs = Input(shape=input_shape, name='main_input')
    
    # User metadata input for FiLM conditioning
    if use_film:
        user_meta_input = Input(shape=(9,), name='user_metadata')
        model_inputs = [inputs, user_meta_input]
    else:
        user_meta_input = None
        model_inputs = [inputs]
    
    # CNN branch for local feature extraction
    x_cnn = Conv1D(64, 5, activation='relu', padding='same')(inputs)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = MaxPooling1D(2)(x_cnn)
    x_cnn = Conv1D(128, 3, activation='relu', padding='same')(x_cnn)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = MaxPooling1D(2)(x_cnn)
    x_cnn = Conv1D(256, 3, activation='relu', padding='same')(x_cnn)
    x_cnn = GlobalAveragePooling1D()(x_cnn)
    
    # Transformer branch for global dependencies
    if game_name == "Game-1":
        embedding_dim = 32
        # Convert inputs to integer for embedding
        inputs_int = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.int32))(inputs)
        x_transformer = Embedding(input_dim=2, output_dim=embedding_dim, input_length=input_shape[0])(inputs_int)
        x_transformer = Reshape((input_shape[0], embedding_dim))(x_transformer)
    else:
        x_transformer = Dense(64, activation='relu')(inputs)
        embedding_dim = 64
    
    # Add positional encoding
    x_transformer = ImprovedPositionalEncoding(input_shape[0], embedding_dim)(x_transformer)
    
    # Transformer layers
    for i in range(2):
        attention_output = MultiHeadAttention(
            num_heads=4, key_dim=embedding_dim//4, dropout=0.1
        )(x_transformer, x_transformer)
        x_transformer = LayerNormalization(epsilon=1e-6)(x_transformer + attention_output)
        
        ffn = Dense(embedding_dim * 2, activation='relu')(x_transformer)
        ffn = Dropout(0.1)(ffn)
        ffn = Dense(embedding_dim)(ffn)
        x_transformer = LayerNormalization(epsilon=1e-6)(x_transformer + ffn)
    
    x_transformer = GlobalAveragePooling1D()(x_transformer)
    
    # Combine CNN and Transformer features
    combined = Concatenate()([x_cnn, x_transformer])
    
    # Final classification layers
    x = Dense(256, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Add FiLM conditioning if requested
    if use_film and user_meta_input is not None:
        # Apply FiLM conditioning
        film_layer = FiLMLayer(128)
        x = film_layer([x, user_meta_input])
    
    # Output layer
    if loss_type == 'ce':
        outputs = Dense(num_classes, activation='softmax')(x)
    elif loss_type == 'focal':
        outputs = Dense(num_classes, activation='sigmoid')(x)
    else:
        outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=model_inputs, outputs=outputs, name='multimodal_transformer')
    
    # Compile model
    if loss_type == 'ce':
        loss = 'categorical_crossentropy'
    elif loss_type == 'focal':
        loss = focal_loss(alpha=0.25, gamma=2.0)
    elif loss_type == 'balanced_softmax':
        loss = balanced_softmax_loss(y_train)
    elif loss_type == 'logit_adjust':
        loss = logit_adjust_loss(y_train)
    else:
        loss = 'categorical_crossentropy'
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy']
    )
    
    print(f"✅ Multimodal Transformer model built successfully")
    print(f"   📊 Input shape: {input_shape}")
    print(f"   📊 Output classes: {num_classes}")
    print(f"   📊 Loss function: {loss_type}")
    
    return model

def advanced_preprocessing(data, game_name, user_ids=None, use_per_user_norm=False):
    """Advanced preprocessing with feature engineering and optional per-user normalization"""
    
    if game_name == "Game-1":
        # Binary data - no scaling needed
        return data
    
    elif game_name == "Game-2":
        # Game-2 has sensor + binary features (4 features: 3 sensor + 1 binary)
        sensor_data = data[:, :, :3]  # Sensor data (accel or gyro)
        binary_data = data[:, :, 3:]  # Binary features
        
        if use_per_user_norm:
            # Data is already normalized per-user, just return as is
            return data
        else:
            # Scale sensor data only (not binary)
            scaler = StandardScaler()
            sensor_reshaped = sensor_data.reshape(-1, 3)
            sensor_scaled = scaler.fit_transform(sensor_reshaped).reshape(sensor_data.shape)
            
            # Reconstruct data with scaled sensor + original binary
            processed_data = np.concatenate([sensor_scaled, binary_data], axis=2)
            return processed_data
    
    elif game_name == "Game-3":
        # Game-3 has full IMU + binary features (7 features: 6 IMU + 1 binary)
        imu_data = data[:, :, :6]  # IMU data (accel + gyro)
        binary_data = data[:, :, 6:]  # Binary features
        
        if use_per_user_norm:
            # Data is already normalized per-user, just return as is
            return data
        else:
            # Scale IMU data only (not binary)
            scaler = StandardScaler()
            imu_reshaped = imu_data.reshape(-1, 6)
            imu_scaled = scaler.fit_transform(imu_reshaped).reshape(imu_data.shape)
            
            # Reconstruct data with scaled IMU + original binary
            processed_data = np.concatenate([imu_scaled, binary_data], axis=2)
            return processed_data
    
    return data

def train_and_evaluate_advanced_model(X_train, y_train, X_test, y_test, model_type, game_name, split_name, 
                                     use_class_weights=False, use_per_user_norm=False, featureset='base', 
                                     use_film=False, use_focal_loss=False, use_label_smoothing=False, 
                                     label_smoothing=0.05, loss_type='ce', adjacency_mode='fixed', 
                                     gate_alpha_init=0.1, hierarchical=False, hier_lambda=0.3,
                                     sampler='random', warmup_epochs=0, select_by='accuracy', 
                                     train_user_metadata=None, test_user_metadata=None, placement='all', activity_names=None):
    """Train and evaluate advanced models with semantic features and training improvements"""
    
    # Get activity names for loss functions (define early to avoid scope issues)
    if activity_names is None:
        from games.STM.config import VALID_ACTIVITIES
        activity_names = list(VALID_ACTIVITIES)
    
    print(f"\n🚀 Training {model_type} model for {game_name}...")
    print(f"📊 Training data: {X_train.shape[0]} samples")
    print(f"📊 Test data: {X_test.shape[0]} samples")
    print(f"🎯 Classes: {len(np.unique(y_train))}")
    if use_per_user_norm:
        print(f"🔧 Using per-user normalization")
    if use_film:
        print(f"🎭 Using FiLM conditioning with user metadata")
    if loss_type != 'ce':
        print(f"🎯 Using {loss_type} loss")
    if adjacency_mode != 'fixed':
        print(f"🔗 Using {adjacency_mode} adjacency mode")
    if hierarchical:
        print(f"🏗️ Using hierarchical head (λ={hier_lambda})")
    if sampler == 'class_uniform':
        print(f"⚖️ Using class-uniform sampling")
    if warmup_epochs > 0:
        print(f"🔥 Using {warmup_epochs} warmup epochs")
    print("=" * 60)
    
    # Advanced preprocessing
    X_train_processed = advanced_preprocessing(X_train, game_name, use_per_user_norm=use_per_user_norm)
    X_test_processed = advanced_preprocessing(X_test, game_name, use_per_user_norm=use_per_user_norm)
    
    # Use stratified sampling for validation split to balance minority classes
    from sklearn.model_selection import train_test_split
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_processed, y_train, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train  # Ensure balanced representation of all classes
    )
    
    print(f"   📊 Training samples: {X_train_final.shape[0]}")
    print(f"   📊 Validation samples: {X_val.shape[0]}")
    print(f"   📊 Test samples: {X_test_processed.shape[0]}")
    
    # Print class distribution
    from collections import Counter
    train_dist = Counter(y_train_final)
    val_dist = Counter(y_val)
    test_dist = Counter(y_test)
    print(f"   🎯 Class distribution:")
    print(f"      Train: {dict(train_dist)}")
    print(f"      Val: {dict(val_dist)}")
    print(f"      Test: {dict(test_dist)}")
    
    # Build model with new parameters
    print(f"🔧 Building {model_type} architecture...")
    print(f"   📊 Model type: {model_type}")
    print(f"   📊 Featureset: {featureset}")
    print(f"   📊 Model type comparison: '{model_type}' == 'continuous_late_fusion': {model_type == 'continuous_late_fusion'}")
    if model_type == "advanced_cnn":
        model = build_advanced_cnn(
            X_train_processed.shape[1:], len(np.unique(y_train)), game_name, featureset, use_film,
            loss_type=loss_type, label_smoothing=label_smoothing, y_train=y_train_final,
            adjacency_mode=adjacency_mode, gate_alpha_init=gate_alpha_init,
            hierarchical=hierarchical, hier_lambda=hier_lambda, activity_names=activity_names
        )
    elif model_type == "lstm_attention":
        model = build_lstm_attention(
            X_train_processed.shape[1:], len(np.unique(y_train)), game_name, featureset, use_film,
            loss_type=loss_type, label_smoothing=label_smoothing, y_train=y_train_final,
            adjacency_mode=adjacency_mode, gate_alpha_init=gate_alpha_init,
            hierarchical=hierarchical, hier_lambda=hier_lambda, activity_names=activity_names
        )
    elif model_type == "multimodal_transformer":
        model = build_multimodal_transformer(
            X_train_processed.shape[1:], len(np.unique(y_train)), game_name, featureset, use_film,
            loss_type=loss_type, label_smoothing=label_smoothing, y_train=y_train_final,
            adjacency_mode=adjacency_mode, gate_alpha_init=gate_alpha_init,
            hierarchical=hierarchical, hier_lambda=hier_lambda, activity_names=activity_names
        )
    elif model_type == "hybrid_cnn_transformer":
        model = build_hybrid_cnn_transformer(
            X_train_processed.shape[1:], len(np.unique(y_train)), game_name, featureset, use_film,
            loss_type=loss_type, label_smoothing=label_smoothing, y_train=y_train_final,
            adjacency_mode=adjacency_mode, gate_alpha_init=gate_alpha_init,
            hierarchical=hierarchical, hier_lambda=hier_lambda, activity_names=activity_names
        )
    # Multi-sensor fusion models with new parameters
    elif model_type == "binary_late_fusion":
        model = build_binary_late_fusion(
            X_train_processed.shape[1:], len(np.unique(y_train)), game_name, featureset, use_film,
            loss_type=loss_type, label_smoothing=label_smoothing, y_train=y_train_final,
            adjacency_mode=adjacency_mode, gate_alpha_init=gate_alpha_init,
            hierarchical=hierarchical, hier_lambda=hier_lambda
        )
    elif model_type == "binary_weighted_fusion":
        model = build_binary_weighted_fusion(
            X_train_processed.shape[1:], len(np.unique(y_train)), game_name, featureset, use_film,
            loss_type=loss_type, label_smoothing=label_smoothing, y_train=y_train_final,
            adjacency_mode=adjacency_mode, gate_alpha_init=gate_alpha_init,
            hierarchical=hierarchical, hier_lambda=hier_lambda
        )
    elif model_type == "binary_sensor_token_attention":
        model = build_binary_sensor_token_attention(
            X_train_processed.shape[1:], len(np.unique(y_train)), game_name, featureset, use_film,
            loss_type=loss_type, label_smoothing=label_smoothing, y_train=y_train_final,
            adjacency_mode=adjacency_mode, gate_alpha_init=gate_alpha_init,
            hierarchical=hierarchical, hier_lambda=hier_lambda
        )
    elif model_type == "continuous_late_fusion":
        model = build_continuous_late_fusion(
            X_train_processed.shape[1:], len(np.unique(y_train)), game_name, featureset, use_film,
            loss_type=loss_type, label_smoothing=label_smoothing, y_train=y_train_final,
            adjacency_mode=adjacency_mode, gate_alpha_init=gate_alpha_init,
            hierarchical=hierarchical, hier_lambda=hier_lambda
        )
    elif model_type == "continuous_weighted_fusion":
        model = build_continuous_weighted_fusion(
            X_train_processed.shape[1:], len(np.unique(y_train)), game_name, featureset, use_film,
            loss_type=loss_type, label_smoothing=label_smoothing, y_train=y_train_final,
            adjacency_mode=adjacency_mode, gate_alpha_init=gate_alpha_init,
            hierarchical=hierarchical, hier_lambda=hier_lambda
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Calculate class weights if requested
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(y_train_final, method='balanced')
        print(f"   ⚖️ Using class weights: {class_weights}")
    
    # Loss conflict resolution logic
    # Ensure only one of {class weights, focal, label smoothing, balanced_softmax, logit_adjust} is active
    active_imbalance_methods = []
    if class_weights:
        active_imbalance_methods.append("class_weights")
    if use_focal_loss:
        active_imbalance_methods.append("focal_loss")
    if use_label_smoothing:
        active_imbalance_methods.append("label_smoothing")
    if loss_type in ['balanced_softmax', 'logit_adjust']:
        active_imbalance_methods.append(loss_type)
    
    if len(active_imbalance_methods) > 1:
        print(f"   ⚠️ WARNING: Multiple imbalance methods detected: {active_imbalance_methods}")
        print(f"   🔧 Auto-disabling conflicting methods to use only: {active_imbalance_methods[0]}")
        
        # Keep only the first method, disable others
        if active_imbalance_methods[0] != "class_weights":
            class_weights = None
            print(f"   ❌ Disabled: class_weights")
        if active_imbalance_methods[0] != "focal_loss":
            use_focal_loss = False
            print(f"   ❌ Disabled: focal_loss")
        if active_imbalance_methods[0] != "label_smoothing":
            use_label_smoothing = False
            label_smoothing = 0.0
            print(f"   ❌ Disabled: label_smoothing")
        if active_imbalance_methods[0] not in ['balanced_softmax', 'logit_adjust']:
            loss_type = 'ce'
            print(f"   ❌ Disabled: {loss_type}")
    
    # Compute and log class priors for imbalance-aware losses
    if loss_type in ['balanced_softmax', 'logit_adjust']:
        class_priors = compute_class_priors(y_train_final)
        class_counts = compute_class_counts(y_train_final)
        print(f"   📊 Class Priors: {class_priors}")
        print(f"   📊 Class Counts: {class_counts}")
        print(f"   📊 Class Imbalance Ratio: {np.max(class_counts) / np.min(class_counts):.2f}:1")
    
    # Log class distribution
    log_class_distribution(y_train_final, "training")
    log_class_distribution(y_val, "validation")
    log_class_distribution(y_test, "test")
    
    # Configure loss function (only for non-fusion models that don't have built-in loss)
    if model_type not in ["binary_late_fusion", "binary_weighted_fusion", "binary_sensor_token_attention", "continuous_late_fusion", "continuous_weighted_fusion"]:
        loss_function = 'categorical_crossentropy'
        if use_focal_loss:
            loss_function = FocalLoss(gamma=2.0, from_logits=False)
            print(f"   🎯 Using Focal Loss (gamma=2.0)")
        elif use_label_smoothing:
            loss_function = LabelSmoothingLoss(smoothing=label_smoothing, from_logits=False)
            print(f"   🎯 Using Label Smoothing Loss (smoothing={label_smoothing})")
        elif loss_type == 'balanced_softmax':
            class_priors = compute_class_priors(y_train_final)
            loss_function = BalancedSoftmaxLoss(class_priors=class_priors, from_logits=True)
            print(f"   🎯 Using Balanced Softmax Loss")
        elif loss_type == 'logit_adjust':
            class_counts = compute_class_counts(y_train_final)
            loss_function = LogitAdjustedLoss(class_counts=class_counts, from_logits=True)
            print(f"   🎯 Using Logit-Adjusted Loss")
        
        # Recompile model with new loss function if needed
        if loss_function != 'categorical_crossentropy':
            model.compile(
                optimizer=model.optimizer,
                loss=loss_function,
                metrics=['accuracy']
            )
    
    # Create checkpoint directory with configuration details to prevent overwrites
    checkpoint_dir = os.path.join("checkpoints", model_type, game_name, split_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create results directory for model saving with configuration details to prevent overwrites
    # Include placement, featureset, and other key config details in the path
    config_suffix = f"{placement}_{featureset}"
    if use_film:
        config_suffix += "_film"
    if use_focal_loss:
        config_suffix += "_focal"
    if use_label_smoothing:
        config_suffix += f"_ls{label_smoothing}"
    
    results_dir = os.path.join("results", "STM", game_name, model_type, split_name, config_suffix)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create callbacks with improved monitoring
    monitor_metric = 'val_macro_f1' if select_by == 'macro_f1' else 'val_accuracy'
    callbacks = create_training_callbacks(
        validation_data=(X_val, to_categorical(y_val)),
        monitor=monitor_metric,
        patience=8,
        restore_best_weights=True,
        model=model
    )
    
    # Add model checkpoint with correct path
    mode = 'max' if monitor_metric == 'val_macro_f1' else 'min'
    callbacks.append(ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "best_model.keras.h5"),
        monitor=monitor_metric, mode=mode, save_best_only=True, verbose=1
    ))
    
    # Add learning rate warmup if requested
    if warmup_epochs > 0:
        from tensorflow.keras.callbacks import LearningRateScheduler
        def warmup_cosine_decay(epoch):
            if epoch < warmup_epochs:
                return 0.001 * (epoch + 1) / warmup_epochs  # Linear warmup
            else:
                # Cosine decay after warmup
                decay_epochs = TRAIN_CONFIGS["epochs"] - warmup_epochs
                decay_epoch = epoch - warmup_epochs
                return 0.001 * 0.5 * (1 + np.cos(np.pi * decay_epoch / decay_epochs))
        
        callbacks.append(LearningRateScheduler(warmup_cosine_decay, verbose=1))
        print(f"   🔥 Added learning rate warmup for {warmup_epochs} epochs")
    
    # Train model with optimized parameters and stratified validation
    from games.STM.config import TRAIN_CONFIGS
    print(f"🎯 Training {model_type}...")
    
    # Prepare training data with user metadata if FiLM is enabled
    if use_film and train_user_metadata is not None:
        # Split user metadata for validation
        X_train_final_meta, X_val_meta, _, _ = train_test_split(
            train_user_metadata, y_train, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_train
        )
        
        # Prepare hierarchical targets if needed
        if hierarchical:
            y_train_coarse = create_coarse_targets(y_train_final, activity_names)
            y_val_coarse = create_coarse_targets(y_val, activity_names)
            
            fit_kwargs = {
                'x': [X_train_final, X_train_final_meta],  # Multiple inputs for FiLM
                'y': [to_categorical(y_train_final), to_categorical(y_train_coarse)],  # Multiple outputs
                'validation_data': ([X_val, X_val_meta], [to_categorical(y_val), to_categorical(y_val_coarse)]),
                'epochs': TRAIN_CONFIGS["epochs"],
                'batch_size': TRAIN_CONFIGS["batch_size"],
                'callbacks': callbacks,
                'verbose': 1
            }
        else:
            fit_kwargs = {
                'x': [X_train_final, X_train_final_meta],  # Multiple inputs for FiLM
                'y': to_categorical(y_train_final),
                'validation_data': ([X_val, X_val_meta], to_categorical(y_val)),
                'epochs': TRAIN_CONFIGS["epochs"],
                'batch_size': TRAIN_CONFIGS["batch_size"],
                'callbacks': callbacks,
                'verbose': 1
            }
    else:
        # Prepare hierarchical targets if needed
        if hierarchical:
            y_train_coarse = create_coarse_targets(y_train_final, activity_names)
            y_val_coarse = create_coarse_targets(y_val, activity_names)
            
            fit_kwargs = {
                'x': X_train_final,
                'y': [to_categorical(y_train_final), to_categorical(y_train_coarse)],  # Multiple outputs
                'validation_data': (X_val, [to_categorical(y_val), to_categorical(y_val_coarse)]),
                'epochs': TRAIN_CONFIGS["epochs"],
                'batch_size': TRAIN_CONFIGS["batch_size"],
                'callbacks': callbacks,
                'verbose': 1
            }
        else:
            fit_kwargs = {
                'x': X_train_final,
                'y': to_categorical(y_train_final),
                'validation_data': (X_val, to_categorical(y_val)),
                'epochs': TRAIN_CONFIGS["epochs"],
                'batch_size': TRAIN_CONFIGS["batch_size"],
                'callbacks': callbacks,
                'verbose': 1
            }
    
    if class_weights:
        fit_kwargs['class_weight'] = class_weights
    
    print(f"   🏋️ Starting training for {TRAIN_CONFIGS['epochs']} epochs...")

    history = model.fit(**fit_kwargs)

    print(f"🎯 Evaluating {model_type} on test set...")
    
    # Evaluate with user metadata if FiLM is enabled
    if use_film and test_user_metadata is not None:
        predictions = model.predict([X_test_processed, test_user_metadata])
        if hierarchical:
            y_pred = np.argmax(predictions[0], axis=1)  # Use fine predictions
        else:
            y_pred = np.argmax(predictions, axis=1)
    else:
        predictions = model.predict(X_test_processed)
        if hierarchical:
            y_pred = np.argmax(predictions[0], axis=1)  # Use fine predictions
        else:
            y_pred = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # Check for class collapse (zero recall classes)
    from sklearn.metrics import recall_score
    per_class_recall = recall_score(y_test, y_pred, average=None)
    zero_recall_classes = np.where(per_class_recall == 0)[0]
    
    has_zero_recall = len(zero_recall_classes) > 0
    if has_zero_recall:
        print(f"⚠️ WARNING: {len(zero_recall_classes)} classes have zero recall: {zero_recall_classes}")
        
        # Save confusion matrix to failures directory
        failures_dir = os.path.join("results", "experiments", "failures")
        os.makedirs(failures_dir, exist_ok=True)
        
        # Create run_id for filename
        run_id = f"{model_type}_{game_name}_{split_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Plot and save confusion matrix
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {run_id}\nZero Recall Classes: {zero_recall_classes}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = os.path.join(failures_dir, f"{run_id}_cm.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confusion matrix saved to: {cm_path}")
    else:
        print("✅ No class collapse detected - all classes have non-zero recall")
    
    # Print best epoch metrics - handle both regular and hierarchical models
    if "val_accuracy" in history.history:
        # Regular model
        best_epoch = np.argmax(history.history["val_accuracy"])
        best_val_acc = history.history["val_accuracy"][best_epoch]
        accuracy_metric = "val_accuracy"
    elif "val_fine_output_accuracy" in history.history:
        # Hierarchical model - use fine output accuracy
        best_epoch = np.argmax(history.history["val_fine_output_accuracy"])
        best_val_acc = history.history["val_fine_output_accuracy"][best_epoch]
        accuracy_metric = "val_fine_output_accuracy"
    else:
        # Fallback to first available accuracy metric
        accuracy_metrics = [k for k in history.history.keys() if "accuracy" in k and "val" in k]
        if accuracy_metrics:
            accuracy_metric = accuracy_metrics[0]
            best_epoch = np.argmax(history.history[accuracy_metric])
            best_val_acc = history.history[accuracy_metric][best_epoch]
        else:
            # No accuracy metric found, use loss
            best_epoch = np.argmin(history.history["val_loss"])
            best_val_acc = 0.0
            accuracy_metric = "val_loss"
    
    best_val_loss = history.history["val_loss"][best_epoch]
    print(f"📌 Best Epoch: {best_epoch + 1}")
    print(f"   🟢 Best Val Accuracy: {best_val_acc:.4f}")
    print(f"   🔴 Best Val Loss: {best_val_loss:.4f}")
    
    # Store best epoch info for post-hoc analysis
    best_epoch_info = {
        "best_epoch": int(best_epoch + 1),
        "best_val_accuracy": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "total_epochs": int(len(history.history[accuracy_metric])),
        "early_stopped": bool(len(history.history[accuracy_metric]) < TRAIN_CONFIGS["epochs"])
    }
    
    # Print classification report
    from sklearn.metrics import classification_report
    print("📊 Classification Report on Test Set:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, model_type, game_name, split_name)
    
    # Save the final model to results directory for ensemble use
    model_path = os.path.join(results_dir, "model.keras.h5")
    model.save(model_path)
    print(f"💾 Model saved to {model_path}")
    
    print(f"✅ {model_type} - Accuracy: {accuracy:.4f}")
    
    # Return results without the model object to avoid JSON serialization issues
    return y_pred, accuracy, history, best_epoch_info, has_zero_recall, zero_recall_classes.tolist()

def save_advanced_results(y_pred, y_test, accuracy, model, history, model_name, game_name, split_name, results_root, best_epoch_info=None, featureset='base', use_film=False, use_focal_loss=False, use_label_smoothing=False, label_smoothing=0.0, placement='all'):
    """Save results with advanced metrics and best epoch info"""
    
    # Create output directory with configuration details to prevent overwrites
    config_suffix = f"{placement}_{featureset}"
    if use_film:
        config_suffix += "_film"
    if use_focal_loss:
        config_suffix += "_focal"
    if use_label_smoothing:
        config_suffix += f"_ls{label_smoothing}"
    
    out_dir = os.path.join(results_root, game_name, model_name, split_name, config_suffix)
    os.makedirs(out_dir, exist_ok=True)
    
    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, mutual_info_score
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='macro')  # Use macro-F1 for class-balanced evaluation
    mi = mutual_info_score(y_test, y_pred)
    
    # Calculate KL divergence
    from scipy.stats import entropy
    from collections import Counter
    
    def calculate_kl_divergence(y_true, y_pred):
        true_counts = Counter(y_true)
        pred_counts = Counter(y_pred)
        
        total_true = len(y_true)
        total_pred = len(y_pred)
        
        # Get the maximum class index to ensure consistent array sizes
        max_class = max(max(y_true) + 1, max(y_pred) + 1, len(np.unique(y_true)), len(np.unique(y_pred)))
        
        true_probs = np.zeros(max_class)
        pred_probs = np.zeros(max_class)
        
        # Fill in probabilities
        for i in range(max_class):
            true_probs[i] = true_counts.get(i, 0) / total_true
            pred_probs[i] = pred_counts.get(i, 0) / total_pred
        
        # Add epsilon to avoid log(0)
        epsilon = 1e-10
        true_probs += epsilon
        pred_probs += epsilon
        
        # Normalize to ensure they sum to 1
        true_probs /= np.sum(true_probs)
        pred_probs /= np.sum(pred_probs)
        
        return float(entropy(true_probs, pred_probs))
    
    kl_div = calculate_kl_divergence(y_test, y_pred)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Save metrics with best epoch info
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "mutual_information": float(mi),
        "kl_divergence": float(kl_div),
        "confusion_matrix": cm.tolist()
    }
    
    # Add best epoch info if provided
    if best_epoch_info:
        metrics["best_epoch_info"] = best_epoch_info
    
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save model if provided (for ensemble use)
    if model is not None:
        model_path = os.path.join(out_dir, "model.keras.h5")
        model.save(model_path)
        metrics["model_path"] = model_path
    else:
        # If model is None, we assume it was already saved during training
        model_path = os.path.join(out_dir, "model.keras.h5")
        if os.path.exists(model_path):
            metrics["model_path"] = model_path
    
    # Save training history
    history_path = os.path.join(out_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    
    print(f"💾 Results saved to {out_dir}")
    print(f"📈 Accuracy: {accuracy:.4f}")
    print(f"📈 F1 Score: {f1:.4f}")
    
    # Add model path to metrics for ensemble use
    metrics["model_path"] = model_path
    
    # Don't include the model object in the returned metrics to avoid JSON serialization issues
    # The model is already saved to disk at model_path
    
    return metrics


# ============================================================================
# Multi-Sensor Fusion Models for Game-1-multi-binary
# ============================================================================

def build_binary_late_fusion(input_shape, num_classes, game_name, featureset='base', use_film=False,
                           loss_type='ce', label_smoothing=0.0, y_train=None, 
                           adjacency_mode='fixed', gate_alpha_init=0.1,
                           hierarchical=False, hier_lambda=0.3, activity_names=None):
    """Late fusion model: separate encoders per placement + majority vote with semantic features support"""
    
    # Get activity names for loss functions (define early to avoid scope issues)
    if activity_names is None:
        from games.STM.config import VALID_ACTIVITIES
        activity_names = list(VALID_ACTIVITIES)
    
    print(f"🔧 Building binary_late_fusion with input_shape: {input_shape}")
    
    # Handle different input shapes for binary-only data
    if len(input_shape) == 3:
        # Input shape: (time_steps, placements, features) = (100, 5, 1)
        time_steps, num_placements, features_per_placement = input_shape
        has_summary_token = False
        summary_token_dim = 0
    elif len(input_shape) == 2:
        # Check if we have summary token (T+1, P+semantic_features)
        # For semantic features: (101, 19) = (100+1, 5+14) or (101, 99) = (100+1, 5+94)
        has_summary_token = input_shape[0] > 100  # Assuming window_size=100
        if has_summary_token:
            # Extract summary token and main data
            # input_shape[1] = 19 (5+14) or 99 (5+94) = 5 placements + semantic features
            summary_token_dim = input_shape[1] - 5  # 14 or 94 semantic features
            time_steps = input_shape[0] - 1  # 100 time steps (remove summary token)
            num_placements = 5
            features_per_placement = 1
        else:
            # Fallback for 2D input without summary token
            time_steps, num_placements = input_shape
            features_per_placement = 1
            summary_token_dim = 0
    else:
        raise ValueError(f"Unsupported input shape for binary_late_fusion: {input_shape}")
    
    print(f"   📊 Time steps: {time_steps}, Placements: {num_placements}, Features per placement: {features_per_placement}")
    
    # For binary-only data, we always have 1 feature per placement
    is_continuous = features_per_placement > 1
    
    # Shared temporal encoder for binary sequences
    def create_placement_encoder():
        # For binary features: (T, 1) - single binary feature per placement
        encoder_input = Input(shape=(time_steps, 1))  # (T, 1) for single placement
        
        # Small temporal encoder for binary sequences
        x = Conv1D(16, 3, activation='relu', padding='same')(encoder_input)
        x = BatchNormalization()(x)
        x = Conv1D(32, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.GRU(16, return_sequences=False)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        return Model(encoder_input, x)
    
    # Create shared encoder
    shared_encoder = create_placement_encoder()
    
    # Main input: (B, T, P, C) for binary data
    inputs = Input(shape=input_shape)
    
    # User metadata input for FiLM conditioning
    if use_film:
        user_meta_input = Input(shape=(9,), name='user_metadata')  # 9 features: 5 numeric + 2 dom_hand + 2 dom_foot
        model_inputs = [inputs, user_meta_input]
    else:
        user_meta_input = None
        model_inputs = [inputs]
    
    if has_summary_token:
        # Extract summary token and main data
        summary_token = tf.keras.layers.Lambda(lambda x: x[:, 0, 5:])(inputs)  # (B, semantic_features)
        main_data = tf.keras.layers.Lambda(lambda x: x[:, 1:, :5])(inputs)     # (B, T, 5)
    else:
        summary_token = None
        main_data = inputs
    
    # Process each placement separately
    placement_outputs = []
    for i in range(num_placements):
        # For binary features: extract single placement: (B, T, 1)
        if has_summary_token:
            # For semantic features: main_data shape is (B, T, P+semantic_features)
            # Extract placement i from the first 5 features (placements)
            placement_input = tf.keras.layers.Lambda(
                lambda x, idx=i: x[:, :, idx:idx+1], 
                output_shape=(time_steps, 1)
            )(main_data)
        else:
            # Original format: (B, T, P, C)
            placement_input = tf.keras.layers.Lambda(
                lambda x, idx=i: x[:, :, idx, :], 
                output_shape=(time_steps, features_per_placement)
            )(main_data)
        
        # Encode placement
        placement_encoded = shared_encoder(placement_input)
        placement_outputs.append(placement_encoded)
    
    # Stack placement outputs: (B, P, 32)
    stacked_outputs = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(placement_outputs)
    
    # Add placement semantics if requested
    if 'semantics' in featureset:
        graph_semantics = GraphSemantics(
            hidden_dim=64, 
            adjacency_mode=adjacency_mode,
            gate_alpha_init=gate_alpha_init
        )
        stacked_outputs = graph_semantics(stacked_outputs)
        # Ensure the output has a known shape for downstream layers
        stacked_outputs = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(stacked_outputs)
        # Force shape inference for downstream layers
        stacked_outputs = tf.keras.layers.Lambda(lambda x: tf.reshape(x, tf.shape(x)))(stacked_outputs)
    
    # Global pooling across placements
    x = GlobalAveragePooling1D()(stacked_outputs)
    
    # Add summary token if available
    if summary_token is not None:
        # Project summary token to match placement encoding dimension
        summary_projection = Dense(64, activation='relu')(summary_token)
        x = Concatenate()([x, summary_projection])
    
    # Add FiLM conditioning if requested
    if use_film and user_meta_input is not None:
        film_layer = AnatomyConditionedFiLM(output_dim=x.shape[-1])
        x = film_layer(x, user_meta_input)
        print(f"film: on, gamma/beta shape = ({x.shape[-1]}, {x.shape[-1]})")
    
    # Final classifier
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Create outputs based on hierarchical setting
    if hierarchical:
        fine_outputs, coarse_outputs, _ = create_hierarchical_head(x, num_classes, hier_lambda)
        outputs = [fine_outputs, coarse_outputs]
    else:
        outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=model_inputs, outputs=outputs)
    
    # Create loss function
    if y_train is not None:
        loss_function = create_loss_function(loss_type, y_train, label_smoothing)
    else:
        loss_function = 'categorical_crossentropy'
    
    # Compile model
    if hierarchical:
        # For hierarchical model, we need custom loss that combines fine and coarse
        def hierarchical_loss(y_true, y_pred):
            # Handle tensor unpacking safely
            if isinstance(y_pred, list):
                fine_pred, coarse_pred = y_pred[0], y_pred[1]
            else:
                # If not a list, assume it's the fine prediction
                fine_pred, coarse_pred = y_pred, y_pred
            
            if isinstance(y_true, list):
                fine_true, coarse_true = y_true[0], y_true[1]
            else:
                # If not a list, assume it's the fine target
                fine_true, coarse_true = y_true, y_true
            
            # Fine loss
            fine_loss = tf.keras.losses.categorical_crossentropy(fine_true, fine_pred)
            
            # Coarse loss
            coarse_loss = tf.keras.losses.categorical_crossentropy(coarse_true, coarse_pred)
            
            # Combined loss - use tf.constant to avoid symbolic tensor issues
            total_loss = fine_loss + tf.constant(hier_lambda, dtype=tf.float32) * coarse_loss
            return total_loss
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=hierarchical_loss,
            metrics={'fine_output': 'accuracy', 'coarse_output': 'accuracy'}
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=loss_function,
            metrics=['accuracy']
        )
    
    return model


def build_binary_weighted_fusion(input_shape, num_classes, game_name, featureset='base', use_film=False,
                               loss_type='ce', label_smoothing=0.0, y_train=None,
                               adjacency_mode='fixed', gate_alpha_init=0.1,
                               hierarchical=False, hier_lambda=0.3, activity_names=None):
    """Weighted fusion model: separate encoders + learned linear weights with semantic features support"""
    
    # Get activity names for loss functions (define early to avoid scope issues)
    if activity_names is None:
        from games.STM.config import VALID_ACTIVITIES
        activity_names = list(VALID_ACTIVITIES)
    
    print(f"🔧 Building binary_weighted_fusion with input_shape: {input_shape}")
    
    # Handle different input shapes for binary-only data
    if len(input_shape) == 3:
        # Input shape: (time_steps, placements, features) = (100, 5, 1)
        time_steps, num_placements, features_per_placement = input_shape
        has_summary_token = False
        summary_token_dim = 0
    elif len(input_shape) == 2:
        # Check if we have summary token (T+1, P+semantic_features)
        # For semantic features: (101, 19) = (100+1, 5+14) or (101, 99) = (100+1, 5+94)
        has_summary_token = input_shape[0] > 100  # Assuming window_size=100
        if has_summary_token:
            # Extract summary token and main data
            # input_shape[1] = 19 (5+14) or 99 (5+94) = 5 placements + semantic features
            summary_token_dim = input_shape[1] - 5  # 14 or 94 semantic features
            time_steps = input_shape[0] - 1  # 100 time steps (remove summary token)
            num_placements = 5
            features_per_placement = 1
        else:
            # Fallback for 2D input without summary token
            time_steps, num_placements = input_shape
            features_per_placement = 1
            summary_token_dim = 0
    else:
        raise ValueError(f"Unsupported input shape for binary_weighted_fusion: {input_shape}")
    
    print(f"   📊 Time steps: {time_steps}, Placements: {num_placements}, Features per placement: {features_per_placement}")
    
    # For binary-only data, we always have 1 feature per placement
    is_continuous = features_per_placement > 1
    
    # Shared temporal encoder for binary sequences
    def create_placement_encoder():
        # For binary features: (T, 1) - single binary feature per placement
        encoder_input = Input(shape=(time_steps, 1))  # (T, 1) for single placement
        
        # Small temporal encoder for binary sequences
        x = Conv1D(16, 3, activation='relu', padding='same')(encoder_input)
        x = BatchNormalization()(x)
        x = Conv1D(32, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.GRU(16, return_sequences=False)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        return Model(encoder_input, x)
    
    shared_encoder = create_placement_encoder()
    
    # Main input: (B, T, P) or (B, T+1, P+semantic_features)
    inputs = Input(shape=input_shape)
    
    # User metadata input for FiLM conditioning
    if use_film:
        user_meta_input = Input(shape=(9,), name='user_metadata')  # 9 features: 5 numeric + 2 dom_hand + 2 dom_foot
        model_inputs = [inputs, user_meta_input]
    else:
        user_meta_input = None
        model_inputs = [inputs]
    
    if has_summary_token:
        # Extract summary token and main data
        summary_token = tf.keras.layers.Lambda(lambda x: x[:, 0, 5:])(inputs)  # (B, semantic_features)
        main_data = tf.keras.layers.Lambda(lambda x: x[:, 1:, :5])(inputs)     # (B, T, 5)
    else:
        summary_token = None
        main_data = inputs
    
    # Process each placement separately
    placement_outputs = []
    for i in range(num_placements):
        # For binary features: extract single placement: (B, T, 1)
        if has_summary_token:
            # For semantic features: main_data shape is (B, T, P+semantic_features)
            # Extract placement i from the first 5 features (placements)
            placement_input = tf.keras.layers.Lambda(
                lambda x, idx=i: x[:, :, idx:idx+1], 
                output_shape=(time_steps, 1)
            )(main_data)
        else:
            # Original format: (B, T, P, C)
            placement_input = tf.keras.layers.Lambda(
                lambda x, idx=i: x[:, :, idx, :], 
                output_shape=(time_steps, features_per_placement)
            )(main_data)
        
        placement_encoded = shared_encoder(placement_input)
        placement_outputs.append(placement_encoded)
    
    # Stack placement outputs for weighted fusion: (B, P, 32)
    stacked_outputs = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(placement_outputs)
    
    # Add placement semantics if requested
    if 'semantics' in featureset:
        graph_semantics = GraphSemantics(
            hidden_dim=64, 
            adjacency_mode=adjacency_mode,
            gate_alpha_init=gate_alpha_init
        )
        stacked_outputs = graph_semantics(stacked_outputs)
        # Ensure the output has a known shape for downstream layers
        stacked_outputs = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(stacked_outputs)
        # Force shape inference for downstream layers
        stacked_outputs = tf.keras.layers.Lambda(lambda x: tf.reshape(x, tf.shape(x)))(stacked_outputs)
    
    # Global pooling across placements
    x = GlobalAveragePooling1D()(stacked_outputs)
    
    # Add summary token if available
    if summary_token is not None:
        # Project summary token to match placement encoding dimension
        summary_projection = Dense(64, activation='relu')(summary_token)
        x = Concatenate()([x, summary_projection])
    
    # Add FiLM conditioning if requested
    if use_film and user_meta_input is not None:
        film_layer = AnatomyConditionedFiLM(output_dim=x.shape[-1])
        x = film_layer(x, user_meta_input)
        print(f"film: on, gamma/beta shape = ({x.shape[-1]}, {x.shape[-1]})")
    
    # Learnable fusion weights
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Create outputs based on hierarchical setting
    if hierarchical:
        fine_outputs, coarse_outputs, _ = create_hierarchical_head(x, num_classes, hier_lambda)
        outputs = [fine_outputs, coarse_outputs]
    else:
        outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=model_inputs, outputs=outputs)
    
    # Create loss function
    if y_train is not None:
        loss_function = create_loss_function(loss_type, y_train, label_smoothing)
    else:
        loss_function = 'categorical_crossentropy'
    
    # Compile model
    if hierarchical:
        # For hierarchical model, we need custom loss that combines fine and coarse
        def hierarchical_loss(y_true, y_pred):
            # Handle tensor unpacking safely
            if isinstance(y_pred, list):
                fine_pred, coarse_pred = y_pred[0], y_pred[1]
            else:
                # If not a list, assume it's the fine prediction
                fine_pred, coarse_pred = y_pred, y_pred
            
            if isinstance(y_true, list):
                fine_true, coarse_true = y_true[0], y_true[1]
            else:
                # If not a list, assume it's the fine target
                fine_true, coarse_true = y_true, y_true
            
            # Fine loss
            fine_loss = tf.keras.losses.categorical_crossentropy(fine_true, fine_pred)
            
            # Coarse loss
            coarse_loss = tf.keras.losses.categorical_crossentropy(coarse_true, coarse_pred)
            
            # Combined loss - use tf.constant to avoid symbolic tensor issues
            total_loss = fine_loss + tf.constant(hier_lambda, dtype=tf.float32) * coarse_loss
            return total_loss
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=hierarchical_loss,
            metrics={'fine_output': 'accuracy', 'coarse_output': 'accuracy'}
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=loss_function,
            metrics=['accuracy']
        )
    
    return model


def build_binary_sensor_token_attention(input_shape, num_classes, game_name, featureset='base', use_film=False,
                                      loss_type='ce', label_smoothing=0.0, y_train=None,
                                      adjacency_mode='fixed', gate_alpha_init=0.1,
                                      hierarchical=False, hier_lambda=0.3, activity_names=None):
    """Sensor-token attention: treat placements as tokens with attention and semantic features support"""
    
    # Get activity names for loss functions (define early to avoid scope issues)
    if activity_names is None:
        from games.STM.config import VALID_ACTIVITIES
        activity_names = list(VALID_ACTIVITIES)
    
    print(f"🔧 Building binary_sensor_token_attention with input_shape: {input_shape}")
    
    # Handle different input shapes for binary-only data
    if len(input_shape) == 3:
        # Input shape: (time_steps, placements, features) = (100, 5, 1)
        time_steps, num_placements, num_features_per_placement = input_shape
        has_continuous_features = num_features_per_placement > 1
        has_summary_token = False
        summary_token_dim = 0
    elif len(input_shape) == 2:
        # Check if we have summary token (T+1, P+semantic_features)
        # For semantic features: (101, 19) = (100+1, 5+14) or (101, 99) = (100+1, 5+94)
        has_summary_token = input_shape[0] > 100  # Assuming window_size=100
        if has_summary_token:
            # Extract summary token and main data
            # input_shape[1] = 19 (5+14) or 99 (5+94) = 5 placements + semantic features
            summary_token_dim = input_shape[1] - 5  # 14 or 94 semantic features
            time_steps = input_shape[0] - 1  # 100 time steps (remove summary token)
            num_placements = 5
            num_features_per_placement = 1
        else:
            # Fallback for 2D input without summary token
            time_steps, num_placements = input_shape
            num_features_per_placement = 1
            summary_token_dim = 0
        has_continuous_features = False
    else:
        raise ValueError(f"Unsupported input shape for binary_sensor_token_attention: {input_shape}")
    
    print(f"   📊 Time steps: {time_steps}, Placements: {num_placements}, Features per placement: {num_features_per_placement}")
    print(f"   🔧 Binary features detected: {num_placements} placements")
    d_model = 64  # Small model as specified
    
    # Per-placement temporal encoder for binary sequences
    def create_placement_encoder():
        # For binary features: (T, 1) - single binary feature per placement
        encoder_input = Input(shape=(time_steps, 1))
        
        # Small temporal encoder for binary sequences
        x = Conv1D(32, 5, activation='relu', padding='same')(encoder_input)
        x = BatchNormalization()(x)
        x = Conv1D(64, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.GRU(32, return_sequences=False)(x)
        x = Dense(d_model, activation='relu')(x)  # Output d_model dimensions
        
        return Model(encoder_input, x)
    
    # Create separate encoders for each placement to avoid Lambda issues
    placement_encoders = []
    for i in range(num_placements):
        placement_encoders.append(create_placement_encoder())
    
    # Main input: (B, T, P) or (B, T+1, P+semantic_features)
    inputs = Input(shape=input_shape)
    
    # User metadata input for FiLM conditioning
    if use_film:
        user_meta_input = Input(shape=(9,), name='user_metadata')  # 9 features: 5 numeric + 2 dom_hand + 2 dom_foot
        model_inputs = [inputs, user_meta_input]
    else:
        user_meta_input = None
        model_inputs = [inputs]
    
    if has_summary_token:
        # Extract summary token and main data
        summary_token = tf.keras.layers.Lambda(lambda x: x[:, 0, 5:])(inputs)  # (B, semantic_features)
        main_data = tf.keras.layers.Lambda(lambda x: x[:, 1:, :5])(inputs)     # (B, T, 5)
    else:
        summary_token = None
        main_data = inputs
    
    # Process each placement to get token embeddings: (B, P, d_model)
    placement_tokens = []
    
    # For binary features: extract single feature for each placement
    for i in range(num_placements):
        if has_summary_token:
            # For semantic features: main_data shape is (B, T, P+semantic_features)
            # Extract placement i from the first 5 features (placements)
            placement_input = tf.keras.layers.Lambda(
                lambda x, idx=i: x[:, :, idx:idx+1], 
                output_shape=(time_steps, 1)
            )(main_data)
        else:
            # Original format: (B, T, P, C)
            placement_input = tf.keras.layers.Lambda(
                lambda x, idx=i: x[:, :, idx, :], 
                output_shape=(time_steps, features_per_placement)
            )(main_data)
        placement_encoded = placement_encoders[i](placement_input)
        placement_tokens.append(placement_encoded)
    
    # Stack tokens: (B, P, d_model)
    token_embeddings = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(placement_tokens)
    
    # Add placement semantics if requested
    if 'semantics' in featureset:
        # Support for hybrid adjacency
        if adjacency_mode == 'hybrid':
            from ..features.enhanced_semantic_features import create_hybrid_adjacency
            hybrid_adj = create_hybrid_adjacency(num_placements, fixed_weight=0.7, learned_weight=0.3)
            print(f"   🔧 Using hybrid adjacency matrix")
            
            graph_semantics = GraphSemantics(
                hidden_dim=d_model, 
                adjacency_mode='fixed',  # Use fixed mode but with custom adjacency
                gate_alpha_init=gate_alpha_init,
                custom_adjacency=hybrid_adj
            )
        else:
            graph_semantics = GraphSemantics(
                hidden_dim=d_model, 
                adjacency_mode=adjacency_mode,
                gate_alpha_init=gate_alpha_init
            )
        
        token_embeddings = graph_semantics(token_embeddings)
        # Ensure the output has a known shape for downstream layers
        token_embeddings = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(token_embeddings)
        # Force shape inference for MultiHeadAttention
        token_embeddings = tf.keras.layers.Lambda(lambda x: tf.reshape(x, tf.shape(x)))(token_embeddings)
    
    # Add positional encoding for placement tokens
    pos_encoding = ImprovedPositionalEncoding(num_placements, d_model)(token_embeddings)
    
    # Multi-head attention across placement tokens
    attention_output = MultiHeadAttention(
        num_heads=2, key_dim=d_model//2, dropout=0.1
    )(pos_encoding, pos_encoding)
    
    x = LayerNormalization(epsilon=1e-6)(pos_encoding + attention_output)
    
    # Feed-forward network
    ffn = Dense(d_model * 2, activation='relu')(x)
    ffn = Dropout(0.1)(ffn)
    ffn = Dense(d_model)(ffn)
    x = LayerNormalization(epsilon=1e-6)(x + ffn)
    
    # Global pooling across placement tokens
    x = GlobalAveragePooling1D()(x)
    
    # Add summary token if available
    if summary_token is not None:
        # Project summary token to match placement encoding dimension
        summary_projection = Dense(64, activation='relu')(summary_token)
        x = Concatenate()([x, summary_projection])
    
    # Add FiLM conditioning if requested
    if use_film and user_meta_input is not None:
        film_layer = AnatomyConditionedFiLM(output_dim=x.shape[-1])
        x = film_layer(x, user_meta_input)
        print(f"film: on, gamma/beta shape = ({x.shape[-1]}, {x.shape[-1]})")
    
    # Final classifier
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Create outputs based on hierarchical setting
    if hierarchical:
        fine_outputs, coarse_outputs, _ = create_hierarchical_head(x, num_classes, hier_lambda)
        outputs = [fine_outputs, coarse_outputs]
    else:
        outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=model_inputs, outputs=outputs)
    
    # Create loss function
    if y_train is not None:
        loss_function = create_loss_function(loss_type, y_train, label_smoothing)
    else:
        loss_function = 'categorical_crossentropy'
    
    # Compile model
    if hierarchical:
        # For hierarchical model, we need custom loss that combines fine and coarse
        def hierarchical_loss(y_true, y_pred):
            # Handle tensor unpacking safely
            if isinstance(y_pred, list):
                fine_pred, coarse_pred = y_pred[0], y_pred[1]
            else:
                # If not a list, assume it's the fine prediction
                fine_pred, coarse_pred = y_pred, y_pred
            
            if isinstance(y_true, list):
                fine_true, coarse_true = y_true[0], y_true[1]
            else:
                # If not a list, assume it's the fine target
                fine_true, coarse_true = y_true, y_true
            
            # Fine loss
            fine_loss = tf.keras.losses.categorical_crossentropy(fine_true, fine_pred)
            
            # Coarse loss
            coarse_loss = tf.keras.losses.categorical_crossentropy(coarse_true, coarse_pred)
            
            # Combined loss - use tf.constant to avoid symbolic tensor issues
            total_loss = fine_loss + tf.constant(hier_lambda, dtype=tf.float32) * coarse_loss
            return total_loss
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=hierarchical_loss,
            metrics={'fine_output': 'accuracy', 'coarse_output': 'accuracy'}
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=loss_function,
            metrics=['accuracy']
        )
    
    return model 

def build_continuous_late_fusion(input_shape, num_classes, game_name, featureset='base', use_film=False,
                               loss_type='ce', label_smoothing=0.0, y_train=None, 
                               adjacency_mode='fixed', gate_alpha_init=0.1,
                               hierarchical=False, hier_lambda=0.3, activity_names=None):
    """Late fusion model for continuous features: separate encoders per placement with proper feature handling"""
    
    # Get activity names for loss functions (define early to avoid scope issues)
    if activity_names is None:
        from games.STM.config import VALID_ACTIVITIES
        activity_names = list(VALID_ACTIVITIES)
    
    # Check if we have summary token (T+1, P+semantic_features)
    has_summary_token = len(input_shape) == 2 and input_shape[0] > 100  # Assuming window_size=100
    if has_summary_token:
        # Extract summary token and main data
        summary_token_dim = input_shape[1] - 5  # Assuming 5 placements
        main_input_shape = (input_shape[0] - 1, 5)  # Remove summary token, keep only placements
    else:
        main_input_shape = input_shape
        summary_token_dim = 0
    
    # Handle 4D input shape (None, 100, 5, 7) - this is the actual data shape
    # 100 = time steps, 5 = placements, 7 = features per placement
    if len(main_input_shape) == 3:
        # Input shape is (100, 5, 7) - this is correct
        num_placements = main_input_shape[1]  # 5 placements
        features_per_placement = main_input_shape[2]  # 7 features per placement
    else:
        # Fallback
        num_placements = main_input_shape[1] if len(main_input_shape) > 1 else 1
        features_per_placement = main_input_shape[0] if len(main_input_shape) > 0 else 1
    
    # Shared temporal encoder for continuous features
    def create_placement_encoder():
        # For continuous features: (T, features_per_placement)
        encoder_input = Input(shape=(main_input_shape[0], features_per_placement))
        
        # Temporal encoder for continuous features
        x = Conv1D(32, 3, activation='relu', padding='same')(encoder_input)
        x = BatchNormalization()(x)
        x = Conv1D(64, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.GRU(32, return_sequences=False)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        return Model(encoder_input, x)
    
    # Create shared encoder
    shared_encoder = create_placement_encoder()
    
    # Main input: (B, T, P, F) where T=time, P=placements, F=features
    inputs = Input(shape=input_shape)
    
    # User metadata input for FiLM conditioning
    if use_film:
        user_meta_input = Input(shape=(9,), name='user_metadata')  # 9 features: 5 numeric + 2 dom_hand + 2 dom_foot
        model_inputs = [inputs, user_meta_input]
    else:
        user_meta_input = None
        model_inputs = [inputs]
    
    if has_summary_token:
        # Extract summary token and main data
        summary_token = tf.keras.layers.Lambda(lambda x: x[:, 0, 5:])(inputs)  # (B, semantic_features)
        main_data = tf.keras.layers.Lambda(lambda x: x[:, 1:, :5])(inputs)     # (B, T, 5)
    else:
        summary_token = None
        main_data = inputs
    
    # Process each placement separately
    placement_outputs = []
    for i in range(num_placements):
        # Extract single placement: (B, T, F) where F=features_per_placement
        placement_input = tf.keras.layers.Lambda(lambda x, idx=i: x[:, :, idx, :])(main_data)
        
        # Encode placement
        placement_encoded = shared_encoder(placement_input)
        placement_outputs.append(placement_encoded)
    
    # Stack placement outputs: (B, P, 64)
    stacked_outputs = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(placement_outputs)
    
    # Add placement semantics if requested
    if 'semantics' in featureset:
        graph_semantics = GraphSemantics(
            hidden_dim=64, 
            adjacency_mode=adjacency_mode,
            gate_alpha_init=gate_alpha_init
        )
        stacked_outputs = graph_semantics(stacked_outputs)
        # Ensure the output has a known shape for downstream layers
        stacked_outputs = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(stacked_outputs)
        # Force shape inference for downstream layers
        stacked_outputs = tf.keras.layers.Lambda(lambda x: tf.reshape(x, tf.shape(x)))(stacked_outputs)
    
    # Global pooling across placements
    x = GlobalAveragePooling1D()(stacked_outputs)
    
    # Add summary token if available
    if summary_token is not None:
        # Project summary token to match placement encoding dimension
        summary_projection = Dense(64, activation='relu')(summary_token)
        x = Concatenate()([x, summary_projection])
    
    # Add FiLM conditioning if requested
    if use_film and user_meta_input is not None:
        film_layer = AnatomyConditionedFiLM(output_dim=x.shape[-1])
        x = film_layer(x, user_meta_input)
        print(f"film: on, gamma/beta shape = ({x.shape[-1]}, {x.shape[-1]})")
    
    # Final classifier
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Create outputs based on hierarchical setting
    if hierarchical:
        fine_outputs, coarse_outputs, _ = create_hierarchical_head(x, num_classes, hier_lambda)
        outputs = [fine_outputs, coarse_outputs]
    else:
        outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=model_inputs, outputs=outputs)
    
    # Create loss function
    if y_train is not None:
        loss_function = create_loss_function(loss_type, y_train, label_smoothing)
    else:
        loss_function = 'categorical_crossentropy'
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss_function,
        metrics=['accuracy']
    )
    
    return model

def build_continuous_weighted_fusion(input_shape, num_classes, game_name, featureset='base', use_film=False,
                               loss_type='ce', label_smoothing=0.0, y_train=None, 
                               adjacency_mode='fixed', gate_alpha_init=0.1,
                               hierarchical=False, hier_lambda=0.3, activity_names=None):
    """Weighted fusion model for continuous features: separate encoders + learned linear weights"""
    
    # Get activity names for loss functions (define early to avoid scope issues)
    if activity_names is None:
        from games.STM.config import VALID_ACTIVITIES
        activity_names = list(VALID_ACTIVITIES)
    
    # Check if we have summary token (T+1, P+semantic_features)
    has_summary_token = len(input_shape) == 2 and input_shape[0] > 100  # Assuming window_size=100
    if has_summary_token:
        # Extract summary token and main data
        summary_token_dim = input_shape[1] - 5  # Assuming 5 placements
        main_input_shape = (input_shape[0] - 1, 5)  # Remove summary token, keep only placements
    else:
        main_input_shape = input_shape
        summary_token_dim = 0
    
    # Handle 4D input shape (None, 100, 5, 7) - this is the actual data shape
    # 100 = time steps, 5 = placements, 7 = features per placement
    if len(main_input_shape) == 3:
        # Input shape is (100, 5, 7) - this is correct
        num_placements = main_input_shape[1]  # 5 placements
        features_per_placement = main_input_shape[2]  # 7 features per placement
    else:
        # Fallback
        num_placements = main_input_shape[1] if len(main_input_shape) > 1 else 1
        features_per_placement = main_input_shape[0] if len(main_input_shape) > 0 else 1
    
    # Shared temporal encoder for continuous features
    def create_placement_encoder():
        # For continuous features: (T, features_per_placement)
        encoder_input = Input(shape=(main_input_shape[0], features_per_placement))
        
        # Temporal encoder for continuous features
        x = Conv1D(32, 3, activation='relu', padding='same')(encoder_input)
        x = BatchNormalization()(x)
        x = Conv1D(64, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.GRU(32, return_sequences=False)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        return Model(encoder_input, x)
    
    # Create separate encoders for each placement
    placement_encoders = []
    for i in range(num_placements):
        placement_encoders.append(create_placement_encoder())
    
    # Main input: (B, T, P, F) where T=time, P=placements, F=features
    inputs = Input(shape=input_shape)
    
    # User metadata input for FiLM conditioning
    if use_film:
        user_meta_input = Input(shape=(9,), name='user_metadata')  # 9 features: 5 numeric + 2 dom_hand + 2 dom_foot
        model_inputs = [inputs, user_meta_input]
    else:
        user_meta_input = None
        model_inputs = [inputs]
    
    if has_summary_token:
        # Extract summary token and main data
        summary_token = tf.keras.layers.Lambda(lambda x: x[:, 0, 5:])(inputs)  # (B, semantic_features)
        main_data = tf.keras.layers.Lambda(lambda x: x[:, 1:, :5])(inputs)     # (B, T, 5)
    else:
        summary_token = None
        main_data = inputs
    
    # Process each placement separately
    placement_outputs = []
    for i in range(num_placements):
        # Extract single placement: (B, T, F) where F=features_per_placement
        placement_input = tf.keras.layers.Lambda(lambda x, idx=i: x[:, :, idx, :])(main_data)
        
        # Encode placement
        placement_encoded = placement_encoders[i](placement_input)
        placement_outputs.append(placement_encoded)
    
    # Stack placement outputs: (B, P, 64)
    stacked_outputs = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(placement_outputs)
    
    # Add placement semantics if requested
    if 'semantics' in featureset:
        graph_semantics = GraphSemantics(
            hidden_dim=64, 
            adjacency_mode=adjacency_mode,
            gate_alpha_init=gate_alpha_init
        )
        stacked_outputs = graph_semantics(stacked_outputs)
        # Ensure the output has a known shape for downstream layers
        stacked_outputs = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(stacked_outputs)
        # Force shape inference for downstream layers
        stacked_outputs = tf.keras.layers.Lambda(lambda x: tf.reshape(x, tf.shape(x)))(stacked_outputs)
    
    # Global pooling across placements
    x = GlobalAveragePooling1D()(stacked_outputs)
    
    # Add summary token if available
    if summary_token is not None:
        # Project summary token to match placement encoding dimension
        summary_projection = Dense(64, activation='relu')(summary_token)
        x = Concatenate()([x, summary_projection])
    
    # Add FiLM conditioning if requested
    if use_film and user_meta_input is not None:
        film_layer = AnatomyConditionedFiLM(output_dim=x.shape[-1])
        x = film_layer(x, user_meta_input)
        print(f"film: on, gamma/beta shape = ({x.shape[-1]}, {x.shape[-1]})")
    
    # Learnable fusion weights
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Create outputs based on hierarchical setting
    if hierarchical:
        fine_outputs, coarse_outputs, _ = create_hierarchical_head(x, num_classes, hier_lambda)
        outputs = [fine_outputs, coarse_outputs]
    else:
        outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=model_inputs, outputs=outputs)
    
    # Create loss function
    if y_train is not None:
        loss_function = create_loss_function(loss_type, y_train, label_smoothing)
    else:
        loss_function = 'categorical_crossentropy'
    
    # Compile model
    if hierarchical:
        # For hierarchical model, we need custom loss that combines fine and coarse
        def hierarchical_loss(y_true, y_pred):
            # Handle tensor unpacking safely
            if isinstance(y_pred, list):
                fine_pred, coarse_pred = y_pred[0], y_pred[1]
            else:
                # If not a list, assume it's the fine prediction
                fine_pred, coarse_pred = y_pred, y_pred
            
            if isinstance(y_true, list):
                fine_true, coarse_true = y_true[0], y_true[1]
            else:
                # If not a list, assume it's the fine target
                fine_true, coarse_true = y_true, y_true
            
            # Fine loss
            fine_loss = tf.keras.losses.categorical_crossentropy(fine_true, fine_pred)
            
            # Coarse loss
            coarse_loss = tf.keras.losses.categorical_crossentropy(coarse_true, coarse_pred)
            
            # Combined loss - use tf.constant to avoid symbolic tensor issues
            total_loss = fine_loss + tf.constant(hier_lambda, dtype=tf.float32) * coarse_loss
            return total_loss
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=hierarchical_loss,
            metrics={'fine_output': 'accuracy', 'coarse_output': 'accuracy'}
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=loss_function,
            metrics=['accuracy']
        )
    
    return model