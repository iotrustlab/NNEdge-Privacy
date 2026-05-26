"""
FiLM (Feature-wise Linear Modulation) for anatomy-conditioned normalization.

Uses per-user anatomical measurements to condition the model through
learnable scaling and shifting parameters.
"""

import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Optional, Tuple
import numpy as np


class FiLMMLP(keras.layers.Layer):
    """
    MLP that generates FiLM parameters (γ, β) from user metadata.
    
    Takes anatomical measurements as input and outputs scaling and shifting
    parameters for feature-wise linear modulation.
    """
    
    def __init__(self, output_dim: int, hidden_dims: List[int] = [64, 32], 
                 dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build MLP layers
        self.mlp_layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            self.mlp_layers.append(keras.layers.Dense(
                hidden_dim, 
                activation='relu',
                name=f"film_mlp_{i}"
            ))
            self.mlp_layers.append(keras.layers.Dropout(dropout_rate))
        
        # Output layers for gamma and beta
        self.gamma_layer = keras.layers.Dense(output_dim, name="film_gamma")
        self.beta_layer = keras.layers.Dense(output_dim, name="film_beta")
        
        # Layer normalization for input
        self.input_norm = keras.layers.LayerNormalization(name="film_input_norm")
    
    def call(self, user_meta: tf.Tensor, training: Optional[bool] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generate FiLM parameters from user metadata.
        
        Args:
            user_meta: Tensor of shape (batch_size, meta_dim) with user measurements
            training: Training flag for dropout
        
        Returns:
            Tuple of (gamma, beta) tensors, each of shape (batch_size, output_dim)
        """
        # Normalize input metadata
        x = self.input_norm(user_meta)
        
        # Pass through MLP
        for layer in self.mlp_layers:
            x = layer(x, training=training)
        
        # Generate gamma and beta
        gamma = self.gamma_layer(x)  # (batch_size, output_dim)
        beta = self.beta_layer(x)    # (batch_size, output_dim)
        
        return gamma, beta
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate
        })
        return config


class FiLMLayer(keras.layers.Layer):
    """
    Feature-wise Linear Modulation layer.
    
    Applies learned scaling (γ) and shifting (β) parameters to input features.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs: tf.Tensor, gamma: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
        """
        Apply FiLM modulation to input features.
        
        Args:
            inputs: Input tensor of any shape
            gamma: Scaling parameters, must be broadcastable to inputs
            beta: Shifting parameters, must be broadcastable to inputs
        
        Returns:
            Modulated tensor: gamma * inputs + beta
        """
        # Ensure gamma and beta have the right shape for broadcasting
        # If inputs is (batch, seq_len, features), gamma/beta should be (batch, 1, features)
        if len(inputs.shape) > 2:
            # Add dimensions for broadcasting
            for _ in range(len(inputs.shape) - len(gamma.shape)):
                gamma = tf.expand_dims(gamma, axis=1)
                beta = tf.expand_dims(beta, axis=1)
        
        # Apply modulation
        modulated = gamma * inputs + beta
        
        return modulated
    
    def get_config(self):
        return super().get_config()


class AnatomyConditionedFiLM(keras.layers.Layer):
    """
    Complete anatomy-conditioned FiLM layer.
    
    Combines the MLP for parameter generation with the FiLM modulation.
    """
    
    def __init__(self, output_dim: int, hidden_dims: List[int] = [64, 32], 
                 dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # FiLM MLP for parameter generation
        self.film_mlp = FiLMMLP(output_dim, hidden_dims, dropout_rate)
        
        # FiLM layer for modulation
        self.film_layer = FiLMLayer()
    
    def call(self, inputs: tf.Tensor, user_meta: tf.Tensor, 
             training: Optional[bool] = None) -> tf.Tensor:
        """
        Apply anatomy-conditioned FiLM modulation.
        
        Args:
            inputs: Input tensor to be modulated
            user_meta: User metadata tensor
            training: Training flag
        
        Returns:
            Modulated tensor
        """
        # Generate FiLM parameters from user metadata
        gamma, beta = self.film_mlp(user_meta, training=training)
        
        # Apply FiLM modulation
        modulated = self.film_layer(inputs, gamma, beta)
        
        return modulated
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate
        })
        return config


# User metadata schema and processing
USER_META_FIELDS = [
    'height_in', 'leg_len_in', 'arm_len_in', 'torso_len_in', 
    'shoe_size', 'dom_hand_right', 'dom_hand_left',
    'dom_foot_right', 'dom_foot_left'
]

USER_META_DIM = len(USER_META_FIELDS)


def create_user_meta_schema() -> Dict[str, str]:
    """Create schema for user metadata CSV."""
    return {
        'user_id': 'int',
        'height_in': 'float',
        'leg_len_in': 'float', 
        'arm_len_in': 'float',
        'torso_len_in': 'float',
        'shoe_size': 'float',
        'dom_hand_right': 'int',  # 0 or 1
        'dom_hand_left': 'int',   # 0 or 1
        'dom_foot_right': 'int',  # 0 or 1
        'dom_foot_left': 'int'    # 0 or 1
    }


def normalize_user_meta(user_meta: np.ndarray) -> np.ndarray:
    """
    Normalize user metadata using z-scoring.
    
    Args:
        user_meta: Array of shape (num_users, meta_dim)
    
    Returns:
        Normalized array
    """
    mean = np.mean(user_meta, axis=0, keepdims=True)
    std = np.std(user_meta, axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)  # Avoid division by zero
    
    normalized = (user_meta - mean) / std
    return normalized


def create_film_model(input_shape: Tuple[int, int], 
                     meta_dim: int = USER_META_DIM,
                     hidden_dims: List[int] = [64, 32]) -> keras.Model:
    """
    Create a standalone FiLM model for testing.
    
    Args:
        input_shape: (seq_len, feature_dim)
        meta_dim: Dimension of user metadata
        hidden_dims: Hidden dimensions for FiLM MLP
    
    Returns:
        Keras model
    """
    inputs = keras.Input(shape=input_shape)
    user_meta = keras.Input(shape=(meta_dim,))
    
    film_layer = AnatomyConditionedFiLM(
        output_dim=input_shape[-1],
        hidden_dims=hidden_dims
    )
    
    outputs = film_layer(inputs, user_meta)
    
    return keras.Model(
        inputs=[inputs, user_meta], 
        outputs=outputs, 
        name="film_test"
    )


# Utility functions for user metadata
def get_user_meta_fields() -> List[str]:
    """Get list of user metadata field names."""
    return USER_META_FIELDS.copy()


def get_user_meta_dim() -> int:
    """Get dimension of user metadata."""
    return USER_META_DIM


def validate_user_meta(user_meta: np.ndarray) -> bool:
    """
    Validate user metadata array.
    
    Args:
        user_meta: Array to validate
    
    Returns:
        True if valid, False otherwise
    """
    if user_meta.shape[-1] != USER_META_DIM:
        return False
    
    # Check that one-hot encoded fields sum to 1
    dom_hand_sum = user_meta[:, 5] + user_meta[:, 6]  # dom_hand_right + dom_hand_left
    dom_foot_sum = user_meta[:, 7] + user_meta[:, 8]  # dom_foot_right + dom_foot_left
    
    if not np.allclose(dom_hand_sum, 1.0) or not np.allclose(dom_foot_sum, 1.0):
        return False
    
    return True
