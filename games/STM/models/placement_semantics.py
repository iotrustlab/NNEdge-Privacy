"""
Placement semantics for STM games.

Implements learnable placement embeddings, metadata flags, and graph attention
for modeling relationships between sensor placements.
"""

import tensorflow as tf
import keras
import numpy as np
from typing import Optional


# Placement metadata flags
PLACEMENT_META = {
    "left-wrist":  {"left": 1, "right": 0, "upper": 1, "lower": 0, "core": 0},
    "right-wrist": {"left": 0, "right": 1, "upper": 1, "lower": 0, "core": 0},
    "left-ankle":  {"left": 1, "right": 0, "upper": 0, "lower": 1, "core": 0},
    "right-ankle": {"left": 0, "right": 1, "upper": 0, "lower": 1, "core": 0},
    "right-pocket": {"left": 0, "right": 1, "upper": 0, "lower": 1, "core": 1}
}

# Graph edges between placements
PLACEMENT_EDGES = [
    ("left-wrist", "right-wrist"),      # Symmetric wrists
    ("left-ankle", "right-ankle"),      # Symmetric ankles
    ("left-wrist", "left-ankle"),       # Ipsilateral wrist-ankle
    ("right-wrist", "right-ankle"),     # Ipsilateral wrist-ankle
    ("right-pocket", "left-wrist"),     # Pocket to wrist
    ("right-pocket", "right-wrist"),    # Pocket to wrist
    ("right-pocket", "left-ankle"),     # Pocket to ankle
    ("right-pocket", "right-ankle"),    # Pocket to ankle
]

# Placement order (must match data loading order)
PLACEMENT_ORDER = ["left-wrist", "right-wrist", "right-pocket", "right-ankle", "left-ankle"]


class GraphSemantics(keras.layers.Layer):
    """
    Residual-gated GraphSemantics layer with adjacency ablations.
    
    Inputs: x of shape (B,P,D) or (B,T,P,D)
    Outputs: H of shape (B,P,H) or (B,T,P,H) where H=64
    
    Uses residual connection: H = X + α * dropout(f(X), rate=0.1)
    where f(X) is the graph convolution and α is a trainable scalar gate.
    """
    
    def __init__(self, hidden_dim=64, adjacency_mode='fixed', gate_alpha_init=0.1, 
                 dropout_rate=0.1, custom_adjacency=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.adjacency_mode = adjacency_mode
        self.gate_alpha_init = gate_alpha_init
        self.dropout_rate = dropout_rate
        self.custom_adjacency = custom_adjacency
        
        # Trainable scalar gate α ≥ 0
        self.alpha = None  # Will be initialized in build()
        
        # Adjacency matrix - will be set based on mode
        self.A_norm = None  # Will be initialized in build()
        
        # Parameters
        self.W_self = None  # Will be initialized in build()
        self.W_neigh = None  # Will be initialized in build()
        self.b = None  # Will be initialized in build()
        
        # Learnable adjacency matrix (for 'learned' mode)
        self.learned_adjacency = None  # Will be initialized in build()
        
        # Fallback layer for non-placement inputs
        self.fallback_dense = None  # Will be initialized in build()
        
        # Input projection layer for residual connection
        self.input_projection = None  # Will be initialized in build()
        
        # Dropout layer
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def build(self, input_shape):
        # input_shape is (B, P, D) or (B, T, P, D)
        feature_dim = input_shape[-1]
        
        # Initialize trainable scalar gate α ≥ 0
        self.alpha = self.add_weight(
            name='alpha_gate',
            shape=(),
            initializer=tf.keras.initializers.Constant(self.gate_alpha_init),
            constraint=tf.keras.constraints.NonNeg(),  # Ensure α ≥ 0
            trainable=True
        )
        
        # Initialize adjacency matrix based on mode
        if self.custom_adjacency is not None:
            # Use custom adjacency matrix
            A = self.custom_adjacency.astype(np.float32)
            # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
            D = np.sum(A, axis=1)
            D_inv_sqrt = np.power(D, -0.5)
            D_inv_sqrt = np.diag(D_inv_sqrt)
            A_norm = D_inv_sqrt @ A @ D_inv_sqrt
            self.A_norm = tf.constant(A_norm, dtype=tf.float32)
        elif self.adjacency_mode == 'identity':
            # Identity matrix (no cross-talk)
            A = np.eye(5, dtype=np.float32)
            self.A_norm = tf.constant(A, dtype=tf.float32)
        elif self.adjacency_mode == 'fixed':
            # Fixed hand-crafted edges with symmetric normalization
            A = np.array([
                # right_wrist, left_wrist, right_pocket, right_ankle, left_ankle
                [1, 1, 1, 1, 0],  # right_wrist connects to left_wrist, right_pocket, right_ankle
                [1, 1, 0, 0, 1],  # left_wrist connects to right_wrist, left_ankle
                [1, 0, 1, 1, 1],  # right_pocket connects to right_wrist, right_ankle, left_ankle
                [1, 0, 1, 1, 1],  # right_ankle connects to right_wrist, right_pocket, left_ankle
                [0, 1, 1, 1, 1],  # left_ankle connects to left_wrist, right_pocket, right_ankle
            ], dtype=np.float32)
            
            # Make it symmetric (undirected graph)
            A = np.maximum(A, A.T)
            
            # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
            D = np.sum(A, axis=1)
            D_inv_sqrt = np.power(D, -0.5)
            D_inv_sqrt = np.diag(D_inv_sqrt)
            A_norm = D_inv_sqrt @ A @ D_inv_sqrt
            self.A_norm = tf.constant(A_norm, dtype=tf.float32)
        elif self.adjacency_mode == 'learned':
            # Start from identity, add small learnable matrix E
            self.learned_adjacency = self.add_weight(
                name='learned_adjacency',
                shape=(5, 5),
                initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                trainable=True
            )
            # Initialize A_norm as identity for now, will be computed in call()
            self.A_norm = tf.eye(5, dtype=tf.float32)
        else:
            raise ValueError(f"Unknown adjacency_mode: {self.adjacency_mode}")
        
        # Initialize parameters
        self.W_self = self.add_weight(
            name='W_self',
            shape=(feature_dim, self.hidden_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.W_neigh = self.add_weight(
            name='W_neigh',
            shape=(feature_dim, self.hidden_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.b = self.add_weight(
            name='bias',
            shape=(self.hidden_dim,),
            initializer='zeros',
            trainable=True
        )
        
        # Initialize fallback dense layer
        self.fallback_dense = tf.keras.layers.Dense(self.hidden_dim, name='fallback_dense')
        
        # Initialize input projection layer for residual connection
        self.input_projection = tf.keras.layers.Dense(self.hidden_dim, name='input_projection')
        
        super().build(input_shape)
    
    def _compute_learned_adjacency(self):
        """Compute learned adjacency matrix with constraints."""
        # Start from identity
        A = tf.eye(5, dtype=tf.float32)
        
        # Add learned matrix with constraints
        E = self.learned_adjacency
        
        # Apply tanh + normalization to keep values reasonable
        E = tf.nn.tanh(E)
        E = E / tf.reduce_max(tf.abs(E)) * 0.1  # Scale down to small values
        
        # Add to identity
        A = A + E
        
        # Make symmetric
        A = (A + tf.transpose(A)) / 2
        
        # Apply softmax row-wise for normalization
        A = tf.nn.softmax(A, axis=-1)
        
        return A
    
    def call(self, inputs, training=None):
        """
        Forward pass with residual gating.
        
        Args:
            inputs: Tensor of shape (B, P, D) or (B, T, P, D)
            training: Training mode flag
        
        Returns:
            Tensor of shape (B, P, H) or (B, T, P, H)
        """
        x = inputs
        
        # Update learned adjacency if needed
        if self.adjacency_mode == 'learned':
            self.A_norm = self._compute_learned_adjacency()
        
        # Get input shape and rank
        input_shape = tf.shape(x)
        input_rank = tf.rank(x)
        
        # For now, just handle 3D input (B, P, D) which is what we're getting
        # Compute graph convolution
        H_msg = self._compute_graph_convolution(x)
        
        # Apply dropout and residual connection
        H_msg = self.dropout(H_msg, training=training)
        # Project input to match output dimension for residual connection
        x_projected = self.input_projection(x)
        H = x_projected + tf.nn.relu(self.alpha) * H_msg
        
        return H
    
    def _compute_graph_convolution(self, x):
        """Compute graph convolution for 3D input (B, P, D)."""
        # For 3D input: (B, P, D)
        H_self = tf.tensordot(x, self.W_self, axes=[[-1], [0]])
        X_neigh = tf.einsum('ij,bjp->bip', self.A_norm, x)
        H_neigh = tf.tensordot(X_neigh, self.W_neigh, axes=[[-1], [0]])
        H = tf.nn.relu(H_self + H_neigh + self.b)
        return H
    
    def _compute_residual_gated_output(self, x, training):
        """Compute residual-gated output for dynamic shapes."""
        # Compute graph convolution
        H_msg = self._compute_graph_convolution(x)
        
        # Apply dropout and residual connection
        H_msg = self.dropout(H_msg, training=training)
        # Project input to match output dimension for residual connection
        x_projected = self.input_projection(x)
        H = x_projected + tf.nn.relu(self.alpha) * H_msg
        
        return H
    
    def compute_output_shape(self, input_shape):
        """Compute the output shape."""
        if len(input_shape) == 3:
            # (B, P, D) -> (B, P, H)
            return (input_shape[0], input_shape[1], self.hidden_dim)
        elif len(input_shape) == 4:
            # (B, T, P, D) -> (B, T, P, H)
            return (input_shape[0], input_shape[1], input_shape[2], self.hidden_dim)
        else:
            raise ValueError(f"Expected 3D or 4D input shape, got {len(input_shape)}D")
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'adjacency_mode': self.adjacency_mode,
            'gate_alpha_init': self.gate_alpha_init,
            'dropout_rate': self.dropout_rate
        })
        return config
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim
        })
        return config


# Keep the old classes for backward compatibility but mark as deprecated
class PlacementEmbedding(keras.layers.Layer):
    """Placement embedding layer (deprecated - use GraphSemantics instead)."""
    
    def __init__(self, embedding_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.embedding = keras.layers.Embedding(5, embedding_dim + 5)  # 5 placements + 5 metadata flags
    
    def call(self, placement_indices):
        return self.embedding(placement_indices)
    
    def get_config(self):
        config = super().get_config()
        config.update({'embedding_dim': self.embedding_dim})
        return config


class GraphAttentionLayer(keras.layers.Layer):
    """Graph attention layer (deprecated - use GraphSemantics instead)."""
    
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.attention = keras.layers.MultiHeadAttention(num_heads, d_model)
    
    def call(self, inputs):
        return self.attention(inputs, inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads
        })
        return config


class PlacementSemanticsLayer(keras.layers.Layer):
    """
    DEPRECATED: Use GraphSemantics instead.
    This layer has graph-mode compatibility issues.
    """
    
    def __init__(self, embedding_dim=16, d_model=32, num_heads=4, use_graph=True, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_graph = use_graph
        
        # Placement embedding layer
        self.placement_embedding = PlacementEmbedding(embedding_dim)
        
        # Projection to d_model
        self.projection = keras.layers.Dense(d_model, name="placement_projection")
        
        # Placement feature projection
        self.placement_projection = keras.layers.Dense(d_model, name="placement_feature_projection")
        
        # Graph attention layer - only create if use_graph is True
        self.graph_attention = None
        if use_graph:
            self.graph_attention = GraphAttentionLayer(d_model, num_heads)
        
        # Layer normalization
        self.layer_norm = keras.layers.LayerNormalization(name="placement_layer_norm")
    
    def call(self, inputs: tf.Tensor, placement_indices: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        DEPRECATED: This method has graph-mode compatibility issues.
        Use GraphSemantics instead.
        """
        raise NotImplementedError(
            "PlacementSemanticsLayer is deprecated due to graph-mode compatibility issues. "
            "Use GraphSemantics instead."
        )
    
    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        if len(input_shape) == 3:
            # (batch_size, num_placements, feature_dim) -> (batch_size, num_placements, d_model)
            return (input_shape[0], input_shape[1], self.d_model)
        elif len(input_shape) == 4:
            # (batch_size, time_steps, num_placements, feature_dim) -> (batch_size, time_steps, num_placements, d_model)
            return (input_shape[0], input_shape[1], input_shape[2], self.d_model)
        else:
            raise ValueError(f"Expected 3D or 4D input shape, got {len(input_shape)}D")
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'use_graph': self.use_graph
        })
        return config


def create_placement_semantics_model(input_shape: tf.TensorShape, 
                                   embedding_dim: int = 8,
                                   d_model: int = 64,
                                   num_heads: int = 1,
                                   use_graph: bool = True) -> keras.Model:
    """
    Create a standalone placement semantics model for testing.
    
    Args:
        input_shape: (num_placements, feature_dim)
        embedding_dim: Dimension of placement embeddings
        d_model: Model dimension
        num_heads: Number of attention heads
        use_graph: Whether to use graph attention
    
    Returns:
        Keras model
    """
    inputs = keras.Input(shape=input_shape)
    
    placement_semantics = GraphSemantics(
        hidden_dim=d_model
    )
    
    outputs = placement_semantics(inputs)
    
    return keras.Model(inputs=inputs, outputs=outputs, name="placement_semantics_test")


# Utility functions
def get_placement_metadata(placement_name: str) -> dict:
    """Get metadata flags for a placement."""
    return PLACEMENT_META.get(placement_name, {})


def get_placement_index(placement_name: str) -> int:
    """Get index of placement in the order."""
    return PLACEMENT_ORDER.index(placement_name)


def get_placement_edges() -> list:
    """Get list of placement edges."""
    return PLACEMENT_EDGES.copy()
