"""
Enhanced semantic features with optimization techniques.

This module provides improved semantic feature integration with:
1. Feature normalization and scaling
2. Attention-based fusion mechanisms
3. Dynamic gating parameter scheduling
4. Hybrid fusion strategies
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Tuple, Optional, Union

from .binary_features import compute_multi_scale_features
from .cross_features import compute_cross_features


class SemanticFeatureNormalizer:
    """Normalize semantic features for better integration."""
    
    def __init__(self, method='standard', feature_groups=None):
        """
        Initialize normalizer.
        
        Args:
            method: 'standard', 'minmax', or 'robust'
            feature_groups: Dict mapping group names to feature indices
        """
        self.method = method
        self.feature_groups = feature_groups or {}
        self.scalers = {}
        self.is_fitted = False
        
    def fit(self, semantic_features_list: List[np.ndarray]):
        """Fit normalizers on semantic features."""
        if not semantic_features_list:
            return
            
        # Stack all features
        all_features = np.vstack(semantic_features_list)
        
        if self.feature_groups:
            # Fit per-group normalizers
            for group_name, indices in self.feature_groups.items():
                if self.method == 'standard':
                    scaler = StandardScaler()
                elif self.method == 'minmax':
                    scaler = MinMaxScaler()
                elif self.method == 'robust':
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()
                else:
                    raise ValueError(f"Unknown normalization method: {self.method}")
                
                group_features = all_features[:, indices]
                scaler.fit(group_features)
                self.scalers[group_name] = scaler
        else:
            # Fit single normalizer for all features
            if self.method == 'standard':
                scaler = StandardScaler()
            elif self.method == 'minmax':
                scaler = MinMaxScaler()
            elif self.method == 'robust':
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization method: {self.method}")
            
            scaler.fit(all_features)
            self.scalers['all'] = scaler
            
        self.is_fitted = True
        
    def transform(self, semantic_features: np.ndarray) -> np.ndarray:
        """Transform semantic features using fitted normalizers."""
        if not self.is_fitted:
            return semantic_features
            
        if self.feature_groups:
            # Transform per-group
            normalized_features = semantic_features.copy()
            for group_name, indices in self.feature_groups.items():
                if group_name in self.scalers:
                    group_features = semantic_features[:, indices]
                    normalized_group = self.scalers[group_name].transform(group_features)
                    normalized_features[:, indices] = normalized_group
            return normalized_features
        else:
            # Transform all features
            if 'all' in self.scalers:
                return self.scalers['all'].transform(semantic_features)
            return semantic_features


class AttentionBasedFusion:
    """Attention-based fusion for semantic features."""
    
    def __init__(self, semantic_dim: int, hidden_dim: int = 64, num_heads: int = 4):
        """
        Initialize attention fusion.
        
        Args:
            semantic_dim: Dimension of semantic features
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
        """
        self.semantic_dim = semantic_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
    def build_attention_layer(self):
        """Build multi-head attention layer."""
        return tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.hidden_dim // self.num_heads
        )
    
    def build_fusion_model(self, input_shape: Tuple[int, ...]) -> tf.keras.Model:
        """Build attention-based fusion model."""
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Separate sensor and semantic features
        sensor_features = inputs[:, :, :5]  # First 5 dimensions are sensor features
        semantic_features = inputs[:, :, 5:]  # Remaining are semantic features
        
        # Project semantic features to hidden dimension
        semantic_projection = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(semantic_features)
        
        # Apply self-attention to semantic features
        attention_layer = self.build_attention_layer()
        attended_semantic = attention_layer(semantic_projection, semantic_projection)
        
        # Add residual connection
        attended_semantic = tf.keras.layers.Add()([semantic_projection, attended_semantic])
        
        # Project back to original semantic dimension
        fused_semantic = tf.keras.layers.Dense(self.semantic_dim, activation='relu')(attended_semantic)
        
        # Concatenate with sensor features
        outputs = tf.keras.layers.Concatenate(axis=-1)([sensor_features, fused_semantic])
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)


class DynamicGatingScheduler:
    """Dynamic scheduling for gating parameter α."""
    
    def __init__(self, initial_alpha: float = 0.1, schedule_type: str = 'cosine'):
        """
        Initialize dynamic gating scheduler.
        
        Args:
            initial_alpha: Initial α value
            schedule_type: 'cosine', 'linear', 'exponential', or 'adaptive'
        """
        self.initial_alpha = initial_alpha
        self.schedule_type = schedule_type
        self.current_epoch = 0
        self.alpha_history = []
        
    def get_alpha(self, epoch: int, val_metric: Optional[float] = None) -> float:
        """Get α value for current epoch."""
        self.current_epoch = epoch
        
        if self.schedule_type == 'cosine':
            # Cosine annealing
            alpha = self.initial_alpha * (1 + np.cos(np.pi * epoch / 100)) / 2
        elif self.schedule_type == 'linear':
            # Linear increase
            alpha = min(1.0, self.initial_alpha + epoch * 0.01)
        elif self.schedule_type == 'exponential':
            # Exponential decay
            alpha = self.initial_alpha * np.exp(-epoch * 0.05)
        elif self.schedule_type == 'adaptive':
            # Adaptive based on validation metric
            if val_metric is not None and len(self.alpha_history) > 0:
                if val_metric > max(self.alpha_history[-5:]):  # Improving
                    alpha = min(1.0, self.alpha_history[-1] * 1.1)
                else:  # Not improving
                    alpha = max(0.01, self.alpha_history[-1] * 0.9)
            else:
                alpha = self.initial_alpha
        else:
            alpha = self.initial_alpha
            
        self.alpha_history.append(alpha)
        return alpha


def compute_enhanced_semantic_features(window_data: np.ndarray, 
                                     placement_names: List[str],
                                     featureset: str = 'micro+cross+semantics',
                                     normalizer: Optional[SemanticFeatureNormalizer] = None,
                                     use_attention: bool = False) -> Dict:
    """
    Compute enhanced semantic features with optimization techniques.
    
    Args:
        window_data: Array of shape (window_size, num_placements) with binary data
        placement_names: List of placement names in order
        featureset: Feature set to compute
        normalizer: Optional normalizer for semantic features
        use_attention: Whether to use attention-based fusion
    
    Returns:
        Dictionary with enhanced semantic features
    """
    if featureset == 'base':
        return {'summary_token': None, 'semantic_features': {}, 'enhanced': False}
    
    # Extract binary sequences for each placement
    bits_by_placement = {}
    for i, placement in enumerate(placement_names):
        if i < window_data.shape[1]:
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
        
        # Apply normalization if provided
        if normalizer is not None and normalizer.is_fitted:
            summary_token = normalizer.transform(summary_token.reshape(1, -1)).flatten()
        
        enhanced = normalizer is not None or use_attention
    else:
        summary_token = None
        enhanced = False
    
    return {
        'summary_token': summary_token,
        'semantic_features': semantic_features,
        'enhanced': enhanced
    }


def create_hybrid_adjacency(num_placements: int, 
                          fixed_weight: float = 0.7,
                          learned_weight: float = 0.3) -> np.ndarray:
    """
    Create hybrid adjacency matrix combining fixed and learned components.
    
    Args:
        num_placements: Number of placements
        fixed_weight: Weight for fixed adjacency component
        learned_weight: Weight for learned adjacency component
    
    Returns:
        Hybrid adjacency matrix
    """
    # Fixed adjacency (hand-crafted relationships)
    fixed_adj = np.zeros((num_placements, num_placements))
    
    # Define hand-crafted relationships based on placement proximity
    placement_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Adjacent placements
        (0, 2), (1, 3), (2, 4),          # Skip-one relationships
        (0, 3), (1, 4),                  # Skip-two relationships
    ]
    
    for i, j in placement_pairs:
        if i < num_placements and j < num_placements:
            fixed_adj[i, j] = 1.0
            fixed_adj[j, i] = 1.0
    
    # Add self-connections
    np.fill_diagonal(fixed_adj, 1.0)
    
    # Learned adjacency (random initialization)
    learned_adj = np.random.rand(num_placements, num_placements) * 0.1
    learned_adj = (learned_adj + learned_adj.T) / 2  # Make symmetric
    np.fill_diagonal(learned_adj, 1.0)
    
    # Combine with weights
    hybrid_adj = fixed_weight * fixed_adj + learned_weight * learned_adj
    
    # Normalize
    row_sums = hybrid_adj.sum(axis=1, keepdims=True)
    hybrid_adj = hybrid_adj / (row_sums + 1e-8)
    
    return hybrid_adj


def create_weighted_adjacency(num_placements: int, 
                            confidence_weights: Optional[Dict[Tuple[int, int], float]] = None) -> np.ndarray:
    """
    Create weighted adjacency matrix with confidence-based edge weights.
    
    Args:
        num_placements: Number of placements
        confidence_weights: Dict mapping (i, j) pairs to confidence weights
    
    Returns:
        Weighted adjacency matrix
    """
    # Start with fixed adjacency
    adj = np.zeros((num_placements, num_placements))
    
    # Define base relationships
    placement_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Adjacent placements
        (0, 2), (1, 3), (2, 4),          # Skip-one relationships
        (0, 3), (1, 4),                  # Skip-two relationships
    ]
    
    # Apply confidence weights
    for i, j in placement_pairs:
        if i < num_placements and j < num_placements:
            weight = confidence_weights.get((i, j), 1.0) if confidence_weights else 1.0
            adj[i, j] = weight
            adj[j, i] = weight
    
    # Add self-connections with high confidence
    np.fill_diagonal(adj, 2.0)
    
    # Normalize
    row_sums = adj.sum(axis=1, keepdims=True)
    adj = adj / (row_sums + 1e-8)
    
    return adj


def combine_loss_functions(base_loss: str = 'balanced_softmax',
                          focal_gamma: float = 2.0,
                          focal_alpha: float = 0.25) -> callable:
    """
    Create combined loss function.
    
    Args:
        base_loss: Base loss function ('balanced_softmax', 'logit_adjust')
        focal_gamma: Focal loss gamma parameter
        focal_alpha: Focal loss alpha parameter
    
    Returns:
        Combined loss function
    """
    def combined_loss(y_true, y_pred):
        # Base loss
        if base_loss == 'balanced_softmax':
            from ..models.training_utils import BalancedSoftmaxLoss
            base_loss_fn = BalancedSoftmaxLoss()
            base_loss_val = base_loss_fn(y_true, y_pred)
        elif base_loss == 'logit_adjust':
            from ..models.training_utils import LogitAdjustedLoss
            base_loss_fn = LogitAdjustedLoss()
            base_loss_val = base_loss_fn(y_true, y_pred)
        else:
            base_loss_val = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Focal loss component
        ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        pt = tf.exp(-ce_loss)
        focal_loss = focal_alpha * tf.pow(1 - pt, focal_gamma) * ce_loss
        
        # Combine with equal weights
        combined = 0.5 * base_loss_val + 0.5 * focal_loss
        
        return combined
    
    return combined_loss


def create_hierarchical_edge_priors(num_coarse_classes: int,
                                  num_fine_classes: int,
                                  coarse_to_fine_mapping: Dict[int, List[int]]) -> np.ndarray:
    """
    Create hierarchical edge priors for coarse-to-fine classification.
    
    Args:
        num_coarse_classes: Number of coarse classes
        num_fine_classes: Number of fine classes
        coarse_to_fine_mapping: Mapping from coarse to fine class indices
    
    Returns:
        Edge prior matrix of shape (num_fine_classes, num_fine_classes)
    """
    edge_priors = np.zeros((num_fine_classes, num_fine_classes))
    
    # Create edges between fine classes within the same coarse class
    for coarse_class, fine_classes in coarse_to_fine_mapping.items():
        for i in fine_classes:
            for j in fine_classes:
                if i != j:
                    edge_priors[i, j] = 1.0
    
    # Normalize
    row_sums = edge_priors.sum(axis=1, keepdims=True)
    edge_priors = edge_priors / (row_sums + 1e-8)
    
    return edge_priors


def apply_temporal_jittering(window_data: np.ndarray,
                           jitter_prob: float = 0.1,
                           max_shift: int = 2) -> np.ndarray:
    """
    Apply temporal jittering for data augmentation.
    
    Args:
        window_data: Input data of shape (window_size, num_features)
        jitter_prob: Probability of applying jitter
        max_shift: Maximum temporal shift
    
    Returns:
        Jittered data
    """
    if np.random.random() > jitter_prob:
        return window_data
    
    # Random temporal shift
    shift = np.random.randint(-max_shift, max_shift + 1)
    
    if shift == 0:
        return window_data
    
    # Apply shift with padding
    window_size = window_data.shape[0]
    jittered = np.zeros_like(window_data, dtype=np.float32)
    
    if shift > 0:
        # Shift right
        jittered[shift:] = window_data[:-shift]
        jittered[:shift] = window_data[0]  # Repeat first frame
    else:
        # Shift left
        jittered[:shift] = window_data[-shift:]
        jittered[shift:] = window_data[-1]  # Repeat last frame
    
    return jittered


def inject_sensor_noise(window_data: np.ndarray,
                       noise_prob: float = 0.05,
                       noise_std: float = 0.1) -> np.ndarray:
    """
    Inject sensor noise for data augmentation.
    
    Args:
        window_data: Input data
        noise_prob: Probability of adding noise to each element
        noise_std: Standard deviation of noise
    
    Returns:
        Noisy data
    """
    # Convert to float32 to avoid casting issues
    noisy_data = window_data.astype(np.float32)
    
    noise_mask = np.random.random(window_data.shape) < noise_prob
    noise = np.random.normal(0, noise_std, window_data.shape).astype(np.float32)
    
    noisy_data[noise_mask] += noise[noise_mask]
    
    # Clip to valid range [0, 1] for binary features
    noisy_data = np.clip(noisy_data, 0, 1)
    
    return noisy_data


def create_class_rebalancing_augmentation(X: np.ndarray,
                                         y: np.ndarray,
                                         target_samples_per_class: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create class-rebalancing augmentation.
    
    Args:
        X: Input data
        y: Labels
        target_samples_per_class: Target number of samples per class
    
    Returns:
        Augmented (X, y) with balanced classes
    """
    unique_classes = np.unique(y)
    augmented_X = []
    augmented_y = []
    
    # Preserve the original dtype of y
    y_dtype = y.dtype
    
    for class_idx in unique_classes:
        class_mask = y == class_idx
        class_samples = X[class_mask]
        class_count = len(class_samples)
        
        if class_count < target_samples_per_class:
            # Need to augment this class
            samples_needed = target_samples_per_class - class_count
            
            # Random sampling with replacement
            indices = np.random.choice(class_count, samples_needed, replace=True)
            augmented_samples = class_samples[indices]
            
            # Apply augmentation to some samples
            for i in range(len(augmented_samples)):
                if np.random.random() < 0.3:  # 30% chance of augmentation
                    augmented_samples[i] = apply_temporal_jittering(augmented_samples[i])
                if np.random.random() < 0.2:  # 20% chance of noise injection
                    augmented_samples[i] = inject_sensor_noise(augmented_samples[i])
            
            # Combine original and augmented samples
            all_samples = np.vstack([class_samples, augmented_samples])
            all_labels = np.full(len(all_samples), class_idx, dtype=y_dtype)
        else:
            # Class has enough samples, use random subset
            indices = np.random.choice(class_count, target_samples_per_class, replace=False)
            all_samples = class_samples[indices]
            all_labels = np.full(len(all_samples), class_idx, dtype=y_dtype)
        
        augmented_X.append(all_samples)
        augmented_y.append(all_labels)
    
    # Ensure consistent data types
    final_X = np.vstack(augmented_X).astype(np.float32)
    final_y = np.concatenate(augmented_y)
    
    return final_X, final_y
