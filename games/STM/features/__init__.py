"""
Features module for STM games.

Contains binary micro-features, cross-placement feature extraction, and enhanced semantic features.
"""

from .binary_features import (
    compute_binary_micro_features,
    compute_multi_scale_features,
    compute_all_placement_features,
    get_feature_names,
    get_all_feature_names,
    FEATURES_PER_PLACEMENT,
    TOTAL_MICRO_FEATURES
)

from .cross_features import (
    compute_cross_features,
    compute_all_cross_features,
    get_cross_feature_names,
    TOTAL_CROSS_FEATURES
)

from .enhanced_semantic_features import (
    SemanticFeatureNormalizer,
    AttentionBasedFusion,
    DynamicGatingScheduler,
    compute_enhanced_semantic_features,
    create_hybrid_adjacency,
    create_weighted_adjacency,
    combine_loss_functions,
    create_hierarchical_edge_priors,
    apply_temporal_jittering,
    inject_sensor_noise,
    create_class_rebalancing_augmentation
)

__all__ = [
    # Binary features
    'compute_binary_micro_features',
    'compute_multi_scale_features', 
    'compute_all_placement_features',
    'get_feature_names',
    'get_all_feature_names',
    'FEATURES_PER_PLACEMENT',
    'TOTAL_MICRO_FEATURES',
    
    # Cross features
    'compute_cross_features',
    'compute_all_cross_features',
    'get_cross_feature_names',
    'TOTAL_CROSS_FEATURES',
    
    # Enhanced semantic features
    'SemanticFeatureNormalizer',
    'AttentionBasedFusion',
    'DynamicGatingScheduler',
    'compute_enhanced_semantic_features',
    'create_hybrid_adjacency',
    'create_weighted_adjacency',
    'combine_loss_functions',
    'create_hierarchical_edge_priors',
    'apply_temporal_jittering',
    'inject_sensor_noise',
    'create_class_rebalancing_augmentation'
]
