"""
Cross-placement feature extraction for STM games.

Computes relationships between different sensor placements including
cross-correlations, symmetry indices, and phase relationships.
"""

import numpy as np
from typing import Dict, List, Tuple
import warnings
from .binary_features import compute_binary_micro_features


# Define placement pairs for cross-feature computation
PLACEMENT_PAIRS = [
    ('left-wrist', 'right-wrist'),      # Symmetric wrists
    ('left-ankle', 'right-ankle'),      # Symmetric ankles
    ('left-wrist', 'left-ankle'),       # Ipsilateral wrist-ankle
    ('right-wrist', 'right-ankle'),     # Ipsilateral wrist-ankle
    ('right-pocket', 'left-wrist'),     # Pocket to wrist
    ('right-pocket', 'right-wrist'),    # Pocket to wrist
    ('right-pocket', 'left-ankle'),     # Pocket to ankle
    ('right-pocket', 'right-ankle'),    # Pocket to ankle
]

# Symmetric pairs for asymmetry computation
SYMMETRIC_PAIRS = [
    ('left-wrist', 'right-wrist'),
    ('left-ankle', 'right-ankle'),
]


def compute_cross_features(bits_by_placement: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute cross-placement features from binary sequences.
    
    Args:
        bits_by_placement: Dictionary mapping placement names to binary sequences
    
    Returns:
        Dictionary of cross-placement features
    """
    features = {}
    
    # 2.1 Cross-correlation features for all pairs
    for p1, p2 in PLACEMENT_PAIRS:
        if p1 in bits_by_placement and p2 in bits_by_placement:
            try:
                xcorr_features = _compute_cross_correlation_features(
                    bits_by_placement[p1], bits_by_placement[p2], p1, p2
                )
                features.update(xcorr_features)
            except Exception as e:
                warnings.warn(f"Failed to compute cross-correlation for {p1}-{p2}: {e}")
    
    # 2.2 Asymmetry indices for symmetric pairs
    for p1, p2 in SYMMETRIC_PAIRS:
        if p1 in bits_by_placement and p2 in bits_by_placement:
            try:
                asym_features = _compute_asymmetry_features(
                    bits_by_placement[p1], bits_by_placement[p2], p1, p2
                )
                features.update(asym_features)
            except Exception as e:
                warnings.warn(f"Failed to compute asymmetry for {p1}-{p2}: {e}")
    
    return features


def _compute_cross_correlation_features(b1: np.ndarray, b2: np.ndarray, 
                                      p1: str, p2: str) -> Dict[str, float]:
    """
    Compute cross-correlation features between two binary sequences.
    
    Args:
        b1, b2: Binary sequences
        p1, p2: Placement names for feature naming
    
    Returns:
        Dictionary of cross-correlation features
    """
    # Compute flip series
    flip1 = np.abs(np.diff(b1, prepend=b1[0]))
    flip2 = np.abs(np.diff(b2, prepend=b2[0]))
    
    # Cross-correlation over lag range [-20, 20]
    max_lag = min(20, len(flip1) // 2, len(flip2) // 2)
    lags = range(-max_lag, max_lag + 1)
    
    xcorr_values = []
    for lag in lags:
        if lag >= 0:
            # b1[t] vs b2[t+lag]
            if len(flip1) > lag and len(flip2) > lag and len(flip1[:-lag]) > 0 and len(flip2[lag:]) > 0:
                try:
                    corr = np.corrcoef(flip1[:-lag], flip2[lag:])[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                except:
                    corr = 0.0
            else:
                corr = 0.0
        else:
            # b1[t-lag] vs b2[t]
            if len(flip1) > -lag and len(flip2) > -lag and len(flip1[-lag:]) > 0 and len(flip2[:-(-lag)]) > 0:
                try:
                    corr = np.corrcoef(flip1[-lag:], flip2[:-(-lag)])[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                except:
                    corr = 0.0
            else:
                corr = 0.0
        
        xcorr_values.append(corr)
    
    if not xcorr_values:
        return {}
    
    # Find peak and lag
    peak_idx = np.argmax(xcorr_values)
    peak_lag = lags[peak_idx]
    peak_value = xcorr_values[peak_idx]
    
    # Create feature names
    pair_name = f"{p1.replace('-', '_')}_{p2.replace('-', '_')}"
    
    return {
        f"xcorr_peak_{pair_name}": float(peak_value),
        f"xcorr_lag_{pair_name}": float(peak_lag),
    }


def _compute_asymmetry_features(b1: np.ndarray, b2: np.ndarray, 
                              p1: str, p2: str) -> Dict[str, float]:
    """
    Compute asymmetry indices for symmetric pairs.
    
    Args:
        b1, b2: Binary sequences from symmetric placements
        p1, p2: Placement names for feature naming
    
    Returns:
        Dictionary of asymmetry features
    """
    # Compute micro-features for both placements
    features1 = compute_binary_micro_features(b1)
    features2 = compute_binary_micro_features(b2)
    
    # Compute asymmetry for relevant features
    asym_features = {}
    pair_name = f"{p1.replace('-', '_')}_{p2.replace('-', '_')}"
    
    # Asymmetry in flip rate
    flip_rate1 = features1.get('flip_rate', 0.0)
    flip_rate2 = features2.get('flip_rate', 0.0)
    asym_flip = _compute_asymmetry_index(flip_rate1, flip_rate2)
    asym_features[f"asym_flip_rate_{pair_name}"] = asym_flip
    
    # Asymmetry in duty cycle
    duty1 = features1.get('duty', 0.0)
    duty2 = features2.get('duty', 0.0)
    asym_duty = _compute_asymmetry_index(duty1, duty2)
    asym_features[f"asym_duty_{pair_name}"] = asym_duty
    
    # Asymmetry in run length means
    on_len1 = features1.get('on_len_mean', 0.0)
    on_len2 = features2.get('on_len_mean', 0.0)
    asym_on_len = _compute_asymmetry_index(on_len1, on_len2)
    asym_features[f"asym_on_len_mean_{pair_name}"] = asym_on_len
    
    off_len1 = features1.get('off_len_mean', 0.0)
    off_len2 = features2.get('off_len_mean', 0.0)
    asym_off_len = _compute_asymmetry_index(off_len1, off_len2)
    asym_features[f"asym_off_len_mean_{pair_name}"] = asym_off_len
    
    return asym_features


def _compute_asymmetry_index(val1: float, val2: float, epsilon: float = 1e-8) -> float:
    """
    Compute asymmetry index: (val1 - val2) / (val1 + val2 + epsilon)
    
    Args:
        val1, val2: Values to compare
        epsilon: Small constant to avoid division by zero
    
    Returns:
        Asymmetry index in [-1, 1]
    """
    denominator = val1 + val2 + epsilon
    if denominator == 0:
        return 0.0
    return (val1 - val2) / denominator


def get_cross_feature_names() -> List[str]:
    """Get list of all cross-feature names."""
    feature_names = []
    
    # Cross-correlation features for all pairs
    for p1, p2 in PLACEMENT_PAIRS:
        pair_name = f"{p1.replace('-', '_')}_{p2.replace('-', '_')}"
        feature_names.extend([
            f"xcorr_peak_{pair_name}",
            f"xcorr_lag_{pair_name}",
        ])
    
    # Asymmetry features for symmetric pairs
    for p1, p2 in SYMMETRIC_PAIRS:
        pair_name = f"{p1.replace('-', '_')}_{p2.replace('-', '_')}"
        feature_names.extend([
            f"asym_flip_rate_{pair_name}",
            f"asym_duty_{pair_name}",
            f"asym_on_len_mean_{pair_name}",
            f"asym_off_len_mean_{pair_name}",
        ])
    
    return feature_names


# Feature dimension constants
TOTAL_CROSS_FEATURES = len(get_cross_feature_names())


def compute_all_cross_features(bits_by_placement: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute all cross-placement features.
    
    Args:
        bits_by_placement: Dictionary mapping placement names to binary sequences
    
    Returns:
        Dictionary of all cross-placement features
    """
    return compute_cross_features(bits_by_placement)
