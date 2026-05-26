"""
Binary micro-features extraction for STM games.

Extracts human-meaningful kinematic cues from binary (0/1) sensor streams
without using raw IMU data. Features include flip rates, cadence, duty cycles,
and run-length statistics at multiple temporal scales.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings


def compute_binary_micro_features(b: np.ndarray, dt: Optional[float] = None) -> Dict[str, float]:
    """
    Compute micro-features from a binary sequence.
    
    Args:
        b: Binary sequence of shape (T,) where T is window length
        dt: Time step in seconds (optional, for cadence in Hz)
    
    Returns:
        Dictionary of computed features
    """
    if len(b.shape) != 1:
        raise ValueError(f"Expected 1D array, got shape {b.shape}")
    
    T = len(b)
    if T == 0:
        raise ValueError("Empty sequence")
    
    features = {}
    
    # 1.1 Flip series & cadence
    flip_series = np.abs(np.diff(b, prepend=b[0]))
    flip_rate = np.mean(flip_series)
    features['flip_rate'] = float(flip_rate)
    
    # Cadence via autocorrelation peak
    cadence_lag = _compute_cadence_lag(flip_series)
    features['cadence_lag'] = float(cadence_lag)
    
    if dt is not None:
        features['cadence_hz'] = float(cadence_lag / (T * dt))
    
    # 1.2 Duty cycle
    duty = np.mean(b)
    features['duty'] = float(duty)
    
    # 1.3 Run-length statistics
    on_runs, off_runs = _compute_run_lengths(b)
    
    if len(on_runs) > 0:
        features['on_len_mean'] = float(np.mean(on_runs))
        features['on_len_var'] = float(np.var(on_runs))
    else:
        features['on_len_mean'] = 0.0
        features['on_len_var'] = 0.0
    
    if len(off_runs) > 0:
        features['off_len_mean'] = float(np.mean(off_runs))
        features['off_len_var'] = float(np.var(off_runs))
    else:
        features['off_len_mean'] = 0.0
        features['off_len_var'] = 0.0
    
    return features


def compute_multi_scale_features(b: np.ndarray, dt: Optional[float] = None) -> Dict[str, float]:
    """
    Compute features at multiple temporal scales.
    
    Args:
        b: Binary sequence of shape (T,)
        dt: Time step in seconds (optional)
    
    Returns:
        Dictionary with features at scales T/2 and T
    """
    T = len(b)
    features = {}
    
    # Full window (T)
    full_features = compute_binary_micro_features(b, dt)
    for key, value in full_features.items():
        features[f"{key}_S100"] = value
    
    # Half window (T/2) - use first half
    if T >= 2:
        half_length = T // 2
        half_features = compute_binary_micro_features(b[:half_length], dt)
        for key, value in half_features.items():
            features[f"{key}_S50"] = value
    else:
        # If window too small, duplicate full features
        for key, value in full_features.items():
            features[f"{key}_S50"] = value
    
    return features


def _compute_cadence_lag(flip_series: np.ndarray) -> int:
    """
    Compute cadence lag via autocorrelation peak.
    
    Args:
        flip_series: Binary flip series (0/1)
    
    Returns:
        Lag value (integer)
    """
    T = len(flip_series)
    
    # Define lag range: 4 to min(40, T/2)
    max_lag = min(40, T // 2)
    if max_lag < 4:
        return 1  # Fallback for very short sequences
    
    lags = range(4, max_lag + 1)
    autocorr_values = []
    
    for lag in lags:
        if lag >= T:
            break
        
        # Compute autocorrelation at this lag
        corr = np.corrcoef(flip_series[:-lag], flip_series[lag:])[0, 1]
        if np.isnan(corr):
            corr = 0.0
        autocorr_values.append(corr)
    
    if not autocorr_values:
        return 1
    
    # Find peak lag
    peak_idx = np.argmax(autocorr_values)
    return lags[peak_idx]


def _compute_run_lengths(b: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Compute run lengths of consecutive 1s and 0s.
    
    Args:
        b: Binary sequence
    
    Returns:
        Tuple of (on_runs, off_runs) lists
    """
    if len(b) == 0:
        return [], []
    
    on_runs = []
    off_runs = []
    
    current_run = 1
    current_value = b[0]
    
    for i in range(1, len(b)):
        if b[i] == current_value:
            current_run += 1
        else:
            if current_value == 1:
                on_runs.append(current_run)
            else:
                off_runs.append(current_run)
            current_run = 1
            current_value = b[i]
    
    # Add final run
    if current_value == 1:
        on_runs.append(current_run)
    else:
        off_runs.append(current_run)
    
    return on_runs, off_runs


def compute_all_placement_features(bits_by_placement: Dict[str, np.ndarray], 
                                 dt: Optional[float] = None) -> Dict[str, Dict[str, float]]:
    """
    Compute micro-features for all placements.
    
    Args:
        bits_by_placement: Dictionary mapping placement names to binary sequences
        dt: Time step in seconds (optional)
    
    Returns:
        Dictionary mapping placement names to their feature dictionaries
    """
    features_by_placement = {}
    
    for placement, bits in bits_by_placement.items():
        try:
            features = compute_multi_scale_features(bits, dt)
            features_by_placement[placement] = features
        except Exception as e:
            warnings.warn(f"Failed to compute features for {placement}: {e}")
            features_by_placement[placement] = {}
    
    return features_by_placement


# Feature dimension constants
FEATURES_PER_PLACEMENT = 14  # 7 features × 2 scales (S50, S100)
TOTAL_MICRO_FEATURES = FEATURES_PER_PLACEMENT * 5  # 5 placements


def get_feature_names() -> List[str]:
    """Get list of all micro-feature names for a single placement."""
    base_features = [
        'flip_rate', 'cadence_lag', 'duty', 
        'on_len_mean', 'on_len_var', 'off_len_mean', 'off_len_var'
    ]
    
    feature_names = []
    for scale in ['S50', 'S100']:
        for feature in base_features:
            feature_names.append(f"{feature}_{scale}")
    
    return feature_names


def get_all_feature_names() -> List[str]:
    """Get list of all micro-feature names across all placements."""
    placement_names = ['left-wrist', 'right-wrist', 'right-pocket', 'right-ankle', 'left-ankle']
    feature_names = []
    
    for placement in placement_names:
        for feature in get_feature_names():
            feature_names.append(f"{placement}_{feature}")
    
    return feature_names
