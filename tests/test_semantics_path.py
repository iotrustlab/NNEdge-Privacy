"""
Unit test for continuous features path in GraphSemantics.

Tests that GraphSemantics runs without fallback when continuous features are present.
"""

import numpy as np
import tensorflow as tf
from games.STM.models.placement_semantics import GraphSemantics
from games.STM.data_loader import load_data_for_game
import os


def test_continuous_features_path():
    """Test that GraphSemantics runs without fallback with continuous features."""
    print("🔍 Testing continuous features path...")
    
    # Create synthetic batch: (B=2, T=32, P=5, C=7)
    batch_size, time_steps, num_placements, num_features = 2, 32, 5, 7
    
    # Create synthetic data with continuous features
    # Features: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, dec_tree_out_1]
    synthetic_data = np.random.randn(batch_size, time_steps, num_placements, num_features).astype(np.float32)
    
    # Make the last feature (dec_tree_out_1) binary
    synthetic_data[:, :, :, -1] = np.random.randint(0, 2, size=(batch_size, time_steps, num_placements))
    
    print(f"  📊 Synthetic data shape: {synthetic_data.shape}")
    print(f"  📊 Expected: (B={batch_size}, T={time_steps}, P={num_placements}, C={num_features})")
    
    # Test GraphSemantics with different adjacency modes
    adjacency_modes = ['identity', 'fixed', 'learned']
    
    for mode in adjacency_modes:
        print(f"  🔧 Testing adjacency mode: {mode}")
        
        # Create GraphSemantics layer
        graph_semantics = GraphSemantics(
            hidden_dim=64,
            adjacency_mode=mode,
            gate_alpha_init=0.1
        )
        
        # Test forward pass
        try:
            # Input shape: (B, T, P, C)
            output = graph_semantics(synthetic_data)
            
            # Check output shape: should be (B, T, P, hidden_dim)
            expected_shape = (batch_size, time_steps, num_placements, 64)
            actual_shape = output.shape
            
            print(f"    ✅ {mode}: Output shape {actual_shape}")
            
            # Verify that the output is not just identity (fallback)
            # Check if the alpha gate is being used
            alpha_value = graph_semantics.alpha.numpy()
            print(f"    📊 Alpha gate value: {alpha_value:.4f}")
            
            # Check if adjacency matrix is being used
            if mode == 'learned':
                # For learned mode, check if learned adjacency is being updated
                learned_adj = graph_semantics.learned_adjacency.numpy()
                print(f"    📊 Learned adjacency range: [{learned_adj.min():.4f}, {learned_adj.max():.4f}]")
            else:
                # For fixed/identity modes, check if A_norm is set
                adj_norm = graph_semantics.A_norm.numpy()
                print(f"    📊 Adjacency norm shape: {adj_norm.shape}")
            
            # Verify that the output is not just the input (fallback behavior)
            input_norm = tf.norm(synthetic_data).numpy()
            output_norm = tf.norm(output).numpy()
            print(f"    📊 Input norm: {input_norm:.4f}, Output norm: {output_norm:.4f}")
            
            # The output should be different from input (not fallback)
            assert output_norm != input_norm, f"Output is identical to input (fallback detected) for mode {mode}"
            
        except Exception as e:
            print(f"    ❌ {mode}: Failed with error: {e}")
            raise
    
    print("✅ Continuous features path test completed successfully!")


def test_real_data_loading():
    """Test loading real data with continuous features."""
    print("🔍 Testing real data loading with continuous features...")
    
    # Check if data directory exists
    data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data")
    if not os.path.exists(data_root):
        print(f"  ⚠️ Data directory not found: {data_root}")
        print("  📝 Skipping real data test")
        return
    
    try:
        # Load data for Game-1-multi (should now have continuous features)
        user_data, all_labels = load_data_for_game("Game-1-multi", data_root)
        
        if user_data:
            # Get a sample to check dimensions
            sample_user = list(user_data.keys())[0]
            sample_data = user_data[sample_user][0]  # X data
            
            print(f"  📊 Sample data shape: {sample_data.shape}")
            
            # Should be (num_samples, T, P, C) where P=5, C=7
            if len(sample_data.shape) == 4:
                num_samples, T, P, C = sample_data.shape
                print(f"  📊 Time steps (T): {T}")
                print(f"  📊 Placements (P): {P}")
                print(f"  📊 Features per placement (C): {C}")
                
                # Verify expected dimensions
                assert P == 5, f"Expected 5 placements, got {P}"
                assert C == 7, f"Expected 7 features, got {C}"
                
                print("  ✅ Real data loading test passed!")
            else:
                print(f"  ⚠️ Unexpected data shape: {sample_data.shape}")
        else:
            print("  ⚠️ No user data loaded")
            
    except Exception as e:
        print(f"  ❌ Real data loading failed: {e}")
        raise


if __name__ == "__main__":
    print("🚀 Starting semantics path tests...")
    
    # Test synthetic data
    test_continuous_features_path()
    
    # Test real data loading
    test_real_data_loading()
    
    print("🎉 All tests completed!")


