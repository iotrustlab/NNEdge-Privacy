#!/usr/bin/env python3
"""
Example: Running Game-1 Frequency Study with 50Hz Respect
"""

from games.STM.run_game1_freq_refit_50hz import Game1FreqRefit50Hz

def main():
    """Run a simple frequency study example"""
    print("🔬 Running Game-1 Frequency Study Example...")
    
    # Create study instance
    study = Game1FreqRefit50Hz("results/example_frequency_study")
    
    # Run with minimal parameters for quick testing
    study.sampling_intervals = [20, 50]  # ms
    study.window_sizes = [20, 30]
    study.accel_thresholds = [250]
    study.gyro_thresholds = [15000]
    study.models = ['lstm_attention']
    study.seeds = [42]
    
    # Run the study
    results = study.run_full_study()
    
    if results is not None:
        output_path = study.save_results()
        print(f"✅ Example completed! Results saved to: {output_path}")
    else:
        print("❌ Example failed")

if __name__ == "__main__":
    main()
