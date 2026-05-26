#!/usr/bin/env python3
"""
Sampling Audit Tool for Game-1 Frequency Study
Verifies the physical sampling rate of the IMU sensors.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import glob
from pathlib import Path
from datetime import datetime

def audit_sampling_rate(raw_root, output_file):
    """Audit the sampling rate of raw IMU files"""
    print(f"🔍 Auditing sampling rates in: {raw_root}")
    
    # Find all raw IMU files
    raw_files = []
    for pattern in ["**/Processed/**/*.csv", "**/Processed/**/*.txt"]:
        raw_files.extend(glob.glob(os.path.join(raw_root, pattern), recursive=True))
    
    if not raw_files:
        print(f"❌ No raw files found in {raw_root}")
        return False
    
    print(f"📁 Found {len(raw_files)} raw files")
    
    # Audit results
    audit_results = {
        "timestamp": datetime.now().isoformat(),
        "raw_root": raw_root,
        "total_files": len(raw_files),
        "files_audited": 0,
        "files_with_time": 0,
        "sampling_stats": {},
        "per_file_stats": {},
        "summary": {}
    }
    
    all_delta_t = []
    
    for file_path in raw_files[:10]:  # Sample first 10 files for efficiency
        try:
            print(f"   📄 Auditing: {os.path.basename(file_path)}")
            
            # Try to read the file
            df = pd.read_csv(file_path)
            
            # Look for time column
            time_cols = [col for col in df.columns if 'time' in col.lower() or 'timestamp' in col.lower()]
            
            if not time_cols:
                print(f"     ⚠️ No time column found")
                continue
            
            time_col = time_cols[0]
            audit_results["files_with_time"] += 1
            
            # Calculate time differences
            time_values = pd.to_numeric(df[time_col], errors='coerce')
            time_values = time_values.dropna()
            
            if len(time_values) < 2:
                print(f"     ⚠️ Insufficient time data")
                continue
            
            # Calculate delta_t in milliseconds
            if time_values.max() > 1e12:  # Likely microseconds
                delta_t_ms = np.diff(time_values) / 1000
            elif time_values.max() > 1e9:  # Likely nanoseconds
                delta_t_ms = np.diff(time_values) / 1e6
            else:  # Likely milliseconds
                delta_t_ms = np.diff(time_values)
            
            # Filter out extreme outliers (keep 99% of data)
            # Handle case where std is 0 (perfectly uniform sampling)
            std_val = np.std(delta_t_ms)
            if std_val > 0:
                delta_t_ms = delta_t_ms[np.abs(delta_t_ms - np.median(delta_t_ms)) < 3 * std_val]
            # If std is 0, keep all samples (they're all identical)
            
            if len(delta_t_ms) == 0:
                print(f"     ⚠️ No valid time differences")
                continue
            
            # Calculate statistics
            file_stats = {
                "mean_delta_t_ms": float(np.mean(delta_t_ms)),
                "median_delta_t_ms": float(np.median(delta_t_ms)),
                "std_delta_t_ms": float(np.std(delta_t_ms)),
                "min_delta_t_ms": float(np.min(delta_t_ms)),
                "max_delta_t_ms": float(np.max(delta_t_ms)),
                "num_samples": len(delta_t_ms),
                "sampling_rate_hz": 1000.0 / float(np.median(delta_t_ms))
            }
            
            audit_results["per_file_stats"][os.path.basename(file_path)] = file_stats
            all_delta_t.extend(delta_t_ms)
            
            print(f"     ✅ Median: {file_stats['median_delta_t_ms']:.2f}ms ({file_stats['sampling_rate_hz']:.1f}Hz)")
            
        except Exception as e:
            print(f"     ❌ Error reading {file_path}: {e}")
            continue
    
    # Calculate overall statistics
    if all_delta_t:
        all_delta_t = np.array(all_delta_t)
        audit_results["sampling_stats"] = {
            "mean_delta_t_ms": float(np.mean(all_delta_t)),
            "median_delta_t_ms": float(np.median(all_delta_t)),
            "std_delta_t_ms": float(np.std(all_delta_t)),
            "min_delta_t_ms": float(np.min(all_delta_t)),
            "max_delta_t_ms": float(np.max(all_delta_t)),
            "sampling_rate_hz": 1000.0 / float(np.median(all_delta_t))
        }
        
        # Check if median is close to 20ms (50Hz)
        median_delta_t = audit_results["sampling_stats"]["median_delta_t_ms"]
        is_valid = 19.5 <= median_delta_t <= 20.5
        
        audit_results["summary"] = {
            "is_valid_50hz": is_valid,
            "expected_50hz": True,
            "actual_median_ms": median_delta_t,
            "actual_rate_hz": audit_results["sampling_stats"]["sampling_rate_hz"],
            "conclusion": "VALID" if is_valid else "INVALID - Check sensor configuration"
        }
        
        print(f"\n📊 Sampling Audit Summary:")
        print(f"   Median Δt: {median_delta_t:.2f}ms")
        print(f"   Sampling Rate: {audit_results['sampling_stats']['sampling_rate_hz']:.1f}Hz")
        print(f"   Status: {'✅ VALID (50Hz)' if is_valid else '❌ INVALID'}")
        
        if not is_valid:
            print(f"   ⚠️ Expected 20.0ms ± 0.5ms for 50Hz sensor")
    else:
        # No valid time differences found
        audit_results["summary"] = {
            "is_valid_50hz": False,
            "expected_50hz": True,
            "actual_median_ms": None,
            "actual_rate_hz": None,
            "conclusion": "INVALID - No valid time differences found"
        }
        
        print(f"\n❌ Sampling Audit Failed:")
        print(f"   No valid time differences found in {audit_results['files_audited']} files")
        print(f"   Check if files contain proper time columns")
    
    # Save audit results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(audit_results, f, indent=2)
    
    print(f"📝 Audit results saved to: {output_file}")
    return audit_results["summary"]["is_valid_50hz"]

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Audit IMU sampling rates")
    parser.add_argument("--raw-root", required=True, help="Path to raw IMU data")
    parser.add_argument("--out", required=True, help="Output JSON file for audit results")
    
    args = parser.parse_args()
    
    # Run audit
    is_valid = audit_sampling_rate(args.raw_root, args.out)
    
    if is_valid:
        print("✅ Sampling audit PASSED - 50Hz sensor confirmed")
        sys.exit(0)
    else:
        print("❌ Sampling audit FAILED - Check sensor configuration")
        sys.exit(1)

if __name__ == "__main__":
    main()
