import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import argparse

def segment_signal(signal, window_size, stride):
    segments = []
    for start in range(0, len(signal) - window_size + 1, stride):
        segment = signal[start:start + window_size]
        if len(segment) == window_size:
            segments.append(segment)
    return segments

def preprocess_motionsense(raw_path, save_path, window_size=250, stride=125):
    os.makedirs(os.path.join(save_path, "acc"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "gyro"), exist_ok=True)

    all_labels = []
    sample_id = 0

    for activity_folder in sorted(os.listdir(raw_path)):
        folder_path = os.path.join(raw_path, activity_folder)
        if not os.path.isdir(folder_path):
            continue

        # Extract activity name (everything before '_')
        activity_label = activity_folder.split('_')[0]
        # Normalize label name for consistency
        if activity_label == 'std':
            activity_label = 'stand'

        for csv_file in sorted(os.listdir(folder_path)):
            if not csv_file.endswith(".csv"):
                continue
            
            file_path = os.path.join(folder_path, csv_file)
            df = pd.read_csv(file_path)

            # Drop first column if it's just an index
            if df.columns[0] == '':
                df = df.drop(df.columns[0], axis=1)

            # Extract accelerometer and gyroscope data
            acc = df[['userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z']].values
            gyro = df[['rotationRate.x', 'rotationRate.y', 'rotationRate.z']].values

            acc_segments = segment_signal(acc, window_size, stride)
            gyro_segments = segment_signal(gyro, window_size, stride)

            for a_seg, g_seg in zip(acc_segments, gyro_segments):
                np.save(os.path.join(save_path, "acc", f"{sample_id}.npy"), a_seg)
                np.save(os.path.join(save_path, "gyro", f"{sample_id}.npy"), g_seg)

                all_labels.append(activity_label)
                sample_id += 1

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(all_labels)
    np.save(os.path.join(save_path, "label.npy"), y)

    print(f"✅ Preprocessing complete: {sample_id} samples saved in {save_path}")
    print(f"Activities: {list(le.classes_)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=str, required=True, help="Path to raw MotionSense data")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save processed data")
    parser.add_argument("--window_size", type=int, default=250, help="Samples per window (default=250)")
    parser.add_argument("--stride", type=int, default=125, help="Stride for sliding window (default=125)")
    args = parser.parse_args()

    preprocess_motionsense(args.raw_path, args.save_path, args.window_size, args.stride)
