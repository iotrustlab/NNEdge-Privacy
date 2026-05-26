import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import argparse

activity_map = {
    11111: 'walk',
    11112: 'stand',
    11113: 'jog',
    11114: 'sit',
    11115: 'bike',
    11116: 'upstairs',
    11117: 'downstairs',
    11118: 'type',
    11119: 'write',
    11120: 'coffee',
    11121: 'talk',
    11122: 'smoke',
    11123: 'eat'
}

# Activities we keep (to match MotionSense/RealWorld)
target_activities = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']

def segment_signal(signal, window_size, stride):
    segments = []
    for start in range(0, len(signal) - window_size + 1, stride):
        segment = signal[start:start + window_size]
        if len(segment) == window_size:
            segments.append(segment)
    return segments

def preprocess_shoaib(csv_path, save_path, window_size=100, stride=50):
    os.makedirs(os.path.join(save_path, "acc"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "gyro"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "mag"), exist_ok=True)

    df = pd.read_csv(csv_path)
    
    # Extract columns
    acc = df.iloc[:, 1:4].values   # accelerometer x,y,z
    gyro = df.iloc[:, 7:10].values # gyroscope x,y,z
    mag = df.iloc[:, 10:13].values # magnetometer x,y,z
    labels_raw = df.iloc[:, -1].values # activity codes
    
    # Map codes to names and filter
    labels_mapped = [activity_map.get(code) for code in labels_raw]
    
    # Filter by target activities
    mask = [label in target_activities for label in labels_mapped]
    acc, gyro, mag = acc[mask], gyro[mask], mag[mask]
    labels_filtered = np.array([label for label in labels_mapped if label in target_activities])

    # Segment
    all_labels = []
    sample_id = 0
    for start in range(0, len(labels_filtered) - window_size + 1, stride):
        window_labels = labels_filtered[start:start + window_size]
        if len(set(window_labels)) == 1:  # pure window (one activity)
            a_seg = acc[start:start + window_size]
            g_seg = gyro[start:start + window_size]
            m_seg = mag[start:start + window_size]

            np.save(os.path.join(save_path, "acc", f"{sample_id}.npy"), a_seg)
            np.save(os.path.join(save_path, "gyro", f"{sample_id}.npy"), g_seg)
            np.save(os.path.join(save_path, "mag", f"{sample_id}.npy"), m_seg)

            all_labels.append(window_labels[0])
            sample_id += 1

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(all_labels)
    np.save(os.path.join(save_path, "label.npy"), y)

    print(f"✅ Preprocessing complete: {sample_id} samples saved in {save_path}")
    print(f"Activities: {list(le.classes_)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to wrist or pocket CSV file")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save processed data")
    parser.add_argument("--window_size", type=int, default=100, help="Samples per window")
    parser.add_argument("--stride", type=int, default=50, help="Stride for sliding window")
    args = parser.parse_args()

    preprocess_shoaib(args.csv_path, args.save_path, args.window_size, args.stride)
