import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import argparse

# Accepted activities and label map
ACTIVITY_MAP = {
    "climbingdown": "downstairs",
    "climbingup": "upstairs",
    "running": "jog",
    "sitting": "sit",
    "standing": "stand",
    "walking": "walk"
}

# Sensor target location
SENSOR_LOCATION = "waist"

def segment_signal(data, window_size, stride):
    segments = []
    for start in range(0, len(data) - window_size + 1, stride):
        seg = data[start:start + window_size]
        if len(seg) == window_size:
            segments.append(seg)
    return segments

def process_csv_file(csv_path):
    df = pd.read_csv(csv_path)
    return df[['attr_x', 'attr_y', 'attr_z']].values

def preprocess_realworld(dataset_path, save_path, window_size=250, stride=125):
    os.makedirs(os.path.join(save_path, "acc"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "gyro"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "mag"), exist_ok=True)

    sample_id = 0
    label_list = []

    for proband in sorted(os.listdir(dataset_path)):
        proband_path = os.path.join(dataset_path, proband, "data")
        if not os.path.exists(proband_path):
            continue

        for raw_activity in ACTIVITY_MAP:
            label = ACTIVITY_MAP[raw_activity]

            try:
                acc_dir = unzip_to_temp(os.path.join(proband_path, f"acc_{raw_activity}_csv.zip"))
                gyro_dir = unzip_to_temp(os.path.join(proband_path, f"gyr_{raw_activity}_csv.zip"))
                mag_dir = unzip_to_temp(os.path.join(proband_path, f"mag_{raw_activity}_csv.zip"))

                acc_file = os.path.join(acc_dir, f"acc_{raw_activity}_{SENSOR_LOCATION}.csv")
                gyro_file = os.path.join(gyro_dir, f"Gyroscope_{raw_activity}_{SENSOR_LOCATION}.csv")
                mag_file = os.path.join(mag_dir, f"MagneticField_{raw_activity}_{SENSOR_LOCATION}.csv")

                if not all(os.path.exists(f) for f in [acc_file, gyro_file, mag_file]):
                    continue

                acc_data = process_csv_file(acc_file)
                gyro_data = process_csv_file(gyro_file)
                mag_data = process_csv_file(mag_file)

                # Segment and sync
                acc_segs = segment_signal(acc_data, window_size, stride)
                gyro_segs = segment_signal(gyro_data, window_size, stride)
                mag_segs = segment_signal(mag_data, window_size, stride)
                n_segs = min(len(acc_segs), len(gyro_segs), len(mag_segs))

                for i in range(n_segs):
                    np.save(os.path.join(save_path, "acc", f"{sample_id}.npy"), acc_segs[i])
                    np.save(os.path.join(save_path, "gyro", f"{sample_id}.npy"), gyro_segs[i])
                    np.save(os.path.join(save_path, "mag", f"{sample_id}.npy"), mag_segs[i])
                    label_list.append(label)
                    sample_id += 1

            except Exception as e:
                print(f"⚠️ Error processing {proband}/{raw_activity}: {e}")

    # Encode labels
    le = LabelEncoder()
    le.fit(['downstairs', 'jog', 'sit', 'stand', 'upstairs', 'walk'])  # enforce consistent order
    y = le.transform(label_list)
    np.save(os.path.join(save_path, "label.npy"), y)

    print(f"\n✅ Done! {sample_id} samples saved to {save_path}")
    print("Label counts:", {cls: sum(np.array(label_list) == cls) for cls in le.classes_})

def unzip_to_temp(zip_path):
    extract_dir = zip_path.replace("_csv.zip", "_extracted")
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
    return extract_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to realworld2016_dataset/")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save preprocessed .npy files")
    parser.add_argument("--window_size", type=int, default=250)
    parser.add_argument("--stride", type=int, default=125)
    args = parser.parse_args()

    preprocess_realworld(args.dataset_path, args.save_path, args.window_size, args.stride)
