import os
import shutil
import pandas as pd

src_root = 'Data'
dst_root = 'STM_MotionSense'

def convert_and_rename_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        # Check for STM columns
        if 'acc_x[mg]' in df.columns and 'gyro_x[mdps]' in df.columns:
            # Convert units
            df['x'] = df['acc_x[mg]'] / 1000.0
            df['y'] = df['acc_y[mg]'] / 1000.0
            df['z'] = df['acc_z[mg]'] / 1000.0
            df['gyro_x'] = df['gyro_x[mdps]'] / 1000.0
            df['gyro_y'] = df['gyro_y[mdps]'] / 1000.0
            df['gyro_z'] = df['gyro_z[mdps]'] / 1000.0
            # Keep dec_tree_out_1 as well
            keep_cols = ['x', 'y', 'z', 'gyro_x', 'gyro_y', 'gyro_z', 'dec_tree_out_1']
            if 'time[us]' in df.columns:
                keep_cols = ['time[us]'] + keep_cols
            df = df[keep_cols]
            df.to_csv(file_path, index=False)
            print(f"Converted and renamed columns in {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def copy_measurements(user_path, user, dst_root):
    src_file = os.path.join(user_path, 'measurements.txt')
    if os.path.isfile(src_file):
        dst_dir = os.path.join(dst_root, user)
        os.makedirs(dst_dir, exist_ok=True)
        dst_file = os.path.join(dst_dir, 'measurements.txt')
        shutil.copy2(src_file, dst_file)
        print(f"Copied {src_file} -> {dst_file}")

def copy_processed(user_path, user, dst_root):
    processed_path = os.path.join(user_path, 'Processed')
    if not os.path.isdir(processed_path):
        return
    for activity in os.listdir(processed_path):
        if 'laying' in activity.lower():
            continue  # Skip laying activity
        activity_path = os.path.join(processed_path, activity)
        if not os.path.isdir(activity_path):
            continue
        for item in os.listdir(activity_path):
            item_path = os.path.join(activity_path, item)
            if os.path.isfile(item_path):
                if 'pocket' in item.lower():
                    dst_dir = os.path.join(dst_root, user, 'Processed', activity)
                    os.makedirs(dst_dir, exist_ok=True)
                    dst_file = os.path.join(dst_dir, item)
                    shutil.copy2(item_path, dst_file)
                    print(f"Copied {item_path} -> {dst_file}")
                    convert_and_rename_csv(dst_file)
            elif os.path.isdir(item_path):
                if 'pocket' in item.lower():
                    for subfile in os.listdir(item_path):
                        subfile_path = os.path.join(item_path, subfile)
                        if os.path.isfile(subfile_path):
                            dst_dir = os.path.join(dst_root, user, 'Processed', activity, item)
                            os.makedirs(dst_dir, exist_ok=True)
                            dst_file = os.path.join(dst_dir, subfile)
                            shutil.copy2(subfile_path, dst_file)
                            print(f"Copied {subfile_path} -> {dst_file}")
                            convert_and_rename_csv(dst_file)

for user in os.listdir(src_root):
    user_path = os.path.join(src_root, user)
    if not os.path.isdir(user_path):
        continue
    copy_measurements(user_path, user, dst_root)
    copy_processed(user_path, user, dst_root)
