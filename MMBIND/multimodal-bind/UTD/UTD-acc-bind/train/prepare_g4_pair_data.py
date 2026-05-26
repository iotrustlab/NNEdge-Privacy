from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


PAIR_DIRS = ("train_skeleton_paired_AB", "train_gyro_paired_AB")


def parse_args():
    parser = argparse.ArgumentParser("Prepare UTD G4 paired-data ablations")
    parser.add_argument("--source_root", type=str, default="./save_mmbind",
                        help="root containing canonical train_skeleton_paired_AB and train_gyro_paired_AB")
    parser.add_argument("--output_root", type=str, default="./save_g4_pairs",
                        help="root where condition-specific pair directories are written")
    parser.add_argument("--condition_name", type=str, required=True,
                        help="condition name used under output_root")
    parser.add_argument("--paired_fraction", type=float, default=1.0,
                        help="fraction of aligned pair data to retain")
    parser.add_argument("--pairing", type=str, default="aligned",
                        choices=["aligned", "mismatched"], help="how to pair private-modality samples")
    parser.add_argument("--seed", type=int, default=41, help="random seed")
    return parser.parse_args()


def load_array(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def hash_array(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr).tobytes()


def subset_indices(labels: np.ndarray, paired_fraction: float, seed: int) -> np.ndarray:
    if paired_fraction >= 1.0:
        return np.arange(labels.shape[0])

    all_indices = np.arange(labels.shape[0])
    selected, _ = train_test_split(
        all_indices,
        train_size=paired_fraction,
        stratify=labels,
        random_state=seed,
    )
    return np.sort(selected)


def infer_labels_from_source(modality: str, samples: list[np.ndarray], source_root: Path) -> np.ndarray:
    utd_root = source_root.parents[2] / "UTD-split-222"
    if modality == "gyro":
        raw_labels = load_array(utd_root / "train_B" / "label.npy")
        raw_dir = utd_root / "train_B" / "inertial"
        raw_samples = [load_array(raw_dir / f"{idx}.npy")[:, 3:6] for idx in range(raw_labels.shape[0])]
    else:
        raw_labels = load_array(utd_root / "train_A" / "label.npy")
        raw_dir = utd_root / "train_A" / "skeleton"
        raw_samples = [load_array(raw_dir / f"{idx}.npy") for idx in range(raw_labels.shape[0])]

    label_lookup = {hash_array(sample): raw_labels[idx] for idx, sample in enumerate(raw_samples)}
    private_labels = []
    for sample in samples:
        key = hash_array(sample)
        if key not in label_lookup:
            raise KeyError(f"Could not recover {modality} label for paired sample")
        private_labels.append(label_lookup[key])
    return np.array(private_labels)


def best_derangement(reference_labels: np.ndarray, private_labels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    size = reference_labels.shape[0]
    base_indices = np.arange(size)
    best_perm = None
    best_same_class = size + 1

    for _ in range(5000):
        perm = rng.permutation(size)
        if np.any(perm == base_indices):
            continue
        same_class = int(np.sum(private_labels[perm] == reference_labels))
        if same_class < best_same_class:
            best_same_class = same_class
            best_perm = perm.copy()
            if same_class == 0:
                break

    if best_perm is None:
        raise RuntimeError("Failed to construct a derangement for mismatched pairing")

    return best_perm


def write_pair_dir(
    output_root: Path,
    pair_dir_name: str,
    skeleton_samples: list[np.ndarray],
    gyro_samples: list[np.ndarray],
    labels: np.ndarray,
    similarity: np.ndarray,
    pairing_record: np.ndarray,
):
    target_dir = output_root / pair_dir_name
    if target_dir.exists():
        shutil.rmtree(target_dir)
    (target_dir / "skeleton").mkdir(parents=True)
    (target_dir / "gyro").mkdir(parents=True)

    for idx, sample in enumerate(skeleton_samples):
        np.save(target_dir / "skeleton" / f"{idx}.npy", sample)
    for idx, sample in enumerate(gyro_samples):
        np.save(target_dir / "gyro" / f"{idx}.npy", sample)

    np.save(target_dir / "label.npy", labels)
    np.save(target_dir / "similarity.npy", similarity)
    np.save(target_dir / "pairing_record.npy", pairing_record)


def load_pair_dir(source_root: Path, pair_dir_name: str):
    pair_dir = source_root / pair_dir_name
    labels = load_array(pair_dir / "label.npy")
    similarity = load_array(pair_dir / "similarity.npy")
    pairing_record = load_array(pair_dir / "pairing_record.npy") if (pair_dir / "pairing_record.npy").exists() else None
    skeleton_samples = [load_array(pair_dir / "skeleton" / f"{idx}.npy") for idx in range(labels.shape[0])]
    gyro_samples = [load_array(pair_dir / "gyro" / f"{idx}.npy") for idx in range(labels.shape[0])]
    return labels, similarity, pairing_record, skeleton_samples, gyro_samples


def prepare_pair_dir(source_root: Path, output_root: Path, pair_dir_name: str, paired_fraction: float, pairing: str, seed: int):
    labels, similarity, original_pairing_record, skeleton_samples, gyro_samples = load_pair_dir(source_root, pair_dir_name)
    selected = subset_indices(labels, paired_fraction, seed)

    selected_labels = labels[selected]
    selected_similarity = similarity[selected]
    selected_skeleton = [skeleton_samples[idx] for idx in selected]
    selected_gyro = [gyro_samples[idx] for idx in selected]

    if pairing == "mismatched":
        skeleton_labels = infer_labels_from_source("skeleton", selected_skeleton, source_root)
        if pair_dir_name == "train_skeleton_paired_AB":
            fixed_imu_labels = infer_labels_from_source("gyro", selected_gyro, source_root)
        else:
            fixed_imu_labels = selected_labels

        perm = best_derangement(fixed_imu_labels, skeleton_labels, seed)
        selected_skeleton = [selected_skeleton[idx] for idx in perm]
        pairing_record = (skeleton_labels[perm] == fixed_imu_labels).astype(np.float32)
    else:
        pairing_record = original_pairing_record[selected] if original_pairing_record is not None else np.ones(selected_labels.shape[0], dtype=np.float32)

    write_pair_dir(
        output_root=output_root,
        pair_dir_name=pair_dir_name,
        skeleton_samples=selected_skeleton,
        gyro_samples=selected_gyro,
        labels=selected_labels,
        similarity=selected_similarity,
        pairing_record=pairing_record,
    )


def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve() / args.condition_name
    output_root.mkdir(parents=True, exist_ok=True)

    for offset, pair_dir_name in enumerate(PAIR_DIRS):
        prepare_pair_dir(
            source_root=source_root,
            output_root=output_root,
            pair_dir_name=pair_dir_name,
            paired_fraction=args.paired_fraction,
            pairing=args.pairing,
            seed=args.seed + offset,
        )

    print(output_root)


if __name__ == "__main__":
    main()
