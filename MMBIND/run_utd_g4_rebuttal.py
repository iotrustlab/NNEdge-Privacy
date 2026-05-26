from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
MMBIND_ROOT = ROOT / "MMBIND" / "multimodal-bind"
TRAIN_DIR = MMBIND_ROOT / "UTD" / "UTD-acc-bind" / "train"
EVAL_DIR = MMBIND_ROOT / "UTD" / "UTD-acc-bind" / "evaluation"
PYTHON = MMBIND_ROOT / ".venv-mmbind" / "bin" / "python"
RESULTS_DIR = ROOT / "results"
LOG_DIR = RESULTS_DIR / "g4_utd_mhad_rebuttal_logs"
PRIVATE_MODALITY = "Skeleton"
LABEL_SETS = ["label_216", "label_162", "label_108", "label_54"]
SEEDS = [42, 43, 44, 45, 46]
LATE_EPOCH_START = 80
PAPER_ACC_TARGET = 78.86
PAPER_F1_TARGET = 0.763
MAX_PARALLEL_CANONICAL_EVAL = 2
MAX_PARALLEL_STAGE3 = int(os.environ.get("G4_MAX_PARALLEL_STAGE3", "1"))
MAX_PARALLEL_IMU_EVAL = int(os.environ.get("G4_MAX_PARALLEL_IMU_EVAL", "1"))


def run(cmd: list[str], cwd: Path):
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_path(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def run_many(jobs: list[dict], max_parallel: int):
    if not jobs:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    running: list[dict] = []

    while pending or running:
        while pending and len(running) < max_parallel:
            job = pending.pop(0)
            log_handle = job["log_path"].open("w")
            proc = subprocess.Popen(
                job["cmd"],
                cwd=job["cwd"],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            print(f"[start] {job['name']} -> {job['log_path']}")
            running.append({
                "name": job["name"],
                "proc": proc,
                "log_handle": log_handle,
                "log_path": job["log_path"],
            })

        next_running = []
        for job in running:
            returncode = job["proc"].poll()
            if returncode is None:
                next_running.append(job)
                continue

            job["log_handle"].close()
            if returncode != 0:
                raise RuntimeError(f"{job['name']} failed; see {job['log_path']}")
            if "done_path" in job:
                job["done_path"].write_text("done\n")
            print(f"[done] {job['name']}")

        running = next_running
        if pending or running:
            time.sleep(1)


def canonical_stage1_ckpt() -> Path:
    return TRAIN_DIR / "save_mmbind" / "save_train_AB_acc_AE" / "models" / "single_train_AB_lr_0.001_decay_0.0001_bsz_128" / "last.pth"


def canonical_stage3_weighted_ckpt() -> Path:
    return TRAIN_DIR / "save_mmbind" / "save_train_all_paired_AB_incomplete_contrastive_weighted_no_pretrain" / "models" / "lr_0.0005_decay_0.0001_bsz_64" / "last.pth"


def condition_stage3_ckpt(save_name: str) -> Path:
    return TRAIN_DIR / "save_mmbind" / save_name / "models" / "lr_0.0005_decay_0.0001_bsz_64" / "last.pth"


def ensure_stage1():
    if canonical_stage1_ckpt().exists():
        return
    run([str(PYTHON), "main_mmbind_1_acc_autencoder.py"], cwd=TRAIN_DIR)


def ensure_stage2_pairs():
    skeleton_pair_dir = TRAIN_DIR / "save_mmbind" / "train_skeleton_paired_AB_test"
    gyro_pair_dir = TRAIN_DIR / "save_mmbind" / "train_gyro_paired_AB_test"
    if not skeleton_pair_dir.exists() or not gyro_pair_dir.exists():
        run([str(PYTHON), "main_mmbind_2_measure_similarity.py", "--reference_modality", "skeleton"], cwd=TRAIN_DIR)
        run([str(PYTHON), "main_mmbind_2_measure_similarity.py", "--reference_modality", "gyro"], cwd=TRAIN_DIR)

    for src_name, dst_name in (
        ("train_skeleton_paired_AB_test", "train_skeleton_paired_AB"),
        ("train_gyro_paired_AB_test", "train_gyro_paired_AB"),
    ):
        src = TRAIN_DIR / "save_mmbind" / src_name
        dst = TRAIN_DIR / "save_mmbind" / dst_name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def ensure_canonical_stage3():
    if canonical_stage3_weighted_ckpt().exists():
        return
    run(
        [str(PYTHON), "main_mmbind_3_incomplete_contrastive_weighted.py", "--print_freq", "1000"],
        cwd=TRAIN_DIR,
    )


def prepare_pair_condition(condition_name: str, paired_fraction: float, pairing: str):
    output_root = TRAIN_DIR / "save_g4_pairs" / condition_name
    if output_root.exists():
        shutil.rmtree(output_root)
    run(
        [
            str(PYTHON),
            "prepare_g4_pair_data.py",
            "--source_root",
            str(TRAIN_DIR / "save_mmbind"),
            "--output_root",
            str(TRAIN_DIR / "save_g4_pairs"),
            "--condition_name",
            condition_name,
            "--paired_fraction",
            str(paired_fraction),
            "--pairing",
            pairing,
            "--seed",
            "41",
        ],
        cwd=TRAIN_DIR,
    )


def ensure_condition_stage3(condition_name: str):
    ckpt = condition_stage3_ckpt(f"g4_{condition_name}")
    if ckpt.exists():
        return
    run(
        [
            str(PYTHON),
            "main_mmbind_3_incomplete_contrastive_weighted.py",
            "--print_freq",
            "1000",
            "--paired_data_root",
            str(TRAIN_DIR / "save_g4_pairs" / condition_name),
            "--save_name",
            f"g4_{condition_name}",
        ],
        cwd=TRAIN_DIR,
    )


def canonical_repro_result_dir(label_set: str, trial_id: int) -> Path:
    return EVAL_DIR / "save_test_train_C" / label_set / "mmbind_all_weighted_incomplete_no_pretrain" / f"trial_{trial_id}" / "results"


def imu_result_dir(condition_name: str, label_set: str, trial_id: int) -> Path:
    dataset_token = f"train_C_{label_set}"
    return EVAL_DIR / "save_g4_imu" / condition_name / dataset_token / f"trial_{trial_id}" / "results"


def job_done_path(job_name: str) -> Path:
    return LOG_DIR / f"{job_name}.done"


def ensure_canonical_reproduction():
    jobs = []
    for label_set in LABEL_SETS:
        job_name = f"canonical_eval_{label_set}"
        if job_done_path(job_name).exists():
            continue
        jobs.append({
            "name": job_name,
            "cmd": [
                str(PYTHON),
                "main_fuse_sup_mmbind_incomplete_contrastive_weighted.py",
                "--dataset",
                f"train_C/{label_set}/",
                "--print_freq",
                "1000",
            ],
            "cwd": EVAL_DIR,
            "log_path": LOG_DIR / f"{job_name}.log",
            "done_path": job_done_path(job_name),
        })
    run_many(jobs, MAX_PARALLEL_CANONICAL_EVAL)


def ensure_imu_condition(condition_name: str, acc_ckpt: str = "", gyro_ckpt: str = ""):
    jobs = []
    for label_set in LABEL_SETS:
        job_name = f"imu_eval_{condition_name}_{label_set}"
        if job_done_path(job_name).exists():
            continue
        cmd = [
            str(PYTHON),
            "main_fuse_sup_g4_imu_only.py",
            "--dataset",
            f"train_C/{label_set}/",
            "--condition_name",
            condition_name,
            "--print_freq",
            "1000",
        ]
        if acc_ckpt:
            cmd.extend(["--acc_ckpt", acc_ckpt])
        if gyro_ckpt:
            cmd.extend(["--gyro_ckpt", gyro_ckpt])
        jobs.append({
            "name": job_name,
            "cmd": cmd,
            "cwd": EVAL_DIR,
            "log_path": LOG_DIR / f"{job_name}.log",
            "done_path": job_done_path(job_name),
        })
    run_many(jobs, MAX_PARALLEL_IMU_EVAL)


def late_epoch_mean(path: Path) -> float:
    arr = np.loadtxt(path)
    return float(np.mean(arr[LATE_EPOCH_START:]))


def aggregate_condition(result_dir_fn, condition_name: str):
    per_seed = []
    detailed = []
    for seed_idx, seed in enumerate(SEEDS):
        acc_vals = []
        f1_vals = []
        for label_set in LABEL_SETS:
            result_dir = result_dir_fn(label_set, seed_idx)
            acc = late_epoch_mean(result_dir / "test_accuracy.txt")
            f1 = late_epoch_mean(result_dir / "test_f1.txt")
            acc_vals.append(acc)
            f1_vals.append(f1)
            detailed.append({
                "condition_name": condition_name,
                "seed": seed,
                "train_c_subset": label_set,
                "accuracy": acc,
                "macro_f1": f1,
            })
        per_seed.append({
            "seed": seed,
            "accuracy": float(np.mean(acc_vals)),
            "macro_f1": float(np.mean(f1_vals)),
        })
    return per_seed, detailed


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    ensure_path(path)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_metric(accuracy_mean, accuracy_std, macro_f1_mean, macro_f1_std):
    return f"{accuracy_mean:.2f} ± {accuracy_std:.2f} / {macro_f1_mean:.3f} ± {macro_f1_std:.3f}"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "seed_list": SEEDS,
        "late_epoch_aggregation_start": LATE_EPOCH_START,
        "notes": [
            "Canonical UTD acc-bind reproduction uses the weighted MMbind path because train_mmbind.sh maps to weighted stage 3 and eval_mmbind_no_weighted.sh points at the weighted evaluation entrypoint.",
            "Original reproduction uses Skeleton + Gyro at train_C/test time because load_data_incomplete zeros Acc on train_C and test in evaluation/shared_files/data_pre.py.",
            "Stage 2 writes train_*_paired_AB_test, so reproduction copies those directories to train_*_paired_AB for the stage-3 loader.",
            "PyTorch 2.6 compatibility required explicit weights_only=False in checkpoint loads.",
            "Weighted stage 3 also required moving similarity weights to CUDA before the contrastive loss call.",
        ],
        "entrypoints": {
            "canonical_train": [
                str(TRAIN_DIR / "main_mmbind_1_acc_autencoder.py"),
                str(TRAIN_DIR / "main_mmbind_2_measure_similarity.py --reference_modality skeleton"),
                str(TRAIN_DIR / "main_mmbind_2_measure_similarity.py --reference_modality gyro"),
                str(TRAIN_DIR / "main_mmbind_3_incomplete_contrastive_weighted.py"),
            ],
            "canonical_eval": str(EVAL_DIR / "main_fuse_sup_mmbind_incomplete_contrastive_weighted.py"),
            "imu_only_eval": str(EVAL_DIR / "main_fuse_sup_g4_imu_only.py"),
            "pair_prep": str(TRAIN_DIR / "prepare_g4_pair_data.py"),
        },
    }

    ensure_stage1()
    ensure_stage2_pairs()
    ensure_canonical_stage3()
    ensure_canonical_reproduction()

    stage3_jobs = []
    for condition_name, paired_fraction, pairing in (
        ("mismatched_100", 1.0, "mismatched"),
        ("aligned_50", 0.5, "aligned"),
        ("aligned_25", 0.25, "aligned"),
    ):
        prepare_pair_condition(condition_name, paired_fraction, pairing)
        ckpt = condition_stage3_ckpt(f"g4_{condition_name}")
        if ckpt.exists():
            continue
        stage3_jobs.append({
            "name": f"stage3_{condition_name}",
            "cmd": [
                str(PYTHON),
                "main_mmbind_3_incomplete_contrastive_weighted.py",
                "--print_freq",
                "1000",
                "--paired_data_root",
                str(TRAIN_DIR / "save_g4_pairs" / condition_name),
                "--save_name",
                f"g4_{condition_name}",
            ],
            "cwd": TRAIN_DIR,
            "log_path": LOG_DIR / f"stage3_{condition_name}.log",
        })
    run_many(stage3_jobs, MAX_PARALLEL_STAGE3)

    ensure_imu_condition("imu_only_no_alignment")
    canonical_ckpt = str(canonical_stage3_weighted_ckpt())
    ensure_imu_condition("aligned_100", acc_ckpt=canonical_ckpt, gyro_ckpt=canonical_ckpt)
    ensure_imu_condition("mismatched_100", acc_ckpt=str(condition_stage3_ckpt("g4_mismatched_100")), gyro_ckpt=str(condition_stage3_ckpt("g4_mismatched_100")))
    ensure_imu_condition("aligned_50", acc_ckpt=str(condition_stage3_ckpt("g4_aligned_50")), gyro_ckpt=str(condition_stage3_ckpt("g4_aligned_50")))
    ensure_imu_condition("aligned_25", acc_ckpt=str(condition_stage3_ckpt("g4_aligned_25")), gyro_ckpt=str(condition_stage3_ckpt("g4_aligned_25")))

    canonical_per_seed, canonical_detailed = aggregate_condition(
        lambda label_set, trial_id: canonical_repro_result_dir(label_set, trial_id),
        "canonical_reference_weighted",
    )
    canonical_acc_mean = float(np.mean([row["accuracy"] for row in canonical_per_seed]))
    canonical_acc_std = float(np.std([row["accuracy"] for row in canonical_per_seed], ddof=0))
    canonical_f1_mean = float(np.mean([row["macro_f1"] for row in canonical_per_seed]))
    canonical_f1_std = float(np.std([row["macro_f1"] for row in canonical_per_seed], ddof=0))

    condition_meta = {
        "imu_only_no_alignment": {"pairing": "none", "paired_data_fraction": 0.0},
        "aligned_100": {"pairing": "correct/aligned", "paired_data_fraction": 1.0},
        "mismatched_100": {"pairing": "shuffled/mismatched", "paired_data_fraction": 1.0},
        "aligned_50": {"pairing": "correct/aligned", "paired_data_fraction": 0.5},
        "aligned_25": {"pairing": "correct/aligned", "paired_data_fraction": 0.25},
    }

    result_rows = []
    summary_rows = []
    detailed_rows = []
    condition_summaries = {}

    for condition_name in condition_meta:
        per_seed, detailed = aggregate_condition(
            lambda label_set, trial_id, cn=condition_name: imu_result_dir(cn, label_set, trial_id),
            condition_name,
        )
        detailed_rows.extend(detailed)

        acc_values = [row["accuracy"] for row in per_seed]
        f1_values = [row["macro_f1"] for row in per_seed]
        acc_mean = float(np.mean(acc_values))
        acc_std = float(np.std(acc_values, ddof=0))
        f1_mean = float(np.mean(f1_values))
        f1_std = float(np.std(f1_values, ddof=0))

        condition_summaries[condition_name] = {
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "macro_f1_mean": f1_mean,
            "macro_f1_std": f1_std,
        }

        for row in per_seed:
            result_rows.append({
                "condition_name": condition_name,
                "pairing": condition_meta[condition_name]["pairing"],
                "paired_data_fraction": condition_meta[condition_name]["paired_data_fraction"],
                "test_time_input": "IMU only",
                "private_modality": PRIVATE_MODALITY,
                "seed": row["seed"],
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
            })

        summary_rows.append({
            "condition_name": condition_name,
            "pairing": condition_meta[condition_name]["pairing"],
            "paired_data_fraction": condition_meta[condition_name]["paired_data_fraction"],
            "test_time_input": "IMU only",
            "private_modality": PRIVATE_MODALITY,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "macro_f1_mean": f1_mean,
            "macro_f1_std": f1_std,
        })

    write_csv(
        RESULTS_DIR / "g4_utd_mhad_rebuttal_results.csv",
        ["condition_name", "pairing", "paired_data_fraction", "test_time_input", "private_modality", "seed", "accuracy", "macro_f1"],
        result_rows,
    )
    write_csv(
        RESULTS_DIR / "g4_utd_mhad_rebuttal_summary.csv",
        ["condition_name", "pairing", "paired_data_fraction", "test_time_input", "private_modality", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"],
        summary_rows,
    )
    write_csv(
        RESULTS_DIR / "g4_utd_mhad_rebuttal_results_detailed.csv",
        ["condition_name", "seed", "train_c_subset", "accuracy", "macro_f1"],
        detailed_rows + canonical_detailed,
    )

    latex = "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{G4 semantic leakage under cross-modal alignment ablations on UTD-MHAD. The adversary observes IMU only at inference time and predicts the activity/gesture label.}",
        "\\label{tab:g4_utd_mhad_ablation}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Setting & Pairing & Paired data & Acc. / Macro-F1 \\\\",
        "\\midrule",
        f"IMU-only baseline & None & 0\\% & {format_metric(**condition_summaries['imu_only_no_alignment'])} \\\\",
        f"G4 aligned & Correct & 100\\% & {format_metric(**condition_summaries['aligned_100'])} \\\\",
        f"G4 mismatched & Shuffled & 100\\% & {format_metric(**condition_summaries['mismatched_100'])} \\\\",
        f"G4 reduced & Correct & 50\\% & {format_metric(**condition_summaries['aligned_50'])} \\\\",
        f"G4 reduced & Correct & 25\\% & {format_metric(**condition_summaries['aligned_25'])} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    (RESULTS_DIR / "g4_utd_mhad_rebuttal_table.tex").write_text(latex)

    aligned = condition_summaries["aligned_100"]
    baseline = condition_summaries["imu_only_no_alignment"]
    mismatched = condition_summaries["mismatched_100"]
    aligned_50 = condition_summaries["aligned_50"]
    aligned_25 = condition_summaries["aligned_25"]

    supports_baseline = aligned["accuracy_mean"] > baseline["accuracy_mean"] and aligned["macro_f1_mean"] > baseline["macro_f1_mean"]
    supports_mismatch = mismatched["accuracy_mean"] < aligned["accuracy_mean"] and mismatched["macro_f1_mean"] < aligned["macro_f1_mean"]
    supports_fraction = (
        aligned_50["accuracy_mean"] < aligned["accuracy_mean"]
        and aligned_25["accuracy_mean"] < aligned["accuracy_mean"]
        and aligned_50["macro_f1_mean"] < aligned["macro_f1_mean"]
        and aligned_25["macro_f1_mean"] < aligned["macro_f1_mean"]
    )
    support_count = sum([supports_baseline, supports_mismatch, supports_fraction])
    conclusion_word = "support" if support_count == 3 else "partially support" if support_count >= 1 else "do not support"

    acc_match = abs(canonical_acc_mean - PAPER_ACC_TARGET)
    f1_match = abs(canonical_f1_mean - PAPER_F1_TARGET)
    matches_paper = acc_match <= 2.0 and f1_match <= 0.03

    interpretation = "\n".join([
        "# UTD-MHAD G4 Rebuttal Interpretation",
        "",
        "## Original Reference Reproduction",
        f"- Canonical reproduction path: weighted Acc-binding MMbind with `main_mmbind_1_acc_autencoder.py`, `main_mmbind_2_measure_similarity.py --reference_modality skeleton`, `main_mmbind_2_measure_similarity.py --reference_modality gyro`, `main_mmbind_3_incomplete_contrastive_weighted.py`, and `main_fuse_sup_mmbind_incomplete_contrastive_weighted.py`.",
        f"- Reproduced canonical reference: {canonical_acc_mean:.2f} ± {canonical_acc_std:.2f} accuracy / {canonical_f1_mean:.3f} ± {canonical_f1_std:.3f} macro-F1.",
        f"- Paper comparability target: {PAPER_ACC_TARGET:.2f} accuracy / {PAPER_F1_TARGET:.3f} macro-F1.",
        f"- Match status: {'matches the cited result within expected variance' if matches_paper else 'does not fully match the cited result within the tolerance used here'} (|Δacc|={acc_match:.2f}, |ΔF1|={f1_match:.3f}).",
        "- Important protocol note: the original canonical evaluation is not IMU-only at test time. In `evaluation/shared_files/data_pre.py`, `load_data_incomplete` zeros `Acc` on `train_C` and `test`, so the reproduced reference evaluates `Skeleton + Gyro`, not IMU only.",
        "",
        "## IMU-Only G4 Ablations",
        f"1. `aligned_100` vs `imu_only_no_alignment`: {'aligned_100 outperforms imu_only_no_alignment' if supports_baseline else 'aligned_100 does not outperform imu_only_no_alignment'} ({aligned['accuracy_mean']:.2f}/{aligned['macro_f1_mean']:.3f} vs {baseline['accuracy_mean']:.2f}/{baseline['macro_f1_mean']:.3f}).",
        f"2. `mismatched_100` vs `aligned_100`: {'mismatched_100 drops relative to aligned_100' if supports_mismatch else 'mismatched_100 does not drop relative to aligned_100'} ({mismatched['accuracy_mean']:.2f}/{mismatched['macro_f1_mean']:.3f} vs {aligned['accuracy_mean']:.2f}/{aligned['macro_f1_mean']:.3f}).",
        f"3. `aligned_50` and `aligned_25` vs `aligned_100`: {'both reduced paired-data conditions degrade relative to aligned_100' if supports_fraction else 'the reduced paired-data conditions do not both degrade relative to aligned_100'} ({aligned_50['accuracy_mean']:.2f}/{aligned_50['macro_f1_mean']:.3f} and {aligned_25['accuracy_mean']:.2f}/{aligned_25['macro_f1_mean']:.3f} vs {aligned['accuracy_mean']:.2f}/{aligned['macro_f1_mean']:.3f}).",
        "",
        f"These results {conclusion_word} the claim that G4 leakage arises from shared semantic structure learned through cross-modal alignment rather than from model capacity alone.",
        "",
    ])
    (RESULTS_DIR / "g4_utd_mhad_rebuttal_interpretation.md").write_text(interpretation)

    manifest["canonical_reference_weighted"] = {
        "accuracy_mean": canonical_acc_mean,
        "accuracy_std": canonical_acc_std,
        "macro_f1_mean": canonical_f1_mean,
        "macro_f1_std": canonical_f1_std,
    }
    manifest["imu_only_summary"] = condition_summaries
    (RESULTS_DIR / "g4_utd_mhad_rebuttal_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
