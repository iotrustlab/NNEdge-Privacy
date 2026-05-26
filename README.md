# MotionSense Branch

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Paths
- Dataset and output paths are configured near the top of each script.
- The main variables are `motionsense_root`, `stm_root`, `reconstructed_root`, and `results_dir`.
- The default repo-relative layout uses `MotionSense/`, `STM_MotionSense/`, `STM_MotionSense_Reconstructed/`, and `results/`.

## Run
```bash
.venv/bin/python game-1-accel-gyro-reconstruction-inference.py
.venv/bin/python game-1-accel-reconstruction-inference.py
.venv/bin/python game-2-accel-cross_dataset_inference.py
.venv/bin/python game-2-gyro-cross_dataset_inference.py
.venv/bin/python game-2-gyro-reconstruction-based.py
.venv/bin/python game-3_stm_motionsense.py
```

## Results
- `results/game-1-accel-gyro-reconstruction-inference/`
- `results/game-1-accel-reconstruction-inference/`
- `results/game-2-accel_cross_dataset_inference/`
- `results/game-2-gyro_cross_dataset_inference/`
- `results/game-2-gyro-reconstruction-based/`
- `results/game-3-cross_dataset_inference/`
