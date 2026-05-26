# STM-MotionSense-Cross-Sensor-Cross-Dataset Branch

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Paths
- Dataset roots are resolved in `cross_dataset_utils.py`.
- Set `NNEDGE_PRIVACY_STM_ROOT`, `NNEDGE_PRIVACY_MOTIONSENSE_ROOT`, or `NNEDGE_PRIVACY_DATA_ROOT` if your datasets are not in the repo-relative defaults.
- Output directories are configured by each script through `results_dir` or related top-level settings.

## Run
```bash
.venv/bin/python game-1-cross-sensor-cross-dataset.py
.venv/bin/python game-1-enhanced_filewise.py
.venv/bin/python game-2-accel-cross-sensor-cross-dataset.py
.venv/bin/python game-2-gyro-improved-placement-agnostic.py
.venv/bin/python game-3-cross-dataset-cross-sensor.py
```

## Results
- `results/game-1/STM_MultiPlacement_to_MotionSense/`
- `results/game-1/MotionSense_to_STM_Placements/`
- the `results_dir` configured inside `game-1-enhanced_filewise.py`
- `results/game-2-binary-optimized/`
- `results/game-2-gyro-improved/STM_MultiPlacement_to_MotionSense/`
- `results/game-2-gyro-improved/MotionSense_to_STM_Placements/`
- `results/game-3-multimodal-combined/`
