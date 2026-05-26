# STM-HAR-Cross-Sensor Branch

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Paths
- Dataset and output locations are configured near the top of each script.
- In `game-1-cross-sensor-cross-dataset.py`, update `self.stm_root`, `self.uci_root`, and `self.results_root` if needed.
- The other scripts use their own `results_dir` or `results_root` settings near the top of the file.

## Run
```bash
.venv/bin/python game-1-cross-sensor-cross-dataset.py
.venv/bin/python game-1-cross-sensor-file-wise.py
.venv/bin/python game-1-enhanced-with-accelerometer.py
.venv/bin/python game-2-gyro-stm-uci-cross-sensor-cross-dataset.py
.venv/bin/python game-2-scientific-optimized.py
.venv/bin/python game-3-cross-dataset-cross-sensor.py
```

## Results
- `Results/UCI-HAR/`
- `results/game-1-cross-sensor-cross-dataset-file-wise/`
- `results/game-1-enhanced-accelerometer/`
- `results/game-2-gyro-stm-uci/`
- `results/game-2-scientific/`
- `results/game-3-multimodal-combined/`
