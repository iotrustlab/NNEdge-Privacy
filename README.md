# Identity_Placement_Ablation Branch

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Paths
- Dataset: the scripts read from `Data/User */Processed/`.
- If the dataset is elsewhere, update the `Data/...` references in the `game-*.py` files before running.
- Results: pass `--results-root` or use the default from `cross_sensor_ablation_common.py`, which is `Results/Identity-placement-ablation/50Hz/`.

## Run
```bash
.venv/bin/python game-1-cross-sensor.py --split 8_3 --max-combinations 10
.venv/bin/python game-2-cross-sensor.py --split 8_3 --max-combinations 10
.venv/bin/python game-2-gyro-cross-sensor.py --split 8_3 --max-combinations 10
.venv/bin/python game-3-cross-sensor.py --split 8_3 --max-combinations 10
```

## Results
Outputs are saved under `Results/Identity-placement-ablation/50Hz/<experiment>/`.
