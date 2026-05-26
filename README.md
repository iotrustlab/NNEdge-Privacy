# Identity+Anthrop Branch

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Paths
- Dataset: the scripts read from `Data/User */Processed/` and `Data/User */measurements.txt`.
- If the dataset is elsewhere, update the `Data/...` references in the `game-*.py` files before running.
- Results: each script writes to the `results/` path configured by its `self.base_results_dir` value.

## Run
```bash
.venv/bin/python game-1-user_identity_anthropometrics.py
.venv/bin/python game-2-user_identity_anthropometrics.py
.venv/bin/python game-2-gyro-user_identity_anthropometrics.py
.venv/bin/python game-3-user_identity_anthropometrics.py
```

## Results
- `results/game-1-user_identity_anthropometrics/`
- `results/game-2-user_identity_anthropometrics/`
- `results/game-2-gyro-user_identity_anthropometrics/`
- `results/game-3-user_identity_anthropometrics/`
