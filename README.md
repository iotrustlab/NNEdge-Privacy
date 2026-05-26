# feat/semantics-minigrid-v3 Branch

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Paths
- Update `DATA_ROOT` and `RESULTS_ROOT` in `games/STM/config.py` if needed.
- The frequency-study runner also writes processed intermediates under `Data_processed/`.

## Run
```bash
.venv/bin/python games/STM/experiments.py --grid multi --classes 3 4 5 7 --featuresets concatenated
.venv/bin/python games/STM/runner.py --games Game-1 Game-2-accel Game-2-gyro Game-3 --models advanced_cnn
.venv/bin/python games/STM/run_game1_freq_refit_50hz.py
```

## Results
- `results/experiments/`
- `results/game1_freq_refit_50hz/`
- `results/STM/`
