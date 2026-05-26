# Non-Semantic Branch

## Setup
```bash
python3 -m venv .venv-ml
.venv-ml/bin/python -m pip install -r ML_Models/requirements.txt

python3 -m venv .venv-dl
.venv-dl/bin/python -m pip install -r DL_Models/requirements.txt
```

## Paths
- Dataset: set `NNEDGE_PRIVACY_DATA_ROOT` or use `Data/`.
- Results: set `NNEDGE_PRIVACY_RESULTS_ROOT` or use `Results/`.
- Shared path configuration: `project_paths.py`.

## Run
```bash
for freq in 25 10 5; do
  for script in ML_Models/game-*.py; do
    NNEDGE_PRIVACY_WINDOW_SIZES=25 NNEDGE_PRIVACY_MAX_COMBINATIONS=2 \
      .venv-ml/bin/python "$script" --frequency-hz "$freq"
  done

  for script in \
    DL_Models/game-1-train_cnn.py \
    DL_Models/game-1-train_rnn.py \
    DL_Models/game-1-train_transformer.py \
    DL_Models/game-2-train_cnn.py \
    DL_Models/game-2-train_rnn.py \
    DL_Models/game-2-train_transformer.py \
    DL_Models/game-3-train_cnn.py \
    DL_Models/game-3-train_rnn.py \
    DL_Models/game-3-train_transformer.py; do
    NNEDGE_PRIVACY_WINDOW_SIZES=25 NNEDGE_PRIVACY_MAX_COMBINATIONS=2 \
      .venv-dl/bin/python "$script" --frequency-hz "$freq"
  done
done
```

## Results
Outputs are saved under `Results/<frequency>Hz/`.
