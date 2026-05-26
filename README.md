# Identity Branch

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Paths
- Dataset: set `NNEDGE_PRIVACY_DATA_ROOT` or use `Data/`.
- Results: set `NNEDGE_PRIVACY_RESULTS_ROOT` or use `Results/Identity/`.
- Shared path configuration: `project_paths.py`.

## Run
```bash
for freq in 25 10 5; do
  .venv/bin/python game-1-user_identity_only.py --frequency-hz "$freq" --max-combinations 5
  .venv/bin/python game-2-user_identity_only.py --frequency-hz "$freq" --max-combinations 5
  .venv/bin/python game-2-gyro-user_identity_only.py --frequency-hz "$freq" --max-combinations 5
  .venv/bin/python game-3-user_identity_only.py --frequency-hz "$freq" --max-combinations 5
done
```

## Results
Outputs are saved under `Results/Identity/<frequency>Hz/`.
