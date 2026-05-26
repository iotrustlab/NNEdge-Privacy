# UTD-MHAD Branch

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Paths
- The top-level scripts expect the reorganized dataset at `UTD-MHAD-Reorganized/`.
- If the dataset is elsewhere, update the `data_dir = "UTD-MHAD-Reorganized"` assignment near the bottom of the script you want to run.
- Vulnerability summaries are written by `utd_vulnerability_utils.py` to `Results/`.

## Run
```bash
.venv/bin/python game-1-thigh-targeted-attack.py
.venv/bin/python game-1-wrist-binary-enhanced.py
.venv/bin/python game-1-wrist-binary-script.py
.venv/bin/python game-2-thigh-binary-accel.py
.venv/bin/python game-2-thigh-binary-gyro.py
.venv/bin/python game-2-wrist-comprehensive-evaluation.py
.venv/bin/python game-2-wrist-gyro-binary-comprehensive.py
.venv/bin/python game-3-thigh.py
.venv/bin/python game-3-wrist-multimodal.py
```

## Results
Reports, figures, and vulnerability summaries are written under `Results/`.
