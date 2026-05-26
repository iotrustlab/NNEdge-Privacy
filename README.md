# MMBIND Branch

## Setup
```bash
cd MMBIND/multimodal-bind
python3 -m venv .venv-mmbind
source .venv-mmbind/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install numpy scipy scikit-learn matplotlib pandas opencv-python torch
```

## Paths
- Branch entrypoint: `MMBIND/run_utd_g4_rebuttal.py`.
- Main path configuration: the `ROOT`, `MMBIND_ROOT`, `TRAIN_DIR`, `EVAL_DIR`, and `RESULTS_DIR` constants near the top of that file.
- The underlying UTD loaders resolve their processed split paths from the MMbind files under `MMBIND/multimodal-bind/UTD/UTD-acc-bind/train/shared_files/` and `MMBIND/multimodal-bind/UTD/UTD-acc-bind/evaluation/shared_files/`.

## Run
```bash
MMBIND/multimodal-bind/.venv-mmbind/bin/python MMBIND/run_utd_g4_rebuttal.py
```

## Results
- Final CSV, LaTeX, markdown, manifest, and log outputs: `results/`
- MMbind checkpoints and intermediate evaluation artifacts: `MMBIND/multimodal-bind/UTD/UTD-acc-bind/train/save_*` and `MMBIND/multimodal-bind/UTD/UTD-acc-bind/evaluation/save_*`
