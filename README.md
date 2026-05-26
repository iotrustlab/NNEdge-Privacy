# main Branch

## Setup
```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy pandas scikit-learn matplotlib seaborn tensorflow
```

## Paths
- For the semantic-free pipeline, update `PAMAP2_ROOT`, `UCI_ROOT`, `STM_ROOT`, and `CACHE_DIR` in `Semantic_Free_Adversary/data_loader.py`.
- For the `Game-1/`, `Game-2/`, and `Game-3/` trainers, update the path constants near the top of the specific training script you want to run.

## Run
```bash
.venv/bin/python Semantic_Free_Adversary/data_loader.py
.venv/bin/python Semantic_Free_Adversary/logreg_modality_classifier.py
.venv/bin/python Semantic_Free_Adversary/logreg_activity_classification.py
```

## Results
- `Semantic_Free_Adversary/Cache/`
- `Semantic_Free_Adversary/Results/`
- any script-specific output path configured inside the `Game-1/`, `Game-2/`, or `Game-3/` training file you run
