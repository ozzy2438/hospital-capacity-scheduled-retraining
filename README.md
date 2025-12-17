# hospital-capacity-scheduled-retraining

Apache Airflow DAG for **scheduled ML retraining** with **challenger vs champion evaluation** and **gated auto-promotion**.

This repo is a small reference implementation that demonstrates three core MLOps behaviors:

1. Retrain on a schedule (default: monthly).
2. Evaluate a newly trained challenger against the current champion.
3. Promote the challenger only if it meets performance criteria.

## What it does

The DAG `scheduled_retraining_with_gated_promotion` (see `dags/retraining_dag.py`) runs monthly by default. Each run:

- Trains a **challenger** model.
- Loads the current **champion** model (if one exists).
- Evaluates challenger vs champion on the same holdout set.
- Promotes the challenger only if it improves the configured metric enough.

### Default schedule

The schedule is set to run at **2:00am on the 1st of every month**:

- `0 2 1 * *`

Adjust this in the DAG if you want weekly/daily retraining.

### Promotion policy (gated auto-promotion)

The default gating rule is based on **MAE** (mean absolute error):

- If no champion exists, promote the challenger.
- If a champion exists, promote only if
  - `mae_challenger <= (1 - min_relative_improvement) * mae_champion`

The default minimum relative improvement is **2%**, configurable via env var:

- `PROMOTION_MIN_RELATIVE_IMPROVEMENT` (default: `0.02`)

## Local registry (toy)

To keep this example runnable without external infrastructure, the DAG stores the champion artifact and metadata locally:

- `MODEL_REGISTRY_DIR` (default: `/tmp/model_registry`)
- `champion.joblib`
- `champion_metadata.json`

In production you would replace this with an artifact store / model registry (S3 + Dynamo, MLflow, SageMaker registry, etc.).

## Repo layout

- `dags/retraining_dag.py` — the Airflow DAG
- `tests/` — basic unit tests for promotion gating math

## Notes

This is a reference scaffold. The training code uses a toy regression dataset to keep it self-contained. Swap the following pieces with your real pipeline components:

- Feature extraction and data loading
- Model training
- Validation splits / backtesting
- Model registry and deployment promotion

