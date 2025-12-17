"""Airflow DAG: scheduled model retraining with challenger vs champion evaluation and gated auto-promotion.

This DAG is intentionally self-contained so the repo works as a reference implementation.
In a real deployment, swap the toy training/eval code with your real pipeline steps
(feature extraction, model training, registry lookups, online promotion, etc.).

Key behaviors
- Runs on a schedule (default monthly).
- Trains a challenger model.
- Evaluates challenger vs champion on a holdout set.
- Promotes challenger only if it clears configured thresholds.

Note: This file avoids external services to keep it runnable anywhere.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

from airflow import DAG
from airflow.decorators import task


# Basic local registry paths (toy). In production, replace with MLflow/S3/DB-backed registry.
REGISTRY_DIR = Path(os.environ.get("MODEL_REGISTRY_DIR", "/tmp/model_registry"))
CHAMPION_PATH = REGISTRY_DIR / "champion.joblib"
METADATA_PATH = REGISTRY_DIR / "champion_metadata.json"


def _ensure_registry_dir() -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)


def _load_champion() -> tuple[object | None, dict]:
    """Load current champion model and its metadata if present."""
    if not CHAMPION_PATH.exists() or not METADATA_PATH.exists():
        return None, {"exists": False}

    model_obj = joblib.load(CHAMPION_PATH)
    with METADATA_PATH.open("r", encoding="utf-8") as f_obj:
        meta_obj = json.load(f_obj)
    meta_obj["exists"] = True
    return model_obj, meta_obj


def _save_champion(model_obj: object, meta_obj: dict) -> None:
    _ensure_registry_dir()
    joblib.dump(model_obj, CHAMPION_PATH)
    with METADATA_PATH.open("w", encoding="utf-8") as f_obj:
        json.dump(meta_obj, f_obj, indent=2, sort_keys=True)


default_args = {
    "owner": "mlops",
    "retries": 0,
}


with DAG(
    dag_id="scheduled_retraining_with_gated_promotion",
    description="Monthly retraining with challenger vs champion eval and gated promotion",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="0 2 1 * *",  # 2:00am on the 1st of every month
    catchup=False,
    tags=["retraining", "ml", "promotion"],
) as dag:

    @task
    def train_challenger(seed: int = 42) -> str:
        """Train a challenger model and persist it to disk.

        Returns the path to the serialized challenger artifact.
        """
        _ensure_registry_dir()

        x_vals, y_vals = make_regression(
            n_samples=5000,
            n_features=20,
            noise=15.0,
            random_state=seed,
        )
        x_train, x_test, y_train, y_test = train_test_split(
            x_vals, y_vals, test_size=0.2, random_state=seed
        )

        model_obj = Ridge(alpha=1.0, random_state=seed)
        model_obj.fit(x_train, y_train)

        challenger_path = REGISTRY_DIR / "challenger.joblib"
        joblib.dump(
            {
                "model": model_obj,
                "x_test": x_test,
                "y_test": y_test,
                "seed": seed,
            },
            challenger_path,
        )
        return str(challenger_path)

    @task
    def evaluate_challenger(challenger_artifact_path: str) -> dict:
        """Evaluate challenger vs champion and decide whether it qualifies for promotion.

        Gating rules (defaults):
        - Challenger MAE must be at least 2% better than champion MAE, OR
        - If no champion exists, challenger is eligible by default.
        """
        payload_obj = joblib.load(challenger_artifact_path)
        challenger_model = payload_obj["model"]
        x_test = payload_obj["x_test"]
        y_test = payload_obj["y_test"]

        y_pred_challenger = challenger_model.predict(x_test)
        mae_challenger = float(mean_absolute_error(y_test, y_pred_challenger))

        champion_model, champion_meta = _load_champion()

        min_improvement_fraction = float(
            os.environ.get("PROMOTION_MIN_RELATIVE_IMPROVEMENT", "0.02")
        )

        if champion_model is None:
            return {
                "promote": True,
                "reason": "No champion exists",
                "mae_challenger": mae_challenger,
                "mae_champion": None,
                "min_relative_improvement": min_improvement_fraction,
            }

        y_pred_champion = champion_model.predict(x_test)
        mae_champion = float(mean_absolute_error(y_test, y_pred_champion))

        # Promote if challenger improves by at least configured fraction
        promote_flag = mae_challenger <= (1.0 - min_improvement_fraction) * mae_champion

        return {
            "promote": bool(promote_flag),
            "reason": "Meets threshold" if promote_flag else "Insufficient improvement",
            "mae_challenger": mae_challenger,
            "mae_champion": mae_champion,
            "min_relative_improvement": min_improvement_fraction,
            "champion_metadata": champion_meta,
        }

    @task
    def promote_if_eligible(challenger_artifact_path: str, evaluation: dict) -> dict:
        """Promote challenger to champion if evaluation['promote'] is True."""
        if not evaluation.get("promote", False):
            return {
                "promoted": False,
                "evaluation": evaluation,
            }

        payload_obj = joblib.load(challenger_artifact_path)
        challenger_model = payload_obj["model"]

        meta_obj = {
            "promoted_at": datetime.utcnow().isoformat() + "Z",
            "mae": evaluation.get("mae_challenger"),
            "seed": payload_obj.get("seed"),
        }
        _save_champion(challenger_model, meta_obj)

        return {
            "promoted": True,
            "evaluation": evaluation,
            "new_champion_metadata": meta_obj,
        }

    challenger_path_str = train_challenger()
    eval_dict = evaluate_challenger(challenger_path_str)
    promote_if_eligible(challenger_path_str, eval_dict)
