"""Model registry: every candidate is logged with its calibration and its gate verdict.

A version carries everything the router needs (the two ONNX graphs, the conformal
quantile, the difficulty model, the abstention width) and the evidence for its gate
decision. Only a version that passed the gate gets the ``champion`` alias; the router
loads whatever ``champion`` points to.
"""

from __future__ import annotations

import contextlib
import json
import pickle
import tempfile
from pathlib import Path
from typing import Any

MLFLOW_URI = "sqlite:///mlruns.db"
NAME = "foldgate-variant-effect"


def log_candidate(
    onnx_dir: Path | None,
    calibration: dict[str, Any],
    difficulty: Any,
    metrics: dict[str, float],
    params: dict[str, Any],
    gate: dict[str, Any],
    run_name: str,
) -> dict[str, Any]:
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("foldgate")
    with mlflow.start_run(run_name=run_name) as run, tempfile.TemporaryDirectory() as tmp:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        (Path(tmp) / "calibration.json").write_text(json.dumps(calibration, indent=2))
        (Path(tmp) / "gate.json").write_text(json.dumps(gate, indent=2))
        with (Path(tmp) / "difficulty.pkl").open("wb") as fh:
            pickle.dump(difficulty, fh)
        mlflow.log_artifacts(tmp, "model")
        if onnx_dir is not None:
            mlflow.log_artifacts(str(onnx_dir), "model/onnx")
        run_id = run.info.run_id
    client = mlflow.MlflowClient()
    with contextlib.suppress(mlflow.exceptions.MlflowException):
        client.create_registered_model(NAME)  # raises when it is already there
    version = client.create_model_version(NAME, source=f"runs:/{run_id}/model", run_id=run_id)
    client.set_model_version_tag(NAME, version.version, "gate", "passed" if gate["passed"] else "rejected")
    alias = "champion" if gate["passed"] else "rejected"
    client.set_registered_model_alias(NAME, alias, version.version)
    return {"run_id": run_id, "version": int(version.version), "champion": gate["passed"]}


def champion_uri() -> str:
    return f"models:/{NAME}@champion"


def download(alias: str = "champion") -> Path:
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_URI)
    return Path(mlflow.artifacts.download_artifacts(f"models:/{NAME}@{alias}"))
