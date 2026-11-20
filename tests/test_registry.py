"""The registry: a rejected candidate is kept with its reasons, and never aliased champion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foldgate import registry


@pytest.fixture
def tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(registry, "MLFLOW_URI", f"sqlite:///{tmp_path / 'mlruns.db'}")
    monkeypatch.setattr(registry, "NAME", "foldgate-test")
    return tmp_path


def difficulty() -> object:
    from foldgate.calibration import fit_difficulty

    rng = np.random.default_rng(0)
    x = rng.random((200, 4))
    return fit_difficulty(x, rng.normal(0, 0.2 + x[:, 0]))


def log(name: str, passed: bool, reasons: list[str] | None = None) -> dict[str, object]:
    return registry.log_candidate(
        None,
        {"alpha": 0.1, "q_normalized_quantile": 1.3, "abstain_width": 2.0},
        difficulty(),
        {"spearman": 0.61 if passed else 0.28, "coverage": 0.903},
        {"kind": "lora", "scheme": "random"},
        {"passed": passed, "reasons": reasons or []},
        name,
    )


def test_a_passing_candidate_becomes_champion(tracking: Path) -> None:
    import mlflow

    out = log("honest", passed=True)
    assert out["champion"] is True
    client = mlflow.MlflowClient(tracking_uri=registry.MLFLOW_URI, registry_uri=registry.MLFLOW_URI)
    champion = client.get_model_version_by_alias(registry.NAME, "champion")
    assert int(champion.version) == out["version"]
    assert champion.tags["gate"] == "passed"


def test_a_rejected_candidate_is_logged_but_not_promoted(tracking: Path) -> None:
    import mlflow

    first = log("honest", passed=True)
    second = log("narrow-intervals", passed=False, reasons=["coverage 0.742 is outside 0.90"])
    assert second["champion"] is False
    assert second["version"] == first["version"] + 1
    client = mlflow.MlflowClient(tracking_uri=registry.MLFLOW_URI, registry_uri=registry.MLFLOW_URI)
    champion = client.get_model_version_by_alias(registry.NAME, "champion")
    assert int(champion.version) == first["version"]  # the champion did not move
    rejected = client.get_model_version_by_alias(registry.NAME, "rejected")
    assert int(rejected.version) == second["version"]
    assert rejected.tags["gate"] == "rejected"


def test_the_reasons_and_calibration_travel_with_the_version(tracking: Path) -> None:
    import json

    import mlflow

    out = log("narrow-intervals", passed=False, reasons=["coverage 0.742 is outside 0.90"])
    path = mlflow.artifacts.download_artifacts(
        f"runs:/{out['run_id']}/model/gate.json", tracking_uri=registry.MLFLOW_URI
    )
    gate = json.loads(Path(path).read_text())
    assert gate["reasons"] == ["coverage 0.742 is outside 0.90"]
    path = mlflow.artifacts.download_artifacts(
        f"runs:/{out['run_id']}/model/calibration.json", tracking_uri=registry.MLFLOW_URI
    )
    assert json.loads(Path(path).read_text())["q_normalized_quantile"] == 1.3


def test_the_difficulty_model_can_be_loaded_back(tracking: Path) -> None:
    import pickle

    import mlflow

    out = log("honest", passed=True)
    path = mlflow.artifacts.download_artifacts(
        f"runs:/{out['run_id']}/model/difficulty.pkl", tracking_uri=registry.MLFLOW_URI
    )
    with Path(path).open("rb") as fh:
        model = pickle.load(fh)
    assert model.predict(np.random.default_rng(0).random((5, 4))).shape == (5,)
