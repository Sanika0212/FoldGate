"""End to end on the real backbone: train a little, export, serve the exported graph.

Marked ``model`` because it downloads ESM-2 35M from the Hub. Run with ``-m model``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from foldgate.data import load
from foldgate.export import (
    export_mlm,
    export_regressor,
    parity,
    reference_scores,
    session,
)
from foldgate.model import DEFAULT_BACKBONE, VariantRegressor, zero_shot
from foldgate.router import ModelVersion
from foldgate.train import Collate, TrainConfig, Variants, fit, predict

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def trained(dataset_root: Path) -> tuple[Any, Any, Any]:
    """One short training run on the synthetic assay, shared by the tests in this file."""
    assay = load(["AAA_TEST_2026"], root=dataset_root)["AAA_TEST_2026"]
    v = assay.variants.head(96).copy()
    v["zero_shot"] = zero_shot(
        assay.wild_type, v["position"].to_numpy(), v["wt"].to_numpy(), v["mt"].to_numpy()
    )
    module = fit(v, TrainConfig(epochs=1, batch_size=16, accelerator="cpu", precision="32-true"))
    return module, assay, v


def test_zero_shot_scores_one_variant_per_row(dataset_root: Path) -> None:
    assay = load(["AAA_TEST_2026"], root=dataset_root)["AAA_TEST_2026"]
    v = assay.variants.head(32)
    scores = zero_shot(assay.wild_type, v["position"].to_numpy(), v["wt"].to_numpy(), v["mt"].to_numpy())
    assert scores.shape == (32,)
    assert np.isfinite(scores).all()
    # A substitution is a log-odds against the wild type, so it is almost never favoured.
    assert float(np.mean(scores < 0)) > 0.8


def test_only_the_adapters_and_the_head_train() -> None:
    net = VariantRegressor()
    assert net.trainable_share() < 0.05
    trainable = {n for n, p in net.named_parameters() if p.requires_grad}
    assert all("lora" in n or n.startswith("head") for n in trainable)
    assert any("lora" in n for n in trainable)


def test_predict_returns_a_score_and_an_embedding_per_variant(trained: tuple[Any, Any, Any]) -> None:
    module, _, v = trained
    pred, emb = predict(module, v, embeddings=True)
    assert pred.shape == (len(v),)
    assert emb is not None
    assert emb.shape == (len(v), module.net.head[0].in_features - 1)
    assert np.isfinite(pred).all()


def test_onnx_matches_pytorch_and_serves(trained: tuple[Any, Any, Any], tmp_path: Path) -> None:
    module, assay, v = trained
    rows = v.head(8)
    collate = Collate(DEFAULT_BACKBONE)
    batch = collate([Variants(rows)[i] for i in range(len(rows))])
    reference = reference_scores(module, batch)

    regressor = export_regressor(module, tmp_path / "regressor.onnx", batch)
    mlm = export_mlm(DEFAULT_BACKBONE, tmp_path / "mlm.onnx", batch["input_ids"][:1])
    assert parity(reference, regressor, batch) < 1e-3

    from foldgate.calibration import fit_difficulty
    from foldgate.model import tokenizer

    _, emb = predict(module, v, embeddings=True)
    assert emb is not None
    version = ModelVersion(
        name="test",
        regressor=session(regressor),
        mlm=session(mlm),
        tokenizer=tokenizer(DEFAULT_BACKBONE),
        q=1.5,
        difficulty=fit_difficulty(emb, np.random.default_rng(0).normal(size=len(emb))),
        abstain_width=10.0,
    )
    mutants = rows["mutant"].tolist()[:4]
    out = version.score([(assay.wild_type, m) for m in mutants])
    assert [r["variant"] for r in out] == mutants
    assert all(r["lower"] < r["score"] < r["upper"] for r in out)
    assert all(isinstance(r["abstain"], bool) for r in out)
    # The zero-shot logits of a protein are computed once and reused.
    assert list(version.zero_shot_cache) == [assay.wild_type]


def test_a_variant_that_contradicts_the_protein_is_refused(
    trained: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    module, assay, v = trained
    rows = v.head(4)
    collate = Collate(DEFAULT_BACKBONE)
    batch = collate([Variants(rows)[i] for i in range(len(rows))])
    regressor = export_regressor(module, tmp_path / "regressor.onnx", batch)
    mlm = export_mlm(DEFAULT_BACKBONE, tmp_path / "mlm.onnx", batch["input_ids"][:1])

    from foldgate.calibration import fit_difficulty
    from foldgate.model import tokenizer

    rng = np.random.default_rng(0)
    version = ModelVersion(
        name="test",
        regressor=session(regressor),
        mlm=session(mlm),
        tokenizer=tokenizer(DEFAULT_BACKBONE),
        q=1.5,
        difficulty=fit_difficulty(rng.random((50, 480)), rng.normal(size=50)),
        abstain_width=10.0,
    )
    wrong = "X" if assay.wild_type[0] != "X" else "Y"
    with pytest.raises(ValueError, match="does not match"):
        version.score([(assay.wild_type, f"{wrong}1A")])
