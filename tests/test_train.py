"""Batching for speed must not change what the caller gets back."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
import torch

from foldgate.train import LengthGrouped, predict


def frame(n: int = 97, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(20, 120, n)
    return pd.DataFrame(
        {
            "sequence": ["".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), i)) for i in lengths],
            "position": [int(rng.integers(0, i)) for i in lengths],
            "zero_shot": rng.normal(size=n).astype(np.float32),
            "z": rng.normal(size=n).astype(np.float32),
        }
    )


def test_length_grouped_covers_every_row_exactly_once() -> None:
    df = frame()
    sampler = LengthGrouped(df["sequence"].str.len().to_numpy(), batch_size=16)
    batches = list(iter(sampler))
    assert sorted(i for b in batches for i in b) == list(range(len(df)))
    assert len(batches) == len(sampler)


def test_length_grouped_batches_hold_similar_lengths() -> None:
    df = frame(n=600)
    lengths = df["sequence"].str.len().to_numpy()
    sampler = LengthGrouped(lengths, batch_size=16)
    spreads = [lengths[b].max() - lengths[b].min() for b in iter(sampler)]
    assert float(np.mean(spreads)) < 20  # uniform shuffling would be near the full range


def test_length_grouped_reshuffles_each_epoch() -> None:
    sampler = LengthGrouped(frame()["sequence"].str.len().to_numpy(), batch_size=16)
    assert list(iter(sampler)) != list(iter(sampler))


def test_predict_returns_rows_in_the_callers_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scoring is done shortest first; a prediction must still belong to its own row."""
    df = frame()

    class Fake:
        hparams: ClassVar[dict[str, str]] = {"backbone": "facebook/esm2_t12_35M_UR50D"}

        def to(self, _dev: Any) -> Any:
            return self

        def eval(self) -> Any:
            return self

        def __call__(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
            # Recognisable per row: the length of the sequence it was given.
            return batch["attention_mask"].sum(dim=1).float() - 2

    monkeypatch.setattr("foldgate.model.device", lambda: torch.device("cpu"))
    out, _ = predict(Fake(), df, batch_size=8)  # type: ignore[arg-type]
    assert np.array_equal(out, df["sequence"].str.len().to_numpy())
