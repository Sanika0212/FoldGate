"""The zero-shot cache and the pooled split frame."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from foldgate.data import load
from foldgate.prepare import per_assay, pooled, zero_shot_frame

IDS = ["AAA_TEST_2026", "BBB_TEST_2026"]


@pytest.fixture
def cache(tmp_path: Path, assays: dict[str, Any]) -> Path:
    """A zero-shot cache written by hand, so no test here needs the weights."""
    rng = np.random.default_rng(0)
    frames = [
        pd.DataFrame(
            {
                "dms_id": dms_id,
                "mutant": a.variants["mutant"].to_numpy(),
                "zero_shot": rng.normal(-5, 2, len(a.variants)),
            }
        )
        for dms_id, a in assays.items()
    ]
    path = tmp_path / "zero_shot.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    return path


def test_zero_shot_frame_reads_the_cache_without_loading_a_model(
    cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import foldgate.model as model

    def refuse(*_: object, **__: object) -> None:
        raise AssertionError("the cache should have been used")

    monkeypatch.setattr(model, "zero_shot", refuse)
    df = zero_shot_frame(cache=cache)
    assert set(df.columns) == {"dms_id", "mutant", "zero_shot"}
    assert len(df) > 0


def test_pooled_tags_splits_and_joins_the_zero_shot_score(
    assays: dict[str, Any], cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("foldgate.prepare.ZERO_SHOT", cache)
    df = pooled(assays, IDS, "random", seed=0)
    assert len(df) == sum(len(a.variants) for a in assays.values())
    assert set(df["dms_id"]) == set(IDS)
    assert set(df["split"]) == {"train", "calib", "test"}
    assert df["zero_shot"].notna().all()
    assert df["zero_shot"].nunique() > 1


def test_pooled_keeps_each_assay_split_separately(
    assays: dict[str, Any], cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("foldgate.prepare.ZERO_SHOT", cache)
    df = pooled(assays, IDS, "contiguous", seed=0)
    for dms_id, g in df.groupby("dms_id"):
        held_out = set(g.loc[g["split"] == "test", "position"])
        seen = set(g.loc[g["split"] != "test", "position"])
        assert held_out, dms_id
        assert not held_out & seen, dms_id


def test_per_assay_reports_one_row_per_assay(
    assays: dict[str, Any], cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("foldgate.prepare.ZERO_SHOT", cache)
    df = pooled(assays, IDS, "random", seed=0)
    perfect = per_assay(df, df["z"].to_numpy())
    assert list(perfect["dms_id"]) == sorted(IDS)
    assert perfect["spearman"].round(6).eq(1.0).all()
    assert perfect["ndcg_top10"].round(6).eq(1.0).all()
    assert perfect["n"].sum() == len(df)


def test_load_accepts_a_subset_of_assays(dataset_root: Path) -> None:
    assert list(load(["BBB_TEST_2026"], root=dataset_root)) == ["BBB_TEST_2026"]


def test_a_partial_run_does_not_write_the_cache(
    tmp_path: Path, assays: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache covering two assays would starve every script that expects twelve."""
    cache = tmp_path / "zero_shot.parquet"
    monkeypatch.setattr(
        "foldgate.model.zero_shot",
        lambda wild_type, positions, wt, mt, **kw: np.zeros(len(positions), dtype=np.float32),
    )
    df = zero_shot_frame(assays, cache=cache)
    assert len(df) == sum(len(a.variants) for a in assays.values())
    assert not cache.exists()
