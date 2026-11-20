"""The data layer: wild-type reconstruction, standardization, and honest splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foldgate.data import ASSAYS, HELD_OUT_FAMILIES, load, split
from tests.conftest import WILD_TYPE


def test_load_reconstructs_the_wild_type(dataset_root: Path) -> None:
    assays = load(["AAA_TEST_2026"], root=dataset_root)
    assay = assays["AAA_TEST_2026"]
    assert assay.wild_type == WILD_TYPE
    v = assay.variants
    assert (v["wt"].to_numpy() == np.array([WILD_TYPE[p] for p in v["position"]])).all()
    assert (v["mt"] != v["wt"]).all()
    assert v["sequence"].str.len().eq(len(WILD_TYPE)).all()


def test_scores_are_standardized_within_the_assay(dataset_root: Path) -> None:
    for assay in load(root=dataset_root).values():
        z = assay.variants["z"].to_numpy()
        assert abs(float(z.mean())) < 1e-6
        assert abs(float(z.std(ddof=1)) - 1) < 1e-6


def test_multi_mutants_are_dropped(dataset_root: Path) -> None:
    assay = load(["AAA_TEST_2026"], root=dataset_root)["AAA_TEST_2026"]
    assert not assay.variants["mutant"].str.contains(":").any()


@pytest.mark.parametrize("scheme", ["random", "contiguous"])
def test_splits_partition_every_variant(dataset_root: Path, scheme: str) -> None:
    assay = load(["AAA_TEST_2026"], root=dataset_root)["AAA_TEST_2026"]
    s = split(assay, scheme)
    assert len(s) == len(assay.variants)
    assert set(s.unique()) == {"train", "calib", "test"}


def test_contiguous_split_holds_out_whole_positions(dataset_root: Path) -> None:
    """The point of the hard split: a test position is never trained or calibrated on."""
    assay = load(["AAA_TEST_2026"], root=dataset_root)["AAA_TEST_2026"]
    s = split(assay, "contiguous")
    positions = assay.variants["position"]
    test_positions = set(positions[s == "test"])
    seen = set(positions[s != "test"])
    assert test_positions
    assert not (test_positions & seen)


def test_random_split_is_reproducible_and_seed_dependent(dataset_root: Path) -> None:
    assay = load(["AAA_TEST_2026"], root=dataset_root)["AAA_TEST_2026"]
    assert (split(assay, "random", seed=0) == split(assay, "random", seed=0)).all()
    assert not (split(assay, "random", seed=0) == split(assay, "random", seed=1)).all()


def test_unknown_scheme_is_rejected(dataset_root: Path) -> None:
    assay = load(["AAA_TEST_2026"], root=dataset_root)["AAA_TEST_2026"]
    with pytest.raises(ValueError, match="kfold"):
        split(assay, "kfold")


def test_held_out_families_are_listed_assays_and_excluded_from_training() -> None:
    assert set(HELD_OUT_FAMILIES) < set(ASSAYS)
    in_family = [a for a in ASSAYS if a not in HELD_OUT_FAMILIES]
    assert len(in_family) == len(ASSAYS) - len(HELD_OUT_FAMILIES)
    assert not set(in_family) & set(HELD_OUT_FAMILIES)
