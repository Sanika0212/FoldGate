"""Fixtures: a small synthetic assay on disk, so nothing in the fast tests needs the Hub."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

AMINO = list("ACDEFGHIKLMNPQRSTVWY")
WILD_TYPE = "MKAILVVLLYTFATANADTLCIGYHANNSTDTVDTVLEKNVTVTHSVNLLEDKHNGKLCKL"


def _variants(dms_id: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for position, wt in enumerate(WILD_TYPE):
        for mt in rng.choice([a for a in AMINO if a != wt], 6, replace=False):
            sequence = WILD_TYPE[:position] + mt + WILD_TYPE[position + 1 :]
            rows.append(
                {
                    "mutant": f"{wt}{position + 1}{mt}",
                    "DMS_id": dms_id,
                    "mutated_sequence": sequence,
                    "DMS_score": float(rng.normal(position / 20, 1)),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("proteingym")
    (root / "DMS_substitutions").mkdir()
    df = pd.concat([_variants("AAA_TEST_2026", 1), _variants("BBB_TEST_2026", 2)])
    df.to_parquet(root / "DMS_substitutions" / "train-00000-of-00005.parquet", index=False)
    return root


@pytest.fixture
def assays(dataset_root: Path) -> dict[str, object]:
    from foldgate.data import load

    return dict(load(["AAA_TEST_2026", "BBB_TEST_2026"], root=dataset_root))
