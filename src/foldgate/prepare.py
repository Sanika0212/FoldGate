"""Turn assays into model inputs: the zero-shot cache, pooled splits, per-assay metrics.

The zero-shot score is a feature of the supervised model and the no-label baseline, so it
is computed once for every variant and cached in a parquet file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from foldgate.data import Assay, load, split

ZERO_SHOT = Path("data/zero_shot.parquet")


def zero_shot_frame(
    assays: dict[str, Assay] | None = None, backbone: str | None = None, cache: Path | None = None
) -> pd.DataFrame:
    """One row per variant with its wild-type-marginal score, computed once and cached."""
    from foldgate.model import DEFAULT_BACKBONE, zero_shot

    cache = cache or ZERO_SHOT
    if cache.exists():
        return pd.read_parquet(cache)
    assays = assays or load()
    backbone = backbone or DEFAULT_BACKBONE
    frames = []
    for dms_id, assay in assays.items():
        v = assay.variants
        scores = zero_shot(
            assay.wild_type,
            v["position"].to_numpy(),
            v["wt"].to_numpy(),
            v["mt"].to_numpy(),
            backbone=backbone,
        )
        frames.append(pd.DataFrame({"dms_id": dms_id, "mutant": v["mutant"].to_numpy(), "zero_shot": scores}))
    out = pd.concat(frames, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def pooled(
    assays: dict[str, Assay], ids: list[str], scheme: str = "random", seed: int = 0, cache: Path | None = None
) -> pd.DataFrame:
    """Variants of several assays in one frame, each tagged with its split."""
    zs = zero_shot_frame(assays, cache=cache).set_index(["dms_id", "mutant"])["zero_shot"]
    frames = []
    for dms_id in ids:
        assay = assays[dms_id]
        v = assay.variants.copy()
        v["dms_id"] = dms_id
        v["split"] = split(assay, scheme, seed).to_numpy()
        v["zero_shot"] = [float(zs.loc[(dms_id, m)]) for m in v["mutant"]]
        frames.append(v)
    return pd.concat(frames, ignore_index=True)


def per_assay(df: pd.DataFrame, pred: np.ndarray, column: str = "model") -> pd.DataFrame:
    """Spearman and top-decile NDCG per assay, which is how ProteinGym reports."""
    from foldgate.calibration import ndcg_top, spearman

    out = df.assign(**{column: pred})
    rows = []
    for dms_id, g in out.groupby("dms_id"):
        rows.append(
            {
                "dms_id": dms_id,
                "n": len(g),
                "spearman": spearman(g[column].to_numpy(), g["z"].to_numpy()),
                "ndcg_top10": ndcg_top(g[column].to_numpy(), g["z"].to_numpy()),
            }
        )
    return pd.DataFrame(rows).sort_values("dms_id").reset_index(drop=True)
