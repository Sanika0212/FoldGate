"""Baselines every supervised claim has to beat.

1. **Zero-shot** wild-type marginal from ESM-2, no labels at all.
2. **Ridge on one-hot** mutations (position times amino acid), labels but no language model.
3. **Ridge on zero-shot plus one-hot**, which is the cheapest way to use both.

Run: ``python eval/run_baselines.py --scheme random`` (or ``contiguous``).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from common import IN_FAMILY, per_assay, pooled, write

from foldgate.data import load


def one_hot(df: pd.DataFrame, positions: list[int], amino: list[str]) -> np.ndarray:
    pos_index = {p: i for i, p in enumerate(positions)}
    aa_index = {a: i for i, a in enumerate(amino)}
    x = np.zeros((len(df), len(positions) + len(amino)), dtype=np.float32)
    for row, (p, a) in enumerate(zip(df["position"], df["mt"], strict=True)):
        if int(p) in pos_index:
            x[row, pos_index[int(p)]] = 1.0
        if a in aa_index:
            x[row, len(positions) + aa_index[a]] = 1.0
    return x


def ridge(train: pd.DataFrame, test: pd.DataFrame, with_zero_shot: bool) -> np.ndarray:
    """Fitted per assay: a one-hot position code only means anything inside one protein."""
    from sklearn.linear_model import RidgeCV

    pred = np.zeros(len(test), dtype=np.float32)
    for dms_id, g in test.groupby("dms_id"):
        tr = train[train["dms_id"] == dms_id]
        positions = sorted({int(p) for p in tr["position"]})
        amino = sorted(set(tr["mt"]))
        xt, xs = one_hot(tr, positions, amino), one_hot(g, positions, amino)
        if with_zero_shot:
            xt = np.hstack([xt, tr[["zero_shot"]].to_numpy(np.float32)])
            xs = np.hstack([xs, g[["zero_shot"]].to_numpy(np.float32)])
        model = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(xt, tr["z"].to_numpy())
        pred[test.index.get_indexer(g.index)] = model.predict(xs)
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="random", choices=["random", "contiguous"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assays = load()
    df = pooled(assays, IN_FAMILY, args.scheme, args.seed)
    train = df[df["split"].isin(["train", "calib"])].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    print(f"{args.scheme}: {len(train)} train+calib, {len(test)} test, {df['dms_id'].nunique()} assays")

    results = {
        "zero_shot": per_assay(test, test["zero_shot"].to_numpy()),
        "ridge_one_hot": per_assay(test, ridge(train, test, with_zero_shot=False)),
        "ridge_one_hot_plus_zero_shot": per_assay(test, ridge(train, test, with_zero_shot=True)),
    }
    payload = {"scheme": args.scheme, "seed": args.seed, "n_test": len(test), "baselines": {}}
    for name, table in results.items():
        print(f"\n{name}")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        payload["baselines"][name] = {
            "per_assay": table.to_dict("records"),
            "mean_spearman": float(table["spearman"].mean()),
            "mean_ndcg_top10": float(table["ndcg_top10"].mean()),
        }
        print(f"mean Spearman {table['spearman'].mean():.3f}  mean NDCG@10% {table['ndcg_top10'].mean():.3f}")
    print("\nwrote", write(f"baselines-{args.scheme}", payload))


if __name__ == "__main__":
    main()
