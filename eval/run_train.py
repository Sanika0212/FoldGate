"""Train the LoRA regressor, calibrate it, and ask whether the intervals are honest.

    python eval/run_train.py --scheme random        # or contiguous

Nine assays are pooled (the other three are kept for the shift test). The model trains on
the train split, the conformal quantile comes from the calibration split, and everything
reported comes from the test split.

What gets reported:

1. Accuracy per assay (Spearman, NDCG of the top decile) against the baselines in
   ``runs/baselines-<scheme>/results.json``.
2. Coverage of split-conformal and normalized-conformal intervals at 80/90/95%, with mean
   width, so a narrower interval cannot be mistaken for a better one.
3. Coverage within each difficulty tercile. A plain conformal interval is one width for
   every variant, so it over-covers the easy ones and under-covers the hard ones;
   normalizing by predicted difficulty is meant to even that out.
4. What selective prediction buys a lab that can only test k variants.

Predictions, embeddings and the checkpoint are saved for the shift test and the gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from common import IN_FAMILY, per_assay, pooled, write

from foldgate.calibration import (
    Intervals,
    conformal_quantile,
    fit_difficulty,
    normalized_conformal,
    selective_top_k,
    spearman,
    split_conformal,
)
from foldgate.data import load
from foldgate.model import DEFAULT_BACKBONE
from foldgate.train import TrainConfig, fit, predict

ALPHAS = [0.2, 0.1, 0.05]
ARTIFACTS = Path("artifacts")


def tercile_coverage(iv: Intervals, y: np.ndarray, difficulty: np.ndarray) -> list[float]:
    """Coverage inside each third of the variants, ordered by predicted difficulty."""
    edges = np.quantile(difficulty, [1 / 3, 2 / 3])
    groups = np.digitize(difficulty, edges)
    return [
        float(np.mean((y[groups == g] >= iv.lower[groups == g]) & (y[groups == g] <= iv.upper[groups == g])))
        for g in range(3)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="random", choices=["random", "contiguous"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--duty",
        type=float,
        default=0.5,
        help="share of each second spent working; the rest is idle, to keep the laptop cool",
    )
    args = ap.parse_args()

    df = pooled(load(), IN_FAMILY, args.scheme, args.seed)
    train = df[df["split"] == "train"].reset_index(drop=True)
    calib = df[df["split"] == "calib"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    print(f"{args.scheme}: {len(train)} train, {len(calib)} calib, {len(test)} test")

    cfg = TrainConfig(
        name=args.scheme,
        backbone=args.backbone,
        rank=args.rank,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        duty=args.duty,
    )
    module = fit(train, cfg)
    print(f"trainable share of weights: {module.net.trainable_share():.4f}")

    cal_pred, cal_emb = predict(module, calib, embeddings=True)
    test_pred, test_emb = predict(module, test, embeddings=True)
    assert cal_emb is not None
    assert test_emb is not None
    cal_y, test_y = calib["z"].to_numpy(np.float32), test["z"].to_numpy(np.float32)

    table = per_assay(test, test_pred)
    print("\nLoRA regressor")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"mean Spearman {table['spearman'].mean():.3f}  mean NDCG@10% {table['ndcg_top10'].mean():.3f}")

    baselines = {}
    bpath = Path(f"runs/baselines-{args.scheme}/results.json")
    if bpath.exists():
        loaded = json.loads(bpath.read_text())["baselines"]
        baselines = {k: v["mean_spearman"] for k, v in loaded.items()}
        for name, value in baselines.items():
            print(f"  vs {name}: {table['spearman'].mean() - value:+.3f}")

    # One difficulty model for the whole report: fitted on half the calibration split.
    rng = np.random.default_rng(args.seed)
    half = rng.random(len(cal_y)) < 0.5
    difficulty = fit_difficulty(cal_emb[half], cal_y[half] - cal_pred[half])
    sigma_test = np.exp(difficulty.predict(test_emb))

    coverage = {}
    print("\n                 target  split conformal      normalized conformal")
    print("                         cover  width  terciles      cover  width  terciles")
    for alpha in ALPHAS:
        plain = split_conformal(cal_pred, cal_y, test_pred, alpha)
        norm = normalized_conformal(cal_pred, cal_y, cal_emb, test_pred, test_emb, alpha, seed=args.seed)
        pt = tercile_coverage(plain, test_y, sigma_test)
        nt = tercile_coverage(norm, test_y, sigma_test)
        coverage[f"{1 - alpha:.2f}"] = {
            "split": {"coverage": plain.coverage(test_y), "width": float(plain.width.mean()), "terciles": pt},
            "normalized": {
                "coverage": norm.coverage(test_y),
                "width": float(norm.width.mean()),
                "terciles": nt,
            },
        }
        print(
            f"                 {1 - alpha:>5.0%}   {plain.coverage(test_y):.3f}  "
            f"{plain.width.mean():.2f}  {[round(v, 2) for v in pt]}   "
            f"{norm.coverage(test_y):.3f}  {norm.width.mean():.2f}  {[round(v, 2) for v in nt]}"
        )

    # Selective prediction per assay: a lab picks 10 variants to put on the bench.
    alpha = 0.1
    norm = normalized_conformal(cal_pred, cal_y, cal_emb, test_pred, test_emb, alpha, seed=args.seed)
    selective = []
    for dms_id, g in test.assign(pred=test_pred, width=norm.width).groupby("dms_id"):
        s = selective_top_k(g["pred"].to_numpy(), g["width"].to_numpy(), g["z"].to_numpy(), k=10)
        selective.append({"dms_id": dms_id, **s})
    mean_by_score = float(np.mean([s["ndcg_by_score"] for s in selective]))
    mean_confident = float(np.mean([s["ndcg_confident_only"] for s in selective]))
    hits_by_score = float(np.mean([s["hits_by_score"] for s in selective]))
    hits_confident = float(np.mean([s["hits_confident_only"] for s in selective]))
    print(
        f"\nselective (k=10, keep the most certain half): NDCG {mean_by_score:.3f} "
        f"-> {mean_confident:.3f}, hit rate {hits_by_score:.3f} -> {hits_confident:.3f}"
    )

    out = ARTIFACTS / args.scheme
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": module.state_dict(), "hparams": dict(module.hparams)}, out / "module.pt")
    np.savez(
        out / "predictions.npz",
        cal_pred=cal_pred,
        cal_y=cal_y,
        cal_emb=cal_emb,
        test_pred=test_pred,
        test_y=test_y,
        test_emb=test_emb,
        cal_dms=calib["dms_id"].to_numpy(),
        test_dms=test["dms_id"].to_numpy(),
    )
    calibration = {
        "alpha": alpha,
        "q_plain": conformal_quantile(np.abs(cal_y - cal_pred), alpha),
        "q_normalized": float(np.median(norm.width) / 2),
        "abstain_width": float(np.quantile(norm.width, 0.9)),
        "scheme": args.scheme,
    }
    (out / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")

    payload = {
        "scheme": args.scheme,
        "seed": args.seed,
        "backbone": args.backbone,
        "epochs": args.epochs,
        "rank": args.rank,
        "trainable_share": module.net.trainable_share(),
        "n": {"train": len(train), "calib": len(calib), "test": len(test)},
        "per_assay": table.to_dict("records"),
        "mean_spearman": float(table["spearman"].mean()),
        "mean_ndcg_top10": float(table["ndcg_top10"].mean()),
        "pooled_spearman": spearman(test_pred, test_y),
        "baseline_mean_spearman": baselines,
        "coverage": coverage,
        "selective": {
            "k": 10,
            "per_assay": selective,
            "ndcg_by_score": mean_by_score,
            "ndcg_confident_only": mean_confident,
            "hits_by_score": hits_by_score,
            "hits_confident_only": hits_confident,
        },
        "calibration": calibration,
    }
    print("\nwrote", write(f"train-{args.scheme}", payload), "and", out)


if __name__ == "__main__":
    main()
