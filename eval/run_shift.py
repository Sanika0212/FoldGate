"""What happens on proteins the model never trained on, and how to get the guarantee back.

    python eval/run_shift.py --scheme random

The three held-out families (alpha-synuclein, TPMT, UBC9) stand in for the real case: a lab
sends a protein the model has not seen. Conformal coverage is only guaranteed when test
variants are exchangeable with the calibration ones, and a new protein breaks that. This
script measures how badly, detects the shift without labels, and then shows how few labels
it takes to restore coverage.

1. Accuracy on each held-out assay, against zero-shot on the same variants.
2. Coverage of the in-family intervals there, at the 90% target.
3. MMD between in-family calibration embeddings and the new assay's, with a permutation
   p-value, plus an Evidently drift report on the prediction and embedding summaries.
4. Recalibration: label k variants of the new assay, recompute the quantile, measure
   coverage on the rest. Reported for k = 25, 50, 100, 200, averaged over 20 draws.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from common import pooled, write

from foldgate.calibration import conformal_quantile, fit_difficulty, mmd, ndcg_top, spearman
from foldgate.data import HELD_OUT_FAMILIES, load
from foldgate.train import Module, predict

LABEL_BUDGETS = [25, 50, 100, 200]
DRAWS = 20
ALPHA = 0.1


def load_module(path: Path) -> Module:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    module = Module(**blob["hparams"])
    module.load_state_dict(blob["state_dict"])
    return module.eval()


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, out: Path) -> dict[str, object]:
    """Drift on the prediction and the embedding summaries, as a saved HTML report."""
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        return {"available": False}
    columns = list(reference.columns)
    definition = DataDefinition(numerical_columns=columns)
    ref = Dataset.from_pandas(reference, data_definition=definition)
    cur = Dataset.from_pandas(current, data_definition=definition)
    result = Report([DataDriftPreset()]).run(cur, ref)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(out))
    summary = result.dict()
    counts = [
        m["value"]
        for m in summary.get("metrics", [])
        if "DriftedColumnsCount" in str(m.get("metric_name", "")) and isinstance(m["value"], dict)
    ]
    share = float(counts[0].get("share", float("nan"))) if counts else float("nan")
    columns = {
        str(m["metric_name"]).split("column=")[1].split(",")[0]: float(m["value"])
        for m in summary.get("metrics", [])
        if str(m.get("metric_name", "")).startswith("ValueDrift(")
    }
    return {
        "available": True,
        "html": str(out),
        "share_of_columns_drifted": share,
        "k_s_p_values": columns,
    }


def summarize(emb: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """Embeddings are 480-dimensional; drift reports read better on a few summaries."""
    return pd.DataFrame(
        {
            "prediction": pred,
            "embedding_norm": np.linalg.norm(emb, axis=1),
            "embedding_mean": emb.mean(axis=1),
            "embedding_std": emb.std(axis=1),
            "embedding_pc0": emb[:, 0],
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="random", choices=["random", "contiguous"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    art = Path("artifacts") / args.scheme
    module = load_module(art / "module.pt")
    saved = np.load(art / "predictions.npz", allow_pickle=True)
    cal_pred, cal_y, cal_emb = saved["cal_pred"], saved["cal_y"], saved["cal_emb"]

    rng = np.random.default_rng(args.seed)
    half = rng.random(len(cal_y)) < 0.5
    difficulty = fit_difficulty(cal_emb[half], cal_y[half] - cal_pred[half])
    sigma_cal = np.exp(difficulty.predict(cal_emb[~half]))
    q_norm = conformal_quantile(np.abs(cal_y[~half] - cal_pred[~half]) / sigma_cal, ALPHA)
    q_plain = conformal_quantile(np.abs(cal_y - cal_pred), ALPHA)

    assays = load(HELD_OUT_FAMILIES)
    df = pooled(assays, HELD_OUT_FAMILIES, "random", args.seed)
    rows, payload_assays = [], {}
    for dms_id, g in df.groupby("dms_id"):
        g = g.reset_index(drop=True)
        pred, emb = predict(module, g, embeddings=True)
        assert emb is not None
        y = g["z"].to_numpy(np.float32)
        sigma = np.exp(difficulty.predict(emb))
        plain_cover = float(np.mean(np.abs(y - pred) <= q_plain))
        norm_cover = float(np.mean(np.abs(y - pred) <= q_norm * sigma))
        stat, p = mmd(cal_emb, emb, seed=args.seed)
        report = evidently_report(
            summarize(cal_emb, cal_pred),
            summarize(emb, pred),
            Path("runs") / f"shift-{args.scheme}" / f"{dms_id}.html",
        )

        # Recalibration: k labels from the new assay, coverage measured on everything else.
        recal: dict[str, dict[str, float]] = {}
        for k in LABEL_BUDGETS:
            covers, widths = [], []
            for draw in range(DRAWS):
                r = np.random.default_rng(1000 * draw + args.seed)
                idx = r.choice(len(g), k, replace=False)
                rest = np.setdiff1d(np.arange(len(g)), idx)
                qk = conformal_quantile(np.abs(y[idx] - pred[idx]) / sigma[idx], ALPHA)
                covers.append(float(np.mean(np.abs(y[rest] - pred[rest]) <= qk * sigma[rest])))
                widths.append(float(np.mean(2 * qk * sigma[rest])))
            recal[str(k)] = {
                "coverage": float(np.mean(covers)),
                "coverage_sd": float(np.std(covers)),
                "width": float(np.mean(widths)),
            }

        payload_assays[dms_id] = {
            "n": len(g),
            "spearman_model": spearman(pred, y),
            "spearman_zero_shot": spearman(g["zero_shot"].to_numpy(), y),
            "ndcg_top10_model": ndcg_top(pred, y),
            "ndcg_top10_zero_shot": ndcg_top(g["zero_shot"].to_numpy(), y),
            "coverage_plain": plain_cover,
            "coverage_normalized": norm_cover,
            "mmd": stat,
            "mmd_p_value": p,
            "evidently": report,
            "recalibrated": recal,
        }
        rows.append(
            {
                "dms_id": dms_id,
                "n": len(g),
                "rho_model": payload_assays[dms_id]["spearman_model"],
                "rho_zs": payload_assays[dms_id]["spearman_zero_shot"],
                "cover_plain": plain_cover,
                "cover_norm": norm_cover,
                "mmd_p": p,
                **{f"cover_{k}": recal[str(k)]["coverage"] for k in LABEL_BUDGETS},
            }
        )

    table = pd.DataFrame(rows)
    print(f"\nheld-out families, intervals calibrated in-family at {1 - ALPHA:.0%}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    payload = {
        "scheme": args.scheme,
        "alpha": ALPHA,
        "q_plain": q_plain,
        "q_normalized": q_norm,
        "label_budgets": LABEL_BUDGETS,
        "draws": DRAWS,
        "assays": payload_assays,
    }
    print("\nwrote", write(f"shift-{args.scheme}", payload))
    print(
        json.dumps(
            {
                k: {
                    "coverage_before": v["coverage_normalized"],
                    "coverage_after_200_labels": v["recalibrated"]["200"]["coverage"],
                }
                for k, v in payload_assays.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
