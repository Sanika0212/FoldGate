"""The release gate: three candidates, one promotion.

    python eval/run_gate.py --scheme random

A candidate is promoted to ``champion`` only when it is (1) more accurate than zero-shot on
the test split, (2) honest, meaning its intervals cover within 2 points of the target, and
(3) fast enough, p95 under the serving SLO. Everything is logged to MLflow either way, so a
rejected version keeps its evidence.

Three candidates are put through it:

1. the trained model with its own calibration, which should pass;
2. the same model with a quantile taken at 70% but shipped as 90%, which is the easy way to
   make intervals look impressively narrow. The gate has to catch it;
3. zero-shot predictions dressed up as a model, which is accurate enough to look
   reasonable but no better than the free baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from common import write

from foldgate.calibration import conformal_quantile, fit_difficulty, gate, spearman
from foldgate.registry import NAME, log_candidate

# The SLO is a property of a stated workload, not a round number: one CPU replica serving
# eight concurrent clients, eight variants per request. A forward pass over a 300-residue
# sequence is about 20 GFLOPs per variant, so anything under a second there is arithmetic
# that does not exist on a CPU. The load test reports the same shape of request.
SLO_MS = 3000.0
TARGET = 0.90
TOLERANCE = 0.02


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="random", choices=["random", "contiguous"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    art = Path("artifacts") / args.scheme
    saved = np.load(art / "predictions.npz", allow_pickle=True)
    cal_pred, cal_y, cal_emb = saved["cal_pred"], saved["cal_y"], saved["cal_emb"]
    test_pred, test_y, test_emb = saved["test_pred"], saved["test_y"], saved["test_emb"]
    train = json.loads(Path(f"runs/train-{args.scheme}/results.json").read_text())
    serving_path = Path(f"runs/serving-{args.scheme}/results.json")
    serving = json.loads(serving_path.read_text()) if serving_path.exists() else {}
    warm = [m for m in serving.get("measurements", []) if m["batching"] == "on" and m["cache"] == "warm"]
    p95_ms = float(warm[0]["p95_ms"]) if warm else 0.0
    onnx_dir = art / "onnx" if (art / "onnx").exists() else None

    zs = np.asarray(train["baseline_mean_spearman"].get("zero_shot", 0.0))
    rho_zero_shot = float(zs)

    rng = np.random.default_rng(args.seed)
    half = rng.random(len(cal_y)) < 0.5
    difficulty = fit_difficulty(cal_emb[half], cal_y[half] - cal_pred[half])
    sigma_cal = np.exp(difficulty.predict(cal_emb[~half]))
    sigma_test = np.exp(difficulty.predict(test_emb))
    residual = np.abs(cal_y[~half] - cal_pred[~half]) / sigma_cal

    def coverage(pred: np.ndarray, q: float) -> float:
        return float(np.mean(np.abs(test_y - pred) <= q * sigma_test))

    honest_q = conformal_quantile(residual, 1 - TARGET)
    narrow_q = conformal_quantile(residual, 0.30)  # a 70% quantile shipped as 90%

    # The zero-shot candidate gets its own honest quantile, so it can only fail on accuracy.
    zs_cal, zs_test = zero_shot_candidate(args.scheme, args.seed, cal_y)
    zs_q = conformal_quantile(np.abs(cal_y[~half] - zs_cal[~half]) / sigma_cal, 1 - TARGET)

    candidates = [
        {
            "name": "lora-conformal",
            "pred": test_pred,
            "q": honest_q,
            "params": {
                "kind": "lora",
                "alpha": round(1 - TARGET, 2),
                "scheme": args.scheme,
                "backbone": train["backbone"],
                "rank": train["rank"],
                "epochs": train["epochs"],
            },
        },
        {
            "name": "lora-narrow-intervals",
            "pred": test_pred,
            "q": narrow_q,
            "params": {
                "kind": "lora",
                "alpha": 0.30,
                "shipped_as": round(1 - TARGET, 2),
                "scheme": args.scheme,
            },
        },
        {
            "name": "zero-shot-only",
            "pred": zs_test,
            "q": zs_q,
            "params": {"kind": "zero-shot", "scheme": args.scheme},
        },
    ]

    rows, payload = [], {}
    for c in candidates:
        pred = np.asarray(c["pred"], dtype=np.float32)
        rho = spearman(pred, test_y)
        cov = coverage(pred, float(c["q"]))
        verdict = gate(rho, rho_zero_shot, cov, TARGET, p95_ms, SLO_MS, TOLERANCE)
        metrics = {
            "spearman": rho,
            "spearman_zero_shot": rho_zero_shot,
            "coverage": cov,
            "target_coverage": TARGET,
            "mean_width": float(2 * c["q"] * sigma_test.mean()),
            "p95_ms": p95_ms,
        }
        calibration = {
            "alpha": round(1 - TARGET, 2),
            "q_normalized_quantile": float(c["q"]),
            "abstain_width": float(np.quantile(2 * float(c["q"]) * sigma_test, 0.9)),
        }
        gate_blob = {
            "passed": verdict.passed,
            "reasons": verdict.reasons,
            "slo_ms": SLO_MS,
            "tolerance": TOLERANCE,
            **metrics,
        }
        logged = log_candidate(
            onnx_dir, calibration, difficulty, metrics, dict(c["params"]), gate_blob, str(c["name"])
        )
        rows.append(
            {
                "candidate": c["name"],
                "spearman": rho,
                "coverage": cov,
                "p95_ms": p95_ms,
                "passed": verdict.passed,
                "version": logged["version"],
            }
        )
        payload[str(c["name"])] = {"metrics": metrics, "gate": gate_blob, "registry": logged}
        mark = "promoted" if verdict.passed else "rejected"
        print(f"{c['name']:>22}  rho {rho:.3f}  coverage {cov:.3f}  version {logged['version']}  {mark}")
        for reason in verdict.reasons:
            print(f"{'':>22}  reason: {reason}")

    import pandas as pd

    print()
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    champion = [r for r in rows if r["passed"]]
    print(f"\n{len(champion)} of {len(rows)} candidates carry the champion alias in the '{NAME}' registry")
    print(
        "wrote",
        write(
            f"gate-{args.scheme}",
            {"scheme": args.scheme, "slo_ms": SLO_MS, "target_coverage": TARGET, "candidates": payload},
        ),
    )


def zero_shot_candidate(scheme: str, seed: int, cal_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-shot scores on the calibration and test rows, put on the label scale.

    Zero-shot is a log-likelihood ratio, not a standardized assay score, so it is shifted
    and scaled to the calibration labels. That is the fairest version of the baseline: if
    it still fails the gate, it fails on ranking, not on units.
    """
    from common import IN_FAMILY, pooled

    from foldgate.data import load

    df = pooled(load(), IN_FAMILY, scheme, seed)
    cal = df[df["split"] == "calib"]["zero_shot"].to_numpy(np.float32)
    test = df[df["split"] == "test"]["zero_shot"].to_numpy(np.float32)
    scale = float(cal_y.std()) / float(cal.std())
    shift = float(cal_y.mean()) - scale * float(cal.mean())
    return cal * scale + shift, test * scale + shift


if __name__ == "__main__":
    main()
