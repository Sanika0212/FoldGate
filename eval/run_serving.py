"""Export to ONNX, check it matches PyTorch, and measure the router under load.

    python eval/run_serving.py --scheme random

1. **Export and parity.** LoRA is merged into the weights and the graph is exported twice
   (regressor, masked language model). Parity is the largest absolute difference between
   ONNX and PyTorch scores on a held-out batch; anything above 1e-3 fails the run.
2. **Latency.** The router is driven in-process through ASGI with 16 concurrent clients
   sending 8 variants each, with dynamic batching on and off, and with the zero-shot cache
   cold and warm. Reported as median, p95 and throughput.
3. **A/B and shadow.** The same requests are replayed with a candidate model at a 20%
   share, then in shadow mode, to show the split is stable per experiment and that shadow
   scoring stays off the response path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from common import write
from run_shift import load_module

from foldgate.calibration import fit_difficulty
from foldgate.data import load
from foldgate.export import export_mlm, export_regressor, parity, reference_scores, session
from foldgate.model import DEFAULT_BACKBONE, tokenizer
from foldgate.router import ModelVersion, ab_arm, create_app
from foldgate.train import Collate, Variants

CLIENTS = 8
VARIANTS_PER_REQUEST = 8
REQUESTS_PER_CLIENT = 8
PARITY_TOLERANCE = 1e-3


def model_version(
    name: str, onnx_dir: Path, calibration: dict[str, float], difficulty: object, backbone: str
) -> ModelVersion:
    return ModelVersion(
        name=name,
        regressor=session(onnx_dir / "regressor.onnx"),
        mlm=session(onnx_dir / "mlm.onnx"),
        tokenizer=tokenizer(backbone),
        q=calibration["q_normalized_quantile"],
        difficulty=difficulty,
        abstain_width=calibration["abstain_width"],
    )


async def drive(
    app: object, protein: str, variants: list[str], batching_label: str, experiments: list[str]
) -> dict[str, float | int | dict[str, int]]:
    import httpx

    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    latencies: list[float] = []
    arms: Counter[str] = Counter()
    async with httpx.AsyncClient(transport=transport, base_url="http://router") as client:

        async def one(experiment: str, i: int) -> None:
            picked = variants[i * VARIANTS_PER_REQUEST : (i + 1) * VARIANTS_PER_REQUEST]
            started = time.perf_counter()
            r = await client.post(
                "/predict", json={"protein": protein, "variants": picked, "experiment": experiment}
            )
            latencies.append(time.perf_counter() - started)
            r.raise_for_status()
            arms[r.json()["arm"]] += 1

        async def one_client(c: int) -> None:
            """A client waits for its answer before asking again, so concurrency stays at CLIENTS."""
            for i in range(REQUESTS_PER_CLIENT):
                await one(experiments[(c * REQUESTS_PER_CLIENT + i) % len(experiments)], i)

        started = time.perf_counter()
        await asyncio.gather(*[one_client(c) for c in range(CLIENTS)])
        wall = time.perf_counter() - started
    n = len(latencies)
    return {
        "batching": batching_label,
        "requests": n,
        "variants": n * VARIANTS_PER_REQUEST,
        "median_ms": float(np.median(latencies) * 1000),
        "p95_ms": float(np.quantile(latencies, 0.95) * 1000),
        "throughput_variants_per_s": float(n * VARIANTS_PER_REQUEST / wall),
        "arms": dict(arms),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="random", choices=["random", "contiguous"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    args = ap.parse_args()

    art = Path("artifacts") / args.scheme
    module = load_module(art / "module.pt")
    saved = np.load(art / "predictions.npz", allow_pickle=True)
    cal_pred, cal_y, cal_emb = saved["cal_pred"], saved["cal_y"], saved["cal_emb"]
    base = json.loads((art / "calibration.json").read_text())

    # Difficulty model and normalized quantile, fitted exactly as the gate reports them.
    from foldgate.calibration import conformal_quantile

    rng = np.random.default_rng(args.seed)
    half = rng.random(len(cal_y)) < 0.5
    difficulty = fit_difficulty(cal_emb[half], cal_y[half] - cal_pred[half])
    sigma_cal = np.exp(difficulty.predict(cal_emb[~half]))
    q_norm = conformal_quantile(np.abs(cal_y[~half] - cal_pred[~half]) / sigma_cal, base["alpha"])
    calibration = {
        "alpha": base["alpha"],
        "q_normalized_quantile": q_norm,
        "abstain_width": base["abstain_width"],
    }

    assays = load(["CCDB_ECOLI_Tripathi_2016"])
    assay = assays["CCDB_ECOLI_Tripathi_2016"]
    sample_rows = assay.variants.head(4).assign(zero_shot=0.0)
    collate = Collate(args.backbone)
    batch = collate([Variants(sample_rows)[i] for i in range(len(sample_rows))])

    onnx_dir = art / "onnx"
    reference = reference_scores(module, batch)  # before the export: merging LoRA is in place
    export_regressor(module, onnx_dir / "regressor.onnx", batch)
    export_mlm(args.backbone, onnx_dir / "mlm.onnx", batch["input_ids"][:1])
    gap = parity(reference, onnx_dir / "regressor.onnx", batch)
    sizes = {p.name: p.stat().st_size for p in sorted(onnx_dir.glob("*.onnx"))}
    print(f"parity max |onnx - torch| = {gap:.2e} (tolerance {PARITY_TOLERANCE:g})")
    print("onnx bytes", sizes)
    if gap > PARITY_TOLERANCE:
        raise SystemExit(f"ONNX parity {gap:.2e} exceeds {PARITY_TOLERANCE:g}")

    champion = model_version("champion", onnx_dir, calibration, difficulty, args.backbone)
    candidate = model_version("candidate", onnx_dir, calibration, difficulty, args.backbone)

    protein = assay.wild_type
    variants = assay.variants["mutant"].head(CLIENTS * REQUESTS_PER_CLIENT * VARIANTS_PER_REQUEST).tolist()
    experiments = [f"lab-{i}" for i in range(40)]

    measurements = []
    for batching in (True, False):
        app = create_app(champion, batching=batching)
        label = "on" if batching else "off"
        cold = asyncio.run(drive(app, protein, variants, label, experiments))
        warm = asyncio.run(drive(app, protein, variants, label, experiments))
        cold["cache"], warm["cache"] = "cold", "warm"
        measurements += [cold, warm]
        print(
            f"batching {label:>3} cache {warm['cache']}: median {warm['median_ms']:.1f} ms  "
            f"p95 {warm['p95_ms']:.1f} ms  {warm['throughput_variants_per_s']:.0f} variants/s"
        )

    ab_app = create_app(champion, candidate, candidate_share=0.2)
    ab = asyncio.run(drive(ab_app, protein, variants, "on", experiments))
    shadow_log = Path("runs") / f"serving-{args.scheme}" / "shadow.jsonl"
    shadow_log.parent.mkdir(parents=True, exist_ok=True)
    shadow_log.unlink(missing_ok=True)
    shadow_app = create_app(champion, candidate, shadow=True, shadow_log=shadow_log)
    shadow = asyncio.run(drive(shadow_app, protein, variants, "on", experiments))
    expected = sum(ab_arm(e, 0.2) == "candidate" for e in experiments) / len(experiments)
    print(f"A/B arms {ab['arms']} (share of experiments routed to the candidate {expected:.2f})")
    print(
        f"shadow: median {shadow['median_ms']:.1f} ms, {sum(1 for _ in shadow_log.open())} comparisons logged"
    )

    warm_on = next(m for m in measurements if m["batching"] == "on" and m["cache"] == "warm")
    warm_off = next(m for m in measurements if m["batching"] == "off" and m["cache"] == "warm")
    payload = {
        "scheme": args.scheme,
        "backbone": args.backbone,
        "parity_max_abs_diff": gap,
        "parity_tolerance": PARITY_TOLERANCE,
        "onnx_bytes": sizes,
        "load": {
            "clients": CLIENTS,
            "variants_per_request": VARIANTS_PER_REQUEST,
            "requests_per_client": REQUESTS_PER_CLIENT,
        },
        "measurements": measurements,
        "batching_speedup": warm_off["p95_ms"] / warm_on["p95_ms"],
        "ab": {"arms": ab["arms"], "expected_candidate_share": expected},
        "shadow": {"median_ms": shadow["median_ms"], "comparisons": sum(1 for _ in shadow_log.open())},
        "calibration": calibration,
    }
    print("\nwrote", write(f"serving-{args.scheme}", payload))


if __name__ == "__main__":
    main()
