"""Command line: fetch the data, build the zero-shot cache, score variants, serve a version.

    foldgate download                      # ProteinGym substitution shards
    foldgate zero-shot                     # cache the wild-type-marginal score per variant
    foldgate predict --protein-file wt.txt --variants A23G,L45P
    foldgate serve --scheme random --port 8080

``serve`` reads the deployment's environment, which is how the Helm chart and the KServe
manifest in ``deploy/`` configure it: ``FOLDGATE_ARTIFACTS`` (the version directory an init
container fetched), ``FOLDGATE_CANDIDATE_ARTIFACTS``, ``FOLDGATE_CANDIDATE_SHARE``,
``FOLDGATE_SHADOW``, ``FOLDGATE_MAX_BATCH`` and ``FOLDGATE_MAX_WAIT_MS``. Without them it
serves ``artifacts/<scheme>`` from a local training run.

The experiments live in ``eval/``: training, the shift test, the release gate and the load
test each write a results file under ``runs/``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

ARTIFACTS = Path("artifacts")


def version_dir(scheme: str, which: str = "champion") -> Path:
    """Where a served version's files are: the deployment's mount, else a local run."""
    env = os.environ.get("FOLDGATE_ARTIFACTS" if which == "champion" else "FOLDGATE_CANDIDATE_ARTIFACTS", "")
    return Path(env) if env else ARTIFACTS / scheme


def _version(scheme: str, name: str = "champion") -> Any:
    """Load a served version from the artifacts a training run left behind."""
    import numpy as np

    from foldgate.calibration import conformal_quantile, fit_difficulty
    from foldgate.export import session
    from foldgate.model import DEFAULT_BACKBONE, tokenizer
    from foldgate.router import ModelVersion

    art = version_dir(scheme, name)
    base = json.loads((art / "calibration.json").read_text())
    saved = np.load(art / "predictions.npz", allow_pickle=True)
    cal_pred, cal_y, cal_emb = saved["cal_pred"], saved["cal_y"], saved["cal_emb"]
    rng = np.random.default_rng(0)
    half = rng.random(len(cal_y)) < 0.5
    difficulty = fit_difficulty(cal_emb[half], cal_y[half] - cal_pred[half])
    sigma = np.exp(difficulty.predict(cal_emb[~half]))
    q = conformal_quantile(np.abs(cal_y[~half] - cal_pred[~half]) / sigma, base["alpha"])
    return ModelVersion(
        name=name,
        regressor=session(art / "onnx" / "regressor.onnx"),
        mlm=session(art / "onnx" / "mlm.onnx"),
        tokenizer=tokenizer(DEFAULT_BACKBONE),
        q=q,
        difficulty=difficulty,
        abstain_width=base["abstain_width"],
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="foldgate", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="download the ProteinGym substitution shards")

    zs = sub.add_parser("zero-shot", help="cache the zero-shot score for every variant")
    zs.add_argument("--backbone", default=None)

    pr = sub.add_parser("predict", help="score variants with the exported champion")
    pr.add_argument("--scheme", default="random", choices=["random", "contiguous"])
    pr.add_argument("--protein-file", type=Path, required=True)
    pr.add_argument("--variants", required=True, help="comma separated, e.g. A23G,L45P")

    sv = sub.add_parser("serve", help="serve a version over HTTP")
    sv.add_argument("--scheme", default="random", choices=["random", "contiguous"])
    sv.add_argument("--port", type=int, default=8080)
    sv.add_argument("--host", default="127.0.0.1", help="0.0.0.0 in a container")
    sv.add_argument("--no-batching", action="store_true")
    args = ap.parse_args(argv)

    if args.command == "download":
        from foldgate.data import download

        for path in download():
            print(path)
    elif args.command == "zero-shot":
        from foldgate.prepare import ZERO_SHOT, zero_shot_frame

        df = zero_shot_frame(backbone=args.backbone)
        print(f"{len(df)} variants cached in {ZERO_SHOT}")
    elif args.command == "predict":
        lines = args.protein_file.read_text().splitlines()
        sequence = "".join(line.strip() for line in lines if not line.startswith(">"))
        version = _version(args.scheme)
        rows = version.score([(sequence, v.strip()) for v in args.variants.split(",")])
        print(json.dumps(rows, indent=2))
    elif args.command == "serve":
        import uvicorn

        from foldgate.router import create_app

        share = float(os.environ.get("FOLDGATE_CANDIDATE_SHARE", "0") or 0)
        shadow = os.environ.get("FOLDGATE_SHADOW", "").lower() in {"1", "true", "yes"}
        wants_candidate = share > 0 or shadow
        candidate = _version(args.scheme, "candidate") if wants_candidate else None
        app = create_app(
            _version(args.scheme),
            candidate,
            candidate_share=share,
            shadow=shadow,
            batching=not args.no_batching,
            max_batch=int(os.environ.get("FOLDGATE_MAX_BATCH", "64")),
            max_wait_ms=float(os.environ.get("FOLDGATE_MAX_WAIT_MS", "8")),
        )
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
