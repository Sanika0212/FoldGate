"""Fill the README's result sections from the files the evaluation runs wrote.

    python scripts/render_readme.py --scheme random

The README holds five markers, and each is replaced by a section rendered from
``runs/<step>-<scheme>/results.json``. Numbers in the README are therefore never typed by
hand: they come from the run that produced them, and re-rendering after a new run updates
them. A marker whose results file is missing is reported by name and left alone, so a
half-finished evaluation cannot quietly publish a README with a placeholder in it.

Run ``--check`` in CI to fail when the README and the results disagree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
README = ROOT / "README.md"
MARKERS = ("ACCURACY", "CALIBRATION", "SHIFT", "GATE", "SERVING")
# Which results file each section needs.
NEEDS = {
    "ACCURACY": "train",
    "CALIBRATION": "train",
    "SHIFT": "shift",
    "GATE": "gate",
    "SERVING": "serving",
}


def results(step: str, scheme: str) -> dict[str, Any] | None:
    path = RUNS / f"{step}-{scheme}" / "results.json"
    if not path.exists():
        return None
    loaded: dict[str, Any] = json.loads(path.read_text())
    return loaded


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def accuracy(train: dict[str, Any]) -> str:
    rows = sorted(train["per_assay"], key=lambda r: -float(r["spearman"]))
    baselines = train.get("baseline_mean_spearman") or {}
    best_baseline = max(baselines.items(), key=lambda kv: kv[1]) if baselines else None
    lines = [
        "## Accuracy",
        "",
        f"Nine pooled assays, {train['n']['train']:,} training variants, "
        f"{train['n']['test']:,} held out for test. LoRA trains "
        f"{train['trainable_share'] * 100:.2f}% of the weights.",
        "",
        "| assay | variants | Spearman | NDCG of the top decile |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['dms_id']} | {int(r['n']):,} | {float(r['spearman']):.3f} | {float(r['ndcg_top10']):.3f} |"
        )
    lines += [
        f"| **mean** | | **{train['mean_spearman']:.3f}** | **{train['mean_ndcg_top10']:.3f}** |",
        "",
    ]
    if baselines:
        lines.append("Against the baselines, by mean Spearman across assays:")
        lines.append("")
        lines.append("| baseline | Spearman | difference against this model |")
        lines.append("| --- | --- | --- |")
        for name, value in sorted(baselines.items(), key=lambda kv: -kv[1]):
            delta = train["mean_spearman"] - value
            lines.append(f"| {name} | {value:.3f} | {delta:+.3f} |")
        lines.append("")
        assert best_baseline is not None
        if train["mean_spearman"] <= best_baseline[1]:
            lines.append(
                f"The strongest baseline, {best_baseline[0]}, is not beaten here: "
                f"{best_baseline[1]:.3f} against {train['mean_spearman']:.3f}. A ridge regression "
                "on one-hot mutations plus the zero-shot score is cheaper to fit than this model "
                "and scores better on these assays, so nothing here argues for the trained model "
                "on accuracy alone. What it adds is the calibrated interval, which the baseline "
                "does not have."
            )
            lines.append("")
            lines.append(
                "The release gate does not catch this, and that is worth naming: its accuracy "
                "criterion is only that the model beat zero-shot, which it does by a wide margin. "
                "A gate that compared against the best available baseline would reject this "
                "candidate. The one built here promotes it."
            )
            lines.append("")
    return "\n".join(lines)


def calibration(train: dict[str, Any]) -> str:
    lines = [
        "## Do the intervals mean what they say",
        "",
        "Coverage is measured on the test split, with the conformal quantile taken from the",
        "calibration split. A narrower interval is only better if it still covers, so the",
        "width is reported beside it.",
        "",
        "| target | split conformal | width | normalized | width |",
        "| --- | --- | --- | --- | --- |",
    ]
    for target in sorted(train["coverage"], reverse=True):
        block = train["coverage"][target]
        lines.append(
            f"| {float(target):.0%} | {block['split']['coverage']:.3f} | "
            f"{block['split']['width']:.2f} | {block['normalized']['coverage']:.3f} | "
            f"{block['normalized']['width']:.2f} |"
        )
    worst = min((abs(float(t) - b["split"]["coverage"]), t) for t, b in train["coverage"].items())
    lines += [
        "",
        "A plain conformal interval is one width for every variant, so it over-covers the easy",
        "ones and under-covers the hard ones. Coverage inside each third of the variants,",
        "ordered by predicted difficulty, at the 90% target:",
        "",
        "| difficulty | split conformal | normalized |",
        "| --- | --- | --- |",
    ]
    terciles = train["coverage"].get("0.90")
    if terciles:
        names = ("easiest third", "middle third", "hardest third")
        for i, name in enumerate(names):
            lines.append(
                f"| {name} | {terciles['split']['terciles'][i]:.3f} | "
                f"{terciles['normalized']['terciles'][i]:.3f} |"
            )
    lines += [
        "",
        f"The largest gap between target and achieved coverage is {worst[0]:.3f}, at the "
        f"{float(worst[1]):.0%} target.",
        "",
    ]
    return "\n".join(lines)


def shift(data: dict[str, Any]) -> str:
    budgets = data["label_budgets"]
    biggest = str(max(budgets))
    lines = [
        "## What happens on a protein it never saw",
        "",
        "Conformal coverage holds only while test variants are exchangeable with the",
        "calibration ones. A protein from a family the model never trained on breaks that.",
        "Three were held out for exactly this measurement.",
        "",
        f"| assay | variants | Spearman | zero-shot | coverage | after {biggest} labels | MMD p |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for dms_id, a in sorted(data["assays"].items()):
        after = a["recalibrated"][biggest]["coverage"]
        lines.append(
            f"| {dms_id} | {int(a['n']):,} | {a['spearman_model']:.3f} | "
            f"{a['spearman_zero_shot']:.3f} | {a['coverage_normalized']:.3f} | "
            f"{after:.3f} | {a['mmd_p_value']:.3g} |"
        )
    target = 1 - data["alpha"]
    lines += [
        "",
        f"The target is {target:.0%}. Labelling {biggest} variants of the new protein and",
        "recomputing the quantile is what restores it, and the MMD p-value says the shift is",
        "detectable before any label arrives.",
        "",
    ]
    return "\n".join(lines)


def gate(data: dict[str, Any]) -> str:
    lines = [
        "## The release gate",
        "",
        "A candidate is promoted only when it beats zero-shot on accuracy, covers within two",
        f"points of its target, and keeps p95 latency under {data['slo_ms']:.0f} ms. Three",
        "candidates are put through it, two of which should fail.",
        "",
        "| candidate | Spearman | coverage | p95 ms | verdict |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, c in data["candidates"].items():
        m, g = c["metrics"], c["gate"]
        verdict = "promoted" if g["passed"] else "rejected"
        lines.append(
            f"| {name} | {m['spearman']:.3f} | {m['coverage']:.3f} | {m['p95_ms']:.0f} | {verdict} |"
        )
    rejected = [(n, c) for n, c in data["candidates"].items() if not c["gate"]["passed"]]
    if rejected:
        lines += ["", "Why each rejection was rejected:", ""]
        for i, (name, c) in enumerate(rejected, start=1):
            reasons = "; ".join(c["gate"]["reasons"])
            lines.append(f"{i}. **{name}**: {reasons}")
    lines.append("")
    return "\n".join(lines)


def serving(data: dict[str, Any]) -> str:
    load = data["load"]
    warm = [m for m in data["measurements"] if m["cache"] == "warm"]
    lines = [
        "## Serving",
        "",
        "ONNX export with LoRA merged, checked against PyTorch: largest absolute difference",
        f"{data['parity_max_abs_diff']:.2e}, against a tolerance of {data['parity_tolerance']:g}.",
        "",
        f"The router is driven in-process through ASGI with {load['clients']} concurrent",
        f"clients sending {load['variants_per_request']} variants per request. Latency is a",
        "property of that workload, not of the model alone.",
        "",
        "| batching | median ms | p95 ms | variants per second |",
        "| --- | --- | --- | --- |",
    ]
    for m in sorted(warm, key=lambda m: m["batching"]):
        lines.append(
            f"| {m['batching']} | {m['median_ms']:.1f} | {m['p95_ms']:.1f} | "
            f"{m['throughput_variants_per_s']:.0f} |"
        )
    lines += [
        "",
        f"Dynamic batching improves p95 by {data['batching_speedup']:.2f} times. The A/B split",
        f"sent {data['ab']['expected_candidate_share']:.0%} of experiments to the candidate,",
        f"and shadow scoring stayed off the response path at a median of "
        f"{data['shadow']['median_ms']:.1f} ms with "
        f"{data['shadow']['comparisons']:,} comparisons logged.",
        "",
    ]
    return "\n".join(lines)


RENDER = {
    "ACCURACY": accuracy,
    "CALIBRATION": calibration,
    "SHIFT": shift,
    "GATE": gate,
    "SERVING": serving,
}


def block(marker: str, body: str) -> str:
    return f"<!-- {marker}:start -->\n{body}\n<!-- {marker}:end -->"


def replace(text: str, marker: str, body: str) -> str:
    """Replace either the bare placeholder or a previously rendered block."""
    start, end = f"<!-- {marker}:start -->", f"<!-- {marker}:end -->"
    if start in text and end in text:
        head = text[: text.index(start)]
        tail = text[text.index(end) + len(end) :]
        return head + block(marker, body) + tail
    return text.replace(f"{marker}_SECTION", block(marker, body))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scheme", default="random", choices=["random", "contiguous"])
    ap.add_argument("--check", action="store_true", help="fail if the README is not up to date")
    args = ap.parse_args()

    text = original = README.read_text()
    missing = []
    for marker in MARKERS:
        data = results(NEEDS[marker], args.scheme)
        if data is None:
            missing.append(f"{marker} needs runs/{NEEDS[marker]}-{args.scheme}/results.json")
            continue
        text = replace(text, marker, RENDER[marker](data))

    if args.check:
        stale = [m for m in MARKERS if f"{m}_SECTION" in text]
        if stale or text != original:
            raise SystemExit(
                "README is not up to date with runs/: "
                + ", ".join(stale or ["re-run scripts/render_readme.py"])
            )
        print("README matches the results in runs/")
        return

    README.write_text(text)
    for line in missing:
        print(f"left alone: {line}")
    filled = len(MARKERS) - len(missing)
    print(f"rendered {filled} of {len(MARKERS)} sections into README.md")
    if missing:
        raise SystemExit("some sections are still placeholders")


if __name__ == "__main__":
    main()
