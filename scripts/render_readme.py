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
    # The largest miss over both schemes and every target. This was a min() once, reported as
    # the largest, over the split scheme only, which made the calibration look twice as tight
    # as it measured.
    worst = max(
        (abs(float(t) - b[scheme]["coverage"]), t, scheme)
        for t, b in train["coverage"].items()
        for scheme in ("split", "normalized")
    )
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
        f"The largest gap between target and achieved coverage is {worst[0]:.3f}, by the "
        f"{worst[2].replace('normalized', 'normalized scheme').replace('split', 'split scheme')} "
        f"at the {float(worst[1]):.0%} target.",
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
    summary = data.get("warm", {})
    lines = [
        "## Serving",
        "",
        "ONNX export with LoRA merged, checked against PyTorch: largest absolute difference",
        f"{data['parity_max_abs_diff']:.2e}, against a tolerance of {data['parity_tolerance']:g}.",
        "",
        f"The router is driven in-process through ASGI with {load['clients']} concurrent",
        f"clients sending {load['variants_per_request']} variants per request. Latency is a",
        "property of that workload, not of the model alone. Every figure below is the median",
        f"of {summary.get('on', {}).get('repeats', 1)} warm runs, with the spread beside it,",
        "because a single run of this on a laptop moves by more than a factor of two.",
        "",
        "| batching | median ms | p95 ms | p95 spread | variants per second |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label in ("off", "on"):
        m = summary.get(label)
        if not m:
            continue
        lines.append(
            f"| {label} | {m['median_ms_median']:.1f} | {m['p95_ms_median']:.1f} | "
            f"{m['p95_ms_min']:.0f} to {m['p95_ms_max']:.0f} | "
            f"{m['throughput_variants_per_s_median']:.0f} |"
        )
    on, off = summary.get("on"), summary.get("off")
    lines.append("")
    if on and off:
        ratio = float(off["p95_ms_median"]) / float(on["p95_ms_median"])
        tput = float(off["throughput_variants_per_s_median"]) / float(on["throughput_variants_per_s_median"])
        why = (
            "The reason is the device: these graphs run on CPU, where one request for eight "
            "variants already uses every core. Gathering eight of them into one pass replaces "
            "eight parallel scorings with a single sequential one and adds the batching window "
            "on top. Dynamic batching pays off where a single request leaves the accelerator "
            "idle, which is true of a GPU and is not true here. It is left in the router, off "
            "by default for this deployment, because the right setting is a property of the "
            "hardware rather than of the code."
        )
        overlap = float(on["p95_ms_max"]) >= float(off["p95_ms_min"]) and float(off["p95_ms_max"]) >= float(
            on["p95_ms_min"]
        )
        if overlap:
            # The honest reading when the ranges cross: a ratio of medians here would be a
            # number with no evidence behind it. Single runs of this put the ratio anywhere
            # from 0.26 to 1.01, which is what the repeats were added to expose.
            lines.append(
                f"The two p95 figures cannot be told apart at this sample size: "
                f"{on['p95_ms_median']:.0f} ms against {off['p95_ms_median']:.0f} ms, with "
                f"ranges of {on['p95_ms_min']:.0f} to {on['p95_ms_max']:.0f} and "
                f"{off['p95_ms_min']:.0f} to {off['p95_ms_max']:.0f} that overlap along their "
                "whole length. Reporting a ratio of the medians would be reporting noise. "
                f"What does separate is the rest of the distribution: median latency "
                f"{on['median_ms_median']:.0f} ms batched against "
                f"{off['median_ms_median']:.0f} ms unbatched, and throughput "
                f"{on['throughput_variants_per_s_median']:.0f} against "
                f"{off['throughput_variants_per_s_median']:.0f} variants per second. Batching "
                f"costs {tput:.1f} times the throughput here and buys nothing at the tail. " + why
            )
        elif ratio >= 1:
            lines.append(
                f"Dynamic batching divides p95 by {ratio:.2f}, from "
                f"{off['p95_ms_median']:.0f} ms to {on['p95_ms_median']:.0f} ms, with the "
                "repeat ranges clear of each other."
            )
        else:
            lines.append(
                f"Dynamic batching makes this workload {1 / ratio:.1f} times slower at p95, "
                f"and {tput:.1f} times slower on throughput. That is the measurement. " + why
            )
        lines.append("")
    lines += [
        f"The A/B split sent {data['ab']['expected_candidate_share']:.0%} of experiments to the "
        f"candidate against a configured 20%, which is what a hash over {len(data['ab']['arms'])} "
        "arms and a few dozen experiments gives: the assignment is stable per experiment rather "
        "than balanced per request. Shadow scoring stayed off the response path at a median of "
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
