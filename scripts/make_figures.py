"""Build every README figure, light and dark.

    uv run python scripts/make_figures.py

Inputs are the ``runs/*/results.json`` files the evaluation scripts write. Outputs go to
``docs/assets/<name>-{light,dark}.{png,svg,gif}``. Missing inputs are skipped, so this can
run after any subset of the experiments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch

ASSETS = Path("docs/assets")
RUNS = Path("runs")
DPI = 200
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
GOOD, WARN, BAD = "#0ca30c", "#fab219", "#d03b3b"


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    card: str
    series: tuple[str, ...]


THEMES = {
    "light": Theme(
        "light",
        "#fcfcfb",
        "#0b0b0b",
        "#52514e",
        "#898781",
        "#e1e0d9",
        "#c3c2b7",
        "#f1f0eb",
        ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd1"),
    ),
    "dark": Theme(
        "dark",
        "#1a1a19",
        "#ffffff",
        "#c3c2b7",
        "#898781",
        "#2c2c2a",
        "#383835",
        "#242422",
        ("#3987e5", "#d95926", "#199e70", "#c98500", "#9a73dd"),
    ),
}


def use_theme(mode: str) -> Theme:
    t = THEMES[mode]
    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9.5,
            "figure.facecolor": t.surface,
            "axes.facecolor": t.surface,
            "savefig.facecolor": t.surface,
            "text.color": t.ink,
            "axes.labelcolor": t.ink2,
            "axes.edgecolor": t.axis,
            "xtick.color": t.muted,
            "ytick.color": t.muted,
            "xtick.labelcolor": t.ink2,
            "ytick.labelcolor": t.ink2,
            "grid.color": t.grid,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )
    return t


def title(fig: Any, text: str, sub: str, t: Theme) -> None:
    fig.text(0.012, 0.975, text, ha="left", va="top", fontsize=12.5, weight="bold")
    fig.text(0.012, 0.905, sub, ha="left", va="top", fontsize=9, color=t.ink2)


def save(fig: Any, name: str, mode: str) -> None:
    path = ASSETS / f"{name}-{mode}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path)


def load(name: str) -> Any:
    path = RUNS / name / "results.json"
    return json.loads(path.read_text()) if path.exists() else None


def short(dms_id: str) -> str:
    return dms_id.split("_")[0]


# ------------------------------------------------------------------------ banner


def banner(mode: str) -> str:
    t = THEMES[mode]
    w, h = 1200, 260
    rows = [
        ("K41R", 0.62, 0.30, "act"),
        ("L56P", -1.84, 0.34, "act"),
        ("G68S", 0.11, 1.95, "measure"),
        ("T74A", -0.37, 0.28, "act"),
    ]
    x0, y0, scale = 690, 42, 46.0
    mid = x0 + 230
    shapes = [f'<rect x="{x0}" y="{y0}" width="450" height="184" rx="12" fill="{t.card}"/>']
    for i, (label, score, half, verdict) in enumerate(rows):
        y = y0 + 34 + i * 38
        wide = verdict == "measure"
        color = WARN if wide else t.series[0]
        left = mid + (score - half) * scale
        width = max(6.0, 2 * half * scale)
        shapes.append(
            f'<text x="{x0 + 20}" y="{y + 5}" font-family="{FONT}" font-size="15" '
            f'fill="{t.ink}">{label}</text>'
        )
        shapes.append(
            f'<rect x="{left:.1f}" y="{y - 7}" width="{width:.1f}" height="14" rx="7" '
            f'fill="{color}" fill-opacity="0.85"/>'
        )
        shapes.append(f'<circle cx="{mid + score * scale:.1f}" cy="{y}" r="3.4" fill="{t.surface}"/>')
        if wide:
            shapes.append(
                f'<text x="{x0 + 436}" y="{y + 5}" text-anchor="end" font-family="{FONT}" '
                f'font-size="12" font-weight="700" fill="{WARN}">measure this one</text>'
            )
    label = "FoldGate: variant effect predictions with intervals a lab can act on"
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img" aria-label="{label}">'
    )
    lines = (
        f'<text x="58" y="158" font-family="{FONT}" font-size="22" fill="{t.ink2}">'
        f"Variant effect predictions that say</text>"
        f'<text x="58" y="188" font-family="{FONT}" font-size="22" fill="{t.ink2}">'
        f"when a lab should measure instead.</text>"
    )
    return f"""{head}
  <rect width="{w}" height="{h}" rx="18" fill="{t.surface}"/>
  <text x="56" y="112" font-family="{FONT}" font-size="62" font-weight="700" fill="{t.ink}">FoldGate</text>
  {lines}
  {"".join(shapes)}
</svg>
"""


# --------------------------------------------------------------------- animation


def animate(gate: dict[str, Any], mode: str) -> None:
    """Three candidate versions walking through the release gate, one of them rejected."""
    t = use_theme(mode)
    candidates = list(gate["candidates"].items())
    fig = plt.figure(figsize=(10.4, 3.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    def card(x: float, y: float, w: float, h: float, color: str, alpha: float) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.06,rounding_size=0.12",
                linewidth=1.4,
                edgecolor=color,
                facecolor=color,
                alpha=alpha,
            )
        )

    checks = [
        ("beats zero-shot", "spearman", "spearman_zero_shot"),
        ("intervals honest", "coverage", "target_coverage"),
        ("inside the SLO", "p95_ms", "slo_ms"),
    ]
    frames_per = 10

    def draw(frame: int) -> None:
        ax.clear()
        ax.set_xlim(0, 10.4)
        ax.set_ylim(0, 3.4)
        ax.axis("off")
        ax.text(0.25, 3.05, "The release gate", fontsize=13, weight="bold", color=t.ink)
        ax.text(
            0.25,
            2.76,
            "Every candidate is logged. Only one of these three is allowed to serve.",
            fontsize=9.5,
            color=t.ink2,
        )
        step = frame // frames_per
        for i, (name, blob) in enumerate(candidates):
            y = 1.86 - i * 0.78
            metrics, verdict = blob["metrics"], blob["gate"]
            card(0.25, y - 0.1, 9.9, 0.62, t.card, 1.0)
            ax.text(0.5, y + 0.2, name, fontsize=11, color=t.ink, va="center")
            for j, (label, key, against) in enumerate(checks):
                if step < j + 1:
                    continue
                x = 3.5 + j * 2.1
                if key == "spearman":
                    ok = metrics[key] > metrics[against]
                    value = f"{metrics[key]:.3f} vs {metrics[against]:.3f}"
                elif key == "coverage":
                    ok = abs(metrics[key] - metrics[against]) <= verdict["tolerance"]
                    value = f"{metrics[key]:.3f} for {metrics[against]:.2f}"
                else:
                    ok = metrics[key] <= verdict[against]
                    value = f"{metrics[key]:.0f} ms"
                color = GOOD if ok else BAD
                ax.text(x, y + 0.33, label, fontsize=8.5, color=t.muted, va="center")
                ax.text(x, y + 0.08, value, fontsize=9.5, color=color, va="center", weight="bold")
            if step >= len(checks) + 1:
                passed = verdict["passed"]
                ax.text(
                    10.0,
                    y + 0.2,
                    "champion" if passed else "rejected",
                    fontsize=10,
                    weight="bold",
                    color=GOOD if passed else BAD,
                    ha="right",
                    va="center",
                )
        if step >= len(checks) + 2:
            ax.text(
                0.25,
                0.18,
                "A rejected version keeps its metrics and its reasons in the registry; "
                "it just never gets the alias the router loads.",
                fontsize=9,
                color=t.ink2,
            )

    frames = frames_per * (len(checks) + 3)
    anim = FuncAnimation(fig, draw, frames=frames, interval=120)
    path = ASSETS / f"gate-{mode}.gif"
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=8), dpi=100)
    plt.close(fig)
    print(path)


# ------------------------------------------------------------------------- plots


def plot_accuracy(runs: dict[str, Any], mode: str) -> None:
    """Per-assay Spearman for both splits: zero-shot, ridge, and the fine-tuned model."""
    t = use_theme(mode)
    schemes = [s for s in ("random", "contiguous") if runs.get(f"train-{s}")]
    fig, axes = plt.subplots(1, len(schemes), figsize=(10.4, 3.8), squeeze=False)
    for ax, scheme in zip(axes[0], schemes, strict=True):
        train = runs[f"train-{scheme}"]
        base = runs[f"baselines-{scheme}"]["baselines"]
        rows = train["per_assay"]
        labels = [short(r["dms_id"]) for r in rows]
        xs = np.arange(len(rows))
        series = [
            ("zero-shot", [r["spearman"] for r in base["zero_shot"]["per_assay"]], t.muted),
            (
                "ridge on one-hot",
                [r["spearman"] for r in base["ridge_one_hot"]["per_assay"]],
                t.series[1],
            ),
            ("LoRA + zero-shot", [r["spearman"] for r in rows], t.series[0]),
        ]
        width = 0.8 / len(series)
        for j, (name, values, color) in enumerate(series):
            ax.bar(
                xs + (j - (len(series) - 1) / 2) * width,
                values,
                width=width * 0.92,
                color=color,
                label=name,
            )
        ax.set_xticks(xs, labels, fontsize=8, rotation=35, ha="right")
        ax.set_ylim(min(0.0, min(min(v) for _, v, _ in series) - 0.05), 1.0)
        ax.grid(axis="y")
        ax.axhline(0, color=t.axis, lw=0.8)
        mean = train["mean_spearman"]
        ax.set_title(
            f"{scheme} split: mean Spearman {mean:.3f}",
            loc="left",
            fontsize=9.5,
        )
    axes[0][0].set_ylabel("Spearman on the test split")
    axes[0][0].legend(loc="lower left", fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.74, bottom=0.22, wspace=0.16)
    title(
        fig,
        "Ranking variants a lab has not measured",
        "Nine assays pooled; the contiguous split holds out whole stretches of the protein.",
        t,
    )
    save(fig, "accuracy", mode)


def plot_calibration(train: dict[str, Any], mode: str) -> None:
    t = use_theme(mode)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.5), gridspec_kw={"width_ratios": [1.25, 1]})
    cov = train["coverage"]
    targets = sorted(cov, key=float)
    xs = np.arange(len(targets))
    width = 0.36
    for j, (kind, color) in enumerate((("split", t.series[1]), ("normalized", t.series[0]))):
        values = [cov[k][kind]["coverage"] for k in targets]
        widths = [cov[k][kind]["width"] for k in targets]
        pos = xs + (j - 0.5) * width
        ax.bar(pos, values, width=width * 0.92, color=color, label=f"{kind} conformal")
        for x, v, w in zip(pos, values, widths, strict=True):
            ax.text(x, v + 0.012, f"{w:.1f}", ha="center", fontsize=7.5, color=t.ink2)
    for i, target in enumerate(targets):
        ax.plot([i - 0.5, i + 0.5], [float(target)] * 2, color=t.ink, lw=1.6, zorder=3)
    ax.set_xticks(xs, [f"{float(k):.0%} target" for k in targets])
    ax.set_ylim(0.6, 1.02)
    ax.grid(axis="y")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.set_title("Coverage against the target (mean interval width above each bar)", loc="left", fontsize=9.5)

    key = "0.90" if "0.90" in cov else targets[len(targets) // 2]
    labels = ["easiest third", "middle third", "hardest third"]
    xs2 = np.arange(3)
    for j, (kind, color) in enumerate((("split", t.series[1]), ("normalized", t.series[0]))):
        ax2.bar(
            xs2 + (j - 0.5) * width,
            cov[key][kind]["terciles"],
            width=width * 0.92,
            color=color,
        )
    ax2.axhline(float(key), color=t.ink, lw=1.6)
    ax2.set_xticks(xs2, labels, fontsize=8.5)
    ax2.set_ylim(0.6, 1.02)
    ax2.grid(axis="y")
    ax2.set_title(f"Who the {float(key):.0%} interval actually covers", loc="left", fontsize=9.5)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.76, bottom=0.14, wspace=0.18)
    title(
        fig,
        "Intervals that mean what they say",
        "One width for every variant over-covers the easy ones and under-covers the hard ones.",
        t,
    )
    save(fig, "calibration", mode)


def plot_shift(shift: dict[str, Any], mode: str) -> None:
    t = use_theme(mode)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.5), gridspec_kw={"width_ratios": [1, 1.1]})
    assays = shift["assays"]
    names = sorted(assays)
    budgets = [str(b) for b in shift["label_budgets"]]
    target = 1 - shift["alpha"]

    xs = np.arange(len(names))
    width = 0.8 / (len(budgets) + 1)
    ax.bar(
        xs - 0.4 + width / 2,
        [assays[n]["coverage_normalized"] for n in names],
        width=width * 0.92,
        color=t.series[1],
        label="as calibrated in-family",
    )
    for j, budget in enumerate(budgets):
        ax.bar(
            xs - 0.4 + width * (j + 1.5),
            [assays[n]["recalibrated"][budget]["coverage"] for n in names],
            width=width * 0.92,
            color=t.series[0],
            alpha=0.45 + 0.55 * j / max(1, len(budgets) - 1),
            label=f"after {budget} labels",
        )
    ax.axhline(target, color=t.ink, lw=1.6)
    ax.set_xticks(xs, [short(n) for n in names], fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y")
    ax.legend(loc="lower center", fontsize=7.5, ncol=3)
    ax.set_title(f"Coverage on proteins never trained on (target {target:.0%})", loc="left", fontsize=9.5)

    for i, name in enumerate(names):
        a = assays[name]
        ax2.barh(
            i + 0.18, a["spearman_model"], height=0.33, color=t.series[0], label="model" if i == 0 else None
        )
        ax2.barh(
            i - 0.18,
            a["spearman_zero_shot"],
            height=0.33,
            color=t.muted,
            label="zero-shot" if i == 0 else None,
        )
        ax2.text(
            0.01,
            i + 0.52,
            f"MMD p = {a['mmd_p_value']:.3f}",
            fontsize=7.5,
            color=t.ink2,
            va="center",
        )
    ax2.set_yticks(range(len(names)), [short(n) for n in names], fontsize=8.5)
    ax2.set_xlim(0, max(0.6, max(assays[n]["spearman_model"] for n in names) + 0.12))
    ax2.grid(axis="x")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_title("Ranking there, and whether the drift test noticed", loc="left", fontsize=9.5)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.76, bottom=0.14, wspace=0.2)
    title(
        fig,
        "What a new protein does to the guarantee",
        "Exchangeability breaks, coverage falls, and a few labels buy it back.",
        t,
    )
    save(fig, "shift", mode)


def plot_serving(serving: dict[str, Any], mode: str) -> None:
    t = use_theme(mode)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.4), gridspec_kw={"width_ratios": [1.2, 1]})
    warm = [m for m in serving["measurements"] if m["cache"] == "warm"]
    order = ["on", "off"]
    warm = sorted(warm, key=lambda m: order.index(m["batching"]))
    labels = [f"batching {m['batching']}" for m in warm]
    xs = np.arange(len(warm))
    width = 0.36
    ax.bar(
        xs - width / 2,
        [m["median_ms"] for m in warm],
        width=width * 0.92,
        color=t.series[0],
        label="median",
    )
    ax.bar(
        xs + width / 2,
        [m["p95_ms"] for m in warm],
        width=width * 0.92,
        color=t.series[1],
        label="p95",
    )
    for x, m in zip(xs, warm, strict=True):
        ax.text(x - width / 2, m["median_ms"], f"{m['median_ms']:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(x + width / 2, m["p95_ms"], f"{m['p95_ms']:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("milliseconds per request")
    ax.grid(axis="y")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    load = serving["load"]
    ax.set_title(
        f"{load['clients']} clients, {load['variants_per_request']} variants per request",
        loc="left",
        fontsize=9.5,
    )

    for i, m in enumerate(warm[::-1]):
        ax2.barh(i, m["throughput_variants_per_s"], height=0.52, color=t.series[0])
        ax2.text(
            m["throughput_variants_per_s"] * 1.02,
            i,
            f"{m['throughput_variants_per_s']:.0f}/s",
            va="center",
            fontsize=9,
            color=t.ink2,
        )
    ax2.set_yticks(range(len(warm)), [f"batching {m['batching']}" for m in warm[::-1]])
    ax2.tick_params(axis="y", length=0)
    ax2.set_xlim(0, max(m["throughput_variants_per_s"] for m in warm) * 1.25)
    ax2.grid(axis="x")
    ax2.set_title("Variants scored per second", loc="left", fontsize=9.5)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.76, bottom=0.14, wspace=0.26)
    title(
        fig,
        "Serving the exported graph",
        f"ONNX Runtime on CPU; parity with PyTorch within {serving['parity_max_abs_diff']:.1e}.",
        t,
    )
    save(fig, "serving", mode)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    runs = {
        name: load(name)
        for name in (
            "baselines-random",
            "baselines-contiguous",
            "train-random",
            "train-contiguous",
            "shift-random",
            "serving-random",
            "gate-random",
        )
    }
    for mode in THEMES:
        (ASSETS / f"banner-{mode}.svg").write_text(banner(mode))
        print(ASSETS / f"banner-{mode}.svg")
        if runs["train-random"] and runs["baselines-random"]:
            plot_accuracy(runs, mode)
            plot_calibration(runs["train-random"], mode)
        if runs["shift-random"]:
            plot_shift(runs["shift-random"], mode)
        if runs["serving-random"]:
            plot_serving(runs["serving-random"], mode)
        if runs["gate-random"]:
            animate(runs["gate-random"], mode)


if __name__ == "__main__":
    main()
