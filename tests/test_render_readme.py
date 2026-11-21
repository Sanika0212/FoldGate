"""The README's numbers come from the results files, so the renderer has to survive them.

These run against synthetic payloads shaped exactly like the evaluation scripts write, so a
section that would raise on real results raises here first, in two seconds, rather than
after an hour of training.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_readme.py"


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location("render_readme", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_payload() -> dict[str, Any]:
    return {
        "scheme": "random",
        "trainable_share": 0.0073,
        "n": {"train": 13296, "calib": 4212, "test": 4541},
        "per_assay": [
            {"dms_id": "A", "n": 1200, "spearman": 0.61, "ndcg_top10": 0.72},
            {"dms_id": "B", "n": 900, "spearman": 0.48, "ndcg_top10": 0.65},
        ],
        "mean_spearman": 0.545,
        "mean_ndcg_top10": 0.685,
        "baseline_mean_spearman": {"zero_shot": 0.41, "ridge_one_hot": 0.37},
        "coverage": {
            "0.95": {
                "split": {"coverage": 0.952, "width": 2.6, "terciles": [0.97, 0.95, 0.93]},
                "normalized": {"coverage": 0.949, "width": 2.4, "terciles": [0.95, 0.95, 0.94]},
            },
            "0.90": {
                "split": {"coverage": 0.903, "width": 2.0, "terciles": [0.96, 0.90, 0.84]},
                "normalized": {"coverage": 0.898, "width": 1.9, "terciles": [0.91, 0.90, 0.89]},
            },
            "0.80": {
                "split": {"coverage": 0.804, "width": 1.4, "terciles": [0.88, 0.80, 0.73]},
                "normalized": {"coverage": 0.799, "width": 1.3, "terciles": [0.81, 0.80, 0.79]},
            },
        },
    }


def shift_payload() -> dict[str, Any]:
    assay = {
        "n": 500,
        "spearman_model": 0.33,
        "spearman_zero_shot": 0.30,
        "coverage_plain": 0.71,
        "coverage_normalized": 0.74,
        "mmd": 0.08,
        "mmd_p_value": 0.001,
        "recalibrated": {"25": {"coverage": 0.85}, "200": {"coverage": 0.90}},
    }
    return {"scheme": "random", "alpha": 0.1, "label_budgets": [25, 200], "assays": {"SYN": assay}}


def gate_payload() -> dict[str, Any]:
    return {
        "scheme": "random",
        "slo_ms": 3000.0,
        "target_coverage": 0.9,
        "candidates": {
            "trained": {
                "metrics": {"spearman": 0.55, "coverage": 0.90, "p95_ms": 800.0},
                "gate": {"passed": True, "reasons": []},
                "registry": {"version": "1"},
            },
            "narrow intervals": {
                "metrics": {"spearman": 0.55, "coverage": 0.71, "p95_ms": 790.0},
                "gate": {"passed": False, "reasons": ["coverage 0.710 is outside 0.90 +/- 0.02"]},
                "registry": {"version": "2"},
            },
        },
    }


def serving_payload(on_p95: float = 625.0, on_throughput: float = 200.0) -> dict[str, Any]:
    def row(batching: str, cache: str, p95: float, throughput: float) -> dict[str, Any]:
        return {
            "batching": batching,
            "cache": cache,
            "requests": 64,
            "median_ms": p95 / 2,
            "p95_ms": p95,
            "throughput_variants_per_s": throughput,
        }

    return {
        "parity_max_abs_diff": 3.2e-6,
        "parity_tolerance": 1e-3,
        "onnx_bytes": {"regressor.onnx": 1},
        "load": {"clients": 8, "variants_per_request": 8, "requests_per_client": 8},
        "measurements": [
            row("on", "warm", on_p95, on_throughput),
            row("off", "warm", 900.0, 140.0),
            row("on", "cold", on_p95 * 2, on_throughput / 2),
        ],
        "warm": {
            "on": {
                "repeats": 3,
                "p95_ms_median": on_p95,
                "p95_ms_min": on_p95 * 0.9,
                "p95_ms_max": on_p95 * 1.2,
                "median_ms_median": on_p95 / 2,
                "throughput_variants_per_s_median": on_throughput,
            },
            "off": {
                "repeats": 3,
                "p95_ms_median": 900.0,
                "p95_ms_min": 850.0,
                "p95_ms_max": 980.0,
                "median_ms_median": 450.0,
                "throughput_variants_per_s_median": 140.0,
            },
        },
        "batching_p95_ratio": 900.0 / on_p95,
        "ab": {"arms": {"champion": 50, "candidate": 14}, "expected_candidate_share": 0.2},
        "shadow": {"median_ms": 430.0, "comparisons": 64},
    }


@pytest.mark.parametrize(
    ("marker", "payload"),
    [
        ("ACCURACY", train_payload()),
        ("CALIBRATION", train_payload()),
        ("SHIFT", shift_payload()),
        ("GATE", gate_payload()),
        ("SERVING", serving_payload()),
    ],
)
def test_every_section_renders_from_a_real_shaped_payload(marker: str, payload: dict[str, Any]) -> None:
    body = load_module().RENDER[marker](payload)
    assert body.startswith("## ")
    assert "{" not in body  # no unformatted braces left behind
    assert len(body.splitlines()) > 4


def test_accuracy_reports_a_baseline_it_does_not_beat() -> None:
    """A model that loses to its baseline has to say so, not omit the comparison."""
    module = load_module()
    payload = train_payload()
    payload["mean_spearman"] = 0.30  # worse than zero_shot at 0.41
    body = module.RENDER["ACCURACY"](payload)
    assert "is not beaten here" in body
    assert "-0.110" in body
    # The gate's accuracy criterion is zero-shot, not the best baseline, so a model in this
    # position is still promoted. The section has to admit that rather than imply a rejection.
    assert "The release gate does not catch this" in body
    assert "reject" in body


def test_the_gate_section_gives_a_reason_for_every_rejection() -> None:
    body = load_module().RENDER["GATE"](gate_payload())
    assert "rejected" in body
    assert "coverage 0.710 is outside" in body


def test_calibration_reports_the_largest_miss_not_the_smallest() -> None:
    """This line took a min() and called it the largest, over one scheme of the two."""
    module = load_module()
    payload = train_payload()
    payload["coverage"] = {
        "0.90": {
            "split": {"coverage": 0.880, "width": 2.0, "terciles": [0.9, 0.88, 0.86]},
            "normalized": {"coverage": 0.860, "width": 2.1, "terciles": [0.84, 0.86, 0.88]},
        },
        "0.95": {
            "split": {"coverage": 0.948, "width": 3.0, "terciles": [0.96, 0.95, 0.94]},
            "normalized": {"coverage": 0.951, "width": 3.1, "terciles": [0.94, 0.95, 0.96]},
        },
    }
    body = module.RENDER["CALIBRATION"](payload)
    # 0.90 normalized misses by 0.040; the smallest miss is 0.001 and the largest split-only
    # miss is 0.020. Only the first is the largest gap.
    assert "largest gap between target and achieved coverage is 0.040" in body
    assert "normalized scheme at the 90% target" in body


def test_serving_reports_batching_as_a_gain_when_it_is_one() -> None:
    body = load_module().RENDER["SERVING"](serving_payload())
    assert "divides p95 by 1.44" in body
    assert "slower" not in body


def test_serving_shows_the_spread_of_the_repeats() -> None:
    """One draw of this load test varied by more than 2x, so the range has to be visible."""
    body = load_module().RENDER["SERVING"](serving_payload())
    assert "of 3 warm runs" in body
    assert "562 to 750" in body  # the on-batching p95 spread
    assert "850 to 980" in body


def test_serving_calls_batching_a_slowdown_when_that_is_what_was_measured() -> None:
    """Measured on this machine: batching cost 3.9x at p95, because the CPU was already busy."""
    body = load_module().RENDER["SERVING"](serving_payload(on_p95=3500.0, on_throughput=35.0))
    assert "3.9 times slower at p95" in body
    assert "4.0 times slower on throughput" in body
    assert "divides p95" not in body


def test_a_placeholder_is_replaced_and_then_re_rendered_in_place() -> None:
    """Running twice must update the block, not stack a second copy."""
    module = load_module()
    text = "intro\n\nGATE_SECTION\n\noutro\n"
    once = module.replace(text, "GATE", "## The release gate\n\nfirst")
    twice = module.replace(once, "GATE", "## The release gate\n\nsecond")
    assert once.count("<!-- GATE:start -->") == 1
    assert twice.count("<!-- GATE:start -->") == 1
    assert "second" in twice
    assert "first" not in twice
    assert twice.startswith("intro")
    assert twice.rstrip().endswith("outro")


def test_a_missing_results_file_is_named_rather_than_guessed(tmp_path: Path) -> None:
    module = load_module()
    assert module.results("train", "nonexistent-scheme") is None
    written = tmp_path / "gate-random" / "results.json"
    written.parent.mkdir(parents=True)
    written.write_text(json.dumps(gate_payload()))
    module.RUNS = tmp_path
    assert module.results("gate", "random") is not None
