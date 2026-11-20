"""Calibration is the project's central claim, so it is tested on data with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from foldgate.calibration import (
    conformal_quantile,
    fit_difficulty,
    gate,
    mmd,
    ndcg_top,
    normalized_conformal,
    selective_top_k,
    split_conformal,
)


def heteroscedastic(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predictions whose error grows with one feature: what normalization is for."""
    rng = np.random.default_rng(seed)
    x = rng.random((n, 3))
    noise = 0.2 + 2.5 * x[:, 0]
    pred = rng.normal(0, 1, n)
    y = pred + rng.normal(0, noise)
    return pred, y, x


def test_conformal_quantile_is_conservative_for_small_samples() -> None:
    # With n = 10 the corrected level is above 1, so the largest residual is the quantile.
    assert conformal_quantile(np.arange(1, 11, dtype=float), 0.1) == 10.0
    # With n = 100 it is ceil(101 * 0.9) / 100 = 0.91, rounded up to a residual that exists.
    assert conformal_quantile(np.arange(1, 101, dtype=float), 0.1) == 92.0
    assert conformal_quantile(np.arange(1, 101, dtype=float), 0.2) == 82.0


def test_split_conformal_covers_at_the_target_rate() -> None:
    pred, y, _ = heteroscedastic(4000)
    cal, test = slice(0, 2000), slice(2000, 4000)
    for alpha in (0.2, 0.1, 0.05):
        iv = split_conformal(pred[cal], y[cal], pred[test], alpha)
        assert abs(iv.coverage(y[test]) - (1 - alpha)) < 0.03
        assert float(np.ptp(iv.width)) < 1e-9  # one width for every variant


def test_normalized_conformal_covers_and_varies_its_width() -> None:
    pred, y, x = heteroscedastic(4000)
    cal, test = slice(0, 2000), slice(2000, 4000)
    iv = normalized_conformal(pred[cal], y[cal], x[cal], pred[test], x[test], 0.1)
    assert abs(iv.coverage(y[test]) - 0.9) < 0.04
    assert iv.width.std() > 0.1


def test_normalization_evens_out_coverage_across_difficulty() -> None:
    """A single width over-covers easy variants and under-covers hard ones."""
    pred, y, x = heteroscedastic(6000)
    cal, test = slice(0, 3000), slice(3000, 6000)
    plain = split_conformal(pred[cal], y[cal], pred[test], 0.1)
    norm = normalized_conformal(pred[cal], y[cal], x[cal], pred[test], x[test], 0.1)
    easy = x[test][:, 0] < np.quantile(x[test][:, 0], 1 / 3)
    hard = x[test][:, 0] > np.quantile(x[test][:, 0], 2 / 3)

    def cover(iv: object, mask: np.ndarray) -> float:
        lower, upper = iv.lower[mask], iv.upper[mask]  # type: ignore[attr-defined]
        return float(np.mean((y[test][mask] >= lower) & (y[test][mask] <= upper)))

    plain_gap = abs(cover(plain, easy) - cover(plain, hard))
    norm_gap = abs(cover(norm, easy) - cover(norm, hard))
    assert plain_gap > 0.1
    assert norm_gap < plain_gap / 2


def test_difficulty_model_tracks_the_error() -> None:
    pred, y, x = heteroscedastic(3000)
    model = fit_difficulty(x[:1500], y[:1500] - pred[:1500])
    sigma = np.exp(model.predict(x[1500:]))
    residual = np.abs(y[1500:] - pred[1500:])
    assert np.corrcoef(sigma, residual)[0, 1] > 0.3


def test_ndcg_is_one_for_a_perfect_ranking_and_lower_for_noise() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    assert ndcg_top(y, y) == pytest.approx(1.0)
    assert ndcg_top(rng.normal(size=500), y) < 0.9
    assert 0 <= ndcg_top(-y, y) <= 1


def test_selective_prediction_helps_when_width_tracks_error() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(size=2000)
    width = rng.random(2000) * 4
    pred = y + rng.normal(0, 0.1 + width)  # wide intervals, noisier predictions
    out = selective_top_k(pred, width, y, k=20)
    assert out["ndcg_confident_only"] > out["ndcg_by_score"]
    assert 0 <= out["hits_by_score"] <= 1


def test_mmd_separates_a_shift_from_a_resample() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 8))
    same = rng.normal(size=(500, 8))
    shifted = rng.normal(loc=0.8, size=(500, 8))
    stat_same, p_same = mmd(a, same, permutations=100)
    stat_shift, p_shift = mmd(a, shifted, permutations=100)
    assert p_same > 0.05
    assert p_shift < 0.05
    assert stat_shift > stat_same


def test_gate_passes_an_honest_candidate() -> None:
    verdict = gate(0.62, 0.30, 0.902, 0.90, 120.0, 250.0)
    assert verdict.passed
    assert verdict.reasons == []


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((0.28, 0.30, 0.901, 0.90, 120.0, 250.0), "zero-shot"),
        ((0.62, 0.30, 0.74, 0.90, 120.0, 250.0), "coverage"),
        ((0.62, 0.30, 0.901, 0.90, 800.0, 250.0), "latency"),
    ],
)
def test_gate_rejects_and_says_why(args: tuple[float, ...], expected: str) -> None:
    verdict = gate(*args)
    assert not verdict.passed
    assert any(expected in r for r in verdict.reasons)


def test_gate_rejects_intervals_that_are_too_narrow_to_be_honest() -> None:
    """The failure mode the gate exists for: a 70% quantile shipped as a 90% interval."""
    pred, y, x = heteroscedastic(4000)
    cal, test = slice(0, 2000), slice(2000, 4000)
    narrow = normalized_conformal(pred[cal], y[cal], x[cal], pred[test], x[test], 0.30)
    coverage = narrow.coverage(y[test])
    assert coverage < 0.85
    assert not gate(0.62, 0.30, coverage, 0.90, 120.0, 250.0).passed
