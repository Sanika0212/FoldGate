"""Intervals a lab can trust, abstention, drift, and the release gate.

1. **Split conformal.** The (1 - alpha) quantile of absolute residuals on the calibration
   split, with the finite-sample correction, gives intervals ``prediction +/- q`` that cover
   the true score with probability at least 1 - alpha when test variants are exchangeable
   with calibration ones.
2. **Normalized conformal.** Same guarantee, but residuals are divided by a predicted
   difficulty ``sigma(x)``, so intervals are wide where the model is unsure and narrow
   where it is not. ``sigma`` is fitted on one half of the calibration split and the
   quantile taken on the other half.
3. **Selective prediction.** Abstain on the widest intervals; measure what that is worth to
   a lab choosing the top variants (NDCG of the chosen set).
4. **Drift.** Maximum mean discrepancy between embeddings of the variants the model was
   calibrated on and new ones, with a permutation p-value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import spearmanr


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


@dataclass
class Intervals:
    lower: np.ndarray
    upper: np.ndarray

    @property
    def width(self) -> np.ndarray:
        return np.asarray(self.upper - self.lower)

    def coverage(self, y: np.ndarray) -> float:
        return float(np.mean((y >= self.lower) & (y <= self.upper)))


def split_conformal(cal_pred: np.ndarray, cal_y: np.ndarray, pred: np.ndarray, alpha: float) -> Intervals:
    q = conformal_quantile(np.abs(cal_y - cal_pred), alpha)
    return Intervals(pred - q, pred + q)


def fit_difficulty(features: np.ndarray, residuals: np.ndarray) -> Any:
    """Predicts log |residual| from an embedding: the sigma(x) of normalized conformal."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0).fit(
        features, np.log(np.abs(residuals) + 1e-3)
    )


def normalized_conformal(
    cal_pred: np.ndarray,
    cal_y: np.ndarray,
    cal_x: np.ndarray,
    pred: np.ndarray,
    x: np.ndarray,
    alpha: float,
    seed: int = 0,
) -> Intervals:
    rng = np.random.default_rng(seed)
    half = rng.random(len(cal_y)) < 0.5
    model = fit_difficulty(cal_x[half], cal_y[half] - cal_pred[half])
    sigma_cal = np.exp(model.predict(cal_x[~half]))
    q = conformal_quantile(np.abs(cal_y[~half] - cal_pred[~half]) / sigma_cal, alpha)
    sigma = np.exp(model.predict(x))
    return Intervals(pred - q * sigma, pred + q * sigma)


def spearman(pred: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(pred, y).statistic)


def ndcg_top(pred: np.ndarray, y: np.ndarray, frac: float = 0.1) -> float:
    """NDCG of the top ``frac`` variants by prediction; gains are true scores shifted to >= 0."""
    k = max(1, round(len(y) * frac))
    gains = y - y.min()
    discounts = 1 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains[np.argsort(-pred)[:k]] * discounts))
    ideal = float(np.sum(np.sort(gains)[::-1][:k] * discounts))
    return dcg / ideal if ideal > 0 else 0.0


def selective_top_k(
    pred: np.ndarray, width: np.ndarray, y: np.ndarray, k: int, keep: float = 0.5
) -> dict[str, float]:
    """A lab can test k variants. Pick by score alone, or by score among the most certain half."""
    gains = y - y.min()
    discounts = 1 / np.log2(np.arange(2, k + 2))
    ideal = float(np.sum(np.sort(gains)[::-1][:k] * discounts))
    by_score = np.argsort(-pred)[:k]
    confident = np.where(width <= np.quantile(width, keep))[0]
    by_confident = confident[np.argsort(-pred[confident])[:k]]

    def ndcg(idx: np.ndarray) -> float:
        return float(np.sum(gains[idx] * discounts[: len(idx)])) / ideal if ideal > 0 else 0.0

    top_true = set(np.argsort(-y)[: max(k, int(len(y) * 0.1))])
    return {
        "ndcg_by_score": ndcg(by_score),
        "ndcg_confident_only": ndcg(by_confident),
        "hits_by_score": float(np.mean([i in top_true for i in by_score])),
        "hits_confident_only": float(np.mean([i in top_true for i in by_confident])),
    }


def mmd(
    a: np.ndarray, b: np.ndarray, permutations: int = 200, seed: int = 0, max_n: int = 600
) -> tuple[float, float]:
    """RBF-kernel MMD^2 between two embedding sets and a permutation p-value."""
    rng = np.random.default_rng(seed)
    a = a[rng.choice(len(a), min(len(a), max_n), replace=False)]
    b = b[rng.choice(len(b), min(len(b), max_n), replace=False)]
    z = np.vstack([a, b])
    d2 = np.sum(z**2, 1)[:, None] + np.sum(z**2, 1)[None, :] - 2 * z @ z.T
    gamma = 1.0 / np.median(d2[d2 > 0])
    k = np.exp(-gamma * d2)
    n = len(a)

    def stat(idx: np.ndarray) -> float:
        x, y = idx[:n], idx[n:]
        return float(k[np.ix_(x, x)].mean() + k[np.ix_(y, y)].mean() - 2 * k[np.ix_(x, y)].mean())

    observed = stat(np.arange(len(z)))
    null = [stat(rng.permutation(len(z))) for _ in range(permutations)]
    p = (1 + sum(v >= observed for v in null)) / (1 + permutations)
    return observed, float(p)


@dataclass
class GateResult:
    passed: bool
    reasons: list[str]


def gate(
    spearman_model: float,
    spearman_zero_shot: float,
    coverage: float,
    target: float,
    p95_ms: float,
    slo_ms: float,
    tolerance: float = 0.02,
) -> GateResult:
    """Promote only a candidate that is more accurate than zero-shot, honest, and fast."""
    reasons = []
    if spearman_model <= spearman_zero_shot:
        reasons.append(f"Spearman {spearman_model:.3f} does not beat zero-shot {spearman_zero_shot:.3f}")
    if abs(coverage - target) > tolerance:
        reasons.append(f"coverage {coverage:.3f} is outside {target:.2f} +/- {tolerance:.2f}")
    if p95_ms > slo_ms:
        reasons.append(f"p95 latency {p95_ms:.0f} ms exceeds {slo_ms:.0f} ms")
    return GateResult(not reasons, reasons)
