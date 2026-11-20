"""The router without a model: batching, A/B, shadow, abstention and the metrics endpoint.

A stub stands in for the ONNX sessions so these tests stay fast, and it records every batch
it is asked to score, which is how the dynamic batcher is checked.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from foldgate.router import Batcher, ab_arm, create_app


class StubModel:
    """Scores (protein, mutant) pairs, remembering the size of each batch it was given."""

    def __init__(self, name: str = "stub", offset: float = 0.0, wide: bool = False) -> None:
        self.name = name
        self.offset = offset
        self.wide = wide
        self.batches: list[int] = []

    def score(self, items: list[tuple[str, str]]) -> list[dict[str, Any]]:
        self.batches.append(len(items))
        out = []
        for protein, mutant in items:
            position = int("".join(c for c in mutant if c.isdigit()))
            if position > len(protein):
                raise ValueError(f"{mutant} does not match the protein")
            half = 2.0 if self.wide else 0.1
            score = position / 10 + self.offset
            out.append(
                {
                    "variant": mutant,
                    "score": score,
                    "lower": score - half,
                    "upper": score + half,
                    "abstain": self.wide,
                }
            )
        return out


PROTEIN = "MKAILVVLLYTFATANADTLCIGYHANNSTDTVDTVLEKNVTVTHSVNLLEDKHNGKLCKL"
VARIANTS = [f"{PROTEIN[i]}{i + 1}A" for i in range(10)]


async def post(app: object, variants: list[str], experiment: str = "lab-1") -> dict[str, Any]:
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://router") as client:
        r = await client.post(
            "/predict", json={"protein": PROTEIN, "variants": variants, "experiment": experiment}
        )
        r.raise_for_status()
        body: dict[str, Any] = r.json()
        return body


def test_predict_returns_an_interval_per_variant() -> None:
    model = StubModel()
    body = asyncio.run(post(create_app(model), VARIANTS))  # type: ignore[arg-type]
    assert body["model"] == "stub"
    assert body["arm"] == "champion"
    assert [r["variant"] for r in body["results"]] == VARIANTS
    assert all(r["lower"] < r["score"] < r["upper"] for r in body["results"])


def test_abstention_is_reported_per_variant() -> None:
    body = asyncio.run(post(create_app(StubModel(wide=True)), VARIANTS))  # type: ignore[arg-type]
    assert all(r["abstain"] for r in body["results"])


def test_a_variant_that_does_not_match_the_protein_is_a_422() -> None:
    model = StubModel()
    app = create_app(model)  # type: ignore[arg-type]

    async def go() -> int:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://router") as client:
            r = await client.post("/predict", json={"protein": PROTEIN, "variants": ["A9999G"]})
            return r.status_code

    assert asyncio.run(go()) == 422


def test_batching_merges_concurrent_requests_into_one_call() -> None:
    model = StubModel()
    app = create_app(model, batching=True)  # type: ignore[arg-type]

    async def go() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://router") as client:
            await asyncio.gather(
                *[
                    client.post(
                        "/predict",
                        json={"protein": PROTEIN, "variants": VARIANTS[:2], "experiment": f"lab-{i}"},
                    )
                    for i in range(8)
                ]
            )

    asyncio.run(go())
    assert sum(model.batches) == 16
    assert max(model.batches) > 2  # at least one call carried several requests
    assert len(model.batches) < 8


def test_batching_off_scores_each_request_on_its_own() -> None:
    model = StubModel()
    app = create_app(model, batching=False)  # type: ignore[arg-type]

    async def go() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://router") as client:
            await asyncio.gather(
                *[
                    client.post("/predict", json={"protein": PROTEIN, "variants": VARIANTS[:2]})
                    for _ in range(8)
                ]
            )

    asyncio.run(go())
    assert model.batches == [2] * 8


def test_a_failing_batch_does_not_hang_the_other_requests() -> None:
    model = StubModel()
    app = create_app(model, batching=True)  # type: ignore[arg-type]

    async def go() -> list[int]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://router") as client:
            good = [
                client.post("/predict", json={"protein": PROTEIN, "variants": VARIANTS[:2]}) for _ in range(4)
            ]
            bad = [client.post("/predict", json={"protein": PROTEIN, "variants": ["A9999G"]})]
            return [r.status_code for r in await asyncio.gather(*good, *bad)]

    codes = asyncio.run(go())
    assert codes.count(422) == 1
    assert all(c in (200, 422) for c in codes)


def test_the_batcher_respects_its_maximum_batch() -> None:
    model = StubModel()
    batcher = Batcher(model, max_batch=4, max_wait_ms=50)  # type: ignore[arg-type]

    async def go() -> None:
        await asyncio.gather(*[batcher.submit([(PROTEIN, v) for v in VARIANTS[:3]]) for _ in range(6)])

    asyncio.run(go())
    assert max(model.batches) <= 6  # a batch stops collecting once it is over the maximum
    assert sum(model.batches) == 18


def test_ab_split_is_stable_per_experiment_and_roughly_the_right_size() -> None:
    experiments = [f"lab-{i}" for i in range(2000)]
    arms = [ab_arm(e, 0.2) for e in experiments]
    assert arms == [ab_arm(e, 0.2) for e in experiments]
    share = arms.count("candidate") / len(arms)
    assert 0.17 < share < 0.23
    assert all(ab_arm(e, 0.0) == "champion" for e in experiments[:50])
    assert all(ab_arm(e, 1.0) == "candidate" for e in experiments[:50])


def test_ab_routes_some_experiments_to_the_candidate() -> None:
    champion, candidate = StubModel("champion"), StubModel("candidate", offset=5.0)
    app = create_app(champion, candidate, candidate_share=0.5)  # type: ignore[arg-type]
    arms = {asyncio.run(post(app, VARIANTS[:2], f"lab-{i}"))["model"] for i in range(20)}
    assert arms == {"champion", "candidate"}


def test_shadow_scores_the_candidate_off_the_response_path(tmp_path: Path) -> None:
    champion, candidate = StubModel("champion"), StubModel("candidate", offset=5.0)
    log = tmp_path / "shadow.jsonl"
    app = create_app(champion, candidate, shadow=True, shadow_log=log)  # type: ignore[arg-type]

    async def go() -> dict[str, Any]:
        body = await post(app, VARIANTS, "lab-7")
        for _ in range(100):  # the shadow call runs as a background task
            if candidate.batches and log.exists():
                break
            await asyncio.sleep(0.05)
        return body

    body = asyncio.run(go())
    assert body["model"] == "champion"  # the candidate never answers in shadow mode
    assert candidate.batches  # but it did score the same variants
    record = json.loads(log.read_text().splitlines()[0])
    assert record["experiment"] == "lab-7"
    assert record["n"] == len(VARIANTS)
    assert record["spearman"] == pytest.approx(1.0)  # a constant offset keeps the ranking


def test_healthz_and_metrics() -> None:
    champion, candidate = StubModel("champion"), StubModel("candidate")
    app = create_app(champion, candidate)  # type: ignore[arg-type]

    async def go() -> tuple[dict[str, Any], str]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://router") as client:
            await client.post("/predict", json={"protein": PROTEIN, "variants": VARIANTS[:2]})
            health = await client.get("/healthz")
            metrics = await client.get("/metrics")
            return health.json(), metrics.text

    health, metrics = asyncio.run(go())
    assert health == {"status": "ok", "champion": "champion", "candidate": "candidate"}
    assert "fg_request_seconds" in metrics
    assert "fg_batch_variants" in metrics
