"""The serving router: scores, intervals, abstention, A/B and shadow, with dynamic batching.

    POST /predict {"protein": "MSK...", "variants": ["A23G", ...], "experiment": "lab-7"}

For each variant it returns the predicted (standardized) effect, a conformal interval at
the model's target coverage, and ``abstain: true`` when the interval is wider than the
abstention threshold, i.e. when a lab should measure the variant rather than trust the
score.

1. **A/B.** A hash of the experiment id sends a fixed share of experiments to the candidate
   model; the same experiment always gets the same model.
2. **Shadow.** In shadow mode every request is also scored by the candidate, off the
   response path; agreement (rank correlation), interval width and abstention rate are
   recorded per request, so the candidate can be judged before any lab sees it.
3. **Dynamic batching.** Concurrent requests are gathered for up to ``max_wait_ms`` or
   ``max_batch`` variants and run through ONNX Runtime in one call.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from prometheus_client import Counter, Histogram
from pydantic import BaseModel

from foldgate.data import MUTANT

LATENCY = Histogram(
    "fg_request_seconds", "Predict latency", ["model"], buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5)
)
BATCH = Histogram("fg_batch_variants", "Variants per ONNX run", buckets=(1, 4, 16, 32, 64, 128, 256))
ABSTAINED = Counter("fg_abstained_total", "Variants the model abstained on", ["model"])
Queue = asyncio.Queue[tuple[list[tuple[str, str]], "asyncio.Future[Any]"]]
SHADOW_AGREEMENT = Histogram(
    "fg_shadow_spearman", "Champion vs candidate rank agreement", buckets=(0, 0.5, 0.7, 0.8, 0.9, 0.95, 1)
)


class PredictRequest(BaseModel):
    """Defined here, not inside ``create_app``: FastAPI resolves annotations in module scope."""

    protein: str
    variants: list[str]
    experiment: str = "default"


@dataclass
class ModelVersion:
    """One deployable version: ONNX graphs plus its calibration."""

    name: str
    regressor: Any  # onnxruntime.InferenceSession
    mlm: Any
    tokenizer: Any
    q: float  # conformal quantile for normalized residuals
    difficulty: Any  # sklearn regressor on embeddings, predicts log |residual|
    abstain_width: float
    zero_shot_cache: dict[str, np.ndarray] = field(default_factory=dict)

    def wt_logits(self, protein: str) -> np.ndarray:
        if protein not in self.zero_shot_cache:
            enc = self.tokenizer(protein, return_tensors="np")
            out = self.mlm.run(
                ["log_probs"],
                {
                    "input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64),
                },
            )
            self.zero_shot_cache[protein] = out[0][0]
        return self.zero_shot_cache[protein]

    def score(self, items: list[tuple[str, str]]) -> list[dict[str, Any]]:
        """``items`` are (protein, mutant) pairs; one ONNX run for all of them."""
        vocab = self.tokenizer.get_vocab()
        seqs, pos, zs = [], [], []
        for protein, mutant in items:
            m = MUTANT.match(mutant)
            if not m or int(m.group(2)) > len(protein) or protein[int(m.group(2)) - 1] != m.group(1):
                raise ValueError(f"{mutant} does not match the protein")
            p = int(m.group(2)) - 1
            logits = self.wt_logits(protein)
            zs.append(logits[p + 1, vocab[m.group(3)]] - logits[p + 1, vocab[m.group(1)]])
            seqs.append(protein[:p] + m.group(3) + protein[p + 1 :])
            pos.append(p)
        enc = self.tokenizer(seqs, return_tensors="np", padding=True)
        score, emb = self.regressor.run(
            ["score", "embedding"],
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
                "position": np.array(pos, dtype=np.int64),
                "zero_shot": np.array(zs, dtype=np.float32),
            },
        )
        BATCH.observe(len(items))
        sigma = np.exp(self.difficulty.predict(emb))
        half = self.q * sigma
        out = []
        for (_, mutant), s, h in zip(items, score, half, strict=True):
            abstain = bool(2 * h > self.abstain_width)
            if abstain:
                ABSTAINED.labels(self.name).inc()
            out.append(
                {
                    "variant": mutant,
                    "score": float(s),
                    "lower": float(s - h),
                    "upper": float(s + h),
                    "abstain": abstain,
                }
            )
        return out


class Batcher:
    """Collect concurrent requests for one model and run them together."""

    def __init__(
        self, model: ModelVersion, max_batch: int = 64, max_wait_ms: float = 8.0, enabled: bool = True
    ) -> None:
        self.model, self.max_batch, self.max_wait = model, max_batch, max_wait_ms / 1000
        self.enabled = enabled
        self.queue: Queue = asyncio.Queue()
        self.task: asyncio.Task[None] | None = None
        self.loop: asyncio.AbstractEventLoop | None = None

    async def submit(self, items: list[tuple[str, str]]) -> list[dict[str, Any]]:
        if not self.enabled:
            return await asyncio.to_thread(self.model.score, items)
        loop = asyncio.get_running_loop()
        if self.task is None or self.loop is not loop:
            # A queue and its collector belong to one event loop. Serving uses a single
            # loop; rebinding here keeps the batcher usable when it is driven from another.
            self.loop, self.queue = loop, asyncio.Queue()
            self.task = loop.create_task(self._run())
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        await self.queue.put((items, fut))
        result: list[dict[str, Any]] = await fut
        return result

    async def _run(self) -> None:
        while True:
            first = await self.queue.get()
            batch, deadline = [first], time.monotonic() + self.max_wait
            size = len(first[0])
            while size < self.max_batch and (left := deadline - time.monotonic()) > 0:
                try:
                    nxt = await asyncio.wait_for(self.queue.get(), timeout=left)
                except TimeoutError:
                    break
                batch.append(nxt)
                size += len(nxt[0])
            flat = [it for items, _ in batch for it in items]
            try:
                scored = await asyncio.to_thread(self.model.score, flat)
                i = 0
                for items, fut in batch:
                    fut.set_result(scored[i : i + len(items)])
                    i += len(items)
            except Exception:
                # One client's bad variant must not fail everyone sharing its batch, so
                # the batch is retried a request at a time and only the bad one raises.
                for items, fut in batch:
                    if fut.done():
                        continue
                    try:
                        fut.set_result(await asyncio.to_thread(self.model.score, items))
                    except Exception as exc:
                        fut.set_exception(exc)


def ab_arm(experiment: str, candidate_share: float) -> str:
    h = int(hashlib.sha256(experiment.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "candidate" if h < candidate_share else "champion"


def create_app(
    champion: ModelVersion,
    candidate: ModelVersion | None = None,
    candidate_share: float = 0.0,
    shadow: bool = False,
    batching: bool = True,
    shadow_log: Path | None = None,
    max_batch: int = 64,
    max_wait_ms: float = 8.0,
) -> Any:
    from fastapi import FastAPI, HTTPException, Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    app = FastAPI(title="FoldGate")
    background: set[asyncio.Task[None]] = set()

    def batcher(model: ModelVersion) -> Batcher:
        return Batcher(model, max_batch=max_batch, max_wait_ms=max_wait_ms, enabled=batching)

    batchers = {"champion": batcher(champion)}
    if candidate:
        batchers["candidate"] = batcher(candidate)

    async def shadow_score(req: PredictRequest, primary: list[dict[str, Any]]) -> None:
        from scipy.stats import spearmanr

        other = await batchers["candidate"].submit([(req.protein, v) for v in req.variants])
        a = [r["score"] for r in primary]
        b = [r["score"] for r in other]
        rho = float(spearmanr(a, b).statistic) if len(a) > 2 else float("nan")
        if rho == rho:
            SHADOW_AGREEMENT.observe(rho)
        if shadow_log:
            with shadow_log.open("a") as fh:
                record = {
                    "experiment": req.experiment,
                    "n": len(a),
                    "spearman": rho,
                    "abstain_champion": float(np.mean([r["abstain"] for r in primary])),
                    "abstain_candidate": float(np.mean([r["abstain"] for r in other])),
                }
                fh.write(json.dumps(record) + "\n")

    @app.post("/predict")
    async def predict(req: PredictRequest) -> dict[str, Any]:
        arm = ab_arm(req.experiment, candidate_share) if candidate else "champion"
        started = time.perf_counter()
        try:
            results = await batchers[arm].submit([(req.protein, v) for v in req.variants])
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        LATENCY.labels(arm).observe(time.perf_counter() - started)
        if shadow and candidate and arm == "champion":
            task = asyncio.create_task(shadow_score(req, results))
            background.add(task)
            task.add_done_callback(background.discard)
        model = batchers[arm].model
        return {"model": model.name, "arm": arm, "results": results}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "champion": champion.name, "candidate": candidate.name if candidate else ""}

    return app
