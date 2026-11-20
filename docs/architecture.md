# How FoldGate is put together

The question the system answers is not "what is this variant's effect" but "may a lab act
on this number". Everything below exists to make that answer defensible.

## The path a variant takes

```
ProteinGym DMS assays (parquet, Hugging Face)
        |
        |  data.py       select 12 assays, reconstruct each wild type, standardize scores
        |  prepare.py    cache the zero-shot wild-type marginal per variant
        v
  train / calib / test split, per assay, random or contiguous
        |
        |  model.py      ESM-2 35M + LoRA on query and value, head on the mutated position
        |  train.py      Lightning, MPS or CUDA, early stopping on a slice of train
        v
   predictions + embeddings on calib and test
        |
        |  calibration.py  split conformal, normalized conformal, difficulty model,
        |                  selective prediction, MMD drift, the release gate
        v
   a calibrated version: ONNX graphs + quantile + difficulty model + abstention width
        |
        |  export.py     merge LoRA, export regressor and masked LM, parity check
        |  registry.py   MLflow run, metrics, gate verdict, champion alias
        v
   router.py             /predict with intervals, abstention, A/B, shadow, micro-batching
        |
        |  deploy/       Helm chart, KServe InferenceService
        |  infra/        SageMaker endpoint, alarms, budget (validated, never applied)
        v
   Prometheus: latency, batch size, abstention rate, shadow agreement
```

## The split is the experiment

1. **random** shuffles variants: the model has usually seen other mutations at the same
   position. This is the easy setting, and a one-hot ridge does well in it.
2. **contiguous** holds out whole stretches of positions. Nothing in training or
   calibration touches a test position, which is what happens when a lab scans a new
   region. One-hot features collapse here; the language model is what carries over.
3. **held-out families** (alpha-synuclein, TPMT, UBC9) are never trained on at all. They
   are the shift test, where the conformal guarantee is supposed to break, and the point is
   to catch it without labels and then repair it with a few.

## Why two conformal flavours

Split conformal gives one interval width for every variant. It is honest on average and
wrong in detail: it over-covers the variants the model finds easy and under-covers the hard
ones, which is exactly backwards for a lab deciding what to measure.

Normalized conformal divides residuals by a predicted difficulty, fitted on half the
calibration split from the mutated position's embedding, and takes the quantile on the
other half. The guarantee is the same, but the width now varies with the model's own
uncertainty, so abstention means something: the interval is wide because this variant looks
unlike what the model learned from.

## The gate

A candidate is promoted only when all three hold:

1. it ranks better than the zero-shot score the backbone gives for free;
2. its intervals cover within 2 points of their target on the test split;
3. its p95 latency is inside the serving SLO.

Rejected candidates are logged with their reasons and tagged `rejected` in the registry,
never aliased `champion`. The gate run deliberately includes a candidate whose quantile was
taken at 70% and shipped as 90%, because narrow intervals are the easiest way to look good.

## Serving

The router loads whatever the champion alias points at: two ONNX graphs (the fine-tuned
regressor with LoRA merged, and the base masked language model for the zero-shot term of a
protein it has not seen), the conformal quantile, the difficulty model and the abstention
width.

1. **Micro-batching.** Concurrent requests are collected for up to 8 ms or 64 variants and
   run through ONNX Runtime in one call. A failing batch is retried one request at a time,
   so a single bad variant cannot fail its neighbours.
2. **Zero-shot cache.** The masked language model runs once per protein, not per variant.
3. **A/B.** A hash of the experiment id picks the arm, so one experiment always sees one
   model.
4. **Shadow.** The candidate scores the same request off the response path; rank agreement,
   interval width and abstention rate are recorded per request.

## What runs where

| Piece | Ran on | Status |
| --- | --- | --- |
| Training, calibration, shift test | Apple M-series GPU (MPS), ESM-2 35M | ran, results in the README |
| ONNX export and parity | CPU, ONNX Runtime | ran, in CI |
| Router load test | in-process ASGI, CPU | ran, numbers in the README |
| Helm chart, KServe manifest | rendered and schema checked in CI | not applied to a cluster |
| SageMaker endpoint, alarms, budget | `terraform validate`, `tflint`, `checkov` in CI | never applied, no AWS account used |
