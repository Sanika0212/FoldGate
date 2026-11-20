# Runbook

## Reproduce the results

```bash
uv sync --all-extras
uv run foldgate download                       # ProteinGym substitution shards
uv run foldgate zero-shot                      # cache the wild-type marginal per variant

uv run python eval/run_baselines.py --scheme random
uv run python eval/run_baselines.py --scheme contiguous
uv run python eval/run_train.py --scheme random
uv run python eval/run_train.py --scheme contiguous
uv run python eval/run_shift.py --scheme random
uv run python eval/run_serving.py --scheme random
uv run python eval/run_gate.py --scheme random
```

Each script writes `runs/<name>/results.json`. Training also writes
`artifacts/<scheme>/` with the checkpoint, the saved predictions and embeddings, the
calibration, and (after the serving run) the exported ONNX graphs. On an M-series Mac the
two training runs take about 25 minutes each with the 35M backbone.

The zero-shot cache in `data/zero_shot.parquet` is committed, so the baselines can be
reproduced without downloading the weights.

## Serve a version

```bash
uv run foldgate serve --scheme random --port 8080
curl -s localhost:8080/predict -H 'content-type: application/json' -d '{
  "protein": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
  "variants": ["K6A", "L8P", "R72A"],
  "experiment": "lab-7"
}' | jq
curl -s localhost:8080/metrics | grep fg_
```

`abstain: true` means the interval is wider than the version's abstention width: the model
is saying measure this one rather than trust the score.

In a cluster the same command is driven by the environment instead of flags:
`FOLDGATE_ARTIFACTS` points at the mounted version, `FOLDGATE_CANDIDATE_ARTIFACTS` and
`FOLDGATE_CANDIDATE_SHARE` turn on the A/B split, `FOLDGATE_SHADOW` turns on shadow
scoring, and `FOLDGATE_MAX_BATCH` with `FOLDGATE_MAX_WAIT_MS` tune the micro-batcher.

## Promote or reject a candidate

```bash
uv run python eval/run_gate.py --scheme random          # logs every candidate to MLflow
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db
```

The gate writes the verdict and its reasons next to each version. Only a passing candidate
gets the `champion` alias, which is what the router loads.

## When something is wrong

| Symptom | What it means | What to do |
| --- | --- | --- |
| Abstention rate climbs above a fifth of variants | Inputs are drifting away from the calibration set | Run `eval/run_shift.py` on the new proteins; if MMD confirms the shift, recalibrate with labels |
| Coverage on a new assay is far below target | The new protein is not exchangeable with calibration | Label 100 to 200 variants of it and recompute the quantile; the shift results show what that recovers |
| `/predict` returns 422 | A variant contradicts the protein (wrong wild-type residue, or a position past the end) | Check the sequence the caller sent; the router never scores a variant it cannot verify |
| p95 latency above the SLO | Batching off, or too few replicas for the load | Confirm `FOLDGATE_MAX_BATCH` is set, then scale out; the load numbers in the README are per replica |
| Parity check fails after a change to the model | The exported graph no longer matches PyTorch | Do not promote. Rerun `eval/run_serving.py`, which refuses to continue above 1e-3 |
| Spearman drops but coverage holds | The model got worse honestly | The gate rejects it against the zero-shot baseline; nothing to roll back |

## What has not been run

The Helm chart and the KServe InferenceService are rendered and schema checked in CI, not
applied to a cluster. The SageMaker stack in `infra/terraform/aws` is formatted, validated
and linted in CI, and has never been applied: there is no FoldGate endpoint in any AWS
account, and no cloud spend behind any number in this repository.
