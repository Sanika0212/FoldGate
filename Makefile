.PHONY: install lint test test-model data baselines train shift serving gate eval figures readme readme-check serve

PY ?= .venv/bin/python
SCHEME ?= random

install:            ## dev environment
	uv sync --all-extras

lint:
	.venv/bin/ruff check src tests eval scripts
	.venv/bin/ruff format --check src tests eval scripts
	.venv/bin/mypy

test:               ## everything that does not need the weights
	.venv/bin/pytest -q --cov=foldgate --cov-report=term

test-model:         ## downloads ESM-2 35M and exports it to ONNX
	.venv/bin/pytest -q -m model

data:               ## ProteinGym shards and the zero-shot cache
	$(PY) -m foldgate.cli download
	$(PY) -m foldgate.cli zero-shot

baselines:
	$(PY) eval/run_baselines.py --scheme random
	$(PY) eval/run_baselines.py --scheme contiguous

train:
	$(PY) eval/run_train.py --scheme $(SCHEME)

shift:
	$(PY) eval/run_shift.py --scheme $(SCHEME)

serving:            ## ONNX export, parity, load test
	$(PY) eval/run_serving.py --scheme $(SCHEME)

gate:               ## three candidates, one promotion
	$(PY) eval/run_gate.py --scheme $(SCHEME)

eval: baselines     ## the whole experiment, in order, then the README it produces
	$(PY) eval/run_train.py --scheme random
	$(PY) eval/run_train.py --scheme contiguous
	$(PY) eval/run_shift.py --scheme random
	$(PY) eval/run_serving.py --scheme random
	$(PY) eval/run_gate.py --scheme random
	$(PY) scripts/make_figures.py
	$(PY) scripts/render_readme.py --scheme random

figures:
	$(PY) scripts/make_figures.py

readme:             ## fill the README's result sections from runs/
	$(PY) scripts/render_readme.py --scheme $(SCHEME)

readme-check:       ## fail when the README and runs/ disagree
	$(PY) scripts/render_readme.py --scheme $(SCHEME) --check

serve:
	$(PY) -m foldgate.cli serve --scheme $(SCHEME) --port 8080
