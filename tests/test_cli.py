"""The command line, with the model and the Hub stubbed out."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from foldgate import cli


def test_download_prints_each_shard(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("foldgate.data.download", lambda: [Path("a.parquet"), Path("b.parquet")])
    cli.main(["download"])
    assert capsys.readouterr().out.split() == ["a.parquet", "b.parquet"]


def test_zero_shot_reports_how_many_variants_were_cached(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pd.DataFrame({"dms_id": ["A"] * 3, "mutant": ["A1G"] * 3, "zero_shot": [0.0] * 3})
    monkeypatch.setattr("foldgate.prepare.zero_shot_frame", lambda backbone=None: frame)
    cli.main(["zero-shot"])
    assert "3 variants cached" in capsys.readouterr().out


def test_predict_reads_a_fasta_and_scores_each_variant(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = tmp_path / "wt.fasta"
    fasta.write_text(">my protein\nMKAILVV\nLLYTFATA\n")
    seen: dict[str, Any] = {}

    class Stub:
        def score(self, items: list[tuple[str, str]]) -> list[dict[str, Any]]:
            seen["items"] = items
            return [
                {"variant": m, "score": 0.5, "lower": 0.1, "upper": 0.9, "abstain": False} for _, m in items
            ]

    monkeypatch.setattr(cli, "_version", lambda scheme: Stub())
    cli.main(["predict", "--protein-file", str(fasta), "--variants", "M1A, L9P"])
    rows = json.loads(capsys.readouterr().out)
    assert [r["variant"] for r in rows] == ["M1A", "L9P"]
    # The header line is dropped and the sequence lines are joined.
    assert seen["items"][0][0] == "MKAILVVLLYTFATA"


def test_serve_builds_the_app_and_hands_it_to_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    monkeypatch.setattr(cli, "_version", lambda scheme: f"version-{scheme}")
    monkeypatch.setattr(
        "foldgate.router.create_app",
        lambda champion, candidate, **kw: {"champion": champion, "candidate": candidate, **kw},
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "uvicorn",
        type(
            "M",
            (),
            {"run": staticmethod(lambda app, host, port: calls.update(app=app, host=host, port=port))},
        ),
    )
    cli.main(["serve", "--scheme", "contiguous", "--port", "9123", "--batching"])
    assert calls["port"] == 9123
    assert calls["app"]["champion"] == "version-contiguous"
    assert calls["app"]["candidate"] is None
    assert calls["app"]["batching"] is True


def test_serve_picks_up_the_deployment_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The chart and the KServe manifest configure the router through these variables."""
    calls: dict[str, Any] = {}
    monkeypatch.setattr(cli, "_version", lambda scheme, which="champion": f"{which}-{scheme}")
    monkeypatch.setattr(
        "foldgate.router.create_app",
        lambda champion, candidate, **kw: {"champion": champion, "candidate": candidate, **kw},
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "uvicorn",
        type(
            "M",
            (),
            {"run": staticmethod(lambda app, host, port: calls.update(app=app, host=host, port=port))},
        ),
    )
    monkeypatch.setenv("FOLDGATE_CANDIDATE_SHARE", "0.25")
    monkeypatch.setenv("FOLDGATE_SHADOW", "true")
    monkeypatch.setenv("FOLDGATE_MAX_BATCH", "128")
    monkeypatch.setenv("FOLDGATE_MAX_WAIT_MS", "15")
    cli.main(["serve", "--host", "0.0.0.0"])
    assert calls["host"] == "0.0.0.0"
    app = calls["app"]
    assert app["candidate"] == "candidate-random"
    assert app["candidate_share"] == 0.25
    assert app["shadow"] is True
    assert app["max_batch"] == 128
    assert app["max_wait_ms"] == 15.0
    # Batching is opt-in: it cost throughput and bought no tail latency on this CPU.
    assert app["batching"] is False


def test_version_dir_prefers_the_mounted_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    assert cli.version_dir("random") == Path("artifacts/random")
    monkeypatch.setenv("FOLDGATE_ARTIFACTS", "/models/champion")
    monkeypatch.setenv("FOLDGATE_CANDIDATE_ARTIFACTS", "/models/candidate")
    assert cli.version_dir("random") == Path("/models/champion")
    assert cli.version_dir("random", "candidate") == Path("/models/candidate")


def test_an_unknown_command_is_rejected() -> None:
    with pytest.raises(SystemExit):
        cli.main(["teleport"])
