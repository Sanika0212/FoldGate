"""The throttle that keeps a run from cooking the machine it is running on."""

from __future__ import annotations

import os
import time

import pytest

from foldgate.thermal import DutyCycle, background_qos, limit_threads


def test_limit_threads_sets_every_pool() -> None:
    limit_threads(2)
    assert os.environ["OMP_NUM_THREADS"] == "2"
    assert os.environ["MKL_NUM_THREADS"] == "2"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"

    import torch

    assert torch.get_num_threads() == 2


def test_duty_cycle_pauses_in_proportion_to_the_work() -> None:
    cycle = DutyCycle(duty=0.5)
    started = time.monotonic()
    for _ in range(3):
        cycle.begin()
        time.sleep(0.05)
        cycle.end()
    elapsed = time.monotonic() - started
    assert elapsed > 0.25  # 0.15 s of work plus about as much idling
    assert 0.4 < cycle.share < 0.6


def test_a_full_duty_cycle_never_sleeps() -> None:
    cycle = DutyCycle(duty=1.0)
    started = time.monotonic()
    cycle.begin()
    time.sleep(0.05)
    cycle.end()
    assert time.monotonic() - started < 0.12
    assert cycle.paused == 0.0


def test_a_long_batch_does_not_pause_forever() -> None:
    cycle = DutyCycle(duty=0.1, max_pause=0.05)
    cycle.begin()
    time.sleep(0.02)
    started = time.monotonic()
    cycle.end()
    assert time.monotonic() - started <= 0.1  # capped, not 0.18 s


@pytest.mark.parametrize("duty", [0.0, -0.5, 1.5])
def test_an_impossible_duty_is_refused(duty: float) -> None:
    with pytest.raises(ValueError, match="duty"):
        DutyCycle(duty)


def test_background_qos_reports_whether_it_took() -> None:
    assert isinstance(background_qos(), bool)


def test_memory_headroom_is_a_fraction() -> None:
    from foldgate.thermal import memory_headroom

    value = memory_headroom()
    assert 0.0 <= value <= 1.0


def test_cap_memory_sets_the_metal_watermark(monkeypatch: pytest.MonkeyPatch) -> None:
    from foldgate.thermal import cap_memory

    monkeypatch.delenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", raising=False)
    monkeypatch.delenv("PYTORCH_MPS_LOW_WATERMARK_RATIO", raising=False)
    cap_memory(0.4)
    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.40"
    assert os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] == "0.30"


def test_waiting_for_memory_returns_at_once_when_there_is_room(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import foldgate.thermal as thermal

    monkeypatch.setattr(thermal, "memory_headroom", lambda: 0.9)
    assert thermal.wait_for_memory(0.25, poll=0.01) == 0.0


def test_waiting_for_memory_gives_up_rather_than_blocking_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A machine that never frees memory must not stall a run indefinitely."""
    import foldgate.thermal as thermal

    monkeypatch.setattr(thermal, "memory_headroom", lambda: 0.01)
    waited = thermal.wait_for_memory(0.25, poll=0.01, patience=0.05)
    assert 0.0 < waited <= 0.06


def test_waiting_for_memory_returns_once_the_machine_frees_some(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import foldgate.thermal as thermal

    readings = iter([0.05, 0.05, 0.6, 0.6])
    monkeypatch.setattr(thermal, "memory_headroom", lambda: next(readings))
    waited = thermal.wait_for_memory(0.25, poll=0.01, patience=5.0)
    assert 0.0 < waited < 0.05
