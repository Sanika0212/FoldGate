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


def test_available_memory_is_a_fraction() -> None:
    """Jetsam's own availability figure, which is what the gate decides on."""
    from foldgate.thermal import available_memory

    assert 0.0 <= available_memory() <= 1.0


def test_pressure_level_is_one_of_the_kernels_levels() -> None:
    from foldgate.thermal import pressure_level

    assert pressure_level() in (1, 2, 4)


def test_unreadable_sysctls_never_block_a_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a machine that cannot be measured, the gate must open, not close."""
    import foldgate.thermal as thermal

    monkeypatch.setattr(thermal, "_sysctl_int", lambda _name: None)
    assert thermal.available_memory() == 1.0
    assert thermal.pressure_level() == 1
    assert thermal.memory_is_tight() is False


def test_memory_is_tight_on_low_availability_or_critical_pressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import foldgate.thermal as thermal

    monkeypatch.setattr(thermal, "available_memory", lambda: 0.40)
    monkeypatch.setattr(thermal, "pressure_level", lambda: 2)
    assert thermal.memory_is_tight() is False  # an ordinary busy laptop

    monkeypatch.setattr(thermal, "available_memory", lambda: 0.05)
    assert thermal.memory_is_tight() is True  # Jetsam territory

    monkeypatch.setattr(thermal, "available_memory", lambda: 0.40)
    monkeypatch.setattr(thermal, "pressure_level", lambda: 4)
    assert thermal.memory_is_tight() is True  # critical pressure at any availability


def test_the_gate_opens_on_a_laptop_with_low_free_pages_but_no_real_pressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug this replaced: free pages sit near a quarter on this machine always, so a
    gate on them never opened and the run slept instead of training."""
    import foldgate.thermal as thermal

    monkeypatch.setattr(thermal, "memory_headroom", lambda: 0.20)
    monkeypatch.setattr(thermal, "available_memory", lambda: 0.41)
    monkeypatch.setattr(thermal, "pressure_level", lambda: 2)
    assert thermal.wait_for_memory(poll=0.01) == 0.0


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

    monkeypatch.setattr(thermal, "memory_is_tight", lambda *_a, **_k: False)
    assert thermal.wait_for_memory(poll=0.01) == 0.0


def test_waiting_for_memory_gives_up_rather_than_blocking_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A machine that never recovers must not stall a run indefinitely."""
    import foldgate.thermal as thermal

    monkeypatch.setattr(thermal, "memory_is_tight", lambda *_a, **_k: True)
    waited = thermal.wait_for_memory(poll=0.01, patience=0.05)
    assert 0.0 < waited <= 0.06


def test_waiting_for_memory_returns_once_the_machine_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import foldgate.thermal as thermal

    readings = iter([True, True, False, False])
    monkeypatch.setattr(thermal, "memory_is_tight", lambda *_a, **_k: next(readings))
    assert 0.0 < thermal.wait_for_memory(poll=0.01, patience=5.0) < 0.05


def test_releasing_the_cache_reports_a_pool_size_and_does_not_raise() -> None:
    """On a machine without Metal this is a no-op; on one with it, a number in GiB."""
    from foldgate.thermal import release_cache

    pool = release_cache()
    assert pool >= 0.0


def test_releasing_the_cache_survives_a_torch_without_metal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A machine with no Metal must get a no-op, not an exception mid-epoch."""
    import sys
    from types import SimpleNamespace

    import foldgate.thermal as thermal

    no_metal = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "torch", no_metal)
    assert thermal.release_cache() == 0.0


def test_waiting_for_the_gpu_is_safe_without_metal(monkeypatch: pytest.MonkeyPatch) -> None:
    """The throttle runs on machines with no Metal, where this has to be a no-op."""
    import sys
    from types import SimpleNamespace

    import foldgate.thermal as thermal

    no_metal = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "torch", no_metal)
    assert thermal.wait_for_gpu() is None


def test_waiting_for_the_gpu_does_not_raise_on_this_machine() -> None:
    from foldgate.thermal import wait_for_gpu

    assert wait_for_gpu() is None
