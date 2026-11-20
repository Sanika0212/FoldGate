"""Keep a run inside what the machine can survive.

This was written after three restarts on a 16 GB M4 MacBook Air. The panic log named the
cause: ``userspace watchdog timeout: no successful checkins from WindowServer in 121
seconds``. The machine had run out of memory, and the resulting swap thrash stalled the
display server long enough for the kernel to give up and reboot. The memory-exhaustion
report from the same night named the largest process: the training run.

So the thing to bound is memory first and heat second.

1. ``cap_memory()`` limits how much of the machine a Metal run may take, so an allocation
   fails inside this process instead of pushing the whole system into swap.
2. ``memory_headroom()`` reports the share of memory that is free, and ``wait_for_memory``
   blocks while the machine is short of it. A long run checks before each stretch of work,
   so another application opening is a pause here rather than a reboot.
3. ``limit_threads()`` caps the thread pools that numerical libraries open by default.
4. ``DutyCycle`` idles between batches, which is what keeps a fanless machine cool.
5. ``background_qos()`` is available but costs about ten times the run time on GPU work,
   because it pins to the efficiency cores. Prefer the memory cap and the duty cycle.
"""

from __future__ import annotations

import os
import time
from typing import Any

DEFAULT_THREADS = 2
DEFAULT_MEMORY_SHARE = 0.4
MIN_HEADROOM = 0.25


def cap_memory(share: float = DEFAULT_MEMORY_SHARE) -> None:
    """Refuse to let this process take more than ``share`` of the machine's memory.

    On Apple silicon the GPU allocates from the same pool as everything else, so an
    unbounded Metal run competes with the window server. With a watermark set, PyTorch
    raises inside this process instead, which is a failed run rather than a failed machine.
    """
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", f"{share:.2f}")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", f"{max(0.0, share - 0.1):.2f}")


def memory_headroom() -> float:
    """Share of physical memory that is free or reclaimable, from 0 to 1."""
    import subprocess

    try:
        total = int(
            subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.SubprocessError, ValueError):
        return 1.0
    page = 16384
    counts: dict[str, int] = {}
    for line in out.splitlines():
        if line.startswith("Mach Virtual Memory Statistics"):
            if "page size of" in line:
                page = int(line.split("page size of")[1].split("bytes")[0].strip())
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            digits = value.strip().rstrip(".")
            if digits.isdigit():
                counts[key.strip()] = int(digits)
    free = sum(counts.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative"))
    return min(1.0, free * page / total) if total else 1.0


def wait_for_memory(minimum: float = MIN_HEADROOM, poll: float = 15.0, patience: float = 600.0) -> float:
    """Block while the machine is short of memory; returns the seconds spent waiting.

    After ``patience`` it gives up waiting and returns anyway: the caller decides whether to
    carry on or stop, and a run that waits forever is its own kind of failure.
    """
    waited = 0.0
    while memory_headroom() < minimum and waited < patience:
        time.sleep(poll)
        waited += poll
    return waited


def limit_threads(threads: int = DEFAULT_THREADS) -> None:
    """Cap every thread pool that might otherwise take the whole machine."""
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(name, str(threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass  # interop threads can only be set once per process


def background_qos() -> bool:
    """Ask for background scheduling priority. True when the system accepted it."""
    try:
        import ctypes

        libc = ctypes.CDLL("libc.dylib", use_errno=True)
        # setpriority(PRIO_DARWIN_PROCESS=4, 0, PRIO_DARWIN_BG=0x1000)
        return bool(libc.setpriority(4, 0, 0x1000) == 0)
    except (OSError, AttributeError):
        return False


def load_average() -> float:
    """The one-minute load average, or 0 where the platform does not report it."""
    try:
        return float(os.getloadavg()[0])
    except (OSError, AttributeError):
        return 0.0


def wait_for_calm(limit: float, poll: float = 20.0, patience: float = 900.0) -> float:
    """Block while the machine is busy with something else.

    A laptop that has just rebooted spends a while on Spotlight, Time Machine and cloud
    sync, and piling training on top of that is how it overheats again. This waits for the
    load average to fall below ``limit``, gives up after ``patience`` seconds so a busy
    machine cannot stall a run forever, and returns how long it waited.
    """
    waited = 0.0
    while load_average() > limit and waited < patience:
        time.sleep(poll)
        waited += poll
    return waited


class DutyCycle:
    """Sleep between units of work so the chip idles for part of every second."""

    def __init__(self, duty: float = 0.5, max_pause: float = 5.0) -> None:
        if not 0 < duty <= 1:
            raise ValueError("duty must be in (0, 1]")
        self.duty = duty
        self.max_pause = max_pause
        self.started = time.monotonic()
        self.worked = 0.0
        self.paused = 0.0

    def begin(self) -> None:
        self.started = time.monotonic()

    def end(self) -> None:
        """Pause for as long as this unit of work earned."""
        busy = time.monotonic() - self.started
        self.worked += busy
        if self.duty >= 1:
            return
        pause = min(busy * (1 - self.duty) / self.duty, self.max_pause)
        if pause > 0:
            time.sleep(pause)
            self.paused += pause

    @property
    def share(self) -> float:
        total = self.worked + self.paused
        return self.worked / total if total else 1.0


def lightning_callback(duty: float = 0.5, every: int = 25, headroom: float = MIN_HEADROOM) -> Any:
    """A Lightning callback that applies :class:`DutyCycle` around every training batch.

    It also prints a line every ``every`` batches. A throttled run takes long enough that
    silence is indistinguishable from a hang, and the progress bar is off.
    """
    import lightning as L

    class Throttle(L.Callback):
        def __init__(self) -> None:
            self.cycle = DutyCycle(duty)
            self.batches = 0
            self.waited = 0.0

        def on_train_batch_start(self, *_: Any, **__: Any) -> None:
            if headroom and self.batches % every == 0:
                # Another application may have taken the memory since the last check.
                self.waited += wait_for_memory(headroom)
            self.cycle.begin()

        def on_train_batch_end(self, trainer: Any, *_: Any, **__: Any) -> None:
            self.cycle.end()
            self.batches += 1
            if self.batches % every:
                return
            total = self.cycle.worked + self.cycle.paused
            per_batch = total / self.batches
            left = max(0, (trainer.num_training_batches or 0) - trainer.fit_loop.batch_idx - 1)
            print(
                f"  epoch {trainer.current_epoch} batch {trainer.fit_loop.batch_idx + 1}"
                f"/{trainer.num_training_batches}: {per_batch:.1f} s a batch, "
                f"{left * per_batch / 60:.0f} min left in this epoch, duty {self.cycle.share:.0%}"
                + f", {memory_headroom():.0%} memory free"
                + (f", waited {self.waited / 60:.0f} min for memory" if self.waited else ""),
                flush=True,
            )

        def on_train_epoch_end(self, trainer: Any, _module: Any) -> None:
            print(
                f"  epoch {trainer.current_epoch}: worked {self.cycle.worked / 60:.1f} min, "
                f"paused {self.cycle.paused / 60:.1f} min, duty {self.cycle.share:.0%}",
                flush=True,
            )

    return Throttle()
