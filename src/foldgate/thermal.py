"""Keep a run inside what the machine can survive.

This was written after three restarts on a 16 GB M4 MacBook Air. The panic log named the
cause: ``userspace watchdog timeout: no successful checkins from WindowServer in 121
seconds``. The machine had run out of memory, the resulting swap thrash stalled the display
server long enough for the kernel to give up, and the memory-exhaustion report from the same
night named the largest process: the training run. So the thing to bound is memory first and
heat second.

A first version of this gated on free pages: it waited until "free plus inactive plus
speculative" reached a quarter of physical memory. That was the wrong quantity. macOS keeps
almost nothing free on purpose, because it compresses rather than wastes, so on this machine
the metric sits at about 24% whatever is happening and a 25% threshold never opens. A run
started with that gate slept in it for ten minutes at a time and trained almost nothing.

What the kernel itself uses to decide whether memory is in trouble is better and is readable:

1. ``kern.memorystatus_level`` is the share of memory Jetsam considers available. It read
   41% on the same machine, at the same moment, as the 24% above. Jetsam is what killed
   processes on the reboot night, so this is the number that matters.
2. ``kern.memorystatus_vm_pressure_level`` is 1 normal, 2 warning, 4 critical. Everyday use
   of this laptop sits at 2, so only 4 is worth stopping for.

So the gate waits when Jetsam's availability is genuinely low or pressure is critical, and
its patience is a minute rather than ten, because a gate that parks a run is as bad as no
gate at all. Around that:

3. ``cap_memory()`` limits how much of the machine a Metal run may take, so an allocation
   fails inside this process instead of pushing the whole system into swap. Measured: the cap
   costs nothing, 1.17 s against 1.07 s on the same GPU workload.
4. ``limit_threads()`` caps the thread pools that numerical libraries open by default.
5. ``DutyCycle`` idles between batches, which is what keeps a fanless machine cool.
6. ``background_qos()`` is available but costs about ten times the run time on GPU work,
   because it pins to the efficiency cores. Prefer the memory cap and the duty cycle.
"""

from __future__ import annotations

import os
import time
from typing import Any

DEFAULT_THREADS = 2
DEFAULT_MEMORY_SHARE = 0.4
# Jetsam's own availability figure, below which a run should wait. This laptop idles
# around 40%, and the night it rebooted Jetsam was killing applications outright.
MIN_AVAILABLE = 0.18
# 1 normal, 2 warning, 4 critical. Ordinary use of this machine sits at 2.
CRITICAL_PRESSURE = 4


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


def _sysctl_int(name: str) -> int | None:
    """One integer sysctl, or None where the platform does not have it."""
    import subprocess

    try:
        out = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return int(out) if out.lstrip("-").isdigit() else None


def available_memory() -> float:
    """The share of memory Jetsam considers available, from 0 to 1.

    This is ``kern.memorystatus_level``, the number the kernel uses to decide whether to
    start killing processes, which is what happened on the night this machine rebooted.
    Returns 1.0 where it cannot be read, so a guard built on it never blocks a run on a
    machine it cannot measure.
    """
    level = _sysctl_int("kern.memorystatus_level")
    return 1.0 if level is None else max(0.0, min(1.0, level / 100))


def pressure_level() -> int:
    """1 normal, 2 warning, 4 critical, from ``kern.memorystatus_vm_pressure_level``."""
    level = _sysctl_int("kern.memorystatus_vm_pressure_level")
    return 1 if level is None else level


def memory_is_tight(minimum: float = MIN_AVAILABLE, critical: int = CRITICAL_PRESSURE) -> bool:
    """Whether the machine is in real trouble, as opposed to merely busy."""
    return available_memory() < minimum or pressure_level() >= critical


def wait_for_memory(
    minimum: float = MIN_AVAILABLE,
    poll: float = 5.0,
    patience: float = 60.0,
    critical: int = CRITICAL_PRESSURE,
) -> float:
    """Block while the machine is genuinely short of memory; returns the seconds waited.

    Patience is deliberately short. The point is to pace a run around a burst of pressure,
    not to park it: a gate that waits ten minutes at a time trains nothing, which is how the
    first version of this failed. After ``patience`` it returns and lets the run continue,
    bounded by ``cap_memory`` and by whatever watchdog is outside this process.
    """
    waited = 0.0
    while memory_is_tight(minimum, critical) and waited < patience:
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


def wait_for_gpu() -> None:
    """Block until the GPU has finished the work already submitted to it.

    Metal runs asynchronously: a training step returns once its work is queued, not once the
    work is done. A timer around the step therefore measures submission, and a duty cycle
    built on that timer idles the CPU while the GPU is still busy with the batch it just
    received, which is not a duty cycle at all. Synchronizing first makes the measurement and
    the pause mean what they say.
    """
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.synchronize()
    except (ImportError, AttributeError, RuntimeError):
        pass


def release_cache() -> float:
    """Hand cached GPU blocks back, and report the pool size in GiB before doing it.

    PyTorch's Metal allocator keeps freed blocks to reuse, which is right for throughput and
    wrong under a hard cap: over a hundred or so steps the pool fills with blocks of the
    wrong shape and a 33 MiB request fails against a 4.3 GiB pool. That is what ended the
    third attempt at this run. Releasing during the duty cycle's pause costs nothing, because
    the GPU is idle then anyway.
    """
    try:
        import torch

        if not torch.backends.mps.is_available():
            return 0.0
        pool = torch.mps.driver_allocated_memory() / (1 << 30)
        torch.mps.empty_cache()
        return float(pool)
    except (ImportError, AttributeError, RuntimeError):
        return 0.0


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


def lightning_callback(
    duty: float = 0.5,
    every: int = 25,
    minimum: float = MIN_AVAILABLE,
    release_every: int = 5,
) -> Any:
    """A Lightning callback that applies :class:`DutyCycle` around every training batch.

    It prints a line every ``every`` batches, because a throttled run takes long enough
    that silence is indistinguishable from a hang and the progress bar is off, and it hands
    the GPU pool back every ``release_every`` batches.
    """
    import lightning as L

    class Throttle(L.Callback):
        def __init__(self) -> None:
            self.cycle = DutyCycle(duty)
            self.batches = 0
            self.waited = 0.0
            self.pool = 0.0
            self.peak_pool = 0.0
            # The duty cycle counts from the start of the run, so an epoch's own share has
            # to be taken as a difference. Reporting the running total under an epoch's
            # heading reads as though each epoch were slower than the last.
            self.at_epoch_start = (0.0, 0.0)

        def on_train_batch_start(self, *_: Any, **__: Any) -> None:
            if minimum and self.batches % every == 0:
                # Another application may have taken the memory since the last check.
                self.waited += wait_for_memory(minimum)
            self.cycle.begin()

        def on_train_batch_end(self, trainer: Any, *_: Any, **__: Any) -> None:
            # Before the clock stops, or the pause that follows idles nothing.
            wait_for_gpu()
            self.cycle.end()
            self.batches += 1
            if release_every and self.batches % release_every == 0:
                # The GPU is idle in the pause this just took, so give the pool back here.
                # Measured every 25 batches, the pool reached 4.4 GiB against a 4.74 GiB
                # cap, which is 340 MiB of margin and not enough. Every 5 keeps it low, and
                # emptying the cache costs milliseconds against a batch of 0.7 s.
                pool = release_cache()
                self.peak_pool = max(self.peak_pool, pool)
                self.pool = pool
            if self.batches % every:
                return
            total = self.cycle.worked + self.cycle.paused
            per_batch = total / self.batches
            left = max(0, (trainer.num_training_batches or 0) - trainer.fit_loop.batch_idx - 1)
            print(
                f"  epoch {trainer.current_epoch} batch {trainer.fit_loop.batch_idx + 1}"
                f"/{trainer.num_training_batches}: {per_batch:.1f} s a batch, "
                f"{left * per_batch / 60:.0f} min left in this epoch, duty {self.cycle.share:.0%}"
                + f", {available_memory():.0%} memory available"
                + (f", GPU pool {self.pool:.1f} GiB peak {self.peak_pool:.1f}" if self.pool else "")
                + (f", pressure {pressure_level()}" if pressure_level() > 1 else "")
                + (f", waited {self.waited / 60:.0f} min for memory" if self.waited else ""),
                flush=True,
            )

        def on_train_epoch_end(self, trainer: Any, _module: Any) -> None:
            worked = self.cycle.worked - self.at_epoch_start[0]
            paused = self.cycle.paused - self.at_epoch_start[1]
            self.at_epoch_start = (self.cycle.worked, self.cycle.paused)
            print(
                f"  epoch {trainer.current_epoch}: worked {worked / 60:.1f} min, "
                f"paused {paused / 60:.1f} min "
                f"(run so far {self.cycle.worked / 60:.1f} min worked, "
                f"duty {self.cycle.share:.0%}, GPU peak {self.peak_pool:.1f} GiB)",
                flush=True,
            )

    return Throttle()
