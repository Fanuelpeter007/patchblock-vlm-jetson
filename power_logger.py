#!/usr/bin/env python3
"""
power_logger.py

Energy measurement harness for NVIDIA Jetson Orin Nano, built on tegrastats.
Reads the VDD_IN rail (total module input power) reported by the onboard
INA3221, samples it in a background thread, and integrates power over a timed
window to produce energy in joules.

Built for the PatchBlock-on-VLM energy study: measure joules-per-inference for
(clean / attacked-no-defense / attacked-with-defense) at each nvpmodel power
mode (15W / 25W / MAXN SUPER).

Why repeat-and-average: tegrastats samples at ~100 ms at best. A single
sub-second inference yields only a few samples, which is noisy. measure_energy()
runs the workload many times over a fixed minimum window, then divides total
energy by iteration count. That averages out the coarse sampling.

Reports TOTAL module power (VDD_IN). For defense *overhead*, take the delta
between conditions; idle and board power cancel in the difference.

If it prints 0 samples, tegrastats needs elevated rights on your image:
    sudo ~/patchblock-venv/bin/python power_logger.py
"""

import re
import subprocess
import threading
import time
from dataclasses import dataclass


@dataclass
class EnergyResult:
    joules_per_call: float
    mean_power_w: float
    iters: int
    duration_s: float
    samples: int

    def __str__(self):
        return (
            f"{self.joules_per_call * 1000:.2f} mJ/call | "
            f"{self.mean_power_w:.2f} W mean | "
            f"{self.iters} iters | {self.duration_s:.1f} s | "
            f"{self.samples} power samples"
        )


class PowerSampler:
    """Background sampler parsing one rail (mW) from tegrastats with timestamps."""

    def __init__(self, interval_ms: int = 100, rail: str = "VDD_IN"):
        self.interval_ms = interval_ms
        self.rail = rail
        # tegrastats prints e.g.  "VDD_IN 4523mW/4523mW" (instant/running-avg)
        self._rail_re = re.compile(rf"{re.escape(rail)}\s+(\d+)mW/(\d+)mW")
        self._proc = None
        self._thread = None
        self._running = False
        self._samples = []  # list of (monotonic_time_s, power_w)

    def _reader(self):
        for line in self._proc.stdout:
            if not self._running:
                break
            m = self._rail_re.search(line)
            if m:
                self._samples.append((time.monotonic(), int(m.group(1)) / 1000.0))

    def start(self):
        self._samples = []
        self._running = True
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=2)

    @property
    def samples(self):
        return list(self._samples)


def _integrate(samples):
    """Trapezoidal integration of power(W) over time(s) -> energy(J)."""
    if len(samples) < 2:
        return 0.0, 0.0, 0.0
    energy = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        energy += 0.5 * (p0 + p1) * (t1 - t0)
    duration = samples[-1][0] - samples[0][0]
    mean_power = energy / duration if duration > 0 else 0.0
    return energy, mean_power, duration


def measure_energy(fn, min_duration_s=30.0, min_iters=30, interval_ms=100,
                   warmup=3, rail="VDD_IN") -> EnergyResult:
    """
    Run fn() repeatedly until BOTH min_duration_s and min_iters are met, while
    sampling module power. Returns per-call energy.

    warmup: untimed calls first, to fill caches and reach steady clocks.
    """
    for _ in range(warmup):
        fn()

    sampler = PowerSampler(interval_ms=interval_ms, rail=rail)
    sampler.start()
    time.sleep(0.2)  # let the first sample land

    t_start = time.monotonic()
    iters = 0
    while True:
        fn()
        iters += 1
        if (time.monotonic() - t_start) >= min_duration_s and iters >= min_iters:
            break

    sampler.stop()
    energy, mean_power, duration = _integrate(sampler.samples)
    joules_per_call = energy / iters if iters else 0.0
    return EnergyResult(joules_per_call, mean_power, iters, duration,
                        len(sampler.samples))


if __name__ == "__main__":
    # Verification: idle vs a busy loop. Confirms sampling + integration work,
    # and (run across nvpmodel modes) that mean power tracks the power mode.
    import math

    def idle():
        time.sleep(0.05)

    def busy():
        x = 0.0
        for i in range(300000):
            x += math.sin(i)
        return x

    print("idle :", measure_energy(idle, min_duration_s=8, min_iters=20))
    print("busy :", measure_energy(busy, min_duration_s=8, min_iters=20))
