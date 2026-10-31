#!/usr/bin/env python3
"""
energy_sweep.py  -  Phase 4: energy and latency per decision, per power mode.

Measures, at the CURRENT nvpmodel power mode:
  idle            board power with the model loaded and nothing running
  clean_nodef     clean image   -> SmolVLM                 (baseline)
  attacked_nodef  patched image -> SmolVLM                 (undefended)
  attacked_pb15   patched image -> PatchBlock (15%) -> SmolVLM  (defended)

Each condition runs back-to-back decisions for at least --duration seconds
(cycling through the 10 Phase 2 scenes) while tegrastats samples the VDD_IN
rail every 100 ms. Energy per decision = mean power x duration / decisions.
Defense overhead = attacked_pb15 minus attacked_nodef.

Also logged per condition, to catch throttling and invalid runs:
  GPU clock (mean, min), max temperature, over-current event count change,
  VDD_CPU_GPU_CV and VDD_SOC rail power, and whether every decision matched
  the action recorded in Phase 2 / Phase 3 (proves the intended workload ran).

Run once per power mode and repeat (run_sweep.sh does this). Results append to
energy_results/energy.csv; --resume skips run/mode/condition already done.

METHOD NOTES
- float32, greedy decoding, pinned preprocessing: same as Phases 2 and 3.
- PatchBlock runs on the CPU, SmolVLM on the GPU, sequentially. The paper
  overlaps them in a pipeline; that hides latency but not CPU energy.
- VDD_IN is total module input power. Energy above idle subtracts the idle
  power measured in the same mode, the same session.

Usage:
  python -u energy_sweep.py --resume --run 2
  python -u energy_sweep.py --summary-only
"""

import argparse
import csv
import datetime
import glob
import os
import re
import statistics
import subprocess
import threading
import time

import numpy as np

import patchblock as pb
from attack_eval import DEFAULT_MODEL, get_action, load_rows, power_mode

FIELDS = ["run", "mode", "condition", "decisions", "duration_s", "mean_power_w",
          "energy_per_decision_j", "energy_above_idle_j", "idle_power_w",
          "latency_median_s", "latency_p90_s", "defense_cpu_median_s",
          "cpu_gpu_cv_power_w", "soc_power_w", "gpu_freq_mean_mhz",
          "gpu_freq_min_mhz", "temp_max_c", "oc_events", "power_samples",
          "actions_as_expected", "timestamp"]

RAIL_RE = re.compile(r"(VDD_[A-Z0-9_]+)\s+(\d+)mW/(\d+)mW")
GPU_RE = re.compile(r"GR3D_FREQ\s+(\d+)%(?:@\[?(\d+)\]?)?")
TEMP_RE = re.compile(r"\b([a-z]+\d?)@(-?[\d.]+)C")
DEVFREQ_ROOT = "/sys/class/devfreq"


def find_gpu_freq_file(root=None):
    """GPU current-frequency file in sysfs (Hz). Readable without sudo, unlike
    the clock in tegrastats. Returns None if not found."""
    root = root or DEVFREQ_ROOT
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        name = os.path.basename(d).lower()
        path = os.path.join(d, "cur_freq")
        if any(s in name for s in ("gpu", "ga10b", "gv11b", "gp10b")) and os.path.exists(path):
            return path
    return None


def read_mhz(path):
    try:
        return int(open(path).read().strip()) / 1e6
    except (OSError, ValueError, TypeError):
        return None


class Sampler:
    """Background tegrastats reader: rails (W), GPU clock (MHz), temps (C).
    GPU clock comes from sysfs when available, else from tegrastats."""

    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.samples = []          # (t, {rail: W}, gpu_mhz or None, max_temp or None)
        self._proc = None
        self._thread = None
        self.gpu_file = find_gpu_freq_file()
        self.gpu_source = self.gpu_file or "tegrastats (needs sudo for clock)"

    def _read(self):
        for line in self._proc.stdout:
            t = time.monotonic()
            rails = {m.group(1): int(m.group(2)) / 1000 for m in RAIL_RE.finditer(line)}
            if not rails:
                continue
            if self.gpu_file:
                gpu = read_mhz(self.gpu_file)
            else:
                g = GPU_RE.search(line)
                gpu = int(g.group(2)) if g and g.group(2) else None
            temps = [float(m.group(2)) for m in TEMP_RE.finditer(line)]
            self.samples.append((t, rails, gpu, max(temps) if temps else None))

    def start(self):
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 5
        while not self.samples and time.monotonic() < deadline:
            time.sleep(0.05)
        if not self.samples:
            self.stop()
            raise RuntimeError("tegrastats gave no power readings. Try running with sudo, "
                               "or check that 'tegrastats' prints VDD_IN.")

    def stop(self):
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=2)

    def window(self, t0, t1):
        return [s for s in self.samples if t0 <= s[0] <= t1]


def mean_rail(samples, rail):
    """Time-weighted (trapezoid) mean power of one rail over the samples."""
    pts = [(t, r[rail]) for t, r, _, _ in samples if rail in r]
    if len(pts) < 2:
        return None
    e = sum(0.5 * (p0 + p1) * (t1 - t0) for (t0, p0), (t1, p1) in zip(pts, pts[1:]))
    return e / (pts[-1][0] - pts[0][0])


def oc_event_total():
    """Sum of over-current event counters, or None if the board has none."""
    vals = []
    for path in glob.glob("/sys/class/hwmon/hwmon*/oc*_event_cnt"):
        try:
            vals.append(int(open(path).read().strip()))
        except (OSError, ValueError):
            pass
    return sum(vals) if vals else None


def measure(sampler, fn, duration, min_iters, warmup):
    """Run fn() until both duration and min_iters are met. Returns stats."""
    for _ in range(warmup):
        fn()
    oc0 = oc_event_total()
    lat, defense_t, ok = [], [], 0
    t0 = time.monotonic()
    while True:
        s = time.monotonic()
        matched, dt = fn()
        lat.append(time.monotonic() - s)
        ok += matched
        if dt is not None:
            defense_t.append(dt)
        if time.monotonic() - t0 >= duration and len(lat) >= min_iters:
            break
    t1 = time.monotonic()
    time.sleep(0.25)                       # let the last sample land
    oc1 = oc_event_total()
    win = sampler.window(t0, t1)
    gpu = [g for _, _, g, _ in win if g is not None]
    temps = [x for _, _, _, x in win if x is not None]
    return dict(
        decisions=len(lat), duration_s=t1 - t0, samples=len(win),
        mean_power_w=mean_rail(win, "VDD_IN"),
        cpu_gpu_cv_power_w=mean_rail(win, "VDD_CPU_GPU_CV"),
        soc_power_w=mean_rail(win, "VDD_SOC"),
        latency_median_s=statistics.median(lat),
        latency_p90_s=float(np.percentile(lat, 90)),
        defense_cpu_median_s=statistics.median(defense_t) if defense_t else None,
        gpu_freq_mean_mhz=statistics.mean(gpu) if gpu else None,
        gpu_freq_min_mhz=min(gpu) if gpu else None,
        temp_max_c=max(temps) if temps else None,
        oc_events=(oc1 - oc0) if oc0 is not None and oc1 is not None else None,
        actions_ok=ok)


def fmt(v, nd=3):
    return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def ms(vals, nd=3):
    """'mean +/- sd (n=k)'; sd is the sample standard deviation across runs."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f} (n=1)"
    return f"{statistics.mean(vals):.{nd}f} +/- {statistics.stdev(vals):.{nd}f} (n={len(vals)})"


def summarize(rows):
    MODE_ORDER = {"15W": 0, "25W": 1, "MAXN_SUPER": 2}
    modes = sorted({r["mode"] for r in rows}, key=lambda m: MODE_ORDER.get(m, 9))
    runs = sorted({int(r["run"]) for r in rows})
    conds = ["clean_nodef", "attacked_nodef", "attacked_pb15"]
    get = {(int(r["run"]), r["mode"], r["condition"]): r for r in rows}

    def col(m, c, field):
        return [float(get[(k, m, c)][field]) for k in runs
                if (k, m, c) in get and get[(k, m, c)][field] != ""]

    def overhead(m, field):
        """Per-run defense overhead (pb15 minus undefended), absolute and %."""
        absv, pct = [], []
        for k in runs:
            a, d = get.get((k, m, "attacked_nodef")), get.get((k, m, "attacked_pb15"))
            if a and d:
                va, vd = float(a[field]), float(d[field])
                absv.append(vd - va)
                pct.append(100 * (vd - va) / va)
        return absv, pct

    L = ["PHASE 4 - ENERGY PER DECISION ACROSS POWER MODES",
         f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
         f"Runs: {runs}. Values are mean +/- sample SD across runs.", "",
         "ENERGY PER DECISION (J), total module input power (VDD_IN)", ""]
    for m in modes:
        L.append(f"  {m}")
        L.append(f"    idle power (W)        : {ms(col(m, 'idle', 'mean_power_w'), 2)}")
        for c in conds:
            L.append(f"    {c:<22}: {ms(col(m, c, 'energy_per_decision_j'))}")
        a, p = overhead(m, "energy_per_decision_j")
        L.append(f"    defense overhead      : {ms(a)} J  =  {ms(p, 1)} %")
    L += ["", "LATENCY PER DECISION, median within each run (s)", ""]
    for m in modes:
        L.append(f"  {m}")
        for c in conds:
            L.append(f"    {c:<22}: {ms(col(m, c, 'latency_median_s'), 2)}")
        a, _ = overhead(m, "latency_median_s")
        L.append(f"    defense overhead      : {ms(a, 2)} s")
        L.append(f"    PatchBlock CPU step   : {ms(col(m, 'attacked_pb15', 'defense_cpu_median_s'), 2)} s")

    L += ["", "VALIDITY CHECKS (every row)", ""]
    hdr = f"{'run':<4} {'mode':<11} {'condition':<15} {'decisions':<9} {'actions ok':<11} " \
          f"{'GPU MHz mean/min':<17} {'max C':<6} {'OC':<4}"
    L += [hdr, "-" * len(hdr)]
    problems = []
    for r in sorted(rows, key=lambda r: (int(r["run"]), MODE_ORDER.get(r["mode"], 9))):
        if r["condition"] == "idle":
            continue
        g = f"{r['gpu_freq_mean_mhz'] or '?'}/{r['gpu_freq_min_mhz'] or '?'}"
        L.append(f"{r['run']:<4} {r['mode']:<11} {r['condition']:<15} {r['decisions']:<9} "
                 f"{r['actions_as_expected']:<11} {g:<17} {r['temp_max_c'] or '?':<6} "
                 f"{r['oc_events'] if r['oc_events'] != '' else 'n/a':<4}")
        n_ok, n = r["actions_as_expected"].split("/")
        tag = f"run {r['run']} {r['mode']} {r['condition']}"
        if n_ok != n:
            problems.append(f"{tag}: actions differ from Phase 2/3")
        if r["oc_events"] not in ("", "0"):
            problems.append(f"{tag}: over-current events {r['oc_events']}")
    L += ["", "PROBLEMS: " + ("none detected" if not problems else ""),
          *[f"  - {p}" for p in problems],
          "", "NOTES",
          "  - float32 inference; PatchBlock runs on the CPU before GPU inference (not pipelined).",
          "  - Defense overhead = attacked_pb15 minus attacked_nodef, computed per run, then averaged.",
          "  - clean vs attacked differ partly because the answers differ in length",
          "    ('REPORT.' vs 'STOP.'), not only because of the attack.",
          "  - Over-current (OC) events mean the supply limit was hit; check that",
          "    mode's GPU clock minimum against its other runs for throttling.",
          "  - In attacked_pb15 rows a low GPU clock minimum (~306 MHz) is expected:",
          "    the GPU clocks down while the CPU runs PatchBlock. Not throttling.",
          "  - Run 1 has no GPU clock (tegrastats without sudo)."]
    return "\n".join(L) + "\n"


def migrate_csv(path):
    """Add the 'run' column to a CSV written before it existed (rows -> run 1)."""
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        if "run" in (rdr.fieldnames or []):
            return
        old = list(rdr)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in old:
            w.writerow({**{k: "" for k in FIELDS}, **r, "run": "1"})
    print(f"Migrated {path}: existing {len(old)} rows labelled run 1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--defense", default="defense_results")
    ap.add_argument("--out", default="energy_results")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds per condition")
    ap.add_argument("--min-iters", type=int, default=20)
    ap.add_argument("--idle", type=float, default=20.0, help="seconds of idle measurement")
    ap.add_argument("--warmup-s", type=float, default=30.0, help="GPU warm-up before measuring")
    ap.add_argument("--flag-frac", type=float, default=0.15)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run", type=int, default=1, help="repeat number (1, 2, 3, ...)")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "energy.csv")
    sum_path = os.path.join(args.out, "energy_summary.txt")
    migrate_csv(csv_path)
    if args.summary_only:
        s = summarize(load_rows(csv_path))
        open(sum_path, "w").write(s)
        print(s)
        return

    mode = power_mode()
    if mode == "not recorded":
        print("Could not read the power mode (nvpmodel -q). Stopping so rows are not mislabelled.")
        return
    if os.path.exists(csv_path) and not args.resume:
        print(f"{csv_path} exists. Use --resume to add this mode, or move it away.")
        return
    done = {(r["run"], r["mode"], r["condition"]) for r in load_rows(csv_path)}
    todo = [c for c in ("idle", "clean_nodef", "attacked_nodef", "attacked_pb15")
            if (str(args.run), mode, c) not in done]
    print(f"Run {args.run}, power mode {mode}. To measure: {todo or 'nothing (already done)'}")
    if not todo:
        return

    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor
    assert torch.cuda.is_available(), "No GPU visible. Activate the venv."

    scenes = [r for r in load_rows(os.path.join(args.results, "results.csv"))
              if r["status"] == "EVALUATED"]
    tag = f"pb_f{int(round(args.flag_frac * 100)):02d}"
    pb_act = {r["filename"]: r["action"] for r in load_rows(os.path.join(args.defense, "defense.csv"))
              if r["condition"] == tag and r["image"] == "patched"}
    items = []
    for r in scenes:
        stem = os.path.splitext(r["filename"])[0]
        items.append(dict(
            clean=Image.open(os.path.join(args.results, "clean", f"{stem}.png")).convert("RGB"),
            patched=np.array(Image.open(os.path.join(args.results, "patched", f"{stem}.png")).convert("RGB")),
            clean_action=r["clean_1"], target=r["target"], pb_action=pb_act.get(r["filename"])))
    items_patched_pil = [Image.fromarray(it["patched"]) for it in items]
    print(f"{len(items)} scenes loaded")

    print(f"Loading {args.model} (float32) ...")
    processor = AutoProcessor.from_pretrained(args.model)
    processor.image_processor.do_image_splitting = False
    processor.image_processor.size = {"longest_edge": 512}
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.float32).to("cuda").eval()

    counter = {"i": 0}

    def nxt():
        it = counter["i"] % len(items)
        counter["i"] += 1
        return it

    def run_clean():
        i = nxt()
        a, _ = get_action(processor, model, items[i]["clean"])
        return a == items[i]["clean_action"], None

    def run_attacked():
        i = nxt()
        a, _ = get_action(processor, model, items_patched_pil[i])
        return a == items[i]["target"], None

    def run_defended():
        i = nxt()
        t = time.perf_counter()
        defended, _ = pb.patchblock(items[i]["patched"], flag_frac=args.flag_frac)
        dt = time.perf_counter() - t
        a, _ = get_action(processor, model, Image.fromarray(defended))
        return (items[i]["pb_action"] is None or a == items[i]["pb_action"]), dt

    sampler = Sampler()
    sampler.start()
    print(f"GPU clock source: {sampler.gpu_source}")
    try:
        if "idle" in todo:
            print(f"Idle: {args.idle:.0f} s, model loaded, nothing running ...")
            torch.cuda.synchronize()
            time.sleep(2)
            t0 = time.monotonic()
            time.sleep(args.idle)
            win = sampler.window(t0, time.monotonic())
            temps = [x for _, _, _, x in win if x is not None]
            append_row(csv_path, {**{k: "" for k in FIELDS}, "run": args.run, "mode": mode, "condition": "idle",
                                  "duration_s": fmt(args.idle, 1),
                                  "mean_power_w": fmt(mean_rail(win, "VDD_IN")),
                                  "cpu_gpu_cv_power_w": fmt(mean_rail(win, "VDD_CPU_GPU_CV")),
                                  "soc_power_w": fmt(mean_rail(win, "VDD_SOC")),
                                  "temp_max_c": fmt(max(temps), 1) if temps else "",
                                  "power_samples": len(win),
                                  "timestamp": datetime.datetime.now().isoformat(timespec="seconds")})
            print(f"    idle {mean_rail(win, 'VDD_IN'):.2f} W\n")

        idle_rows = [r for r in load_rows(csv_path) if r["run"] == str(args.run)
                     and r["mode"] == mode and r["condition"] == "idle"]
        idle_w = float(idle_rows[-1]["mean_power_w"]) if idle_rows else None

        print(f"GPU warm-up: {args.warmup_s:.0f} s of inference ...")
        t_end = time.monotonic() + args.warmup_s
        while time.monotonic() < t_end:
            run_clean()

        fns = {"clean_nodef": run_clean, "attacked_nodef": run_attacked,
               "attacked_pb15": run_defended}
        for cond in [c for c in todo if c != "idle"]:
            counter["i"] = 0
            print(f"{cond}: >= {args.duration:.0f} s ...")
            m = measure(sampler, fns[cond], args.duration, args.min_iters, warmup=2)
            e = m["mean_power_w"] * m["duration_s"] / m["decisions"]
            above = (m["mean_power_w"] - idle_w) * m["duration_s"] / m["decisions"] if idle_w else None
            append_row(csv_path, {
                "run": args.run, "mode": mode, "condition": cond, "decisions": m["decisions"],
                "duration_s": fmt(m["duration_s"], 3), "mean_power_w": fmt(m["mean_power_w"]),
                "energy_per_decision_j": fmt(e), "energy_above_idle_j": fmt(above),
                "idle_power_w": fmt(idle_w), "latency_median_s": fmt(m["latency_median_s"]),
                "latency_p90_s": fmt(m["latency_p90_s"]),
                "defense_cpu_median_s": fmt(m["defense_cpu_median_s"]),
                "cpu_gpu_cv_power_w": fmt(m["cpu_gpu_cv_power_w"]),
                "soc_power_w": fmt(m["soc_power_w"]),
                "gpu_freq_mean_mhz": fmt(m["gpu_freq_mean_mhz"], 0),
                "gpu_freq_min_mhz": fmt(m["gpu_freq_min_mhz"], 0),
                "temp_max_c": fmt(m["temp_max_c"], 1), "oc_events": fmt(m["oc_events"]),
                "power_samples": m["samples"],
                "actions_as_expected": f"{m['actions_ok']}/{m['decisions']}",
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds")})
            print(f"    {m['decisions']} decisions, {m['mean_power_w']:.2f} W, "
                  f"{e:.3f} J/decision, median {m['latency_median_s']:.2f} s, "
                  f"actions ok {m['actions_ok']}/{m['decisions']}\n")
    finally:
        sampler.stop()

    s = summarize(load_rows(csv_path))
    open(sum_path, "w").write(s)
    print(s)
    print(f"CSV     : {os.path.abspath(csv_path)}")
    print(f"Summary : {os.path.abspath(sum_path)}")


if __name__ == "__main__":
    main()
