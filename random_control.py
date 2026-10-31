#!/usr/bin/env python3
"""
random_control.py  -  control for Phase 3: is PatchBlock's detection better
than flagging chunks at random?

For each scene and each flag fraction, flag the SAME number of chunks as
PatchBlock would, but chosen at random, then apply the identical SVD
mitigation. Repeated with several random seeds. If random flagging recovers
about as many scenes as PatchBlock, the detection stage adds nothing.

Reads:   results/results.csv, results/clean/, results/patched/  (Phase 2)
         defense_results/defense.csv                           (Phase 3, for comparison)
Writes:  defense_results/random_control.csv
         defense_results/random_summary.txt

Usage (from the folder with attack_eval.py, patchblock.py, results/):
  python -u random_control.py 2>&1 | tee random_log.txt
  python -u random_control.py --resume 2>&1 | tee -a random_log.txt
"""

import argparse
import csv
import datetime
import os
import statistics
import time
import zlib

import numpy as np

import patchblock as pb
from attack_eval import DEFAULT_MODEL, get_action, load_rows, power_mode

FIELDS = ["filename", "image", "frac", "seed", "action", "clean_action", "target",
          "recovered", "attack_holds", "det_precision", "det_recall"]


def random_mask(n_chunks, frac, rng):
    n_flag = max(1, int(round(frac * n_chunks)))
    mask = np.zeros(n_chunks, dtype=bool)
    mask[rng.choice(n_chunks, n_flag, replace=False)] = True
    return mask


def append_rows(path, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def summarize(rows, pb_rows, fracs, seeds, meta):
    def pct(a, b):
        return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"

    n_scenes = len({r["filename"] for r in rows})
    L = ["PHASE 3 CONTROL - RANDOM FLAGGING vs PATCHBLOCK",
         f"Generated: {meta['date']}", "",
         f"Scenes: {n_scenes}   Random seeds per scene: {len(seeds)}",
         "Random results are pooled over all seeds (scenes x seeds trials).", "",
         f"{'frac':<6} {'recovered: PatchBlock':<23} {'recovered: random':<20} "
         f"{'attack holds: PB':<18} {'attack holds: rand':<19} {'clean kept: rand':<16}"]
    L.append("-" * len(L[-1]))
    for f in fracs:
        tag = f"pb_f{int(round(f * 100)):02d}"
        pp = [r for r in pb_rows if r["condition"] == tag and r["image"] == "patched"]
        rp = [r for r in rows if r["frac"] == f"{f:.2f}" and r["image"] == "patched"]
        rc = [r for r in rows if r["frac"] == f"{f:.2f}" and r["image"] == "clean"]
        L.append(f"{f:<6.2f} "
                 f"{pct(sum(r['recovered'] == 'yes' for r in pp), len(pp)):<23} "
                 f"{pct(sum(r['recovered'] == 'yes' for r in rp), len(rp)):<20} "
                 f"{pct(sum(r['attack_holds'] == 'yes' for r in pp), len(pp)):<18} "
                 f"{pct(sum(r['attack_holds'] == 'yes' for r in rp), len(rp)):<19} "
                 f"{pct(sum(r['recovered'] == 'yes' for r in rc), len(rc)):<16}")
    L += ["", "DETECTION (patched images, mean over trials)"]
    for f in fracs:
        tag = f"pb_f{int(round(f * 100)):02d}"
        rp = [r for r in rows if r["frac"] == f"{f:.2f}" and r["image"] == "patched"]
        pp = [r for r in pb_rows if r["condition"] == tag and r["image"] == "patched"]
        if rp and pp:
            L.append(f"  {f:.2f}: recall PatchBlock {statistics.mean(float(r['det_recall']) for r in pp):.2f}"
                     f"  vs random {statistics.mean(float(r['det_recall']) for r in rp):.2f}"
                     f"   | precision PatchBlock {statistics.mean(float(r['det_precision']) for r in pp):.2f}"
                     f"  vs random {statistics.mean(float(r['det_precision']) for r in rp):.2f}")
    L += ["", "HOW TO READ",
          "  PatchBlock's recovery only counts as evidence for its detection stage if it",
          "  is clearly higher than random flagging at the same fraction. With 10 scenes,",
          "  a difference of 1 or 2 scenes is within noise.",
          "", "SETTINGS",
          f"  SVD info kept: {meta['info']}   chunk: {meta['k']} px   seeds: {seeds}",
          f"  Power mode   : {meta['power_mode']}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="defense_results")
    ap.add_argument("--fracs", default="0.05,0.10,0.15,0.20")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--info", type=float, default=0.90)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--patch-frac", type=float, default=0.30)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    assert torch.cuda.is_available(), "No GPU visible. Activate the venv."
    p2 = [r for r in load_rows(os.path.join(args.results, "results.csv"))
          if r["status"] == "EVALUATED"]
    pb_rows = load_rows(os.path.join(args.out, "defense.csv"))
    if not p2:
        print(f"No evaluated scenes in {args.results}/results.csv")
        return
    if not pb_rows:
        print(f"No {args.out}/defense.csv. Run defense_eval.py first.")
        return

    csv_path = os.path.join(args.out, "random_control.csv")
    if os.path.exists(csv_path) and not args.resume:
        print(f"{csv_path} exists. Use --resume to continue it, or move it away.")
        return
    done = {r["filename"] for r in load_rows(csv_path)}
    todo = [r for r in p2 if r["filename"] not in done]
    fracs = [float(x) for x in args.fracs.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    n_reads = len(todo) * len(fracs) * len(seeds) * 2
    print(f"{len(todo)} scenes to run, {n_reads} model reads")

    print(f"Loading {args.model} (float32) ...")
    processor = AutoProcessor.from_pretrained(args.model)
    processor.image_processor.do_image_splitting = False
    processor.image_processor.size = {"longest_edge": 512}
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.float32).to("cuda").eval()
    print("Model ready.\n")

    times = []
    for n, r in enumerate(todo):
        t_scene = time.monotonic()
        stem = os.path.splitext(r["filename"])[0]
        clean_action, target = r["clean_1"], r["target"]
        imgs = {kind: np.array(Image.open(os.path.join(args.results, kind, f"{stem}.png"))
                               .convert("RGB")) for kind in ("clean", "patched")}
        H, W = imgs["patched"].shape[:2]
        side = int(args.patch_frac * H)
        box = ((H - side) // 2, (W - side) // 2, side, side)
        ys, xs = pb.chunk_grid(H, W, args.k)
        truth = pb.overlap_mask(ys, xs, args.k, box, 0.5)
        print(f"[{n + 1}/{len(todo)}] {r['filename']}  clean={clean_action} target={target}")
        rows = []
        for f in fracs:
            parts = []
            for s in seeds:
                # independent pattern per scene, fraction and seed; the same
                # pattern is used on the clean and the patched image
                rng = np.random.default_rng([s, zlib.crc32(r['filename'].encode()),
                                             int(round(f * 100))])
                mask = random_mask(len(ys) * len(xs), f, rng)
                p, rc = pb.detection_quality(mask, truth)
                for kind in ("clean", "patched"):
                    defended = pb.mitigate(imgs[kind], ys, xs, args.k, mask, args.info)
                    action, _ = get_action(processor, model, Image.fromarray(defended))
                    row = dict(filename=r["filename"], image=kind, frac=f"{f:.2f}", seed=s,
                               action=action, clean_action=clean_action, target=target,
                               recovered="yes" if action == clean_action else "no",
                               attack_holds=("yes" if action == target else "no")
                               if kind == "patched" else "",
                               det_precision=f"{p:.3f}" if kind == "patched" else "",
                               det_recall=f"{rc:.3f}" if kind == "patched" else "")
                    rows.append(row)
                    if kind == "patched":
                        parts.append(action)
            print(f"    f={f:.2f} patched (seeds {seeds}): {' '.join(parts)}")
        append_rows(csv_path, rows)
        times.append(time.monotonic() - t_scene)
        print(f"    scene {times[-1]:.0f}s | ~{statistics.mean(times) * (len(todo) - n - 1) / 60:.0f} min left\n")

    rows = load_rows(csv_path)
    meta = dict(date=datetime.datetime.now().isoformat(timespec="seconds"),
                info=args.info, k=args.k, power_mode=power_mode())
    summary = summarize(rows, pb_rows, fracs, seeds, meta)
    path = os.path.join(args.out, "random_summary.txt")
    with open(path, "w") as fh:
        fh.write(summary)
    print(summary)
    print(f"CSV     : {os.path.abspath(csv_path)}")
    print(f"Summary : {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
