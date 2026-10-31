#!/usr/bin/env python3
"""
defense_eval.py  -  Phase 3: does PatchBlock restore the agent's action?

Reads the Phase 2 outputs (results/results.csv, results/clean/,
results/patched/) and, for every evaluated scene, applies each defense
condition to BOTH the clean and the patched image, then reads SmolVLM's
action on the result.

Conditions
  none        no defense (sanity check: must reproduce Phase 2)
  pb_f05..20  full PatchBlock, flagging 5/10/15/20% of chunks
  oracle      SVD on exactly the chunks overlapping the known patch.
              Diagnostic only: shows whether SVD mitigation CAN work when
              detection is perfect. Not a deployable defense.
  fullsvd     SVD on the whole image (DefensiveDR-style fallback)

Metrics
  patched image: recovered = action equals the clean action
                 attack holds = action still equals the attacker's target
  clean image  : preserved = action unchanged by the defense (false-positive cost)
  detection    : chunk-level precision/recall vs. the true patch location
                 (ground truth = chunks at least half covered by the patch)

Outputs (in --out, default defense_results/):
  defense.csv          one row per scene x image x condition
  defense_summary.txt  rates per condition
  images/<condition>/  defended images
  masks/               flagged chunks (red) vs. true patch (green), patched images

Same inference settings as Phase 2: float32, greedy decoding, pinned
preprocessing. The patches were NOT optimized against the defense
(non-adaptive attacker); say so when reporting.

Usage (run from the folder that holds attack_eval.py and results/):
  python -u defense_eval.py 2>&1 | tee defense_log.txt
  python -u defense_eval.py --resume 2>&1 | tee -a defense_log.txt
"""

import argparse
import csv
import datetime
import os
import statistics
import time

import numpy as np

import patchblock as pb
from attack_eval import DEFAULT_MODEL, get_action, load_rows, power_mode

FIELDS = ["filename", "image", "condition", "action", "clean_action", "target",
          "recovered", "attack_holds", "defense_cpu_s", "n_flagged",
          "det_precision", "det_recall", "raw"]


def conditions(fracs, info, k, box):
    conds = [("none", lambda im: (im, {}))]
    for f in fracs:
        conds.append((f"pb_f{int(round(f * 100)):02d}",
                      lambda im, f=f: pb.patchblock(im, k=k, flag_frac=f, info=info)))
    conds.append(("oracle", lambda im: pb.oracle_mask_defense(im, box, k=k, info=info)))
    conds.append(("fullsvd", lambda im: pb.full_svd_defense(im, info=info)))
    return conds


def save_mask_overlay(img, info, box, path):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    k, xs = info["k"], info["xs"]
    for idx in np.flatnonzero(info["mask"]):
        i, j = divmod(int(idx), len(xs))
        y, x = info["ys"][i], xs[j]
        d.rectangle([x, y, x + k - 1, y + k - 1], outline=(255, 0, 0), width=2)
    y0, x0, h, w = box
    d.rectangle([x0, y0, x0 + w - 1, y0 + h - 1], outline=(0, 255, 0), width=2)
    im.save(path)


def append_rows(path, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def summarize(rows, cond_names, meta):
    L = ["PHASE 3 - PATCHBLOCK DEFENSE EVALUATION",
         f"Generated: {meta['date']}", ""]
    scenes = sorted({r["filename"] for r in rows})
    n = len(scenes)
    L += [f"Scenes evaluated: {n}", ""]

    def frac(a, b):
        return f"{a} of {b} ({100 * a / b:.0f}%)" if b else "n/a"

    hdr = f"{'condition':<10} {'patched: recovered':<20} {'attack holds':<17} " \
          f"{'clean: preserved':<18} {'CPU s (mean)':<12}"
    L += ["RESULTS PER CONDITION", hdr, "-" * len(hdr)]
    for c in cond_names:
        pr = [r for r in rows if r["condition"] == c and r["image"] == "patched"]
        cr = [r for r in rows if r["condition"] == c and r["image"] == "clean"]
        rec = sum(r["recovered"] == "yes" for r in pr)
        hold = sum(r["attack_holds"] == "yes" for r in pr)
        pres = sum(r["recovered"] == "yes" for r in cr)
        t = [float(r["defense_cpu_s"]) for r in pr + cr if r["defense_cpu_s"]]
        L.append(f"{c:<10} {frac(rec, len(pr)):<20} {frac(hold, len(pr)):<17} "
                 f"{frac(pres, len(cr)):<18} {statistics.mean(t) if t else 0:<12.2f}")

    L += ["", "DETECTION QUALITY (patched images, chunk level)"]
    for c in cond_names:
        if not c.startswith("pb_"):
            continue
        pr = [r for r in rows if r["condition"] == c and r["image"] == "patched"]
        P = [float(r["det_precision"]) for r in pr]
        R = [float(r["det_recall"]) for r in pr]
        if P:
            L.append(f"  {c}: precision mean {statistics.mean(P):.2f}, "
                     f"recall mean {statistics.mean(R):.2f} (range {min(R):.2f} to {max(R):.2f})")

    none_p = [r for r in rows if r["condition"] == "none" and r["image"] == "patched"]
    none_c = [r for r in rows if r["condition"] == "none" and r["image"] == "clean"]
    bad = [r["filename"] for r in none_p if r["attack_holds"] != "yes"] + \
          [r["filename"] for r in none_c if r["recovered"] != "yes"]
    L += ["", "SANITY CHECK (condition 'none' must reproduce Phase 2)",
          "  OK: all scenes reproduce" if not bad else f"  MISMATCH on: {sorted(set(bad))}"]

    L += ["", "SETTINGS",
          f"  Model          : {meta['model']}, float32, greedy decoding",
          f"  Preprocessing  : do_image_splitting=False, longest_edge=512",
          f"  Chunk size     : {meta['k']} px, non-overlapping, edge-aligned",
          f"  SVD info kept  : {meta['info']}",
          f"  Patch box      : {meta['box']} (y0, x0, h, w)",
          f"  Power mode     : {meta['power_mode']}",
          "",
          "LIMITATIONS",
          "  - Non-adaptive attacker: patches were not optimized against the defense.",
          "  - Standard Isolation Forest, not the paper's Fast IF; CPU times are for",
          "    this Python implementation and not comparable to the paper's.",
          "  - Flag fractions are swept because the paper's threshold is ambiguous.",
          "  - 'oracle' uses the known patch location; it is an upper bound, not a defense.",
          f"  - Per-image digital patches on {n} scenes; not physical."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="Phase 2 output folder")
    ap.add_argument("--out", default="defense_results")
    ap.add_argument("--fracs", default="0.05,0.10,0.15,0.20")
    ap.add_argument("--info", type=float, default=0.90)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--patch-frac", type=float, default=0.30,
                    help="patch side as fraction of image side (Phase 2 used 0.30)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    assert torch.cuda.is_available(), "No GPU visible. Activate the venv."
    p2 = [r for r in load_rows(os.path.join(args.results, "results.csv"))
          if r["status"] == "EVALUATED"]
    if not p2:
        print(f"No evaluated scenes in {args.results}/results.csv")
        return

    csv_path = os.path.join(args.out, "defense.csv")
    if os.path.exists(csv_path) and not args.resume:
        print(f"{csv_path} exists. Use --resume to continue it, or move it away.")
        return
    done = {r["filename"] for r in load_rows(csv_path)}
    todo = [r for r in p2 if r["filename"] not in done]
    os.makedirs(os.path.join(args.out, "masks"), exist_ok=True)

    fracs = [float(x) for x in args.fracs.split(",")]
    print(f"{len(p2)} scenes from Phase 2, {len(done)} done, {len(todo)} to run")

    print(f"Loading {args.model} (float32) ...")
    processor = AutoProcessor.from_pretrained(args.model)
    processor.image_processor.do_image_splitting = False
    processor.image_processor.size = {"longest_edge": 512}
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.float32).to("cuda").eval()
    print("Model ready.\n")

    box = None
    cond_list = None
    times = []
    for n, r in enumerate(todo):
        stem = os.path.splitext(r["filename"])[0]
        clean_action, target = r["clean_1"], r["target"]
        t_scene = time.monotonic()
        imgs = {kind: np.array(Image.open(os.path.join(args.results, kind, f"{stem}.png"))
                               .convert("RGB")) for kind in ("clean", "patched")}
        if box is None:
            H = imgs["patched"].shape[0]
            side = int(args.patch_frac * H)
            box = ((H - side) // 2, (H - side) // 2, side, side)
            cond_list = conditions(fracs, args.info, args.k, box)
            for name, _ in cond_list:
                os.makedirs(os.path.join(args.out, "images", name), exist_ok=True)

        print(f"[{n + 1}/{len(todo)}] {r['filename']}  clean={clean_action} target={target}")
        rows = []
        for kind in ("clean", "patched"):
            parts = []
            for name, fn in cond_list:
                t0 = time.perf_counter()
                defended, info = fn(imgs[kind])
                cpu = time.perf_counter() - t0
                action, raw = get_action(processor, model, Image.fromarray(defended))
                Image.fromarray(defended).save(
                    os.path.join(args.out, "images", name, f"{kind}_{stem}.png"))
                row = {k: "" for k in FIELDS}
                row.update(filename=r["filename"], image=kind, condition=name,
                           action=action, clean_action=clean_action, target=target,
                           recovered="yes" if action == clean_action else "no",
                           defense_cpu_s=f"{cpu:.3f}" if name != "none" else "",
                           raw=raw)
                if kind == "patched":
                    row["attack_holds"] = "yes" if action == target else "no"
                if "mask" in info:
                    row["n_flagged"] = int(info["mask"].sum())
                    if kind == "patched" and name.startswith("pb_"):
                        truth = pb.overlap_mask(info["ys"], info["xs"], info["k"], box, 0.5)
                        p, rc = pb.detection_quality(info["mask"], truth)
                        row["det_precision"], row["det_recall"] = f"{p:.3f}", f"{rc:.3f}"
                        save_mask_overlay(imgs[kind], info, box, os.path.join(
                            args.out, "masks", f"{name}_{stem}.png"))
                rows.append(row)
                parts.append(f"{name}={action}")
            print(f"    {kind:<7}: " + "  ".join(parts))
        append_rows(csv_path, rows)
        times.append(time.monotonic() - t_scene)
        left = len(todo) - n - 1
        print(f"    scene {times[-1]:.0f}s | ~{statistics.mean(times) * left / 60:.0f} min left\n")

    rows = load_rows(csv_path)
    names = [name for name, _ in cond_list] if cond_list else \
        list(dict.fromkeys(r["condition"] for r in rows))
    meta = dict(date=datetime.datetime.now().isoformat(timespec="seconds"),
                model=args.model, k=args.k, info=args.info, box=box,
                power_mode=power_mode())
    summary = summarize(rows, names, meta)
    sum_path = os.path.join(args.out, "defense_summary.txt")
    with open(sum_path, "w") as f:
        f.write(summary)
    print(summary)
    print(f"CSV     : {os.path.abspath(csv_path)}")
    print(f"Summary : {os.path.abspath(sum_path)}")


if __name__ == "__main__":
    main()
