#!/usr/bin/env python3
"""
fetch_scenes.py  -  download a fixed, citable scene set from COCO val2017.

Uses fixed image IDs, so the set is reproducible and each image can be cited.
Downloads until --num images succeed; failed IDs are reported, not hidden.

Usage:
    python fetch_scenes.py              # 10 images into scenes_coco/
    python attack_eval.py --scenes scenes_coco
"""

import argparse
import os
import urllib.request

# COCO val2017 image IDs: mixed indoor, outdoor, objects, people, animals.
COCO_IDS = [
    139, 285, 632, 724, 776, 785, 802, 872, 885, 1000,
    1268, 1296, 1353, 1425, 1490, 1503,
]
URL = "http://images.cocodataset.org/val2017/{:012d}.jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, default=10)
    ap.add_argument("--out", default="scenes_coco")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ok, failed = [], []
    for cid in COCO_IDS:
        if len(ok) >= args.num:
            break
        path = os.path.join(args.out, f"coco_{cid:012d}.jpg")
        if os.path.exists(path):
            ok.append(cid)
            continue
        try:
            urllib.request.urlretrieve(URL.format(cid), path)
            ok.append(cid)
            print(f"  saved {path}")
        except Exception as e:
            failed.append(cid)
            print(f"  FAILED id {cid}: {type(e).__name__}: {e}")
            if os.path.exists(path):
                os.remove(path)

    print(f"\n{len(ok)} images in {args.out}/  (failed IDs: {failed or 'none'})")
    if len(ok) < args.num:
        print(f"Only {len(ok)} of {args.num} downloaded. Check the network.")


if __name__ == "__main__":
    main()
