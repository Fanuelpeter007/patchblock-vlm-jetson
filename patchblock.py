#!/usr/bin/env python3
"""
patchblock.py  -  reimplementation of PatchBlock (Chattopadhyay et al.,
arXiv:2601.00367, DATE 2026) from its published description.

Three stages, run on the CPU on a uint8 RGB image (H, W, 3):
  1. Chunking   : split the image into k x k chunks (paper: k = 50 px)
  2. Separating : score each chunk with localized mutual information (MI),
                  run an Isolation Forest on the scores, flag the most
                  anomalous chunks
  3. Mitigating : replace each flagged chunk with a low-rank SVD
                  reconstruction that keeps a fraction `info` of its energy
                  (paper: 85-90%)

DEVIATIONS FROM THE PAPER (state these in the report)
- Standard Isolation Forest (scikit-learn), not the paper's "Fast Isolation
  Forest" with gradient-based targeted cuts. The targeted cut is a speed
  optimization; detection logic is the same idea. Runtime here is therefore
  NOT comparable to the paper's 0.317 s.
- The paper describes MI two ways: across colour channels within a chunk
  (Section I, Fig. 2) and between a chunk and its neighbours (Section III-B).
  Both are used here as features: MI(R,G), MI(G,B), MI(R,B) and the mean MI
  of the chunk with its 4 grid neighbours, pixel-aligned, on grayscale.
- Stride is not given in the paper. Default is non-overlapping (stride = k);
  the last row/column of chunks is aligned to the image edge, and overlapping
  writes are averaged.
- The outlier threshold c is ambiguous in the paper (Algorithm 1 picks
  (1 - c) * n chunks with c = 1%, which would flag 99% of the image). Here the
  flagged fraction is an explicit parameter and is swept in the evaluation.
- SVD is applied per colour channel to the raw (uncentred) chunk; `info` is
  the fraction of squared singular values kept.
"""

import numpy as np
from sklearn.ensemble import IsolationForest


# ------------------------------------------------------------- 1. chunking
def chunk_grid(h, w, k=50, stride=None):
    """Top-left corners of the chunk grid. Last row/col aligned to the edge."""
    stride = stride or k
    ys = list(range(0, h - k + 1, stride))
    xs = list(range(0, w - k + 1, stride))
    if ys[-1] != h - k:
        ys.append(h - k)
    if xs[-1] != w - k:
        xs.append(w - k)
    return ys, xs


# ------------------------------------------------------------- 2. separating
def mutual_info(a, b, bins=32):
    """Histogram estimate of MI (nats) between two equal-size uint8 arrays."""
    h, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins,
                             range=[[0, 256], [0, 256]])
    pxy = h / h.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])))


def chunk_features(img, ys, xs, k):
    """4 features per chunk: 3 inter-channel MI values + mean neighbour MI."""
    gray = (0.299 * img[..., 0] + 0.587 * img[..., 1]
            + 0.114 * img[..., 2]).astype(np.uint8)
    feats = np.zeros((len(ys), len(xs), 4))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            c = img[y:y + k, x:x + k]
            feats[i, j, 0] = mutual_info(c[..., 0], c[..., 1])
            feats[i, j, 1] = mutual_info(c[..., 1], c[..., 2])
            feats[i, j, 2] = mutual_info(c[..., 0], c[..., 2])
            g = gray[y:y + k, x:x + k]
            nbr = []
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < len(ys) and 0 <= jj < len(xs):
                    yy, xx = ys[ii], xs[jj]
                    nbr.append(mutual_info(g, gray[yy:yy + k, xx:xx + k]))
            feats[i, j, 3] = float(np.mean(nbr))
    return feats.reshape(-1, 4)


def flag_chunks(feats, flag_frac, seed=0):
    """Isolation Forest on standardized features; flag the most anomalous
    `flag_frac` of chunks. Returns (boolean mask, anomaly scores)."""
    z = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-9)
    forest = IsolationForest(n_estimators=100, max_samples=0.3,
                             random_state=seed).fit(z)
    score = -forest.score_samples(z)          # higher = more anomalous
    n_flag = max(1, int(round(flag_frac * len(score))))
    mask = np.zeros(len(score), dtype=bool)
    mask[np.argsort(score)[::-1][:n_flag]] = True
    return mask, score


# ------------------------------------------------------------- 3. mitigating
def svd_reduce(block, info=0.90):
    """Per-channel low-rank reconstruction keeping `info` of the energy
    (sum of squared singular values)."""
    out = np.empty(block.shape, dtype=np.float64)
    for ch in range(block.shape[2]):
        m = block[..., ch].astype(np.float64)
        u, s, vt = np.linalg.svd(m, full_matrices=False)
        energy = np.cumsum(s ** 2) / max(np.sum(s ** 2), 1e-12)
        r = min(int(np.searchsorted(energy, info)) + 1, len(s))
        out[..., ch] = (u[:, :r] * s[:r]) @ vt[:r]
    return out


def mitigate(img, ys, xs, k, mask, info=0.90):
    """Replace flagged chunks with their SVD reconstruction. Overlapping
    writes (edge chunks) are averaged."""
    acc = np.zeros(img.shape, dtype=np.float64)   # sum of reconstructions
    count = np.zeros(img.shape[:2])               # reconstructions per pixel
    flat = mask.reshape(len(ys), len(xs))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if flat[i, j]:
                acc[y:y + k, x:x + k] += svd_reduce(img[y:y + k, x:x + k], info)
                count[y:y + k, x:x + k] += 1
    out = img.astype(np.float64).copy()
    hit = count > 0
    out[hit] = acc[hit] / count[hit][:, None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# ------------------------------------------------------------- pipelines
def patchblock(img, k=50, stride=None, flag_frac=0.10, info=0.90, seed=0):
    """Full PatchBlock. Returns (defended image, info dict)."""
    ys, xs = chunk_grid(img.shape[0], img.shape[1], k, stride)
    feats = chunk_features(img, ys, xs, k)
    mask, score = flag_chunks(feats, flag_frac, seed)
    out = mitigate(img, ys, xs, k, mask, info)
    return out, {"ys": ys, "xs": xs, "k": k, "mask": mask, "score": score}


def oracle_mask_defense(img, box, k=50, stride=None, info=0.90):
    """Diagnostic upper bound: mitigate exactly the chunks that overlap the
    known patch box (y0, x0, h, w). Separates detection from mitigation."""
    ys, xs = chunk_grid(img.shape[0], img.shape[1], k, stride)
    mask = overlap_mask(ys, xs, k, box)
    return mitigate(img, ys, xs, k, mask, info), {"ys": ys, "xs": xs, "k": k, "mask": mask}


def full_svd_defense(img, info=0.90):
    """SVD on the whole image (DefensiveDR-style), the planned fallback."""
    return np.clip(np.round(svd_reduce(img, info)), 0, 255).astype(np.uint8), {}


def overlap_mask(ys, xs, k, box, min_frac=0.0):
    """Chunks whose overlap with box = (y0, x0, h, w) is more than `min_frac`
    of the chunk area. min_frac=0 means any overlap (used by the oracle, which
    must cover the whole patch); 0.5 means mostly-patch chunks (used as ground
    truth for detection precision/recall)."""
    y0, x0, h, w = box
    m = np.zeros((len(ys), len(xs)), dtype=bool)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            oy = max(0, min(y + k, y0 + h) - max(y, y0))
            ox = max(0, min(x + k, x0 + w) - max(x, x0))
            m[i, j] = oy * ox > min_frac * k * k
    return m.ravel()


def detection_quality(mask, truth):
    """Precision and recall of flagged chunks against the true patch chunks."""
    tp = int(np.sum(mask & truth))
    prec = tp / max(int(mask.sum()), 1)
    rec = tp / max(int(truth.sum()), 1)
    return prec, rec
