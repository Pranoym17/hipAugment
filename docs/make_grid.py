#!/usr/bin/env python3
"""Build the README image grid: each sample image next to the output of every op.

    python3 docs/make_grid.py <hipaug binary> [--backend cpu|hip]

Runs `hipaug run` for each op on docs/samples and writes docs/grid.png.
"""
import argparse
import os
import subprocess
import tempfile

import cv2
import numpy as np

OPS = ["resize", "gaussian_blur", "crop_mirror_normalize", "fused_rcmn"]
CELL = 224


def fit(img):
    """Scale an image to fit a CELL x CELL tile on a white background, without stretching."""
    h, w = img.shape[:2]
    s = CELL / max(h, w)
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    tile = np.full((CELL, CELL, 3), 255, np.uint8)
    y, x = (CELL - small.shape[0]) // 2, (CELL - small.shape[1]) // 2
    tile[y:y + small.shape[0], x:x + small.shape[1]] = small
    return tile


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("hipaug")
    ap.add_argument("--backend", default="cpu")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    samples = os.path.join(here, "samples")
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(samples) if f.endswith(".png"))

    pad, head = 12, 34
    cols = ["input"] + OPS
    grid = np.full((head + len(names) * (CELL + pad), len(cols) * (CELL + pad) + pad, 3), 255,
                   np.uint8)
    for c, label in enumerate(cols):
        cv2.putText(grid, label, (pad + c * (CELL + pad), 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (40, 40, 40), 1, cv2.LINE_AA)
    with tempfile.TemporaryDirectory() as tmp:
        for op in OPS:
            subprocess.run([args.hipaug, "run", "--op", op, "--batch", "4", "--size", "224x224",
                            "--mirror", "--sigma", "2.0", "--backend", args.backend, samples, tmp],
                           check=True)
        for r, name in enumerate(names):
            tiles = [cv2.imread(os.path.join(samples, name + ".png"))]
            tiles += [cv2.imread(os.path.join(tmp, f"{name}_{op}.png")) for op in OPS]
            for c, t in enumerate(tiles):
                y, x = head + r * (CELL + pad), pad + c * (CELL + pad)
                grid[y:y + CELL, x:x + CELL] = fit(t)
    out = os.path.join(here, "grid.png")
    cv2.imwrite(out, grid)
    print("wrote", out)


if __name__ == "__main__":
    main()
