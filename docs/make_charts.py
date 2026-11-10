#!/usr/bin/env python3
"""Draw the README charts from a benchmark CSV.

    python3 docs/make_charts.py docs/results.csv

Writes next to the CSV:
    throughput.png   images per second for the resize + normalize chain, by batch size
    bandwidth.png    achieved memory bandwidth of each HIP kernel at the largest batch
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

# Colour follows the implementation, in a fixed order, whatever subset is present.
SERIES = [
    ("hip_fused", "HIP fused_rcmn", "#2a78d6"),
    ("hip_separate", "HIP resize + cmn", "#eb6834"),
    ("cpu_ref_fused", "CPU reference (fused)", "#1baf7a"),
    ("opencv_cpu", "OpenCV (CPU)", "#eda100"),
    ("torchvision_cpu", "torchvision (CPU)", "#e87ba4"),
]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=10)
    ax.set_axisbelow(True)


def fmt_rate(v):
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.0f}k" if v >= 1e4 else f"{v / 1e3:.1f}k"
    return f"{v:.0f}"


def throughput_chart(rows, path):
    batches = sorted({int(r["batch"]) for r in rows if r["pipeline"] == "resize_cmn"})
    if not batches:
        return False
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    xs = list(range(len(batches)))
    ends = []  # (last value, label) for direct labels
    for impl, label, color in SERIES:
        vals = {int(r["batch"]): float(r["images_per_s"]) for r in rows
                if r["pipeline"] == "resize_cmn" and r["impl"] == impl}
        if not vals:
            continue
        pts = [(x, vals[b]) for x, b in zip(xs, batches) if b in vals]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=2, marker="o",
                markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ends.append((pts[-1][1], pts[-1][0], fmt_rate(pts[-1][1])))
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlim(-0.25, len(batches) - 1 + 0.55)
    ax.set_xlabel("Batch size", color=INK_2, fontsize=10)
    ax.set_ylabel("Images per second (log scale)", color=INK_2, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    # Value at the largest batch, in ink, beside each line's end.
    for y, x, text in ends:
        ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=10, color=INK)
    ax.set_title("Resize to 224x224, mirror, normalize: throughput by batch size", loc="left",
                 color=INK, fontsize=12, fontweight="bold", pad=14)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3, frameon=False,
                    fontsize=10)
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return True


def bandwidth_chart(rows, path):
    gpu = [r for r in rows if r["impl"].startswith("hip")]
    if not gpu:
        return False
    batch = max(int(r["batch"]) for r in gpu)
    gpu = [r for r in gpu if int(r["batch"]) == batch]
    names = {
        ("resize", "hip"): "resize",
        ("crop_mirror_normalize", "hip"): "crop_mirror_normalize",
        ("gaussian_blur_3x3", "hip_direct"): "blur 3x3, direct",
        ("gaussian_blur_3x3", "hip_lds"): "blur 3x3, LDS",
        ("gaussian_blur_5x5", "hip_direct"): "blur 5x5, direct",
        ("gaussian_blur_5x5", "hip_lds"): "blur 5x5, LDS",
        ("resize_cmn", "hip_separate"): "resize + cmn (two kernels)",
        ("resize_cmn", "hip_fused"): "fused_rcmn",
    }
    items = [(names[(r["pipeline"], r["impl"])], float(r["gb_per_s"]), float(r["peak_share_pct"]))
             for r in gpu if (r["pipeline"], r["impl"]) in names]
    items.reverse()  # first row at the top
    fig, ax = plt.subplots(figsize=(8.4, 0.5 * len(items) + 1.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ys = list(range(len(items)))
    ax.barh(ys, [i[1] for i in items], height=0.55, color="#2a78d6")
    ax.set_yticks(ys)
    ax.set_yticklabels([i[0] for i in items], color=INK, fontsize=10)
    top = max(i[1] for i in items)
    ax.set_xlim(0, top * 1.28)
    for y, (_, gbps, share) in zip(ys, items):
        ax.text(gbps + top * 0.012, y, f"{gbps:,.0f} GB/s  ({share:.1f}% of peak)", va="center",
                fontsize=10, color=INK)
    ax.set_xlabel("Achieved bandwidth: (bytes read + bytes written) / median time, GB/s",
                  color=INK_2, fontsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines["left"].set_visible(False)
    device = gpu[0].get("device", "").strip('"')
    ax.set_title(f"Memory bandwidth per HIP kernel, batch {batch}" +
                 (f"\n{device}" if device else ""), loc="left", color=INK, fontsize=12,
                 fontweight="bold", pad=12)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return True


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    with open(src, newline="") as f:
        rows = list(csv.DictReader(f))
    out_dir = os.path.dirname(os.path.abspath(src))
    for name, fn in (("throughput.png", throughput_chart), ("bandwidth.png", bandwidth_chart)):
        path = os.path.join(out_dir, name)
        print(("wrote " if fn(rows, path) else "no data for ") + path)


if __name__ == "__main__":
    main()
