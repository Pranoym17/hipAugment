#!/usr/bin/env python3
"""CPU baselines for the hipAugment benchmark: OpenCV and torchvision.

Times the same two pipelines `hipaug bench` reports, on batches of mixed-size
images drawn from the same size range, and appends rows in the same CSV format:

    resize       bilinear resize of each image to the output size
    resize_cmn   resize, mirror, normalize, to float NCHW (the training chain)

OpenCV works on uint8 HWC arrays. torchvision works on uint8 CHW tensors, which is
the form its decoders produce, so no layout conversion is timed for it.
torchvision is skipped if it is not installed.
"""
import argparse
import csv
import os
import statistics
import time

import cv2
import numpy as np

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

FIELDS = ["pipeline", "impl", "batch", "out_w", "out_h", "src_min", "src_max", "runs",
          "median_ms", "min_ms", "p90_ms", "images_per_s", "bytes_read", "bytes_written",
          "gb_per_s", "peak_share_pct", "device"]


def make_images(n, src_min, src_max, seed):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(src_min, src_max + 1, size=(n, 2))
    images = [rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8) for w, h in sizes]
    mirror = rng.integers(0, 2, size=n).astype(bool)
    return images, mirror


def time_fn(fn, warmup, runs):
    for _ in range(warmup):
        fn()
    ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t0) * 1e3)
    return sorted(ms)


def opencv_pipelines(images, mirror, out_w, out_h):
    def resize():
        return [cv2.resize(im, (out_w, out_h), interpolation=cv2.INTER_LINEAR) for im in images]

    def resize_cmn():
        out = np.empty((len(images), 3, out_h, out_w), dtype=np.float32)
        for i, im in enumerate(images):
            r = cv2.resize(im, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            if mirror[i]:
                r = cv2.flip(r, 1)
            out[i] = ((r.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)
        return out

    return {"resize": resize, "resize_cmn": resize_cmn}


def torchvision_pipelines(images, mirror, out_w, out_h):
    import torch
    from torchvision.transforms.v2 import functional as F

    tensors = [torch.from_numpy(im).permute(2, 0, 1).contiguous() for im in images]
    mean, std = MEAN.tolist(), STD.tolist()
    bilinear = F.InterpolationMode.BILINEAR

    def resize():
        # antialias=False keeps plain bilinear sampling, the same convention hipAugment uses.
        return [F.resize(t, [out_h, out_w], interpolation=bilinear, antialias=False)
                for t in tensors]

    def resize_cmn():
        out = []
        for i, t in enumerate(tensors):
            r = F.resize(t, [out_h, out_w], interpolation=bilinear, antialias=False)
            if mirror[i]:
                r = F.hflip(r)
            out.append(F.normalize(r.to(torch.float32), mean, std))
        return torch.stack(out)

    return {"resize": resize, "resize_cmn": resize_cmn}, torch.get_num_threads()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--append", required=True, help="CSV file written by `hipaug bench`")
    ap.add_argument("--batches", default="1,16,64,256")
    ap.add_argument("--size", default="224x224")
    ap.add_argument("--src-min", type=int, default=256)
    ap.add_argument("--src-max", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--peak-gbps", type=float, default=5300.0)
    ap.add_argument("--pipelines", default="resize,resize_cmn",
                    help="comma separated pipelines; only resize and resize_cmn have baselines")
    args = ap.parse_args()
    pipelines = {p for p in args.pipelines.split(",") if p} or {"resize", "resize_cmn"}
    if not pipelines.intersection({"resize", "resize_cmn"}):
        print("baselines.py: no selected pipelines have CPU baselines")
        return
    out_w, out_h = (int(v) for v in args.size.split("x"))
    out_px = out_w * out_h

    rows = []
    for n in (int(b) for b in args.batches.split(",")):
        images, mirror = make_images(n, args.src_min, args.src_max, seed=n)
        # Same byte accounting as bench/bench.cpp.
        resize_read = sum(min(im.shape[0] * im.shape[1], 4 * out_px) * 3 for im in images)
        nbytes = {"resize": (resize_read, n * out_px * 3),
                  "resize_cmn": (resize_read, n * out_px * 12)}

        impls = {"opencv_cpu": (opencv_pipelines(images, mirror, out_w, out_h),
                                f"cpu, OpenCV {cv2.__version__}, {cv2.getNumThreads()} threads")}
        try:
            import torchvision
            pipes, threads = torchvision_pipelines(images, mirror, out_w, out_h)
            impls["torchvision_cpu"] = (
                pipes, f"cpu, torchvision {torchvision.__version__}, {threads} threads")
        except ImportError:
            if n == int(args.batches.split(",")[0]):
                print("baselines.py: torchvision not installed, skipping it")

        for impl, (pipes, device) in impls.items():
            for pipeline, fn in pipes.items():
                if pipeline not in pipelines:
                    continue
                ms = time_fn(fn, args.warmup, args.runs)
                median = statistics.median(ms)
                rd, wr = nbytes[pipeline]
                gbps = (rd + wr) / (median * 1e-3) / 1e9
                rows.append({
                    "pipeline": pipeline, "impl": impl, "batch": n, "out_w": out_w,
                    "out_h": out_h, "src_min": args.src_min, "src_max": args.src_max,
                    "runs": len(ms), "median_ms": f"{median:.6f}", "min_ms": f"{ms[0]:.6f}",
                    "p90_ms": f"{ms[min(len(ms) - 1, int(0.9 * len(ms)))]:.6f}",
                    "images_per_s": f"{n / (median * 1e-3):.1f}", "bytes_read": rd,
                    "bytes_written": wr, "gb_per_s": f"{gbps:.3f}",
                    "peak_share_pct": f"{100.0 * gbps / args.peak_gbps:.3f}", "device": device,
                })
                print(f"{pipeline:<12} {impl:<16} batch {n:>4}  median {median:9.3f} ms")

    new_file = not os.path.exists(args.append) or os.path.getsize(args.append) == 0
    with open(args.append, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerows(rows)
    print(f"appended {len(rows)} rows to {args.append}")


if __name__ == "__main__":
    main()
