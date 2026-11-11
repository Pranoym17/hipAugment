#!/usr/bin/env python3
"""Join rocprofv3 kernel statistics with the byte counts from `hipaug bench`.

Reads the directory tree scripts/profile.sh writes:

    <profile_dir>/b<batch>/<pipeline>/*kernel_stats.csv   rocprofv3 per-kernel statistics
    <profile_dir>/b<batch>/<pipeline>/results.csv         hipaug bench output

and writes one row per (batch, pipeline, kernel): calls, average kernel time, bytes
moved, bandwidth and share of the device peak.

Kernel time here is the dispatch time rocprofv3 measured on the device. It excludes
launch overhead, so it is shorter than the hipEvent time in results.csv.
"""
import argparse
import csv
import glob
import os
import sys

# Kernel name fragment -> (pipeline, impl) whose results.csv row holds that kernel's bytes.
KERNEL_BYTES = {
    "resize_kernel": ("resize", "hip"),
    "cmn_kernel": ("crop_mirror_normalize", "hip"),
    "blur_direct_kernel": (None, "hip_direct"),  # pipeline is the session's own
    "blur_lds_kernel": (None, "hip_lds"),
    "fused_rcmn_kernel": ("resize_cmn", "hip_fused"),
}


def find_column(header, *needles):
    """Index of the first column whose name contains every needle (case-insensitive)."""
    for i, name in enumerate(header):
        low = name.lower()
        if all(n in low for n in needles):
            return i
    return None


def read_kernel_stats(path):
    """Return {kernel name: (calls, average ns)} from a rocprofv3 kernel stats CSV."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return {}
    header = rows[0]
    name_i = find_column(header, "name")
    calls_i = find_column(header, "calls")
    avg_i = find_column(header, "average")
    if name_i is None or calls_i is None or avg_i is None:
        sys.exit(f"{path}: unexpected columns {header}")
    return {r[name_i]: (int(float(r[calls_i])), float(r[avg_i])) for r in rows[1:] if r}


def read_results(path):
    """Return {(pipeline, impl): row dict} from a hipaug bench CSV."""
    with open(path, newline="") as f:
        return {(r["pipeline"], r["impl"]): r for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("profile_dir")
    ap.add_argument("--md", default="docs/profile.md")
    ap.add_argument("--csv", default="docs/profile.csv")
    ap.add_argument("--peak-gbps", type=float, default=5300.0,
                    help="device peak memory bandwidth in GB/s (MI300X: 5300)")
    args = ap.parse_args()

    batch_dirs = sorted(glob.glob(os.path.join(args.profile_dir, "b*")),
                        key=lambda d: int(os.path.basename(d)[1:]))
    if not batch_dirs:
        sys.exit(f"no b<batch> directories under {args.profile_dir}")

    out_rows = []
    device = ""
    for bdir in batch_dirs:
        batch = int(os.path.basename(bdir)[1:])
        # Byte counts for every (pipeline, impl) at this batch size, from all sessions.
        results = {}
        for path in glob.glob(os.path.join(bdir, "*", "results.csv")):
            results.update(read_results(path))
        for pdir in sorted(glob.glob(os.path.join(bdir, "*"))):
            pipeline = os.path.basename(pdir)
            stats_files = glob.glob(os.path.join(pdir, "**", "*kernel_stats.csv"), recursive=True)
            if not stats_files:
                print(f"warning: no kernel stats in {pdir}", file=sys.stderr)
                continue
            for name, (calls, avg_ns) in sorted(read_kernel_stats(stats_files[0]).items()):
                # Prefer the most specific fragment: fused_rcmn_kernel also contains
                # cmn_kernel. Keep substring matching for decorated profiler names.
                key = next((k for k in sorted(KERNEL_BYTES, key=len, reverse=True)
                            if k in name), None)
                if key is None:
                    continue
                src_pipeline, impl = KERNEL_BYTES[key]
                row = results.get((src_pipeline or pipeline, impl))
                if row is None:
                    print(f"warning: no bench row for {name} in {pdir}", file=sys.stderr)
                    continue
                device = row.get("device", device)
                nbytes = float(row["bytes_read"]) + float(row["bytes_written"])
                gbps = nbytes / (avg_ns * 1e-9) / 1e9
                out_rows.append({
                    "batch": batch, "pipeline": pipeline, "kernel": key, "calls": calls,
                    "avg_us": avg_ns / 1e3, "bytes": int(nbytes), "gb_per_s": gbps,
                    "peak_share_pct": 100.0 * gbps / args.peak_gbps,
                })

    if not out_rows:
        sys.exit("no kernels matched; check the kernel names in the stats files")

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for r in out_rows:
            w.writerow({**r, "avg_us": f"{r['avg_us']:.3f}", "gb_per_s": f"{r['gb_per_s']:.3f}",
                        "peak_share_pct": f"{r['peak_share_pct']:.3f}"})

    with open(args.md, "w") as f:
        f.write("# rocprofv3 kernel profile\n\n")
        if device:
            f.write(f"Device: {device}. ")
        f.write(f"Peak bandwidth used for the share column: {args.peak_gbps:.0f} GB/s.\n\n")
        f.write("| Batch | Pipeline | Kernel | Calls | Avg time (us) | Bytes moved | GB/s | Share of peak |\n")
        f.write("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for r in out_rows:
            f.write(f"| {r['batch']} | {r['pipeline']} | `{r['kernel']}` | {r['calls']} | "
                    f"{r['avg_us']:.1f} | {r['bytes']:,} | {r['gb_per_s']:.1f} | "
                    f"{r['peak_share_pct']:.2f}% |\n")

        # Kernel-only fusion speedup: resize + cmn kernel time over fused kernel time.
        f.write("\n## Fused vs separate, kernel time only\n\n")
        f.write("| Batch | resize + cmn (us) | fused (us) | Speedup |\n| ---: | ---: | ---: | ---: |\n")
        for bdir in batch_dirs:
            batch = int(os.path.basename(bdir)[1:])
            t = {r["kernel"]: r["avg_us"] for r in out_rows
                 if r["batch"] == batch and r["pipeline"] == "resize_cmn"}
            if {"resize_kernel", "cmn_kernel", "fused_rcmn_kernel"} <= t.keys():
                sep = t["resize_kernel"] + t["cmn_kernel"]
                f.write(f"| {batch} | {sep:.1f} | {t['fused_rcmn_kernel']:.1f} | "
                        f"{sep / t['fused_rcmn_kernel']:.2f}x |\n")


if __name__ == "__main__":
    main()
