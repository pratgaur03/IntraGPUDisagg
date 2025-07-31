#!/usr/bin/env python3
"""
Usage:  python run_rocprof.py               # assumes metrics.txt & a.py are nearby
Requirements: pandas ≥ 1.0
"""

import os, re, subprocess
from pathlib import Path
import pandas as pd

GPU_ID        = "5"                 # HIP_VISIBLE_DEVICES value
METRICS_FILE  = Path("metrics.txt") # master list you already have
A_PY          = Path("try.py")        # the workload to profile
LOG_FILE      = Path("rocprof_runs.log")

def main() -> None:
    lines = METRICS_FILE.read_text().splitlines()
    try:
        gpu_line = next(l for l in reversed(lines) if l.startswith("gpu:"))
    except StopIteration:
        raise ValueError("metrics.txt needs one line starting with 'gpu:'")

    pmc_lines = [l for l in lines if l.startswith("pmc:")]
    if not pmc_lines:
        raise ValueError("metrics.txt has no lines starting with 'pmc:'")

    LOG_FILE.unlink(missing_ok=True)      # start fresh

    for idx, pmc in enumerate(pmc_lines, 1):
        mfile = Path(f"metrics_{idx}.txt")
        mfile.write_text(f"{pmc}\n{gpu_line}\n")

        cmd = ["rocprofv3", "-i", str(mfile), "--", "python3", str(A_PY)]
        env = {**os.environ, "HIP_VISIBLE_DEVICES": GPU_ID}

        print(f"[+] Run {idx}/{len(pmc_lines)}: {mfile}")
        with subprocess.Popen(
                cmd, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True) as p, \
             open(LOG_FILE, "a") as log:
            log.write(f"# ---- Run {idx}: {pmc} ----\n")
            for line in p.stdout:                       # stream to log live
                log.write(line)
            p.wait()
        if p.returncode not in (0, -11, 139):
        # 0  = success,  -11/139 = seg-fault (signal 11)
            print(f"[!] Unexpected exit code {p.returncode} on run {idx}; continuing")
    # -------- post-process unified log --------
    csv_paths = []
    csv_regex = re.compile(r"(/[^ ]*counter_collection\.csv)")
    with open(LOG_FILE) as log:
        for line in log:
            m = csv_regex.search(line)
            if m:
                csv_paths.append(Path(m.group(1)))
    csv_paths = list(dict.fromkeys(csv_paths))          # deduplicate/preserve order
    if not csv_paths:
        raise RuntimeError("Did not find any counter_collection.csv paths in log.")

    print(f"[+] Merging {len(csv_paths)} CSVs …")
    df_merged = pd.concat((pd.read_csv(p) for p in csv_paths), ignore_index=True)
    out = Path("all_counter_collections.csv")
    df_merged.to_csv(out, index=False)
    print(f"[✓] Wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")

if __name__ == "__main__":
    main()

