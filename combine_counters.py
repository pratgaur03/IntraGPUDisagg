#!/usr/bin/env python3
"""
Profile `standalone_attn.py` under rocprofv3 while passing through
custom workload arguments such as
    --prefill-batch 8 --decode-batch 128 --prefill-len 6144 --no-masking

Examples
--------
# keep defaults
python run_rocprof.py

# change workload sizes and disable CU masking
python run_rocprof.py --prefill-batch 8 --decode-batch 128 \
                      --prefill-len 6144 --no-masking
"""
import os, re, subprocess, argparse
from pathlib import Path
import pandas as pd

# ------------ user-tunable defaults -----------------------------------
GPU_ID       = "5"                 # HIP_VISIBLE_DEVICES value for rocprof
METRICS_FILE = Path("metrics.txt") # master metrics list
A_PY         = Path("standalone_attn.py")  # workload to profile
LOG_FILE     = Path("rocprof_runs.log")
# ----------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="rocprofv3 wrapper for standalone_attn.py"
    )
    # workload parameters (must match those accepted in standalone_attn.py)
    p.add_argument("--prefill-batch", type=int, default=4)
    p.add_argument("--decode-batch",  type=int, default=200)
    p.add_argument("--prefill-len",   type=int, default=4096)
    p.add_argument("--iters",         type=int, default=4)
    p.add_argument("--masking", dest="masking", action="store_true")
    p.add_argument("--no-masking", dest="masking", action="store_false")
    p.set_defaults(masking=True)

    # misc
    p.add_argument("--gpu-id", default=GPU_ID,
                   help="Value for HIP_VISIBLE_DEVICES during profiling")
    return p

def main() -> None:
    args = build_arg_parser().parse_args()

    # ---- build list of workload CLI flags to forward -----------------
    wl_args = [
        "--prefill-batch", str(args.prefill_batch),
        "--decode-batch",  str(args.decode_batch),
        "--prefill-len",   str(args.prefill_len),
        "--iters",         str(args.iters),
    ]
    if args.masking is False:        # user chose --no-masking
        wl_args.append("--no-masking")

    # ---- parse metrics.txt  --------------------------
    lines = METRICS_FILE.read_text().splitlines()
    gpu_line  = next(l for l in reversed(lines) if l.startswith("gpu:"))
    pmc_lines = [l for l in lines if l.startswith("pmc:")]
    if not pmc_lines:
        raise ValueError("metrics.txt has no lines starting with 'pmc:'")

    LOG_FILE.unlink(missing_ok=True)

    for idx, pmc in enumerate(pmc_lines, 1):
        mfile = Path(f"metrics_{idx}.txt")
        mfile.write_text(f"{pmc}\n{gpu_line}\n")

        cmd = ["rocprofv3", "-i", str(mfile), "--",
               "python3", str(A_PY), *wl_args]

        env = {**os.environ, "HIP_VISIBLE_DEVICES": args.gpu_id}

        print(f"[+] Run {idx}/{len(pmc_lines)}  cmd: {' '.join(cmd)}")
        with subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True
        ) as p, open(LOG_FILE, "a") as log:
            log.write(f"# ---- Run {idx}: {pmc} ----\n")
            for line in p.stdout:
                log.write(line)
            p.wait()
        if p.returncode not in (0, -11, 139):
            print(f"[!] Unexpected exit code {p.returncode} on run {idx}; continuing")

    # -------- merge counter_collection.csv outputs --------------------
    csv_paths = []
    csv_regex = re.compile(r"(/[^ ]*counter_collection\.csv)")
    with open(LOG_FILE) as log:
        for line in log:
            m = csv_regex.search(line)
            if m:
                csv_paths.append(Path(m.group(1)))
    csv_paths = list(dict.fromkeys(csv_paths))          # deduplicate

    if not csv_paths:
        raise RuntimeError("No counter_collection.csv paths found in log.")

    print(f"[+] Merging {len(csv_paths)} CSVs …")
    df_merged = pd.concat((pd.read_csv(p) for p in csv_paths), ignore_index=True)
    out = Path("all_counter_collections.csv")
    df_merged.to_csv(out, index=False)
    print(f"[✓] Wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")

if __name__ == "__main__":
    main()
