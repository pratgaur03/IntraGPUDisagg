#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
import pandas as pd

# ------------- user-tunable defaults -----------------------------------
WORKLOADS = [
    Path("standalone_attn_prefill.py"),
    Path("standalone_attn_decode.py"),
    Path("standalone_attn_prefill_and_decode.py"),
]
CSV_PATH  = Path("./Decode Mask Experiment.csv")
LOG_FILE  = Path("rocprof_runs.log")
# -----------------------------------------------------------------------


def build_wl_args(row) -> list[str]:
    """Return the list of CLI flags to forward to the workload."""
    args = [
        "--prefill-batch", str(int(row["Prefill Batch"])),
        "--prefill-len",   str(int(row["Prefill Len"])),
        "--decode-batch",  str(int(row["Decode batch size"])),
        "--decode-len",    str(int(row["Decode len"])),
        "--iters",         "5",
       
    ]
    if type(row["CU mask"])!=int:
        args.append("--no-masking")
    else:
        args += ["--decode-mask", str(int(row["CU mask"]))]
    return args


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    for idx, row in df.iterrows():
        wl_args = build_wl_args(row)

        # A tag that captures the parameter combo in a filename-safe way
        tag = (
            f"{row['Prefill Batch']}_{row['Prefill Len']}_"
            f"{row['Decode batch size']}_{row['Decode len']}_"
            f"{row['CU mask']}"
        )

        for script in WORKLOADS:
            trace_name = f"{script.stem}_{tag}"

            cmd = [
                "rocprofv3", "--kernel-trace",
                "-d", "./profiles",
                "-o", trace_name,
                "--", "python3", str(script), *wl_args,
            ]

            # (Re-)create log file per run
            # LOG_FILE.unlink(missing_ok=True)

            env = {**os.environ, "HIP_VISIBLE_DEVICES": "5"}

            with subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            ) as p, open(LOG_FILE, "a") as log:
                log.write(f"# ---- Run {trace_name} ----\n")
                for line in p.stdout:
                    log.write(line)
                p.wait()

            if p.returncode not in (0, -11, 139):
                print(
                    f"[!] Unexpected exit code {p.returncode} "
                    f"on {script.name} (row {idx}); continuing"
                )


if __name__ == "__main__":
    main()
