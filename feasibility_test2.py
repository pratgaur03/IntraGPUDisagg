#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
# ------------- user-tunable defaults -----------------------------------
# WORKLOADS = [
#     Path("standalone_attn_prefill_3d.py"),
#     Path("standalone_attn_decode_3d.py"),
#     Path("standalone_attn_prefill_and_decode_3d.py"),
# ]
# CSV_PATH  = Path("./attention_kernel.csv")
# LOG_FILE  = Path("rocprof_runs_2.log")

WORKLOADS = [
    Path("standalone_attn_prefill.py")
]
prefill_batch=[1]
prefill_len=[2048]
cu_mask=[np.nan,32,64,96,128,160]
LOG_FILE  = Path("rocprof_runs2.log")
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
        print("No masking")
    else:
        args += ["--decode-mask", str(int(row["CU mask"]))]
    return args


def main() -> None:
    for b in prefill_batch:
        for l in prefill_len:
            for c in cu_mask:
                wl_args = [
                    "--prefill-batch", str(b),
                    "--prefill-len",   str(l),
                    "--iters",         "5"
                ]
                if type(c)!=int:
                    wl_args.append("--no-masking")
                    print("No masking")
                else:
                    wl_args += ["--decode-mask", str(c)]

                tag = (
                    f"testing"
                    f"{b}_{l}_"
                    f"{c}"
                )

                for script in WORKLOADS:
                    trace_name = f"{script.stem}_{tag}"

                    cmd = [
                        "rocprofv3", "--kernel-trace",
                        "-d", "./profiles",
                        "-o", trace_name,
                        "--", "python3", str(script), *wl_args,
                    ]
                    # cmd = [
                    #     "rocprofv3", "-i", "metrics_2.txt",
                    #     "-d", "./profiles",
                    #     "-o", trace_name,
                    #     "--", "python3", str(script), *wl_args,
                    # ]
                    

                    # (Re-)create log file per run
                    # LOG_FILE.unlink(missing_ok=True)

                    env = {**os.environ, "HIP_VISIBLE_DEVICES": "1"}

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
                            f"on {script.name} continuing"
                        )


if __name__ == "__main__":
    main()
