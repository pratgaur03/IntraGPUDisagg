#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
# ------------- user-tunable defaults -----------------------------------
WORKLOADS = [
    Path("standalone_attn_prefill.py"),
]
prefill_batch=[1, 4, 8,16]
prefill_len=[256,512,1024,2048,4096,8192]
cu_mask=[np.nan,128]
LOG_FILE  = Path("rocprof_runs0.log")
# CSV_PATH  = Path("./attention_kernel.csv")
# LOG_FILE  = Path("rocprof_runs_2d.log")
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
    if type(row["CU mask"])!=int or row["CU mask"]==np.nan or row["CU mask"]=="NA":
        args.append("--no-masking")
    else:
        args += ["--decode-mask", str(int(row["CU mask"]))]
    return args


def main() -> None:
    # df = pd.read_csv(CSV_PATH)

    # for idx, row in df.iterrows():
        # wl_args = build_wl_args(row)

        # A tag that captures the parameter combo in a filename-safe way
        # tag = (
        #     f"{row['Prefill Batch']}_{row['Prefill Len']}_"
        #     f"{row['Decode batch size']}_{row['Decode len']}_"
        #     f"{row['CU mask']}_tp8_unified_attention_2d"
        # )
    for pb in prefill_batch:
        for pl in prefill_len:
                    for c in cu_mask:

                        wl_args = [
                            "--prefill-batch", str(pb),
                            "--prefill-len",   str(pl),
                            "--iters",         "5"
                        ]
                        if type(c)!=int:
                            wl_args.append("--no-masking")
                            print("No masking")
                        else:
                            wl_args += ["--decode-mask", str(c)]

                        tag = (
                            f"write_size_{pb}_{pl}_"
                            f"{c}"
                        )

                        for script in WORKLOADS:
                            trace_name = f"{script.stem}_{tag}"

                            cmd = [
                                "rocprofv3",
                                "-i", "/app/IntraGPUDisagg/metrics_1.txt",
                                "-d", "./hwcount",
                                "-o", trace_name,
                                "--", "python3", str(script), *wl_args,
                            ]
                            # cmd = [
                            #     "rocprofv3", "--kernel-trace",
                            #     "-d", "./profiles",
                            #     "-o", trace_name,
                            #     "--", "python3", str(script), *wl_args,
                            # ]

                            # (Re-)create log file per run
                            # LOG_FILE.unlink(missing_ok=True)

                            env = {**os.environ, "HIP_VISIBLE_DEVICES": "0"}

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
                                    f"on {script.name}  continuing"
                                )


if __name__ == "__main__":
    main()
