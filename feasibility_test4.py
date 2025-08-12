#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
# ------------- user-tunable defaults -----------------------------------
WORKLOADS = [
    Path("standalone_attn_decode.py")
]
decode_batch=[32,64,128]
decode_len=[256,512,1024,2048,4096,8192]
cu_mask=[np.nan,32,64,96,128,160]
LOG_FILE  = Path("rocprof_runs3.log")
# -----------------------------------------------------------------------


def main() -> None:
    for b in decode_batch:
        for l in decode_len:
            for c in cu_mask:

                wl_args = [
                    "--decode-batch",  str(b),
                    "--decode-len",    str(l),
                    "--iters",         "5"
                ]
                if type(c)!=int:
                    wl_args.append("--no-masking")
                    print("No masking")
                else:
                    wl_args += ["--decode-mask", str(c)]

                tag = (
                    f"tp8_"
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
                    #     "rocprofv3", "-i", "metrics_1.txt",
                    #     "-d", "./profiles",
                    #     "-o", trace_name,
                    #     "--", "python3", str(script), *wl_args,
                    # ]


                    # (Re-)create log file per run
                    # LOG_FILE.unlink(missing_ok=True)

                    env = {**os.environ, "HIP_VISIBLE_DEVICES": "2"}

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
