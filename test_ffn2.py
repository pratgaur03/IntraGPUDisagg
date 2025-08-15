#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
# ------------- user-tunable defaults -----------------------------------
WORKLOADS = [
    # Path("standalone_ffn_prefill.py")
    Path("standalone_ffn.py")
]
decode_batch=[1,2,4,8,16,32,64,128]
cu_mask=[np.nan,32,64,96,128,160]
prefill_batch=[1,2,4,8,16]
prefill_len=[256,512,1024,2048,4096,8192]
LOG_FILE  = Path("rocprof_ffn2.log")
# -----------------------------------------------------------------------


def main() -> None:
    script=Path("standalone_ffn.py")
    for db in decode_batch:
        for pb in prefill_batch:
            for pl in prefill_len:
                for c in cu_mask:

                    wl_args = [
                        "--decode-batch",  str(db),
                        "--prefill-batch",  str(pb),
                        "--prefill-len",  str(pl),
                        "--iters",         "5"
                    ]
                    if type(c)!=int:
                        wl_args.append("--no-masking")
                        print("No masking")
                    else:
                        wl_args += ["--decode-mask", str(c)]

                    tag = (
                        f"tp8_{pb}_{pl}_{db}_{c}"
                    )

                    # for script in WORKLOADS:
                    trace_name = f"{script.stem}_{tag}"

                    cmd = [
                        "rocprofv3", "--kernel-trace",
                        "-d", "./ffn",
                        "-o", trace_name,
                        "--", "python3", str(script), *wl_args,
                    ]

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
