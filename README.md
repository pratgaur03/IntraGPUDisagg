# IntraGPUDisagg
This repository demonstrates how to launch the two streams separately with `standalone_attn.py`, measure kernel- and hardware‑level metrics, and aggregate them with `combine_counters.py`.

---

## 📝 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Running the demo](#running-the-demo)
3. [Collecting & combining metrics](#collecting--combining-metrics)

---

## Prerequisites

1.  Rocm
2.  vllm
---


## Running the demo

```bash
# defaults (same as your hard-coded values)
python standalone_attn.py
```

```bash
# custom sizes
python run_unified_attn.py --prefill-batch 8 --decode-batch 128 \
                           --prefill-len 6144
```
```bash
# disable CU masking
python run_unified_attn.py --no-masking
```


## Collecting & combining metrics

To gather all counters into a single CSV for analysis:

```bash
#default
$ python3 combine_counters.py 
```

```bash
# change workload sizes and disable CU masking
python run_rocprof.py --prefill-batch 8 --decode-batch 128 \
                      --prefill-len 6144 --no-masking
```

Modify metrics.txt file if necessary

