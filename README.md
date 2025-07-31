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

`standalone_attn.py` its very basic currently. You will have to change the parameters like batch sizes and prefill lens. You can specifiy -s masking

```bash
# Basic run
$ python standalone_attn.py \
        -s masking
```

## Collecting & combining metrics

To gather all counters into a single CSV for analysis:

```bash
$ python3 combine_counters.py 
```

Modify metrics.txt file if necessary


Licensed under the MIT License – see [`LICENSE`](LICENSE) for details.
