# xla_baseline

_OpenXLA compiler benchmarks & debugging tools_

This repository contains small, focused scripts to:
- run **PyTorch baselines**,
- execute the same models through **PyTorch/XLA** (PJRT),
- **export** models to **StableHLO** bundles,
- and **compile/run** StableHLO through the XLA compiler/runtime.

---
## Repository layout

```

xla\_baseline/
├─ env/           # (env helpers, e.g., docker or setup notes if present)
├─ models/        # small reference models (see below)
├─ scripts/       # all entry-point scripts
└─ utils/         # helpers (timing, model loader, I/O, etc.)

````

> Note: The exact contents of `env/` and `utils/` may evolve; see inline comments in those files.

---

## Prerequisites

- **Python**: 3.10–3.11 recommended
- **PyTorch**: a version compatible with your chosen **torch-xla**
- **torch-xla / PJRT**:
  - **CUDA** (NVIDIA) or other PJRT backends supported by OpenXLA
  - For CUDA: a matching **CUDA toolkit / cuDNN** on the host (if running natively)
- Optional for dumps/visualization:
  - **graphviz** (`dot`) if you plan to render `.dot` graphs

---

## Installation

```bash
git clone https://github.com/HaeeunJeong/xla_baseline.git
cd xla_baseline
````

Install **torch-xla** either from **source** (A) or using the **official Docker** image (B). Follow PyTorch/XLA’s recommendations for your platform.

### Option A) Build PyTorch/XLA from source

This path gives you full control and latest changes, but takes longer.

1. Install a PyTorch build compatible with your target (CUDA / CPU).
2. Build and install `torch_xla` following the instructions in the official PyTorch/XLA README (match versions!).
3. Verify:

```bash
python - <<'PY'
import torch, torch_xla
print("torch:", torch.__version__)
print("torch_xla:", torch_xla.__version__)
PY
```

### Option B) Use the official Docker image

This is the quickest way to get a working PJRT/XLA environment for GPU.

Example (CUDA-enabled host):

```bash
# Make sure you have NVIDIA Container Toolkit installed on the host.
# Then run an interactive container with GPUs and mount the repo.
sudo docker run -it --rm \
  --gpus all \
  --name xla_gpu \
  -w /work \
  -v "$PWD":/work \
  us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest \
  bash

# Inside the container:
python -c "import torch, torch_xla; print(torch.__version__, torch_xla.__version__)"
```

### Option C) Use XLA on CPU

This is the quickest way to get a working PJRT/XLA environment for CPU.

```bash
conda env create --file env/xla-cpu/environment.yaml
conda activate xla-cpu
```

---

## Quick start: scripts

All entry points live under `scripts/`. Use `--help` on each script to see available flags (batch size, device, model name, etc.).

> **Environment variables** commonly used with PJRT:
>
> * `PJRT_DEVICE=CUDA` (for NVIDIA GPU), etc.

### 1) `pytorch_baseline.py`

Baseline performance using **native PyTorch** backend (no XLA). Run on CPU or CUDA directly.

```bash
# CUDA (if available)
python scripts/pytorch_baseline.py --device cuda <MODEL>
```
---

### 2) `compare_xla_torch.py`

> **Status:** *Work in progress.*
> Goal: compare **torch backend** vs **XLA backend** on **GPU target**.
> **Known issue:** the “torch backend” code-path currently falls back to **CPU**.

```bash
# Intended usage (WIP):
PJRT_DEVICE=CUDA \
python scripts/compare_xla_torch.py <MODEL>
```
---

### 3) `compile_run_xla.py`

Lower a `torch.nn.Module` **directly to HLO** and execute with **XLA compiler/runtime**.

```bash
PJRT_DEVICE=CUDA \
python scripts/compile_run_xla.py <MODEL>
```
---

### 4) `export_torch_xla_stablehlo.py`

Export a PyTorch module to a **StableHLO bundle** (bytecode + weights).

```bash
python scripts/export_torch_xla_stablehlo.py <MODEL>
```
---
### 5) `shlo_compile_run.py`

Take a **StableHLO bundle** and run it through the **XLA compiler/runtime** (PJRT).

```bash
PJRT_DEVICE=CUDA \
python scripts/shlo_compile_run.py \
  --bundle ./bundles/<MODEL>.stablehlo \
  --batch 32
```

**What it does:**

* loads the StableHLO bundle,
* invokes PJRT to compile and execute,
* prints timing and (optionally) numerical checks.

---

## Models

All reference models live in `models/`. Typical items you’ll find include:

* **Tiny CNNs / MLPs** for smoke tests (e.g., simple Conv→Conv→ReLU blocks),
* Minimal transformer-ish blocks to stress matmul/softmax paths,
* Utility functions `get_model(name)` and `get_dummy_input(name)` used by scripts.

> If you add a new model:
>
> 1. Implement it under `models/` (e.g., `my_model.py`).
> 2. Expose a factory like:
>
>    ```python
>    def get_model(): ...
>    def get_dummy_input(): ...
>    ```
> 3. Register the model name in the script’s model registry (if applicable).

---

## Tips: XLA debugging & dumps

* **Quick IR/HLO dump**

  ```bash
  export XLA_FLAGS="--xla_dump_to=./out"
  ```

  This writes unoptimized/optimized HLO and other artifacts per compilation.

* **Auto-metrics / profiler hints**

  ```bash
  export PT_XLA_DEBUG_LEVEL=2
  ```

  Prints a summary of compile/execute counts, transfer hot spots, and any ops not lowered to XLA.

* **Append multiple XLA flags**

  ```bash
  export XLA_FLAGS="${XLA_FLAGS} --xla_dump_hlo_as_text --xla_gpu_enable_latency_hiding_scheduler=true"
  ```

---

## Troubleshooting

* **Torch and torch-xla version mismatch**

  * Ensure the **major/minor** versions match the PyTorch/XLA release matrix.
* **CUDA backend not used / runs on CPU**

  * Verify `PJRT_DEVICE=CUDA` and that the container/host sees GPUs (`nvidia-smi`).
  * Check PyTorch reports a CUDA device in eager mode.
* **Compilation too frequent (slow)**

  * Avoid shape polymorphism; keep tensor shapes stable per step.
  * Cache or reuse exported StableHLO bundles for repeated runs.
* **Missing ops (not lowered)**

  * The metrics report will list `aten::` ops routed to CPU. Consider model tweaks or newer torch-xla.
