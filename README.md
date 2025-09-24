# xla_baseline

_OpenXLA compiler benchmarks and debugging tools_

This repository provides small, focused scripts to:
- run **PyTorch baselines**,
- execute the same models through **PyTorch/XLA** (PJRT),
- **export** models to **StableHLO** bundles,
- and **compile/run** StableHLO via the XLA compiler/runtime.

---

## Repository layout

```

xla\_baseline/
├─ env/           # environment helpers, e.g., Docker or Conda definitions
├─ models/        # small reference models (see below)
├─ scripts/       # all entry-point scripts
└─ utils/         # helpers

````

> Note: The exact contents of `env/` and `utils/` may evolve; see inline comments in those files.

---

## Prerequisites

- **Python**: ≥ 3.10 recommended  
- **PyTorch / torch-xla**: Match major versions (e.g., `torch 2.8.x` with `torch_xla 2.8.x`)

---

## Installation

Clone the repository:
```bash
git clone https://github.com/HaeeunJeong/xla_baseline.git
cd xla_baseline
````

If you target CPU, use **Option A**. For CUDA, use **Option B** or **Option C**.

### Option A) Create a Conda environment

```bash
conda env create -f env/torch-xla-cpu/environment.yml
conda activate torch-xla-cpu
```

### Option B) Use the official prebuilt Docker image

PyTorch/XLA publishes prebuilt Docker images. List available tags with:

```bash
gcloud artifacts docker tags list us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla
```

Install the Google Cloud SDK as needed: [gcloud SDK install](https://cloud.google.com/sdk/docs/install?hl=ko).

**Example (tested):** Tag `nightly_3.11_cuda_12.8_20250407`

```bash
docker run \
  --net=host \
  --gpus all \
  --shm-size=16g \
  --name torch-xla-prebuilt \
  -itv <host_dir>:<container_dir> \
  -d us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.11_cuda_12.8_20250407 /bin/bash

docker exec -it torch-xla-prebuilt /bin/bash
```

Install CUDA Toolkit inside the container if required: [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
```

Reinstall PyTorch if needed:

```bash
pip uninstall -y torch
pip install torch
```

### Option C) Build PyTorch/XLA from source

**Status:** Work in progress.
Reference guides: [PyTorch/XLA GPU guide](https://docs.pytorch.org/xla/master/gpu.html), [Contributing](https://github.com/pytorch/xla/blob/master/CONTRIBUTING.md).

---

## Quick start: scripts

All entry points live under `scripts/`.

> **Common PJRT environment variables**
>
> * `PJRT_DEVICE=CUDA` for NVIDIA GPUs.

### 1) `pytorch_baseline.py`

Baseline performance using native **PyTorch** (no XLA).

```bash
# CPU
python scripts/pytorch_baseline.py --device cpu <MODEL>

# CUDA (if available)
python scripts/pytorch_baseline.py --device cuda <MODEL>
```

Notes:

* Default `--device` is `cuda`.
* If `<MODEL>` is omitted, all models in `models/` are used.
* Save results to CSV with `--csv_path <PATH>`.

---

### 2) `compare_xla_torch.py`

Compare execution under native PyTorch vs. PyTorch/XLA.

```bash
PJRT_DEVICE=CUDA \
python scripts/compare_xla_torch.py <MODEL>
```

---

### 3) `compile_run_xla.py`

Lower a `torch.nn.Module` to **HLO** and execute with the **XLA compiler/runtime**.

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

Load a **StableHLO bundle** and run it through the **XLA compiler/runtime** (PJRT).

```bash
PJRT_DEVICE=CUDA \
python scripts/shlo_compile_run.py \
  --bundle ./bundles/<MODEL>.stablehlo \
  --batch 32
```

It loads the bundle, invokes PJRT to compile and execute, and prints timing and optional checks.

---

## Models

All reference models live in `models/`.

**Supported models**

| Category        | Key         | Source                                      | Notes                                    |
| --------------- | ----------- | ------------------------------------------- | ---------------------------------------- |
| Simple custom   | `mm`        | `models/mm_block.py`                        | Simple matrix multiplication kernel      |
| Simple custom   | `conv`      | `models/conv_block.py`                      | Conv–Flatten–Linear–ReLU toy block       |
| **GNN**         | `gcn`       | `models/gcn_block.py`                       | Graph Convolution Network                |
| **GNN**         | `graphsage` | PyTorch Geometric                           | GraphSAGE                                |
| **GNN**         | `gat`       | PyTorch Geometric                           | Graph Attention Network                  |
| **GNN**         | `gatv2`     | PyTorch Geometric                           | Graph Attention Network v2               |
| **CNN**         | `resnet`    | `torchvision` ResNet-18                     | Image backbone, ImageNet pretrained      |
| **CNN**         | `mobilenet` | `torchvision` MobileNet v3-S                | Mobile-oriented CNN, ImageNet pretrained |
| **Transformer** | `vit`       | `torchvision` ViT-B/16                      | Vision Transformer baseline              |
| **Transformer** | `bert`      | `bert-base-uncased`                         | Token-level encoder                      |
| **Transformer** | `gpt2`      | `gpt2`                                      | Decoder baseline                         |
| **LLM**         | `llama`     | `meta-llama/Llama-3.2-1B`                   | Compact general-purpose LLM              |
| **LLM**         | `deepseek`  | `Deepseek-ai/deepseek-R1-Distill-Qwen-1.5B` | Distilled math-centric LLM               |

Utilities `get_model(name)` and `get_dummy_input(name)` are used by scripts.

### Enable Hugging Face models

First log in to Hugging Face and create a token with `read` scope.

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
```

### Add a new model

1. Implement the model under `models/` (e.g., `my_model.py`).
2. Expose factories:

```python
def get_model(): ...
def get_dummy_input(): ...

# For Hugging Face models also import:
from ._hf_wrapper import HFWrapper
```

---

## Tips: XLA debugging and dumps

**Quick IR/HLO dump**

```bash
export XLA_FLAGS="--xla_dump_to=./out"
```

Writes unoptimized/optimized HLO and related artifacts per compilation.

**Auto-metrics and profiler hints**

```bash
export PT_XLA_DEBUG_LEVEL=2
```

Prints summaries of compile/execute counts, transfer hot spots, and ops not lowered to XLA.

**Append multiple XLA flags**

```bash
export XLA_FLAGS="${XLA_FLAGS} --xla_dump_hlo_as_text --xla_gpu_enable_latency_hiding_scheduler=true"
```
