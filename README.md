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
├─ env/           # env helpers, e.g., docker or conda environments
├─ models/        # small reference models (see below)
├─ scripts/       # all entry-point scripts
└─ utils/         # helpers

````

> Note: The exact contents of `env/` and `utils/` may evolve; see inline comments in those files.

---

## Prerequisites

- **Python**: 3.10> recommended
- **PyTorch / torch-xla**: Major version of PyTorch and torch-xla should be matched (e.g., torch 2.8.x - torch_xla 2.8.x)

---

## Installation
Clone the repository.
```bash
git clone https://github.com/HaeeunJeong/xla_baseline.git
cd xla_baseline
````
If you want to use for CPU, go to **Option A**, otherwise if you want to use for CUDA, go to **Option B** or **C**

### Option A) Create Conda environment
```bash
conda env create -f env/torch-xla-cpu/environment.yml
conda activate torch-xla-cpu
````

### Option B) Use the official prebuilt Docker image
Torch_XLA provides prebuilt docker images, and you can get tag list by this command below. 
(Install [gcloud](https://cloud.google.com/sdk/docs/install?hl=ko) by following this instruction. )
```bash
gcloud artifacts docker tags list us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla
````
<br/>

I'm currently using `nightly_3.11_cuda_12.8_20250407` tagged docker image. The run command is below. 
```bash
docker run \
--net=host \
--gpus all \
--shm-size=16g \
--name torch-xla-prebuilt \
-itv {host_system_dir}:{container_dir} \
-d us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.11_cuda_12.8_20250407 /bin/bash

docker exec -it torch-xla-prebuilt /bin/bash
````
Then, install [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
````
And uninstall torch, and reinstall torch.
```bash
pip uninstall torch
pip install torch
````

### Option C) Build PyTorch/XLA from source
> **Status:** *Work in progress.*
> I tried to build by following these links, but the OpenXLA/xla's dependencies cannot be resolved.

[How to run with PyTorch/XLA:GPU](https://docs.pytorch.org/xla/master/gpu.html)

[Contribute To PyTorch/XLA](https://github.com/pytorch/xla/blob/master/CONTRIBUTING.md)

---

## Quick start: scripts

All entry points live under `scripts/`.

> **Environment variables** commonly used with PJRT:
>
> * `PJRT_DEVICE=CUDA` (for NVIDIA GPU), etc.

### 1) `pytorch_baseline.py`

Baseline performance using **native PyTorch** backend (no XLA). Run on CPU or CUDA directly.

```bash
# CPU
python scripts/pytorch_baseline.py --device cpu <MODEL>

# CUDA (if available)
python scripts/pytorch_baseline.py --device cuda <MODEL>
```
- Default `--device` value is `cuda`
- If you do not specify `<MODEL>`, all models in `models/` directory will be chosen.
- If you want to store result to .csv format, using `--csv_path <PATH>` 

---

### 2) `compare_xla_torch.py`



```bash
# Intended usage:
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

**🏗️ Supported models**

| Category        | Key                 | Source                                                | Notes                                                                         |
| --------------- | ------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------- |
| Simple custom   | `mm` | `models/mm_block.py` | Simple matrix multiplication kernel |
| Simple custom   | `conv` | `models/conv_block.py` | Conv-Flatten-Linear-ReLU toy block |
| **GNN**         | `gcn` | `models/gcn_block.py` | Graph Convolution Net |
| **GNN**         | `graphsage` | PyTorch Geometric | Graph Sage Net; The milestone of GNN model|
| **GNN**         | `gat` | PyTorch Geometric | Graph Attention Net |
| **GNN**         | `gatv2` | PyTorch Geometric | Graph Attention Net |
| **CNN**         | `resnet` | `torchvision` ResNet‑18 | Classic image backbone; ImageNet pretrained |
| **CNN**         | `mobilenet` | `torchvision` MobileNet v3 S | Mobile‑oriented CNN; ImageNet pretrained |
| **Transformer** | `vit` | `torchvision` ViT‑B/16| Vision Transformer baseline |
| **Transformer** | `bert` | `bert-base-uncased`| Token‑level encoder|
| **Transformer** | `gpt2` | `gpt2`| 1.5 B‑param decoder |
| **LLM**         | `llama` | `meta-llama/Llama-3.2-1B`| Base Llama 3.2 1 B; compact general‑purpose LLM|
| **LLM**         | `deepseek`| `Deepseek-ai/deepseek-R1-Distill-Qwen-1.5B"`| Distilled math‑centric LLM; complex TF graph|

* Utility functions `get_model(name)` and `get_dummy_input(name)` used by scripts.


### Enable huggingface models
Firstly, login huggingface and create new token, token type `read`
```bash
pip install -U "huggingface_hub[cli]"
hf auth login
```


### Add a new model
1. Implement it under `models/` (e.g., `my_model.py`).
2. Expose a factory like:
```python
def get_model(): ...
def get_dummy_input(): ...

# If you want to huggingface model, add this above
from ._hf_wrapper import HFWrapper
```

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
