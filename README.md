<div align="center">

<img src="assets/header.svg" alt="PersonaLive" width="100%">

<h2>PersonaLive - 🔴 AMD HIP/ZLUDA Edition </h2>

> **Forked from the original [PersonaLive](https://github.com/GVCLab/PersonaLive).** 
> *Expressive Portrait Image Animation for Live Streaming — Optimized for native AMD GPU execution (RX 6000/7000 Series) on Windows via AMD HIP SDK + ZLUDA.*

#### [Zhiyuan Li<sup>1,2,3</sup>](https://huai-chang.github.io/) · [Chi-Man Pun<sup>1,📪</sup>](https://cmpun.github.io/) · [Chen Fang<sup>2</sup>](http://fangchen.org/) · [Jue Wang<sup>2</sup>](https://scholar.google.com/citations?user=Bt4uDWMAAAAJ&hl=en) · [Xiaodong Cun<sup>3,📪</sup>](https://vinthony.github.io/academic/) 
<sup>1</sup> University of Macau  &nbsp;&nbsp; <sup>2</sup> [Dzine.ai](https://www.dzine.ai/)  &nbsp;&nbsp; <sup>3</sup> [GVC Lab, Great Bay University](https://gvclab.github.io/)

<a href='https://arxiv.org/abs/2512.11253'><img src='https://img.shields.io/badge/ArXiv-2512.11253-red'></a> <a href='https://huggingface.co/huaichang/PersonaLive'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffc107'></a> <a href='https://modelscope.cn/models/huaichang/PersonaLive'><img src='https://img.shields.io/badge/ModelScope-Model-624AFF'></a> [![GitHub](https://img.shields.io/github/stars/GVCLab/PersonaLive?style=social)](https://github.com/GVCLab/PersonaLive)

<img src="assets/highlight.svg" alt="highlight" width="95%">

<img src="assets/demo_3.gif" width="46%"> &nbsp;&nbsp; <img src="assets/demo_2.gif" width="40.5%">
</div>

## 💡 What's Different in this Fork?

This fork replaces the original NVIDIA CUDA + TensorRT pipeline with a **native AMD HIP/ZLUDA backend**, enabling full hardware-accelerated inference on AMD GPUs through ZLUDA's CUDA→HIP translation layer.

### Key Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | **HIPAttnProcessor** | Native `F.scaled_dot_product_attention()` — single kernel dispatch, full FP16 precision, zero manual slicing |
| 2 | **ZLUDA Backend** | AMD GPU exposed as `cuda:0` — seamless PyTorch CUDA API compatibility without code changes |
| 3 | **~8-10GB Peak VRAM** | VAE slicing + tiling + aggressive memory management keeps 512×512 inference well within 16GB |
| 4 | **Threading Architecture** | Single-thread model sharing, zero serialization — no redundant model loading or pickle overhead |
| 5 | **Lazy Loading UX** | Server starts instantly, frontend shows loading screen while models warm up in background |
| 6 | **HIP JIT Warmup** | Automatic VAE warmup pass pre-compiles HIP kernels before first user frame — eliminates cold-start stutter |
| 7 | **One-Click Launcher** | `run_online.ps1` handles HIP env vars, conda activation, port detection, and browser auto-open |

---

## 🛠️ Installation for AMD Users (Windows)

> For detailed step-by-step instructions, see **[hip_setup_guide.md](hip_setup_guide.md)**.

**1. Install Prerequisites**
- AMD HIP SDK 6.x from [AMD ROCm Hub](https://www.amd.com/en/developer/resources/rocm-hub.html)
- ZLUDA 0.4+ from [GitHub Releases](https://github.com/vosen/ZLUDA/releases)

**2. Create Conda Environment**
```bash
conda create -n personalive python=3.10 -y
conda activate personalive
```

**3. Install PyTorch with ROCm**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

**4. Install Base Requirements**
```bash
pip install -r requirements_base.txt
```

**5. Download Weights**
```bash
python tools/download_weights.py
```

**6. Set Environment Variables** (Critical for RX 6800)
```powershell
$env:HSA_OVERRIDE_GFX_VERSION = "10.3.0"
```

---

## 🚀 Running Offline Inference

```bash
python inference_offline.py \
  --config configs/prompts/personalive_offline.yaml \
  --reference_image demo/ref_image.png \
  --driving_video demo/driving_video.mp4
```

Device is auto-detected. Override with `--device cuda:0` or `--device cpu`.

---

## 📸 Online Inference (WebUI)

### 📦 Setup Web UI
```bash
cd webcam/frontend
npm install
npm run build
cd ../..
```

### ▶️ One-Click Start (Recommended)
```powershell
.\run_online.ps1
```

### ▶️ Manual Start
```bash
python inference_online.py --acceleration none
```
Open `http://localhost:7860` — the UI loads instantly while models warm up in the background.

---

## 🏗️ Architecture Deep Dive

### `HIPAttnProcessor` — Native SDPA Attention

Every attention layer in the diffusion UNet computes: **Q @ Kᵀ × scale → softmax → @ V**.

The `HIPAttnProcessor` uses PyTorch 2.x `F.scaled_dot_product_attention()` which dispatches to the most efficient kernel available:

| Backend | Kernel | Notes |
|---------|--------|-------|
| HIP/ROCm | Flash Attention (via MIOpen) | Optimal for RDNA2/RDNA3 |
| CUDA | Flash Attention v2 | For NVIDIA fallback |
| Fallback | Math-based SDPA | Universal compatibility |

Full FP16 precision throughout — no dtype upcasting overhead. Single kernel dispatch per attention layer instead of chunked loops.

### Threading Architecture

```
┌─── PersonaLive Threading Model ────────────────────────────────┐
│ Main Thread: PersonaLive init once → start server → 30-45s     │
│ Worker Thread: shares same instance, zero serialization        │
│ Communication: queue.Queue (direct memory reference)           │
└────────────────────────────────────────────────────────────────┘
```

PyTorch GPU operations **release the GIL** during kernel execution. The inference thread runs GPU-bound work (matmul, conv2d, etc.) without blocking the WebSocket I/O thread.

### Lazy Loading + JIT Warmup

```
t=0.0s  Server starts (FastAPI + Uvicorn)     → User opens browser, sees loading screen
t=0.1s  Frontend polls GET /api/status         → { is_ready: false, status: "Loading models..." }
t=30s   Models loaded, VAE warmup pass runs    → { is_ready: false, status: "Warming up HIP/ZLUDA..." }
t=35s   HIP kernels pre-compiled               → { is_ready: true, status: "Ready" }
        Frontend transitions to app UI         → User selects portrait, starts streaming
```

### VRAM Management Strategy

| Optimization | Effect |
|-------------|--------|
| VAE Slicing | Processes VAE decode in sequential slices instead of full batch |
| VAE Tiling | Tiles large images to avoid OOM on high resolutions |
| `flush_vram()` | `gc.collect()` + `torch.cuda.empty_cache()` after every model load |
| FP16 Pipeline | Full half-precision inference — no FP32 intermediate buffers |
| `PYTORCH_HIP_ALLOC_CONF` | Tuned garbage collection threshold (0.9) and max split size (512MB) |

---

## ⚙️ Supported AMD GPUs

| GPU | gfx ID | HSA_OVERRIDE_GFX_VERSION | VRAM |
|-----|--------|--------------------------|------|
| RX 6600/XT | gfx1032 | 10.3.0 | 8GB |
| RX 6700 XT | gfx1031 | 10.3.0 | 12GB |
| RX 6800/XT | gfx1030 | 10.3.0 | 16GB |
| RX 6900 XT | gfx1030 | 10.3.0 | 16GB |
| RX 7600 | gfx1102 | 11.0.0 | 8GB |
| RX 7800 XT | gfx1101 | 11.0.0 | 16GB |
| RX 7900 XT/XTX | gfx1100 | 11.0.0 | 20/24GB |

---

## ⚖️ Disclaimer & License

- This project is released for **academic research only**.
- Users must not use this repository to generate harmful, defamatory, or illegal content.
- The authors bear no responsibility for any misuse or legal consequences arising from the use of this tool.
- By using this code, you agree that you are solely responsible for any content generated.

**Acknowledgment:**
This repository is a modified fork of [PersonaLive](https://github.com/GVCLab/PersonaLive). All core research, model architectures, and original codebase belong to the respective authors: Zhiyuan Li, Chi-Man Pun, Chen Fang, Jue Wang, Xiaodong Cun.

**Source Code Contribution:**
**[NirussVn0](https://github.com/NirussVn0)** — AMD HIP/ZLUDA runtime engine, HIPAttnProcessor, threading architecture, lazy loading system, JIT warmup, and one-click launcher.

This project is licensed under the **Apache License 2.0**. See the `LICENSE` file for more details.

---

## 🎬 More Results & Original Visualizations
<table width="100%">
  <tr>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/cdc885ef-5e1c-4139-987a-2fa50fefd6a4" controls="controls" style="max-width: 100%; display: block;"></video>
    </td>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/014f7bae-74ce-4f56-8621-24bc76f3c123" controls="controls" style="max-width: 100%; display: block;"></video>
    </td>
  </tr>
</table>

## ⭐ Citation
If you find PersonaLive useful for your research, welcome to cite our work using the following BibTeX:
```bibtex
@article{li2025personalive,
  title={PersonaLive! Expressive Portrait Image Animation for Live Streaming},
  author={Li, Zhiyuan and Pun, Chi-Man and Fang, Chen and Wang, Jue and Cun, Xiaodong},
  journal={arXiv preprint arXiv:2512.11253},
  year={2025}
}
```
