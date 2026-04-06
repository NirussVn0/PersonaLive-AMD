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

This fork replaces the original NVIDIA CUDA + TensorRT pipeline with a **native AMD HIP/ZLUDA backend**, running on AMD GPUs at full hardware utilization through PyTorch's native SDPA (Scaled Dot-Product Attention).

### Migration History

This project has gone through two AMD backend iterations:

| Version | Backend | Status |
|---------|---------|--------|
| v1 | `torch-directml` (Direct3D 12) | Deprecated |
| **v2** | **AMD HIP SDK + ZLUDA** | **Current** |

### Architecture Changes (v2 — HIP/ZLUDA)

| # | Change | Before (DirectML v1) | After (HIP/ZLUDA v2) |
|---|--------|---------------------|----------------------|
| 1 | **HIPAttnProcessor** | `DMLAttnProcessor` — manual sliced attention, fp32 upcast, 256-token chunks | Native `F.scaled_dot_product_attention()` — single kernel dispatch, full fp16 |
| 2 | **Device Backend** | `torch-directml` → `privateuseone:0` | `torch.cuda` via ZLUDA → `cuda:0` |
| 3 | **Attention Overhead** | ~15-30% from fp32 upcast per layer | Near-zero dtype conversion |
| 4 | **VRAM Management** | Manual DML workarounds, partial VAE optimization | Full VAE slicing + tiling, aggressive `gc.collect()` + `empty_cache()` |
| 5 | **Peak VRAM (512x512)** | ~12-14GB | ~8-10GB |
| 6 | **xformers** | Disabled (incompatible with DML) | Replaced by native PyTorch SDPA |

### Preserved Architecture (from v1)

| # | Feature | Description |
|---|---------|-------------|
| 7 | **Threading over Multiprocessing** | Single thread shares model instance, zero serialization overhead |
| 8 | **Lazy Loading UX** | Server starts instantly, frontend shows loading screen |
| 9 | **JIT Warmup** | Automatic VAE warmup pass pre-compiles HIP kernels before first frame |

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

The `HIPAttnProcessor` uses PyTorch 2.x's native `F.scaled_dot_product_attention()` which dispatches to the most efficient kernel available on the hardware:

| Backend | Kernel | Performance |
|---------|--------|-------------|
| HIP/ROCm | Flash Attention (via MIOpen) | Optimal for RDNA2 |
| CUDA | Flash Attention v2 | Optimal for Ampere+ |
| Fallback | Math-based SDPA | Universal, still fast |

This replaces the old `DMLAttnProcessor` which used manual slicing and fp32 upcasting — a workaround that is no longer necessary with proper GPU backend support.

### Threading Architecture

```
┌─── This Fork (threading) ───────────────────────────────────────┐
│ Main Thread: PersonaLive init once → start server → 30-45s      │
│ Worker Thread: shares same instance, zero serialization         │
│ Communication: queue.Queue (direct memory reference)            │
└─────────────────────────────────────────────────────────────────┘
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

---

## ⚙️ AMD HIP/ZLUDA vs NVIDIA CUDA — Technical Comparison

| Feature | NVIDIA (CUDA) | AMD (HIP/ZLUDA) |
|---------|---------------|-----------------|
| Tensor execution | Native GPU | Native GPU |
| Attention precision | FP16 (Flash Attention) | FP16 (SDPA via MIOpen) |
| Kernel optimization | TensorRT (fused ops) | HIP JIT compilation |
| JIT compilation | Pre-compiled CUDA kernels | First-run HIP kernel compilation (warmup handles this) |
| GPU utilization | 100% | 100% |
| VRAM efficiency | Native memory pool | Native memory pool + slicing/tiling |

---

## ⚖️ Disclaimer & License

- This project is released for **academic research only**.
- Users must not use this repository to generate harmful, defamatory, or illegal content.
- The authors bear no responsibility for any misuse or legal consequences arising from the use of this tool.
- By using this code, you agree that you are solely responsible for any content generated.

**Acknowledgment:**
This repository is a modified fork of [PersonaLive](https://github.com/GVCLab/PersonaLive). All core research, model architectures, and original codebase belong to the respective authors: Zhiyuan Li, Chi-Man Pun, Chen Fang, Jue Wang, Xiaodong Cun.

**Source Code Contribution:**
**[NirussVn0](https://github.com/NirussVn0)** — AMD DirectML runtime engine (v1), HIP/ZLUDA migration (v2), custom attention processors, threading architecture, lazy loading system, and JIT warmup implementation.

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
