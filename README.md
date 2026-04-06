<div align="center">

<img src="assets/header.svg" alt="PersonaLive" width="100%">

<h2>PersonaLive - 🔴 AMD DirectML Edition </h2>

> **Forked from the original [PersonaLive](https://github.com/GVCLab/PersonaLive).** 
> *Expressive Portrait Image Animation for Live Streaming — Patched for native AMD GPU execution (RX 6000/7000 Series) on Windows via Microsoft DirectML.*

#### [Zhiyuan Li<sup>1,2,3</sup>](https://huai-chang.github.io/) · [Chi-Man Pun<sup>1,📪</sup>](https://cmpun.github.io/) · [Chen Fang<sup>2</sup>](http://fangchen.org/) · [Jue Wang<sup>2</sup>](https://scholar.google.com/citations?user=Bt4uDWMAAAAJ&hl=en) · [Xiaodong Cun<sup>3,📪</sup>](https://vinthony.github.io/academic/) 
<sup>1</sup> University of Macau  &nbsp;&nbsp; <sup>2</sup> [Dzine.ai](https://www.dzine.ai/)  &nbsp;&nbsp; <sup>3</sup> [GVC Lab, Great Bay University](https://gvclab.github.io/)

<a href='https://arxiv.org/abs/2512.11253'><img src='https://img.shields.io/badge/ArXiv-2512.11253-red'></a> <a href='https://huggingface.co/huaichang/PersonaLive'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffc107'></a> <a href='https://modelscope.cn/models/huaichang/PersonaLive'><img src='https://img.shields.io/badge/ModelScope-Model-624AFF'></a> [![GitHub](https://img.shields.io/github/stars/GVCLab/PersonaLive?style=social)](https://github.com/GVCLab/PersonaLive)

<img src="assets/highlight.svg" alt="highlight" width="95%">

<img src="assets/demo_3.gif" width="46%"> &nbsp;&nbsp; <img src="assets/demo_2.gif" width="40.5%">
</div>

## 💡 What's Different in this Fork?

The original PersonaLive was built for NVIDIA CUDA + TensorRT. This fork replaces every CUDA-exclusive operation with **DirectML-compatible equivalents**, making it run natively on AMD GPUs at full GPU utilization through `torch-directml`.

### Runtime Patches

| # | Patch | Problem | Solution |
|---|-------|---------|----------|
| 1 | **DMLAttnProcessor** | All 3 built-in diffusers attention processors (`AttnProcessor2_0`, `AttnProcessor`, `SlicedAttnProcessor`) crash on DirectML — they use `F.scaled_dot_product_attention()` or `torch.baddbmm()`, neither of which has a DML kernel | Custom processor using only `torch.matmul()` + `softmax()` with FP32 upcast for numerical stability |
| 2 | **FP16 Model Loading** | `AutoencoderKL` and `UNet` default to FP32, causing 16GB+ RAM freeze | Forced `torch_dtype=torch.float16` on all `.from_pretrained()` calls |
| 3 | **5D Tensor CPU Offload** | DirectML crashes on 5D tensor manipulations (`.repeat()`, `.unsqueeze()`) | Temporarily offload to CPU, compute, push back to VRAM |
| 4 | **Distance Metric Rewrite** | `torch.cdist` triggers a fatal padding bug in DML's compiler | Replaced with `torch.norm` Euclidean distance |
| 5 | **LayerNorm Device Fix** | Custom `_apply` in LivePortrait utils fails on DirectML device transfer | Rewritten to handle `privateuseone` device properly |
| 6 | **Deprecated API Cleanup** | Custom UNet uses `_remove_lora` and `return_deprecated_lora` args removed in diffusers 0.27+ | Removed deprecated kwargs from `set_attn_processor()` and `get_processor()` |

### Architecture Upgrades

| # | Change | Before | After |
|---|--------|--------|-------|
| 7 | **Threading over Multiprocessing** | Child process loads all models a 2nd time from disk (60-90s startup, `OpaqueTensorImpl` crash on Queue serialization) | Single thread shares the same model instance (30-45s startup, zero serialization) |
| 8 | **Lazy Loading UX** | Server blocks until models finish loading — user sees "Connection Refused" for 30s+ | Server starts instantly, frontend shows loading screen, models load in background thread |
| 9 | **DML JIT Warmup** | First inference frame freezes 5-10×  longer while DirectML compiles shaders | Automatic dummy VAE pass after load pre-compiles all shaders before user interaction |

---

## 🛠️ Installation for AMD Users (Windows)

**1. Create Conda Environment**
```bash
conda create -n personalive python=3.10
conda activate personalive
```

**2. Install PyTorch with DirectML**
```bash
pip install torch torchvision torchaudio torch-directml
```

**3. Install Base Requirements**
*(Important: Remove `torch` and `torchvision` from `requirements_base.txt` before running this to avoid downgrading!)*
```bash
pip install -r requirements_base.txt
```

**4. Download Weights**
```bash
python tools/download_weights.py
```

---

## 🚀 Running Offline Inference (DirectML)
Use the `--device privateuseone:0` flag to force AMD execution.

```bash
python inference_offline.py \
  --config configs/prompts/personalive_offline.yaml \
  --reference_image demo/ref_image.png \
  --driving_video demo/driving_video.mp4 \
  --device privateuseone:0
```

---

## 📸 Online Inference (WebUI)
#### 📦 Setup Web UI
```bash
cd webcam/frontend
npm install
npm run build
cd ../..
```

#### ▶️ Start Streaming
```bash
python inference_online.py --acceleration none
```
Open `http://localhost:7860` — the UI loads instantly while models warm up in the background.

---

## 🏗️ Architecture Deep Dive

### `DMLAttnProcessor` — Why a Custom Attention Kernel?

Every attention layer in the diffusion UNet computes: **Q @ Kᵀ × scale → softmax → @ V**.

The 3 built-in diffusers processors implement this differently — and all 3 crash on DirectML:

| Processor | Implementation | Why it crashes on DML |
|-----------|---------------|----------------------|
| `AttnProcessor2_0` | `F.scaled_dot_product_attention()` | DML has no SDPA kernel |
| `AttnProcessor` | `torch.empty()` + `torch.baddbmm()` | DML has no `baddbmm` kernel |
| `SlicedAttnProcessor` | Sliced `torch.baddbmm()` | Same — still `baddbmm` |

Our `DMLAttnProcessor` uses only two fundamental ops that every GPU backend supports:

```python
scores = torch.matmul(Q.float(), K.float().transpose(-1, -2).contiguous()) * scale
probs  = scores.softmax(dim=-1).to(original_dtype)
output = torch.matmul(probs, V)
```

**FP32 upcast** is required because DirectML's FP16 batched matmul has kernel gaps for attention-sized tensors. This adds ~15-30% overhead on attention layers only. The upcast pattern is the same as diffusers' built-in `upcast_attention` flag — well-tested and numerically stable.

### Threading Architecture

```
┌─── Original (multiprocessing) ──────────────────────────────────┐
│ Main Process: spawn child → wait 60-90s                         │
│ Child Process: torch.load × 8 from disk → PersonaLive init     │
│ Communication: Queue.put(tensor) → pickle → OpaqueTensorImpl 💀 │
└─────────────────────────────────────────────────────────────────┘

┌─── This Fork (threading) ───────────────────────────────────────┐
│ Main Thread: PersonaLive init once → start server → 30-45s      │
│ Worker Thread: shares same instance, zero serialization         │
│ Communication: queue.Queue (direct memory reference)            │
└─────────────────────────────────────────────────────────────────┘
```

Why threading works here: PyTorch GPU operations **release the GIL** during kernel execution. The inference thread runs GPU-bound work (matmul, conv2d, etc.) without blocking the WebSocket I/O thread. No GIL contention.

Why multiprocessing broke: DirectML tensors are **opaque** — they have no accessible storage and cannot be serialized via Python's pickle protocol. `Queue.put(dml_tensor)` throws `NotImplementedError: Cannot access storage of OpaqueTensorImpl`. Threading eliminates serialization entirely.

### Lazy Loading + JIT Warmup

```
t=0.0s  Server starts (FastAPI + Uvicorn)     → User opens browser, sees loading screen
t=0.1s  Frontend polls GET /api/status         → { is_ready: false, status: "Loading models..." }
t=30s   Models loaded, VAE warmup pass runs    → { is_ready: false, status: "Warming up DirectML..." }
t=35s   DML shaders pre-compiled               → { is_ready: true, status: "Ready" }
        Frontend transitions to app UI         → User selects portrait, starts streaming
```

The JIT warmup runs a dummy `VAE encode → decode` pass immediately after model loading. This forces DirectML to compile all VAE shader programs before the user's first real frame — eliminating the "First Frame Freeze" where the GPU spends 5-10× longer on initial inference.

---

## ⚙️ AMD vs NVIDIA — Technical Comparison

DirectML is a **native GPU execution layer** built on Direct3D 12. All tensor operations execute directly on AMD GPU compute units at full hardware utilization — it is NOT CPU emulation or a software fallback.

| Feature | NVIDIA (CUDA) | AMD (DirectML) |
|---------|---------------|----------------|
| Tensor execution | Native GPU | Native GPU |
| Attention precision | FP16 (hardware-optimized) | FP32 upcast needed (~15-30% attention overhead) |
| Kernel optimization | TensorRT (fused ops, INT8 quantization) | Raw PyTorch ops (no fusion layer available) |
| JIT compilation | Pre-compiled CUDA kernels | First-run shader compilation (warmup handles this) |
| Inter-process tensor sharing | CUDA IPC (shared memory) | Not available → Threading instead |
| 5D tensor ops | Full native support | CPU offloading required |
| GPU utilization | 100% | 100% |

> **Key insight:** The original repo's "real-time 30fps" performance comes from **TensorRT acceleration** — a software optimization layer on top of CUDA that fuses kernels and quantizes to INT8. Without TensorRT, vanilla CUDA PyTorch is also significantly slower. The performance gap between AMD and NVIDIA on this project is primarily a **TensorRT vs no-TensorRT** gap, not a hardware gap.
>
> This fork has been tested and validated on **AMD RX 6000/7000 series GPUs** on Windows. Actual FPS depends on your GPU model, VRAM size, and resolution settings.

---

## ⚖️ Disclaimer & License

- This project is released for **academic research only**.
- Users must not use this repository to generate harmful, defamatory, or illegal content.
- The authors bear no responsibility for any misuse or legal consequences arising from the use of this tool.
- By using this code, you agree that you are solely responsible for any content generated.

**Acknowledgment:**
This repository is a modified fork of [PersonaLive](https://github.com/GVCLab/PersonaLive). All core research, model architectures, and original codebase belong to the respective authors: Zhiyuan Li, Chi-Man Pun, Chen Fang, Jue Wang, Xiaodong Cun.

**Source Code Contribution:**
**[NirussVn0](https://github.com/NirussVn0)** — AMD DirectML runtime engine, custom attention processor, threading architecture, lazy loading system, and JIT warmup implementation.

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
