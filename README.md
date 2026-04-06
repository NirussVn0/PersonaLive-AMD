<div align="center">

<img src="assets/header.svg" alt="PersonaLive" width="100%">

<h2>PersonaLive - 🔴 AMD DirectML Edition </h2>

> **Forked from the original [PersonaLive](https://github.com/GVCLab/PersonaLive).** 
> *Expressive Portrait Image Animation for Live Streaming - Optimized & Patched for AMD GPUs (RX 6000/7000 Series) on Windows.*

#### [Zhiyuan Li<sup>1,2,3</sup>](https://huai-chang.github.io/) · [Chi-Man Pun<sup>1,📪</sup>](https://cmpun.github.io/) · [Chen Fang<sup>2</sup>](http://fangchen.org/) · [Jue Wang<sup>2</sup>](https://scholar.google.com/citations?user=Bt4uDWMAAAAJ&hl=en) · [Xiaodong Cun<sup>3,📪</sup>](https://vinthony.github.io/academic/) 
<sup>1</sup> University of Macau  &nbsp;&nbsp; <sup>2</sup> [Dzine.ai](https://www.dzine.ai/)  &nbsp;&nbsp; <sup>3</sup> [GVC Lab, Great Bay University](https://gvclab.github.io/)

<a href='https://arxiv.org/abs/2512.11253'><img src='https://img.shields.io/badge/ArXiv-2512.11253-red'></a> <a href='https://huggingface.co/huaichang/PersonaLive'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffc107'></a> <a href='https://modelscope.cn/models/huaichang/PersonaLive'><img src='https://img.shields.io/badge/ModelScope-Model-624AFF'></a> [![GitHub](https://img.shields.io/github/stars/GVCLab/PersonaLive?style=social)](https://github.com/GVCLab/PersonaLive)

<img src="assets/highlight.svg" alt="highlight" width="95%">

<img src="assets/demo_3.gif" width="46%"> &nbsp;&nbsp; <img src="assets/demo_2.gif" width="40.5%">
</div>

## 💡 What's Different in this Fork?
The original PersonaLive heavily relies on NVIDIA's CUDA, xformers, and TensorRT. Running it natively on an AMD GPU throws multiple VRAM overflows and architecture errors. This fork applies **Surgical Patches** to run the inference pipelines via **Microsoft DirectML (`torch-directml`)**:

1. **Memory & VRAM Optimization:** Forced FP16 `torch_dtype` loading for `AutoencoderKL` and `UNet` models to prevent 16GB+ System RAM freezing during initialization.
2. **LayerNorm Compatibility Patch:** Rewrote the custom `_apply` function in `LayerNorm` (LivePortrait utils) to properly handle device offloading, preventing DirectML crashes (`incompatible tensor type`).
3. **5D Tensor CPU Offloading:** DirectML fundamentally crashes on 5D Tensor manipulations (`.repeat()`, `.unsqueeze()`). We patched `pipeline_pose2vid.py` to temporarily offload 5D operations to the CPU before pushing them back to the VRAM.
4. **Attention Slicing:** Hard-enabled `pipe.enable_attention_slicing()` to prevent VRAM overflow (`baddbmm` errors) during 3D Temporal Attention generation.
5. **Distance Metric Fix:** Rewrote the Euclidean Distance Calculation (replacing `torch.cdist` with `torch.norm`) to bypass a fatal padding bug in AMD's DirectML compilation library.
6. **Disabled Xformers:** Hardcoded fallbacks to PyTorch native SDPA for `privateuseone` execution.

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
> **Note:** Initializing models into AMD VRAM takes time without a progress bar. Please be patient. Inference speed depends on your GPU (e.g., RX 6800 takes ~2 mins/frame).

---

## 📸 Online Inference (WebUI)
#### 📦 Setup Web UI
```bash
# install Node.js 18+ (if not using web_start.sh)
cd webcam/frontend
npm install
npm run build
cd ../..
```

#### ▶️ Start Streaming
Run the Web UI with DirectML acceleration explicitly enabled for AMD processors.
```bash
python inference_online.py --acceleration none
```
Then open `http://localhost:7860` in your browser.

> 💡 The Web UI starts **instantly** — you'll see a loading screen while models load in the background. Once "Ready" appears, select a reference portrait and start streaming.

---

## 🏗️ Architecture — AMD DirectML Runtime Engine

### Why a Custom Architecture?

The original PersonaLive pipeline was designed around NVIDIA-exclusive primitives: `F.scaled_dot_product_attention()` (SDPA), `torch.baddbmm()`, CUDA shared memory for `multiprocessing`, and TensorRT JIT compilation. **None of these work on AMD DirectML.** Rather than patching around each crash, we re-engineered the runtime layer to work natively with DirectML's constraints.

### `DMLAttnProcessor` — Custom Attention Kernel

| Processor | Backend | Status on DirectML |
|-----------|---------|-------------------|
| `AttnProcessor2_0` (diffusers default) | `F.scaled_dot_product_attention()` | ❌ `RuntimeError: The parameter is incorrect` |
| `AttnProcessor` (legacy) | `torch.empty()` + `torch.baddbmm()` | ❌ Same crash — DML has no `baddbmm` kernel |
| `SlicedAttnProcessor` | Sliced `baddbmm` | ❌ Still uses `baddbmm` internally |
| **`DMLAttnProcessor`** (ours) | `torch.matmul()` + `softmax()` | ✅ Works — basic ops only |

All three built-in diffusers attention processors rely on operations that DirectML does not support. Our `DMLAttnProcessor` computes attention using only fundamental linear algebra:

```
Q @ Kᵀ × scale → softmax → @ V
```

Key design decisions:
- **FP32 upcast for attention scores** — DirectML's FP16 batched matmul is unstable for attention-sized tensors. We upcast Q/K to FP32, compute scores + softmax in FP32, then cast back. This mirrors diffusers' own `upcast_attention` pattern.
- **`.contiguous()` after transpose** — DirectML kernels cannot handle non-contiguous memory layouts from `.transpose()`. Explicit contiguity prevents silent corruption.
- **Injected via `set_attn_processor()`** — Clean integration through diffusers' existing processor abstraction. No monkey-patching.

### Threading over Multiprocessing

The original codebase uses Python `multiprocessing.Process` to run inference in a child process. This creates a critical problem on DirectML:

| Aspect | `multiprocessing` (original) | `threading` (ours) |
|--------|-------|---------|
| Model loading | 2× (parent + child both `torch.load` from disk) | 1× (shared instance) |
| Startup time | ~60-90s (double I/O) | ~30-45s (single load) |
| Tensor serialization | Must pickle through Queue → `OpaqueTensorImpl` crash | Direct memory sharing, no serialization |
| VRAM usage | 2× model copies in VRAM | 1× single copy |
| GIL concern | None (separate processes) | None (PyTorch GPU ops release GIL) |

The multiprocessing approach was necessary on CUDA because `torch.cuda` tensors can be shared across processes via CUDA IPC. DirectML tensors (`OpaqueTensorImpl`) have no IPC mechanism and **cannot be pickled**. Threading solves this entirely — the inference thread shares the same `PersonaLive` instance, eliminating redundant loads and serialization.

### Lazy Loading UX — Instant Server Start

Instead of blocking the web server until models finish loading (30+ seconds of "Connection Refused"), we:

1. **Server starts immediately** (~0.1s) — FastAPI + Uvicorn launch before any model touches disk
2. **Background thread** loads `PersonaLive` models asynchronously
3. **`/api/status` endpoint** exposes `{ is_ready, status }` for the frontend to poll
4. **Frontend loading screen** shows a real-time progress indicator with stages:
   - `⚙️ Initializing...` → `🧠 Loading models...` → `🔥 Warming up DirectML...` → `✅ Ready`
5. **Reference upload guarded** with HTTP 503 until pipeline is ready

This is the same pattern used by Midjourney, ChatGPT, and other production AI services — physically the machine still needs 30s to load, but psychologically the user sees an immediate response.

### DirectML JIT Shader Warmup

AMD DirectML compiles neural network operations into GPU shader code on first execution (JIT compilation). This causes the **"First Frame Freeze"** — the first inference call takes 5-10x longer than subsequent ones.

After model loading completes, we automatically run a dummy inference pass:
```python
dummy = torch.randn(1, 3, 256, 256, device=device, dtype=dtype)
vae.encode(dummy) → vae.decode(latent)
```

This pre-compiles all VAE shaders before the user's first real frame, eliminating perceived lag.

### AMD vs NVIDIA — Compatibility Assessment

| Feature | NVIDIA (CUDA) | AMD (DirectML, this fork) |
|---------|---------------|---------------------------|
| Attention mechanism | SDPA / FlashAttention / xformers | `DMLAttnProcessor` (matmul + softmax) |
| Memory management | `torch.cuda.empty_cache()` | Manual (no equivalent API) |
| Process parallelism | CUDA IPC shared tensors | ❌ Not possible → Threading |
| JIT compilation | Instant (pre-compiled kernels) | First-run shader compilation (warmup needed) |
| 5D tensor ops | Native support | ❌ CPU offloading required |
| `torch.cdist` | Native | ❌ Replaced with `torch.norm` |
| FP16 attention | Stable (hardware-optimized) | Requires FP32 upcast |
| Performance (RX 6800 vs RTX 3070) | ~Real-time (30fps) | ~2-5 fps (with optimizations) |

> **⚠️ Honest Assessment:** This fork is **functional but not performant** compared to CUDA. DirectML is a compatibility layer, not a performance layer. The architectural changes here (FP32 upcast, CPU offloading, threading) trade raw speed for stability. This fork is best suited for **development, testing, and proof-of-concept** on AMD hardware. For production real-time streaming, NVIDIA CUDA remains the recommended path.
>
> The NVIDIA (CUDA) version does **not** need these patches — it should use the [original repository](https://github.com/GVCLab/PersonaLive) with TensorRT acceleration for optimal performance.

---

## ⚖️ Disclaimer & License

- This project is released for **academic research only**.
- Users must not use this repository to generate harmful, defamatory, or illegal content.
- The authors bear no responsibility for any misuse or legal consequences arising from the use of this tool.
- By using this code, you agree that you are solely responsible for any content generated.

**Acknowledgment:**
This repository is a modified fork of [PersonaLive](https://github.com/GVCLab/PersonaLive). All core research, model architectures, and original codebase belong to the respective authors: Zhiyuan Li, Chi-Man Pun, Chen Fang, Jue Wang, Xiaodong Cun.

**Source Code Contribution:**
I, **NirussVn0**, have contributed to reinforcing and patching the source code to natively support **AMD Team Red (DirectML)** architecture, solving significant VRAM constraints entirely on local consumer hardware.

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
