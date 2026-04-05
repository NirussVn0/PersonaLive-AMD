# PersonaLive - AMD DirectML Edition 🔴🚀

> **Forked from the original [PersonaLive](https://github.com/GVCLab/PersonaLive).** 
> *Expressive Portrait Image Animation for Live Streaming - Optimized & Patched for AMD GPUs (RX 6000/7000 Series) on Windows.*

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

## ⚖️ Acknowledgment & License
This repository is a modified fork of [PersonaLive](https://github.com/GVCLab/PersonaLive). All core research, model architectures, and original codebase belong to the respective authors: Zhiyuan Li, Chi-Man Pun, Chen Fang, Jue Wang, Xiaodong Cun.

This project is licensed under the **Apache License 2.0**. See the `LICENSE` file for more details.
