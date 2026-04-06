# PersonaLive — AMD HIP SDK + ZLUDA Setup Guide (Windows)

Target hardware: **AMD RX 6800** (gfx1030, RDNA2, 16GB VRAM) on **Windows 11**.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Windows | 11 (22H2+) |
| AMD GPU Driver | Adrenalin 24.x+ |
| Python | 3.10.x |
| Conda | Miniconda or Anaconda |
| AMD HIP SDK | 6.x |
| ZLUDA | 0.4+ |

---

## Step 1: Install AMD GPU Driver

Download and install the latest AMD Adrenalin driver from
[AMD Support](https://www.amd.com/en/support).

Reboot after installation.

---

## Step 2: Install AMD HIP SDK 6.x

1. Download from [AMD ROCm for Windows](https://www.amd.com/en/developer/resources/rocm-hub.html)
2. Run the installer, select **HIP SDK** components
3. After installation, verify:

```powershell
hipconfig --version
```

Expected output: `6.x.xxxxx`

4. Add HIP SDK `bin` directory to your system PATH if not done automatically:

```
C:\Program Files\AMD\ROCm\6.x\bin
```

---

## Step 3: Install ZLUDA

ZLUDA translates CUDA API calls to HIP, allowing PyTorch's CUDA backend to run on AMD GPUs.

1. Download ZLUDA from [GitHub Releases](https://github.com/vosen/ZLUDA/releases)
2. Extract the archive
3. Copy these DLLs from the ZLUDA archive to your Python environment's root directory:
   - `nvcuda.dll`
   - `nvml.dll`

For conda environments, the target directory is typically:

```
%USERPROFILE%\miniconda3\envs\personalive\
```

4. Verify ZLUDA is in place:

```powershell
python -c "import ctypes; ctypes.CDLL('nvcuda')"
```

---

## Step 4: Create Conda Environment

```powershell
conda create -n personalive python=3.10 -y
conda activate personalive
```

---

## Step 5: Install PyTorch with ROCm Support

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

---

## Step 6: Set Environment Variables

These are critical for RX 6800 (gfx1030). Set them system-wide via PowerShell:

```powershell
[System.Environment]::SetEnvironmentVariable("HSA_OVERRIDE_GFX_VERSION", "10.3.0", "User")
[System.Environment]::SetEnvironmentVariable("HIP_VISIBLE_DEVICES", "0", "User")
[System.Environment]::SetEnvironmentVariable("PYTORCH_HIP_ALLOC_CONF", "garbage_collection_threshold:0.9,max_split_size_mb:512", "User")
[System.Environment]::SetEnvironmentVariable("HIP_FORCE_DEV_KERNARG", "1", "User")
```

Or use the `run_online.ps1` launcher which sets these automatically per-session.

---

## Step 7: Install PersonaLive Dependencies

```powershell
cd PersonaLive
pip install -r requirements_base.txt
```

---

## Step 8: Download Model Weights

```powershell
python tools/download_weights.py
```

---

## Step 9: Verify Installation

Run the verification script:

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0)); print('VRAM:', torch.cuda.get_device_properties(0).total_mem // 1024**2, 'MB')"
```

Expected output:

```
CUDA available: True
Device: AMD Radeon RX 6800
VRAM: 16384 MB
```

---

## Step 10: Launch PersonaLive

### One-Click Launch (Recommended)

```powershell
.\run_online.ps1
```

### Manual Launch

```powershell
conda activate personalive
$env:HSA_OVERRIDE_GFX_VERSION = "10.3.0"
python inference_online.py --acceleration none
```

Open `http://localhost:7860` in your browser.

### Offline Inference

```powershell
python inference_offline.py --config configs/prompts/personalive_offline.yaml --reference_image demo/ref_image.png --driving_video demo/driving_video.mp4
```

---

## Troubleshooting

### "No CUDA GPUs are available"

1. Verify ZLUDA DLLs are in the correct directory
2. Verify `HSA_OVERRIDE_GFX_VERSION` is set to `10.3.0`
3. Try running with `$env:HIP_VISIBLE_DEVICES = "0"`

### "hipErrorNoBinaryForGpu"

Your `HSA_OVERRIDE_GFX_VERSION` does not match your GPU. For RX 6800 (gfx1030), use `10.3.0`.

| GPU | gfx ID | HSA_OVERRIDE_GFX_VERSION |
|-----|--------|--------------------------|
| RX 6600/XT | gfx1032 | 10.3.0 |
| RX 6700 XT | gfx1031 | 10.3.0 |
| RX 6800/XT | gfx1030 | 10.3.0 |
| RX 6900 XT | gfx1030 | 10.3.0 |
| RX 7600 | gfx1102 | 11.0.0 |
| RX 7800 XT | gfx1101 | 11.0.0 |
| RX 7900 XT/XTX | gfx1100 | 11.0.0 |

### Out of Memory (OOM)

1. Close other GPU-consuming applications (browsers with hardware acceleration, etc.)
2. Reduce inference resolution in the YAML config
3. Verify `PYTORCH_HIP_ALLOC_CONF` is set correctly

### Slow First Inference

The first inference frame compiles HIP kernels (JIT). Subsequent frames will be significantly faster. The `run_online.ps1` launcher includes a warmup pass that pre-compiles these kernels.

---

## Performance Comparison

| Metric | DirectML (old) | HIP/ZLUDA (new) |
|--------|---------------|-----------------|
| Attention kernel | Sliced fp32 (256 tokens/slice) | Native SDPA fp16 (single dispatch) |
| Attention overhead | ~15-30% from fp32 upcast | Near-zero |
| VAE decode | Standard | Sliced + Tiled |
| Memory management | Manual DML workarounds | Native CUDA/HIP memory pool |
| Peak VRAM (512x512) | ~12-14GB | ~8-10GB |
| First frame latency | 5-10s (shader compilation) | 2-4s (HIP JIT) |
