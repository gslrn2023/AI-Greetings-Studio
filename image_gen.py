import os
import gc

# Prevent CUDA memory fragmentation on constrained GPUs (must be set before importing torch)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
import config

# Cross-platform ZeroGPU compatibility wrapper (strictly checks HF ZeroGPU environment variable)
IS_ZEROGPU = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("true", "1")
try:
    import spaces
    gpu_decorator = spaces.GPU(duration=65)
except (ImportError, AttributeError):
    def gpu_decorator(fn):
        return fn

pipe = None


def _is_low_vram_cuda() -> bool:
    """
    Returns True if running on a CUDA GPU (like Kaggle/Azure T4 14.56 GiB or local GPU)
    that is NOT managed by HF ZeroGPU and has less than 22 GiB total VRAM.
    """
    if IS_ZEROGPU or not torch.cuda.is_available():
        return False
    try:
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return total_vram_gb < 22.0
    except Exception:
        return False


def _enable_memory_efficient_vae(pipeline):
    """
    Enables VAE slicing and tiling both at the pipeline level and directly on pipeline.vae
    so 1088x1920 latents decode in small tiles instead of allocating a 7.44 GiB tensor.
    """
    if hasattr(pipeline, "enable_vae_slicing"):
        try:
            pipeline.enable_vae_slicing()
        except Exception:
            pass
    if hasattr(pipeline, "enable_vae_tiling"):
        try:
            pipeline.enable_vae_tiling()
        except Exception:
            pass
    if hasattr(pipeline, "vae") and pipeline.vae is not None:
        if hasattr(pipeline.vae, "enable_slicing"):
            try:
                pipeline.vae.enable_slicing()
            except Exception:
                pass
        if hasattr(pipeline.vae, "enable_tiling"):
            try:
                pipeline.vae.enable_tiling()
            except Exception:
                pass


def _apply_device_placement(pipeline, device: str, low_vram: bool):
    """
    Applies full-speed pipe.to(device) on HF ZeroGPU / high-VRAM GPUs,
    or sequential/model CPU offload + explicit VAE tiling on <22 GiB GPUs.
    """
    if device == "cuda" and low_vram:
        print("Low-VRAM GPU detected (<22 GiB): enabling memory-efficient VAE & CPU offload...")
        _enable_memory_efficient_vae(pipeline)
        try:
            pipeline.enable_sequential_cpu_offload()
            print("Enabled sequential (layer-by-layer) CPU offload.")
        except Exception as e:
            print(f"Sequential offload unavailable ({e}), using model CPU offload...")
            pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device)


def _init_pipeline():
    global pipe
    if pipe is not None:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    low_vram = _is_low_vram_cuda()

    if low_vram:
        gc.collect()
        torch.cuda.empty_cache()

    # Use float16 on older architectures (like T4 sm_75) if bf16 isn't native, else bfloat16
    if device == "cuda":
        dtype = (
            torch.bfloat16
            if (IS_ZEROGPU or torch.cuda.is_bf16_supported())
            else torch.float16
        )
    else:
        dtype = torch.float32

    if config.IMAGE_ENGINE == "Z_IMAGE":
        print("Loading Tongyi-MAI/Z-Image-Turbo pipeline...")
        from diffusers import ZImagePipeline
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=dtype,
            low_cpu_mem_usage=True if low_vram else False,
            token=config.HF_TOKEN or None
        )
        _apply_device_placement(pipe, device, low_vram)
        print("Z-Image-Turbo loaded successfully!")
    else:
        print("Loading black-forest-labs/FLUX.1-schnell pipeline...")
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=dtype,
            low_cpu_mem_usage=True if low_vram else False,
            token=config.HF_TOKEN or None
        )
        _apply_device_placement(pipe, device, low_vram)
        print("FLUX.1-schnell loaded successfully!")


# Initialize the selected engine at boot (required for HF ZeroGPU snapshot pre-warming)
_init_pipeline()


@gpu_decorator
def generate_artwork(prompt: str) -> Image.Image:
    """Generates a native 1088x1920 9:16 vertical artwork using the active engine."""
    print(f"[{config.IMAGE_ENGINE}] Generating 9:16 artwork: {prompt}")
    low_vram = _is_low_vram_cuda()

    if low_vram:
        gc.collect()
        torch.cuda.empty_cache()

    steps = config.Z_IMAGE_STEPS if config.IMAGE_ENGINE == "Z_IMAGE" else config.FLUX_STEPS

    try:
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                width=config.IMAGE_WIDTH,
                height=config.IMAGE_HEIGHT,
                num_inference_steps=steps,
                guidance_scale=0.0
            ).images[0]
        return image
    finally:
        if low_vram:
            gc.collect()
            torch.cuda.empty_cache()