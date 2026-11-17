import os
import gc
import torch
from PIL import Image
import config

# Cross-platform ZeroGPU compatibility wrapper (works on HF, Kaggle, Azure, Docker, or local GPU)
IS_ZEROGPU = False
try:
    import spaces
    gpu_decorator = spaces.GPU(duration=65)
    IS_ZEROGPU = (
        os.environ.get("SPACES_ZERO_GPU", "").lower() in ("true", "1")
        or hasattr(spaces, "GPU")
    )
except (ImportError, AttributeError):
    def gpu_decorator(fn):
        return fn

pipe = None

def _is_low_vram_cuda() -> bool:
    """
    Returns True if running on a real CUDA GPU (like Kaggle T4 14.56 GiB)
    that is NOT managed by HF ZeroGPU and has less than 22 GiB total VRAM.
    """
    if IS_ZEROGPU or not torch.cuda.is_available():
        return False
    try:
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return total_vram_gb < 22.0
    except Exception:
        return False

def _apply_device_placement(pipeline, device: str, low_vram: bool):
    """
    Applies full-speed pipe.to(device) on HF ZeroGPU / high-VRAM GPUs,
    or model CPU offload + VAE tiling on constrained GPUs (e.g., Kaggle T4).
    """
    if device == "cuda" and low_vram:
        print("Low-VRAM GPU detected (<22 GiB): enabling model CPU offload & VAE slicing...")
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
        try:
            pipeline.enable_model_cpu_offload()
        except Exception as e:
            print(f"Model offload fallback ({e}), using sequential CPU offload...")
            pipeline.enable_sequential_cpu_offload()
    else:
        pipeline.to(device)

def _init_pipeline():
    global pipe
    if pipe is not None:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    low_vram = _is_low_vram_cuda()

    if low_vram:
        torch.cuda.empty_cache()
        gc.collect()

    # Use float16 on older architectures (like Kaggle T4 sm_75) if bf16 isn't native, else bfloat16
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
    if config.IMAGE_ENGINE == "Z_IMAGE":
        image = pipe(
            prompt=prompt,
            width=config.IMAGE_WIDTH,
            height=config.IMAGE_HEIGHT,
            num_inference_steps=config.Z_IMAGE_STEPS,
            guidance_scale=0.0
        ).images[0]
    else:
        image = pipe(
            prompt=prompt,
            width=config.IMAGE_WIDTH,
            height=config.IMAGE_HEIGHT,
            num_inference_steps=config.FLUX_STEPS,
            guidance_scale=0.0
        ).images[0]
    return image