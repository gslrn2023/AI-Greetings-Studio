import torch
import gc
from PIL import Image
import config

# Cross-platform ZeroGPU compatibility wrapper (works on HF, Kaggle, Azure, Docker, or local GPU)[cite: 6]
try:
    import spaces
    gpu_decorator = spaces.GPU(duration=65)
except (ImportError, AttributeError):
    def gpu_decorator(fn):
        return fn

_pipe = None

def _get_pipeline():
    """
    Lazy-loads the active diffusion pipeline on-demand with memory optimizations,
    preventing import-time CUDA OOM crashes across Kaggle, Azure, and HF.
    """
    global _pipe
    if _pipe is not None:
        return _pipe

    # Purge stale VRAM fragmentation before allocation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if config.IMAGE_ENGINE == "Z_IMAGE":
        print("Loading Tongyi-MAI/Z-Image-Turbo pipeline...")
        from diffusers import ZImagePipeline
        _pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            token=config.HF_TOKEN or None
        )
        if device == "cuda":
            _pipe.enable_attention_slicing()
        _pipe.to(device)
        print("Z-Image-Turbo loaded successfully!")
    else:
        print("Loading black-forest-labs/FLUX.1-schnell pipeline...")
        from diffusers import FluxPipeline
        _pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            token=config.HF_TOKEN or None
        )
        if device == "cuda":
            _pipe.enable_attention_slicing()
        _pipe.to(device)
        print("FLUX.1-schnell loaded successfully!")

    return _pipe

@gpu_decorator
def generate_artwork(prompt: str) -> Image.Image:
    """Generates a native 1088x1920 9:16 vertical artwork using the active lazy-loaded engine[cite: 6]."""
    print(f"[{config.IMAGE_ENGINE}] Generating 9:16 artwork: {prompt}[cite: 6]")
    pipe = _get_pipeline()
    
    try:
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
    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(f"Artwork generation failed: {e}")