import torch
from PIL import Image
import config

# Cross-platform ZeroGPU compatibility wrapper (works on HF, Kaggle, Azure, Docker, or local GPU)
try:
    import spaces
    gpu_decorator = spaces.GPU(duration=65)
except (ImportError, AttributeError):
    def gpu_decorator(fn):
        return fn

pipe = None

def _init_pipeline():
    global pipe
    if pipe is not None:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if config.IMAGE_ENGINE == "Z_IMAGE":
        print("Loading Tongyi-MAI/Z-Image-Turbo pipeline...")
        from diffusers import ZImagePipeline
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
            token=config.HF_TOKEN or None
        )
        pipe.to(device)
        print("Z-Image-Turbo loaded successfully!")
    else:
        print("Loading black-forest-labs/FLUX.1-schnell pipeline...")
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=dtype,
            token=config.HF_TOKEN or None
        )
        pipe.to(device)
        print("FLUX.1-schnell loaded successfully!")

# Initialize only the selected engine at boot
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