import os
import gc

# Prevent CUDA memory fragmentation on constrained GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

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
    Prevents 32-bit VAE upcast (cutting 7.44 GiB spike in half) and forces 512px VAE tiling
    so 1088x1920 latents decode in small tiles on 14.56 GiB GPUs.
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
        if hasattr(pipeline.vae, "config") and hasattr(pipeline.vae.config, "force_upcast"):
            pipeline.vae.config.force_upcast = False

        if hasattr(pipeline.vae, "enable_slicing"):
            try:
                pipeline.vae.enable_slicing()
            except Exception:
                pass

        if hasattr(pipeline.vae, "enable_tiling"):
            try:
                pipeline.vae.enable_tiling(
                    tile_sample_min_height=512,
                    tile_sample_min_width=512,
                    tile_latent_min_height=64,
                    tile_latent_min_width=64,
                )
            except TypeError:
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

    # Prefer bfloat16 on modern CUDA or T4 when supported to prevent float16 overflow in VAE
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


def _decode_latents_safely(latents: torch.Tensor) -> Image.Image:
    """
    Flushes transformer allocations from VRAM after the diffusion loop finishes,
    then decodes the 1088x1920 latents with VAE tiling (or CPU fallback if VRAM is tight).
    """
    gc.collect()
    torch.cuda.empty_cache()

    shift_factor = getattr(pipe.vae.config, "shift_factor", 0.0) or 0.0
    scaling_factor = getattr(pipe.vae.config, "scaling_factor", 1.0) or 1.0

    # If the pipeline permits unpacking latents (e.g., Flux), unpack first
    if hasattr(pipe, "_unpack_latents") and latents.ndim == 3:
        latents = pipe._unpack_latents(
            latents,
            config.IMAGE_HEIGHT,
            config.IMAGE_WIDTH,
            pipe.vae_scale_factor,
        )

    latents = (latents / scaling_factor) + shift_factor

    try:
        vae_dtype = next(pipe.vae.parameters()).dtype
        latents_in = latents.to(dtype=vae_dtype)
        image_tensor = pipe.vae.decode(latents_in, return_dict=False)[0]
    except torch.OutOfMemoryError:
        print("VRAM tight during VAE decode; falling back to CPU VAE decode...")
        gc.collect()
        torch.cuda.empty_cache()
        pipe.vae.to("cpu", dtype=torch.float32)
        image_tensor = pipe.vae.decode(latents.to("cpu", dtype=torch.float32), return_dict=False)[0]

    return pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]


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
            if not low_vram:
                # Standard fast path for HF ZeroGPU / high-VRAM GPUs
                return pipe(
                    prompt=prompt,
                    width=config.IMAGE_WIDTH,
                    height=config.IMAGE_HEIGHT,
                    num_inference_steps=steps,
                    guidance_scale=0.0,
                ).images[0]

            # Low-VRAM path: generate raw latents first, flush 8.22 GiB transformer cache, then decode VAE
            latents = pipe(
                prompt=prompt,
                width=config.IMAGE_WIDTH,
                height=config.IMAGE_HEIGHT,
                num_inference_steps=steps,
                guidance_scale=0.0,
                output_type="latent",
            ).images

            return _decode_latents_safely(latents)
    finally:
        if low_vram:
            gc.collect()
            torch.cuda.empty_cache()