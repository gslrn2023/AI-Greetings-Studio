import os
import gc

# 1. Prevent CUDA memory fragmentation on constrained GPUs (must be set before importing torch)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from PIL import Image
import config

# 2. Cross-platform ZeroGPU compatibility wrapper (strictly checks HF ZeroGPU environment variable)
IS_ZEROGPU = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("true", "1")
try:
    import spaces
    gpu_decorator = spaces.GPU(duration=65)
except (ImportError, AttributeError):
    def gpu_decorator(fn):
        return fn

_LOW_VRAM_FLAG = None


def _is_low_vram_cuda() -> bool:
    """
    Returns True if running on a CUDA GPU (like Kaggle/Azure T4 14.56 GiB or local GPU)
    that is NOT managed by HF ZeroGPU and has less than 22 GiB total VRAM.
    """
    global _LOW_VRAM_FLAG
    if _LOW_VRAM_FLAG is not None:
        return _LOW_VRAM_FLAG
    if IS_ZEROGPU or not torch.cuda.is_available():
        _LOW_VRAM_FLAG = False
        return False
    try:
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        _LOW_VRAM_FLAG = total_vram_gb < 22.0
    except Exception:
        _LOW_VRAM_FLAG = False
    return _LOW_VRAM_FLAG


# 3. Head-Chunked SDPA Wrapper (Installed before importing diffusers)
# On T4 (sm_75), bfloat16 SDPA falls back to float32 math, allocating 7.44 GiB x 2 for 30 heads at 1088x1920.
# Slicing into 2-head chunks drops peak attention VRAM from 14.88 GiB down to <1.0 GiB with identical output.
_orig_sdpa = F.scaled_dot_product_attention


def _memory_safe_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    if (
        query.is_cuda
        and query.ndim == 4
        and _is_low_vram_cuda()
        and query.shape[2] > 1024
        and query.shape[1] == key.shape[1] == value.shape[1]
    ):
        num_heads = query.shape[1]
        chunk_size = 2
        outputs = []
        for i in range(0, num_heads, chunk_size):
            q_c = query[:, i : i + chunk_size]
            k_c = key[:, i : i + chunk_size]
            v_c = value[:, i : i + chunk_size]
            if attn_mask is not None and attn_mask.ndim == 4 and attn_mask.shape[1] == num_heads:
                m_c = attn_mask[:, i : i + chunk_size]
            else:
                m_c = attn_mask
            outputs.append(
                _orig_sdpa(
                    q_c,
                    k_c,
                    v_c,
                    attn_mask=m_c,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    **kwargs,
                )
            )
        return torch.cat(outputs, dim=1)
    return _orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)


torch.nn.functional.scaled_dot_product_attention = _memory_safe_sdpa
F.scaled_dot_product_attention = _memory_safe_sdpa

# 4. Primary & Fallback Engine State
PRIMARY_ENGINE = config.IMAGE_ENGINE if config.IMAGE_ENGINE in ("FLUX", "Z_IMAGE") else "FLUX"
FALLBACK_ENGINE = "Z_IMAGE" if PRIMARY_ENGINE == "FLUX" else "FLUX"

pipe = None
active_engine = None


def _enable_memory_efficient_vae(pipeline):
    """
    Configures AutoencoderKL tiling via direct attributes so 1088x1920 latents decode in 512px tiles.
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
                pipeline.vae.enable_tiling()
            except Exception:
                pass
        # Explicitly lower tile thresholds on AutoencoderKL (default sample_size is 1024)
        pipeline.vae.tile_sample_min_size = 512
        pipeline.vae.tile_latent_min_size = 64


def _apply_device_placement(pipeline, device: str, low_vram: bool):
    """
    Applies full-speed pipe.to(device) on HF ZeroGPU / high-VRAM GPUs,
    or sequential/model CPU offload + explicit VAE tiling on <22 GiB GPUs.
    """
    if device == "cuda" and low_vram:
        print("Low-VRAM GPU detected (<22 GiB): enabling head-sliced SDPA, tiled VAE & CPU offload...")
        _enable_memory_efficient_vae(pipeline)
        try:
            pipeline.enable_sequential_cpu_offload()
            print("Enabled sequential (layer-by-layer) CPU offload.")
        except Exception as e:
            print(f"Sequential offload unavailable ({e}), using model CPU offload...")
            pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device)


def _load_single_engine(engine_name: str):
    """Loads the specified engine ('FLUX' or 'Z_IMAGE') and applies device placement."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    low_vram = _is_low_vram_cuda()

    if low_vram:
        gc.collect()
        torch.cuda.empty_cache()

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if engine_name == "Z_IMAGE":
        print("Loading Tongyi-MAI/Z-Image-Turbo pipeline...")
        from diffusers import ZImagePipeline
        loaded_pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=dtype,
            low_cpu_mem_usage=True if low_vram else False,
            token=config.HF_TOKEN or None,
        )
        _apply_device_placement(loaded_pipe, device, low_vram)
        print("Z-Image-Turbo loaded successfully!")
        return loaded_pipe
    else:
        print("Loading black-forest-labs/FLUX.1-schnell pipeline...")
        from diffusers import FluxPipeline
        loaded_pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=dtype,
            low_cpu_mem_usage=True if low_vram else False,
            token=config.HF_TOKEN or None,
        )
        _apply_device_placement(loaded_pipe, device, low_vram)
        print("FLUX.1-schnell loaded successfully!")
        return loaded_pipe


def _init_pipeline(force_engine: str = None):
    """
    Initializes the primary image engine configured in config.py, automatically falling back
    to the secondary engine if loading fails (e.g., gated repo token error, deprecation, or OOM).
    """
    global pipe, active_engine
    if pipe is not None and force_engine is None:
        return

    if pipe is not None:
        del pipe
        pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    candidates = [force_engine] if force_engine else [PRIMARY_ENGINE, FALLBACK_ENGINE]
    last_err = None

    for eng in candidates:
        try:
            pipe = _load_single_engine(eng)
            active_engine = eng
            return
        except Exception as e:
            last_err = e
            print(f"[Image Engine] '{eng}' failed to initialize ({e}). Trying fallback engine...")
            pipe = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise RuntimeError(f"All image engines failed to initialize. Last error: {last_err}")


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
    except Exception as e:
        print(f"VRAM tight during VAE decode ({e}); falling back to CPU VAE decode...")
        gc.collect()
        torch.cuda.empty_cache()
        pipe.vae.to("cpu", dtype=torch.float32)
        image_tensor = pipe.vae.decode(latents.to("cpu", dtype=torch.float32), return_dict=False)[0]

    return pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]


def _run_single_inference(prompt: str) -> Image.Image:
    low_vram = _is_low_vram_cuda()
    if low_vram:
        gc.collect()
        torch.cuda.empty_cache()

    steps = config.Z_IMAGE_STEPS if active_engine == "Z_IMAGE" else config.FLUX_STEPS

    try:
        with torch.inference_mode():
            if not low_vram:
                return pipe(
                    prompt=prompt,
                    width=config.IMAGE_WIDTH,
                    height=config.IMAGE_HEIGHT,
                    num_inference_steps=steps,
                    guidance_scale=0.0,
                ).images[0]

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


@gpu_decorator
def generate_artwork(prompt: str) -> Image.Image:
    """
    Generates a native 1088x1920 9:16 vertical artwork using the active engine,
    automatically falling back to the secondary engine if a runtime error occurs.
    """
    global active_engine
    print(f"[{active_engine}] Generating 9:16 artwork: {prompt}")
    try:
        return _run_single_inference(prompt)
    except Exception as primary_err:
        if active_engine == FALLBACK_ENGINE:
            raise primary_err
        print(f"[Image Engine] Runtime error on '{active_engine}' ({primary_err}). Switching to '{FALLBACK_ENGINE}'...")
        _init_pipeline(force_engine=FALLBACK_ENGINE)
        print(f"[{active_engine}] Retrying 9:16 artwork generation: {prompt}")
        return _run_single_inference(prompt)