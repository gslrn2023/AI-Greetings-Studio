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


def _detect_hardware_profile() -> dict:
    """
    Vendor-agnostic hardware profiler. Inspects GPU count, single-card VRAM, and combined VRAM
    to classify the system into HIGH_VRAM, MULTI_GPU, SINGLE_LOW_VRAM, or CPU.
    """
    if IS_ZEROGPU:
        return {
            "tier": "HIGH_VRAM",
            "gpu_count": 1,
            "single_vram_gb": 40.0,
            "combined_vram_gb": 40.0,
            "needs_sdpa_chunking": False,
        }

    if not torch.cuda.is_available():
        return {
            "tier": "CPU",
            "gpu_count": 0,
            "single_vram_gb": 0.0,
            "combined_vram_gb": 0.0,
            "needs_sdpa_chunking": False,
        }

    try:
        gpu_count = torch.cuda.device_count()
        vram_list = [
            torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            for i in range(gpu_count)
        ]
        max_single_vram = max(vram_list) if vram_list else 0.0
        combined_vram = sum(vram_list)
    except Exception:
        gpu_count = 1
        max_single_vram = 14.5
        combined_vram = 14.5

    if max_single_vram >= 22.0:
        tier = "HIGH_VRAM"
    elif gpu_count >= 2 and combined_vram >= 20.0:
        tier = "MULTI_GPU"
    else:
        tier = "SINGLE_LOW_VRAM"

    return {
        "tier": tier,
        "gpu_count": gpu_count,
        "single_vram_gb": round(max_single_vram, 2),
        "combined_vram_gb": round(combined_vram, 2),
        "needs_sdpa_chunking": max_single_vram < 22.0,
    }


HW_PROFILE = _detect_hardware_profile()
print(
    f"[Hardware Profile] Tier={HW_PROFILE['tier']} | GPUs={HW_PROFILE['gpu_count']} | "
    f"Single_VRAM={HW_PROFILE['single_vram_gb']} GiB | Combined_VRAM={HW_PROFILE['combined_vram_gb']} GiB"
)

# 3. Head-Chunked SDPA Wrapper (Installed before importing diffusers)
# On GPUs with <22 GiB per card (e.g., 14.56 GiB T4), 30-head 1088x1920 attention in float32 fallback
# requests 7.44 GiB x 2 = 14.88 GiB at once. Chunking 4 heads at a time caps attention VRAM at ~0.99 GiB
# while keeping the full 12.3 GiB Transformer resident in GPU VRAM.
_orig_sdpa = F.scaled_dot_product_attention


def _memory_safe_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    if (
        HW_PROFILE["needs_sdpa_chunking"]
        and query.is_cuda
        and query.ndim == 4
        and query.shape[2] > 1024
        and query.shape[1] == key.shape[1] == value.shape[1]
    ):
        num_heads = query.shape[1]
        chunk_size = 4
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

# 4. Primary & Fallback Engine Selection
PRIMARY_ENGINE = config.IMAGE_ENGINE if config.IMAGE_ENGINE in ("FLUX", "Z_IMAGE") else "FLUX"
FALLBACK_ENGINE = "Z_IMAGE" if PRIMARY_ENGINE == "FLUX" else "FLUX"

pipe = None
active_engine = None


def _get_engine_boot_order(force_engine: str = None) -> list[str]:
    """
    Respects config.IMAGE_ENGINE on HF ZeroGPU and >=32 GiB systems.
    If combined VRAM is under 32 GiB (e.g., 2x T4 = 29.12 GiB or 1x T4 = 14.56 GiB) and FLUX (32.8 GB)
    is primary, automatically routes to Z_IMAGE (20.3 GB) first so weights fit 100% in VRAM.
    """
    if force_engine:
        return [force_engine]
    if (
        PRIMARY_ENGINE == "FLUX"
        and not IS_ZEROGPU
        and HW_PROFILE["tier"] in ("MULTI_GPU", "SINGLE_LOW_VRAM")
        and HW_PROFILE["combined_vram_gb"] < 32.0
    ):
        print(
            f"[Hardware Router] Combined VRAM ({HW_PROFILE['combined_vram_gb']} GiB) is <32 GiB required "
            f"for 32.8 GB FLUX; routing to Z_IMAGE first for fast in-VRAM execution."
        )
        return ["Z_IMAGE", "FLUX"]
    return [PRIMARY_ENGINE, FALLBACK_ENGINE]


def _enable_memory_efficient_vae(pipeline):
    """
    Prevents 32-bit float VAE upcast and configures 512px AutoencoderKL tiling via direct attributes.
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
        pipeline.vae.tile_sample_min_size = 512
        pipeline.vae.tile_latent_min_size = 64


def _balance_multi_gpu_vae(pipeline):
    """
    When Diffusers' device_map='balanced' distributes 3 components across 2 GPUs, round-robin (0, 1, 0)
    places both 'transformer' (12.3 GiB) and 'vae' (0.16 GiB) on GPU 0.
    This moves 'vae' onto GPU 1 (with 'text_encoder', which has ~6.6 GiB free VRAM) so VAE decoding
    never competes with the Transformer on GPU 0.
    """
    if not hasattr(pipeline, "hf_device_map") or not isinstance(pipeline.hf_device_map, dict):
        return
    print(f"[Multi-GPU] Initial component layout: {pipeline.hf_device_map}")
    t_dev = pipeline.hf_device_map.get("transformer")
    te_dev = pipeline.hf_device_map.get("text_encoder")
    v_dev = pipeline.hf_device_map.get("vae")

    if t_dev is not None and te_dev is not None and t_dev != te_dev and v_dev == t_dev:
        try:
            from accelerate import dispatch_model
            from accelerate.hooks import remove_hook_from_module

            remove_hook_from_module(pipeline.vae, recurse=True)
            pipeline.vae.to(f"cuda:{te_dev}")
            dispatch_model(pipeline.vae, device_map={"": te_dev})
            pipeline.hf_device_map["vae"] = te_dev
            torch.cuda.empty_cache()
            print(f"[Multi-GPU] Optimized component layout: {pipeline.hf_device_map}")
        except Exception as e:
            print(f"[Multi-GPU] VAE placement note: {e}")


def _load_single_engine(engine_name: str):
    """Loads the specified engine ('FLUX' or 'Z_IMAGE') using the optimal hardware tier strategy."""
    tier = HW_PROFILE["tier"]
    dtype = torch.bfloat16 if tier != "CPU" else torch.float32

    if tier in ("MULTI_GPU", "SINGLE_LOW_VRAM"):
        gc.collect()
        torch.cuda.empty_cache()

    model_id = (
        "Tongyi-MAI/Z-Image-Turbo"
        if engine_name == "Z_IMAGE"
        else "black-forest-labs/FLUX.1-schnell"
    )
    print(f"Loading {model_id} pipeline (Tier={tier})...")

    load_kwargs = {
        "torch_dtype": dtype,
        "token": config.HF_TOKEN or None,
    }

    # Tier 2 (Multi-GPU): Shard components directly across cuda:0 and cuda:1 during loading
    if tier == "MULTI_GPU":
        load_kwargs["device_map"] = "balanced"
        load_kwargs["low_cpu_mem_usage"] = True
    elif tier == "SINGLE_LOW_VRAM":
        load_kwargs["low_cpu_mem_usage"] = True

    if engine_name == "Z_IMAGE":
        from diffusers import ZImagePipeline
        loaded_pipe = ZImagePipeline.from_pretrained(model_id, **load_kwargs)
    else:
        from diffusers import FluxPipeline
        loaded_pipe = FluxPipeline.from_pretrained(model_id, **load_kwargs)

    # Apply device placement by Tier
    if tier == "HIGH_VRAM":
        loaded_pipe.to("cuda")
    elif tier == "MULTI_GPU":
        _enable_memory_efficient_vae(loaded_pipe)
        _balance_multi_gpu_vae(loaded_pipe)
    elif tier == "SINGLE_LOW_VRAM":
        _enable_memory_efficient_vae(loaded_pipe)
        # Use fast component-level offload on >=14 GiB GPUs (e.g., 1x T4 14.56 GiB or 16 GB RTX);
        # only use layer-by-layer sequential offload on <14 GiB cards (8 GB / 12 GB laptops)
        if HW_PROFILE["single_vram_gb"] >= 14.0:
            print("Single GPU (14-22 GiB): enabling fast component-level model CPU offload...")
            loaded_pipe.enable_model_cpu_offload()
        else:
            print("Single GPU (<14 GiB): enabling sequential CPU offload...")
            loaded_pipe.enable_sequential_cpu_offload()
    else:
        loaded_pipe.to("cpu")

    print(f"{model_id} loaded successfully!")
    return loaded_pipe


def _init_pipeline(force_engine: str = None):
    """
    Initializes the optimal primary engine and automatically falls back to the secondary engine
    if loading fails.
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

    candidates = _get_engine_boot_order(force_engine)
    last_err = None

    for eng in candidates:
        try:
            pipe = _load_single_engine(eng)
            active_engine = eng
            return
        except Exception as e:
            last_err = e
            print(f"[Image Engine] '{eng}' failed to initialize ({e}). Trying next engine...")
            pipe = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise RuntimeError(f"All image engines failed to initialize. Last error: {last_err}")


# Initialize at boot (required for HF ZeroGPU snapshot pre-warming)
_init_pipeline()


def _run_single_inference(prompt: str) -> Image.Image:
    conserve_vram = HW_PROFILE["tier"] in ("MULTI_GPU", "SINGLE_LOW_VRAM")
    if conserve_vram:
        gc.collect()
        torch.cuda.empty_cache()

    steps = config.Z_IMAGE_STEPS if active_engine == "Z_IMAGE" else config.FLUX_STEPS

    try:
        with torch.inference_mode():
            return pipe(
                prompt=prompt,
                width=config.IMAGE_WIDTH,
                height=config.IMAGE_HEIGHT,
                num_inference_steps=steps,
                guidance_scale=0.0,
            ).images[0]
    finally:
        if conserve_vram:
            gc.collect()
            torch.cuda.empty_cache()


@gpu_decorator
def generate_artwork(prompt: str) -> Image.Image:
    """
    Generates a native 1088x1920 9:16 vertical artwork using the active engine,
    automatically falling back to the secondary engine if a runtime error occurs.
    """
    global active_engine
    print(f"[{active_engine} | Tier={HW_PROFILE['tier']}] Generating 9:16 artwork: {prompt}")
    try:
        return _run_single_inference(prompt)
    except Exception as primary_err:
        fallback = "Z_IMAGE" if active_engine == "FLUX" else "FLUX"
        print(f"[Image Engine] Runtime error on '{active_engine}' ({primary_err}). Switching to '{fallback}'...")
        _init_pipeline(force_engine=fallback)
        print(f"[{active_engine} | Tier={HW_PROFILE['tier']}] Retrying 9:16 artwork: {prompt}")
        return _run_single_inference(prompt)