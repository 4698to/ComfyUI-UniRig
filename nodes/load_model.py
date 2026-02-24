"""
Model loader nodes for UniRig.

Downloads checkpoints, resolves precision/attention config.
Actual model loading happens lazily in inference nodes via direct.py.
"""

import logging
import os
import sys
from pathlib import Path

import torch

log = logging.getLogger("unirig")

# Attention backend options
ATTN_BACKENDS = ['auto', 'flash_attn', 'sdpa']

# Support both relative imports (ComfyUI) and absolute imports (testing)
try:
    from .base import UNIRIG_PATH, UNIRIG_MODELS_DIR
except ImportError:
    from base import UNIRIG_PATH, UNIRIG_MODELS_DIR

# Global config cache (for config dicts only, no model loading)
_MODEL_CACHE = {}


class UniRigLoadSkeletonModel:
    """
    Load and cache the UniRig skeleton extraction model.

    This pre-downloads the model weights and prepares configuration
    for faster skeleton inference. Connect this to UniRigExtractSkeleton
    to avoid model reload on each run.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {
                    "default": "apozz/UniRig-safetensors",
                    "tooltip": "HuggingFace model ID for skeleton model"
                }),
                "cache_to_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model cached on GPU for faster inference. Disable to offload to CPU after inference (saves VRAM)."
                }),
            },
        }

    RETURN_TYPES = ("UNIRIG_SKELETON_MODEL",)
    RETURN_NAMES = ("skeleton_model",)
    FUNCTION = "load_model"
    CATEGORY = "UniRig/Models"

    def load_model(self, model_id="apozz/UniRig-safetensors", cache_to_gpu=True, **kwargs):
        """Download and cache skeleton model configuration. No model loading."""
        log.info("Loading skeleton model config: %s", model_id)

        cache_key = f"skeleton_{model_id}"

        # Check cache
        if cache_key in _MODEL_CACHE:
            cached_model = _MODEL_CACHE[cache_key]
            log.info("Using cached model configuration")
            return (cached_model,)

        # Download checkpoint
        try:
            from .unirig.download import download
            from .unirig.configs import SKELETON_CHECKPOINT

            log.info("Downloading/verifying checkpoint...")
            local_checkpoint = download(SKELETON_CHECKPOINT)
            log.info("Checkpoint ready: %s", local_checkpoint)

            model_wrapper = {
                "type": "skeleton",
                "model_id": model_id,
                "checkpoint_path": local_checkpoint,
                "unirig_path": UNIRIG_PATH,
                "models_dir": str(UNIRIG_MODELS_DIR),
            }

            _MODEL_CACHE[cache_key] = model_wrapper
            log.info("Skeleton model config cached (checkpoint: %s)", local_checkpoint)
            return (model_wrapper,)

        except Exception as e:
            log.error("Error loading model: %s", e, exc_info=True)
            model_wrapper = {
                "type": "skeleton",
                "model_id": model_id,
                "unirig_path": UNIRIG_PATH,
                "models_dir": str(UNIRIG_MODELS_DIR),
            }
            return (model_wrapper,)


class UniRigLoadSkinningModel:
    """
    Load and cache the UniRig skinning weight prediction model.

    This pre-downloads the model weights and prepares configuration
    for faster skinning inference. Connect this to UniRigApplySkinningML
    to avoid model reload on each run.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {
                    "default": "apozz/UniRig-safetensors",
                    "tooltip": "HuggingFace model ID for skinning model"
                }),
                "cache_to_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model cached on GPU for faster inference. Disable to offload to CPU after inference (saves VRAM)."
                }),
            },
        }

    RETURN_TYPES = ("UNIRIG_SKINNING_MODEL",)
    RETURN_NAMES = ("skinning_model",)
    FUNCTION = "load_model"
    CATEGORY = "UniRig/Models"

    def load_model(self, model_id="apozz/UniRig-safetensors", cache_to_gpu=True, **kwargs):
        """Download and cache skinning model configuration. No model loading."""
        log.info("Loading skinning model config: %s", model_id)

        cache_key = f"skinning_{model_id}"

        # Check cache
        if cache_key in _MODEL_CACHE:
            cached_model = _MODEL_CACHE[cache_key]
            log.info("Using cached model configuration")
            return (cached_model,)

        # Download checkpoint
        try:
            from .unirig.download import download
            from .unirig.configs import SKIN_CHECKPOINT

            log.info("Downloading/verifying checkpoint...")
            local_checkpoint = download(SKIN_CHECKPOINT)
            log.info("Checkpoint ready: %s", local_checkpoint)

            model_wrapper = {
                "type": "skinning",
                "model_id": model_id,
                "checkpoint_path": local_checkpoint,
                "unirig_path": UNIRIG_PATH,
                "models_dir": str(UNIRIG_MODELS_DIR),
            }

            _MODEL_CACHE[cache_key] = model_wrapper
            log.info("Skinning model config cached (checkpoint: %s)", local_checkpoint)
            return (model_wrapper,)

        except Exception as e:
            log.error("Error loading model: %s", e, exc_info=True)
            model_wrapper = {
                "type": "skinning",
                "model_id": model_id,
                "unirig_path": UNIRIG_PATH,
                "models_dir": str(UNIRIG_MODELS_DIR),
            }
            return (model_wrapper,)


class UniRigLoadModel:
    """Load UniRig model configuration for the rigging pipeline.

    Downloads checkpoints and resolves precision/attention settings.
    Actual model loading happens lazily in inference nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "precision": (["auto", "bf16", "fp16", "fp32"], {
                    "default": "auto",
                    "tooltip": "Model precision. auto: best for your GPU (bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older)."
                }),
                "attn_backend": (ATTN_BACKENDS, {
                    "default": "auto",
                    "tooltip": "Attention backend. auto: best available (flash_attn > sdpa). flash_attn requires flash-attn package."
                }),
            },
        }

    RETURN_TYPES = ("UNIRIG_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_models"
    CATEGORY = "UniRig"

    def load_models(self, precision="auto", attn_backend="auto", **kwargs):
        """Download checkpoints and resolve precision/attention config."""
        import comfy.model_management as mm
        # Resolve precision
        device = mm.get_torch_device()
        if precision == "auto":
            if mm.should_use_bf16(device):
                dtype = "bf16"
            elif mm.should_use_fp16(device):
                dtype = "fp16"
            else:
                dtype = "fp32"
        else:
            dtype = precision

        log.info("Resolved precision: %s -> %s", precision, dtype)
        log.info("Attention backend: %s", attn_backend)

        model_id = "apozz/UniRig-safetensors"

        # Download skeleton checkpoint
        skeleton_loader = UniRigLoadSkeletonModel()
        skeleton_result = skeleton_loader.load_model(model_id=model_id)
        skeleton_model = skeleton_result[0]

        # Download skinning checkpoint
        skinning_loader = UniRigLoadSkinningModel()
        skinning_result = skinning_loader.load_model(model_id=model_id)
        skinning_model = skinning_result[0]

        # Propagate dtype and attn_backend into sub-model dicts
        # so inference nodes can access them directly
        skeleton_model["dtype"] = dtype
        skeleton_model["attn_backend"] = attn_backend
        skinning_model["dtype"] = dtype
        skinning_model["attn_backend"] = attn_backend

        combined_model = {
            "skeleton_model": skeleton_model,
            "skinning_model": skinning_model,
            "model_id": model_id,
            "dtype": dtype,
            "attn_backend": attn_backend,
        }

        log.info("UniRig model config ready")
        return (combined_model,)


def clear_model_cache():
    """Clear the global model cache (configs, loaded models, MIA models)."""
    import comfy.model_management as mm
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    mm.soft_empty_cache()
    log.info("Model cache cleared")


def get_cached_models():
    """Get list of cached model keys."""
    return list(_MODEL_CACHE.keys())


class MIALoadModel:
    """
    Load Make-It-Animatable models for fast humanoid rigging.

    Downloads models from HuggingFace on first use (~500MB total).
    MIA is optimized for humanoid characters and outputs Mixamo-compatible skeletons.

    Faster than UniRig (<1 second) but only supports humanoid characters with Mixamo skeleton.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "precision": (["auto", "bf16", "fp16", "fp32"], {
                    "default": "auto",
                    "tooltip": "Model precision. auto: best for your GPU (bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older)."
                }),
                "attn_backend": (ATTN_BACKENDS, {
                    "default": "auto",
                    "tooltip": "Attention backend. auto: best available (flash_attn > sdpa). flash_attn requires flash-attn package."
                }),
            },
        }

    RETURN_TYPES = ("MIA_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_models"
    CATEGORY = "UniRig/MIA"

    def load_models(self, precision="auto", attn_backend="auto"):
        """Return MIA config with resolved precision."""
        import comfy.model_management as mm
        device = mm.get_torch_device()
        if precision == "auto":
            if mm.should_use_bf16(device):
                dtype = "bf16"
            elif mm.should_use_fp16(device):
                dtype = "fp16"
            else:
                dtype = "fp32"
        else:
            dtype = precision

        log.info("MIA config: precision=%s -> %s, attn_backend=%s", precision, dtype, attn_backend)
        return ({
            "backend": "mia",
            "dtype": dtype,
            "attn_backend": attn_backend,
        },)
