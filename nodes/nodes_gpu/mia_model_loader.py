"""
MIALoadModel - Load Make-It-Animatable models for fast humanoid rigging.

Uses comfy-env isolated environment for GPU dependencies.
"""

import sys
from pathlib import Path


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
            "required": {
                "cache_to_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep models on GPU for faster inference. Disable to save VRAM."
                }),
            },
        }

    RETURN_TYPES = ("MIA_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_models"
    CATEGORY = "UniRig/MIA"

    def load_models(self, cache_to_gpu=True):
        """Load and cache MIA models."""
        # Lazy imports - only run in isolated worker
        from mia_inference import load_mia_models

        print(f"[MIALoadModel] Loading Make-It-Animatable models...")
        print(f"[MIALoadModel] GPU caching: {'enabled' if cache_to_gpu else 'disabled'}")

        models = load_mia_models(cache_to_gpu=cache_to_gpu)

        print(f"[MIALoadModel] Models loaded successfully")
        return (models,)
