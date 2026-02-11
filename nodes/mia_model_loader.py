"""
MIALoadModel - Load Make-It-Animatable models for fast humanoid rigging.

Uses comfy-env isolated environment for GPU dependencies.
"""

import sys
from pathlib import Path

# Add lib to path for mia module (needed for torch.load unpickling)
LIB_DIR = Path(__file__).parent / "lib"  # lib is inside nodes/
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

# Add utils to path for mia_inference
UTILS_DIR = Path(__file__).parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

try:
    from ..utils.mia_inference import load_mia_models, ensure_mia_models, MIA_PATH
except ImportError:
    from mia_inference import load_mia_models, ensure_mia_models, MIA_PATH


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
        print(f"[MIALoadModel] Loading Make-It-Animatable models...")
        print(f"[MIALoadModel] GPU caching: {'enabled' if cache_to_gpu else 'disabled'}")

        models = load_mia_models(cache_to_gpu=cache_to_gpu)

        print(f"[MIALoadModel] Models loaded successfully")
        return (models,)
