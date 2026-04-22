"""
MIAAutoRig - Fast humanoid rigging using Make-It-Animatable.
"""

import logging
import time
from pathlib import Path

log = logging.getLogger("unirig")

# ComfyUI folder paths
try:
    import folder_paths
    OUTPUT_DIR = Path(folder_paths.get_output_directory())
except ImportError:
    OUTPUT_DIR = Path(__file__).parent.parent / "output"


class MIAAutoRig:
    """
    Fast humanoid rigging using Make-It-Animatable.

    Takes a mesh and outputs a Mixamo-compatible rigged FBX file.
    Optimized for humanoid characters - faster than UniRig (<1 second).

    Outputs FBX with Mixamo skeleton ready for Mixamo animations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "model": ("MIA_MODEL", {
                    "tooltip": "Pre-loaded MIA model (from MIALoadModel)"
                }),
            },
            "optional": {
                "fbx_name": ("STRING", {
                    "default": "",
                    "tooltip": "Custom filename for saved FBX (without extension). If empty, uses auto-generated name."
                }),
                "no_fingers": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Merge finger weights to hand bone. Enable if model doesn't have separate fingers."
                }),
                "use_normal": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use surface normals for better skinning weights. Helps when limbs are close together."
                }),
                "reset_to_rest": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Transform output to T-pose rest position for animation compatibility."
                }),
                "normalize_ground": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Align mesh feet to ground (Y=0) for stable placement."
                }),
                "add_root_bone": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Add a synthetic Root bone at origin and parent top-level bones to it."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fbx_output_path",)
    FUNCTION = "auto_rig"
    CATEGORY = "UniRig/MIA"

    def auto_rig(
        self,
        trimesh,
        model,
        fbx_name="",
        no_fingers=True,
        use_normal=False,
        reset_to_rest=True,
        normalize_ground=True,
        add_root_bone=True,
    ):
        """
        Complete rigging pipeline using Make-It-Animatable.

        1. Sample points from mesh surface
        2. Normalize and localize joints (coarse)
        3. Predict blend weights, joint positions, and pose
        4. Post-process and export FBX
        """
        # Lazy import - only run in isolated worker
        from .mia_inference import load_mia_models, get_cached_models, run_mia_inference

        total_start = time.time()

        # DEBUG: Check visual immediately on receipt (before any MIA code)
        log.debug("Received mesh visual type: %s", type(trimesh.visual).__name__ if hasattr(trimesh, 'visual') else 'NO VISUAL')
        if hasattr(trimesh, 'visual') and hasattr(trimesh.visual, 'material'):
            log.debug("  Material: %s", type(trimesh.visual.material).__name__)

        log.info("Starting Make-It-Animatable rigging pipeline...")
        log.info(
            "Options: no_fingers=%s, use_normal=%s, reset_to_rest=%s, normalize_ground=%s, add_root_bone=%s",
            no_fingers, use_normal, reset_to_rest, normalize_ground, add_root_bone
        )

        # model is a config dict from MIALoadModel - extract settings
        dtype = model.get("dtype", "fp32")
        log.info("Config: dtype=%s", dtype)

        # Load models internally (downloads from HuggingFace if needed)
        cache_key = load_mia_models(dtype=dtype)
        models = get_cached_models(cache_key)

        # Generate output filename
        if fbx_name:
            output_filename = f"{fbx_name}_mia.fbx"
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"rigged_mia_{timestamp}.fbx"

        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(OUTPUT_DIR / output_filename)

        # Run MIA inference with loaded models
        result_path = run_mia_inference(
            mesh=trimesh,
            models=models,
            output_path=output_path,
            no_fingers=no_fingers,
            use_normal=use_normal,
            reset_to_rest=reset_to_rest,
            normalize_ground=normalize_ground,
            add_root_bone=add_root_bone,
        )

        total_time = time.time() - total_start
        log.info("========================================")
        log.info("Make-It-Animatable rigging complete!")
        log.info("Total time: %.2fs", total_time)
        log.info("Output: %s", result_path)
        log.info("========================================")

        return (result_path,)
