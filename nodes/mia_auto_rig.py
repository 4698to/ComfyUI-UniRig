"""
MIAAutoRig - Fast humanoid rigging using Make-It-Animatable.

Uses comfy-env isolated environment for GPU dependencies.
"""

import os
import sys
import time
from pathlib import Path

# ComfyUI folder paths
try:
    import folder_paths
    OUTPUT_DIR = Path(folder_paths.get_output_directory())
except ImportError:
    OUTPUT_DIR = Path(__file__).parent.parent / "output"

# Add utils to path for imports
UTILS_DIR = Path(__file__).parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

try:
    from ..utils.mia_inference import run_mia_inference
except ImportError:
    from mia_inference import run_mia_inference


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
    ):
        """
        Complete rigging pipeline using Make-It-Animatable.

        1. Sample points from mesh surface
        2. Normalize and localize joints (coarse)
        3. Predict blend weights, joint positions, and pose
        4. Post-process and export FBX
        """
        total_start = time.time()
        print(f"[MIAAutoRig] Starting Make-It-Animatable rigging pipeline...")
        print(f"[MIAAutoRig] Options: no_fingers={no_fingers}, use_normal={use_normal}, reset_to_rest={reset_to_rest}")

        # Generate output filename
        if fbx_name:
            output_filename = f"{fbx_name}_mia.fbx"
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"rigged_mia_{timestamp}.fbx"

        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(OUTPUT_DIR / output_filename)

        # Run MIA inference
        result_path = run_mia_inference(
            mesh=trimesh,
            models=model,
            output_path=output_path,
            no_fingers=no_fingers,
            use_normal=use_normal,
            reset_to_rest=reset_to_rest,
        )

        total_time = time.time() - total_start
        print(f"[MIAAutoRig] ========================================")
        print(f"[MIAAutoRig] Make-It-Animatable rigging complete!")
        print(f"[MIAAutoRig] Total time: {total_time:.2f}s")
        print(f"[MIAAutoRig] Output: {result_path}")
        print(f"[MIAAutoRig] ========================================")

        return (result_path,)
