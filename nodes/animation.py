"""
UniRig Animation Node - Apply animations to rigged meshes
"""

import os
import time
from pathlib import Path

# ComfyUI folder paths
try:
    import folder_paths
    COMFYUI_INPUT_FOLDER = folder_paths.get_input_directory()
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except:
    COMFYUI_INPUT_FOLDER = None
    COMFYUI_OUTPUT_FOLDER = None

# Import from base module
try:
    from .base import LIB_DIR
except ImportError:
    from base import LIB_DIR

# Animation templates folder (in input directory, copied by prestartup_script.py)
ANIMATION_TEMPLATES_DIR = Path(COMFYUI_INPUT_FOLDER) / "animation_templates" if COMFYUI_INPUT_FOLDER else None

# Direct animation module (bpy as Python module)
_DIRECT_ANIMATION_MODULE = None


def _get_direct_animation():
    """Get the direct animation module for in-process animation using bpy."""
    global _DIRECT_ANIMATION_MODULE
    if _DIRECT_ANIMATION_MODULE is None:
        animation_path = os.path.join(LIB_DIR, "unirig", "src", "inference", "direct_apply_animation.py")
        if os.path.exists(animation_path):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("unirig_direct_animation", animation_path)
                _DIRECT_ANIMATION_MODULE = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_DIRECT_ANIMATION_MODULE)
                print(f"[UniRig] Loaded direct animation module from {animation_path}")
            except ImportError as e:
                print(f"[UniRig] Direct animation not available (bpy not installed): {e}")
                _DIRECT_ANIMATION_MODULE = False
            except Exception as e:
                print(f"[UniRig] Warning: Could not load direct animation module: {e}")
                _DIRECT_ANIMATION_MODULE = False
        else:
            print(f"[UniRig] Warning: Direct animation module not found at {animation_path}")
            _DIRECT_ANIMATION_MODULE = False
    return _DIRECT_ANIMATION_MODULE if _DIRECT_ANIMATION_MODULE else None


class UniRigApplyAnimation:
    """
    Apply an animation to a rigged FBX model.
    Supports Mixamo FBX animations and SMPL NPZ animations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get animation files for default (mixamo)
        mixamo_files = cls.get_animation_files("mixamo")

        return {
            "required": {
                "model_fbx_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to the rigged model FBX file (from UniRig: Apply Skinning output)"
                }),
                "animation_type": (["mixamo", "smpl"], {
                    "default": "mixamo",
                    "tooltip": "Animation format: mixamo (FBX) or smpl (NPZ). Refresh node after changing."
                }),
                "animation_file": (mixamo_files, {
                    "default": mixamo_files[0] if mixamo_files else "No animations found",
                    "tooltip": "Animation file to apply. Add more animations to input/animation_templates/<type>/"
                }),
            },
            "optional": {
                "output_name": ("STRING", {
                    "default": "",
                    "tooltip": "Custom output filename (without extension). If empty, auto-generates name."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("animated_fbx_path",)
    FUNCTION = "apply_animation"
    CATEGORY = "UniRig/Animation"
    OUTPUT_NODE = True

    @classmethod
    def get_animation_files(cls, animation_type):
        """Get list of animation files for the specified type."""
        animation_files = []

        if ANIMATION_TEMPLATES_DIR is None:
            return ["No animations found"]

        if animation_type == "mixamo":
            folder = ANIMATION_TEMPLATES_DIR / "mixamo"
            extensions = ['.fbx']
        else:  # smpl
            folder = ANIMATION_TEMPLATES_DIR / "smpl"
            extensions = ['.npz']

        if folder.exists():
            for file in folder.iterdir():
                if file.is_file() and file.suffix.lower() in extensions:
                    animation_files.append(file.name)

        return sorted(animation_files) if animation_files else ["No animations found"]

    @classmethod
    def IS_CHANGED(cls, model_fbx_path, animation_type, animation_file, output_name=""):
        """Force re-execution when inputs change."""
        if ANIMATION_TEMPLATES_DIR is None:
            return f"{model_fbx_path}:{animation_type}:{animation_file}"

        anim_path = ANIMATION_TEMPLATES_DIR / animation_type / animation_file
        if anim_path.exists():
            return f"{model_fbx_path}:{os.path.getmtime(anim_path)}"
        return f"{model_fbx_path}:{animation_type}:{animation_file}"

    def apply_animation(self, model_fbx_path, animation_type, animation_file, output_name=""):
        """Apply animation to rigged model using Blender."""

        # Validate inputs
        if not model_fbx_path or not model_fbx_path.strip():
            raise ValueError("Model FBX path cannot be empty")

        if animation_file == "No animations found":
            raise ValueError(f"No animation files found for type: {animation_type}")

        if not os.path.exists(model_fbx_path):
            raise ValueError(f"Model FBX not found: {model_fbx_path}")

        if ANIMATION_TEMPLATES_DIR is None:
            raise ValueError("Animation templates directory not found")

        # Get animation file path
        animation_path = ANIMATION_TEMPLATES_DIR / animation_type / animation_file

        if not animation_path.exists():
            raise ValueError(f"Animation file not found: {animation_path}")

        # Check direct animation module availability
        direct_animation = _get_direct_animation()
        if not direct_animation:
            raise RuntimeError(
                "Direct animation module not available.\n"
                "Ensure bpy is installed in the unirig environment."
            )

        print(f"[UniRigApplyAnimation] Model: {model_fbx_path}")
        print(f"[UniRigApplyAnimation] Animation type: {animation_type}")
        print(f"[UniRigApplyAnimation] Animation: {animation_file}")

        # Early validation: Check if model filename suggests skeleton template compatibility
        model_basename = os.path.basename(model_fbx_path).lower()
        if animation_type == "mixamo":
            # Mixamo animations require mixamo skeleton template
            if "_smpl.fbx" in model_basename or "_smpl_" in model_basename:
                raise ValueError(
                    "Skeleton template mismatch!\n"
                    f"Model appears to use SMPL skeleton (based on filename: {os.path.basename(model_fbx_path)})\n"
                    "but Mixamo animations require the 'mixamo' skeleton template.\n\n"
                    "To fix: Re-run skeleton extraction with skeleton_template='mixamo'"
                )
            if "_vroid.fbx" in model_basename or "_vroid_" in model_basename:
                raise ValueError(
                    "Skeleton template mismatch!\n"
                    f"Model appears to use VRoid skeleton (based on filename: {os.path.basename(model_fbx_path)})\n"
                    "but Mixamo animations require the 'mixamo' skeleton template.\n\n"
                    "To fix: Re-run skeleton extraction with skeleton_template='mixamo'"
                )
            # Positive check: should have mixamo in name
            if "_mixamo" not in model_basename:
                print(f"[UniRigApplyAnimation] Warning: Model filename does not contain '_mixamo'. "
                      f"Make sure you used skeleton_template='mixamo' when extracting the skeleton.")

        # Handle different animation types
        if animation_type == "smpl":
            # TODO: SMPL animation support requires different processing
            raise NotImplementedError(
                "SMPL animation support is not yet implemented. "
                "Please use Mixamo animations for now."
            )

        # Determine output path
        output_dir = COMFYUI_OUTPUT_FOLDER if COMFYUI_OUTPUT_FOLDER else os.path.dirname(model_fbx_path)

        if output_name and output_name.strip():
            output_filename = output_name.strip()
            if not output_filename.lower().endswith('.fbx'):
                output_filename += '.fbx'
        else:
            # Auto-generate name from model + animation
            model_name = os.path.splitext(os.path.basename(model_fbx_path))[0]
            anim_name = os.path.splitext(animation_file)[0]
            output_filename = f"{model_name}_{anim_name}.fbx"

        output_path = os.path.join(output_dir, output_filename)
        print(f"[UniRigApplyAnimation] Output: {output_path}")

        # Run direct animation module
        print(f"[UniRigApplyAnimation] Applying animation...")
        start_time = time.time()

        try:
            result_path = direct_animation.apply_mixamo_animation(
                model_fbx=model_fbx_path,
                animation_fbx=str(animation_path),
                output_fbx=output_path,
            )

            if not os.path.exists(output_path):
                raise RuntimeError("Animation module did not produce output FBX file")

            elapsed = time.time() - start_time
            print(f"[UniRigApplyAnimation] Animation applied in {elapsed:.2f}s")
            print(f"[UniRigApplyAnimation] Output: {output_path}")

            return (output_path,)

        except RuntimeError as e:
            # Check for skeleton mismatch errors
            error_str = str(e)
            if "mixamorig:" in error_str or "bone names" in error_str.lower():
                raise RuntimeError(
                    "Skeleton mismatch error!\n"
                    "The model's skeleton does not match the animation skeleton.\n\n"
                    "For Mixamo animations, the model must use skeleton_template='mixamo'.\n"
                    "Please re-run UniRig: Extract Skeleton with the correct template.\n\n"
                    f"Details:\n{error_str}"
                )
            raise
        except Exception as e:
            raise RuntimeError(f"Animation application failed: {str(e)}")
