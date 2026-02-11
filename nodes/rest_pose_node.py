"""
UniRig Extract Rest Pose Node
Extract skeleton rest pose from FBX or SMPL parameters, output as FBX.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import folder_paths

# Support both relative imports (ComfyUI) and absolute imports (testing)
try:
    from .base import LIB_DIR
except ImportError:
    from base import LIB_DIR

# SMPL canonical skeleton (24 joints)
SMPL_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
]

SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
]

# T-pose canonical positions (Y-up, meters, ~1.7m height)
SMPL_REST_POSITIONS = np.array([
    [0.0, 0.0, 0.0],        # 0 Pelvis
    [0.09, -0.065, 0.0],    # 1 L_Hip
    [-0.09, -0.065, 0.0],   # 2 R_Hip
    [0.0, 0.09, 0.0],       # 3 Spine1
    [0.09, -0.49, 0.0],     # 4 L_Knee
    [-0.09, -0.49, 0.0],    # 5 R_Knee
    [0.0, 0.20, 0.0],       # 6 Spine2
    [0.09, -0.87, 0.0],     # 7 L_Ankle
    [-0.09, -0.87, 0.0],    # 8 R_Ankle
    [0.0, 0.32, 0.0],       # 9 Spine3
    [0.09, -0.92, 0.12],    # 10 L_Foot
    [-0.09, -0.92, 0.12],   # 11 R_Foot
    [0.0, 0.46, 0.0],       # 12 Neck
    [0.06, 0.40, 0.0],      # 13 L_Collar
    [-0.06, 0.40, 0.0],     # 14 R_Collar
    [0.0, 0.57, 0.0],       # 15 Head
    [0.18, 0.40, 0.0],      # 16 L_Shoulder
    [-0.18, 0.40, 0.0],     # 17 R_Shoulder
    [0.45, 0.40, 0.0],      # 18 L_Elbow
    [-0.45, 0.40, 0.0],     # 19 R_Elbow
    [0.70, 0.40, 0.0],      # 20 L_Wrist
    [-0.70, 0.40, 0.0],     # 21 R_Wrist
    [0.78, 0.40, 0.0],      # 22 L_Hand
    [-0.78, 0.40, 0.0],     # 23 R_Hand
], dtype=np.float32)


# Direct extraction module cache
_DIRECT_REST_POSE_MODULE = None


def _get_direct_rest_pose_module():
    """Get the direct rest pose extraction module for in-process extraction using bpy."""
    global _DIRECT_REST_POSE_MODULE
    if _DIRECT_REST_POSE_MODULE is None:
        module_path = os.path.join(os.path.dirname(__file__), "lib", "direct_extract_rest_pose.py")
        if os.path.exists(module_path):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("direct_extract_rest_pose", module_path)
                _DIRECT_REST_POSE_MODULE = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_DIRECT_REST_POSE_MODULE)
                print(f"[UniRig] Loaded direct rest pose module from {module_path}")
            except ImportError as e:
                print(f"[UniRig] Direct rest pose module not available (bpy not installed): {e}")
                _DIRECT_REST_POSE_MODULE = False
            except Exception as e:
                print(f"[UniRig] Warning: Could not load direct rest pose module: {e}")
                _DIRECT_REST_POSE_MODULE = False
        else:
            print(f"[UniRig] Warning: Direct rest pose module not found at {module_path}")
            _DIRECT_REST_POSE_MODULE = False
    return _DIRECT_REST_POSE_MODULE if _DIRECT_REST_POSE_MODULE else None


class UniRigExtractRestPose:
    """
    Extract skeleton rest pose from FBX file or SMPL parameters.

    Outputs an FBX file path containing the skeleton in T-pose,
    compatible with UniRigCompareSkeletons for side-by-side comparison.

    For FBX source: Imports FBX, strips all animation data, exports T-pose
    For SMPL source: Creates armature from canonical SMPL joint positions
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_type": (["fbx", "smpl"], {
                    "default": "fbx",
                    "tooltip": "Source type: FBX file or SMPL parameters"
                }),
                "output_name": ("STRING", {
                    "default": "rest_pose",
                    "tooltip": "Output filename (without extension)"
                }),
            },
            "optional": {
                "fbx_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to input FBX file (when source_type=fbx)"
                }),
                "smpl_params": ("SMPL_PARAMS", {
                    "tooltip": "SMPL parameters dict (when source_type=smpl) - uses betas for body shape if provided"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("fbx_path", "info")
    FUNCTION = "extract_rest_pose"
    CATEGORY = "unirig"

    def extract_rest_pose(
        self,
        source_type: str,
        output_name: str,
        fbx_path: str = "",
        smpl_params: Optional[Dict] = None,
    ) -> Tuple[str, str]:
        """Extract rest pose skeleton and save as FBX."""

        print(f"[UniRig ExtractRestPose] Source type: {source_type}")

        # Setup output path
        output_dir = folder_paths.get_output_directory()
        if not output_name.endswith('.fbx'):
            output_name = f"{output_name}.fbx"
        output_path = os.path.join(output_dir, output_name)

        # Get the direct module
        direct_module = _get_direct_rest_pose_module()
        if not direct_module:
            raise RuntimeError(
                "bpy module not available. Make sure you're running with the unirig environment."
            )

        if source_type == "fbx":
            # Validate FBX path
            if not fbx_path:
                raise ValueError("fbx_path is required when source_type=fbx")

            # Handle relative paths
            if not os.path.isabs(fbx_path):
                # Check in input directory first, then output
                input_dir = folder_paths.get_input_directory()
                if os.path.exists(os.path.join(input_dir, fbx_path)):
                    fbx_path = os.path.join(input_dir, fbx_path)
                elif os.path.exists(os.path.join(output_dir, fbx_path)):
                    fbx_path = os.path.join(output_dir, fbx_path)

            if not os.path.exists(fbx_path):
                raise FileNotFoundError(f"FBX file not found: {fbx_path}")

            print(f"[UniRig ExtractRestPose] Input FBX: {fbx_path}")

            # Extract rest pose from FBX
            bone_count = direct_module.extract_rest_pose_from_fbx(fbx_path, output_path)
            source_info = f"FBX: {os.path.basename(fbx_path)}"

        else:  # smpl
            print(f"[UniRig ExtractRestPose] Creating SMPL rest pose skeleton")

            # Use canonical SMPL positions (or compute from betas if provided)
            joint_positions = SMPL_REST_POSITIONS.copy()

            # If smpl_params has betas, we could compute actual joint positions
            # For now, use canonical T-pose
            if smpl_params and 'betas' in smpl_params:
                print(f"[UniRig ExtractRestPose] Using canonical T-pose (betas shape computation not implemented)")

            # Create SMPL skeleton
            bone_count = direct_module.create_smpl_skeleton_fbx(
                joint_positions=joint_positions,
                joint_names=SMPL_JOINT_NAMES,
                parent_indices=SMPL_PARENTS,
                output_path=output_path,
            )
            source_info = "SMPL canonical T-pose"

        print(f"[UniRig ExtractRestPose] Output: {output_path}")
        print(f"[UniRig ExtractRestPose] Bones: {bone_count}")

        info = (
            f"Source: {source_info}\n"
            f"Bones: {bone_count}\n"
            f"Output: {output_name}"
        )

        return (output_path, info)
