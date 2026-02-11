"""UniRig Nodes - Unified isolated environment with bpy + CUDA."""

from pathlib import Path

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

nodes_dir = Path(__file__).parent

# === Non-isolated nodes (no heavy dependencies) ===
from .mesh_io import UniRigLoadMesh, UniRigSaveMesh
NODE_CLASS_MAPPINGS.update({
    "UniRigLoadMesh": UniRigLoadMesh,
    "UniRigSaveMesh": UniRigSaveMesh,
})
NODE_DISPLAY_NAME_MAPPINGS.update({
    "UniRigLoadMesh": "UniRig: Load Mesh",
    "UniRigSaveMesh": "UniRig: Save Mesh",
})
print(f"[UniRig] Main nodes loaded (2 nodes)")

# === Isolated nodes (bpy + CUDA dependencies) ===
# These run in isolated Python 3.11 environment
try:
    from comfy_env import wrap_isolated_nodes

    # Import node classes (these don't import bpy at module level)
    # GPU/ML nodes
    from .model_loaders import UniRigLoadModel
    from .auto_rig import UniRigAutoRig
    from .skeleton_extraction import UniRigExtractSkeletonNew
    from .skinning import UniRigApplySkinningMLNew
    from .mia_model_loader import MIALoadModel
    from .mia_auto_rig import MIAAutoRig

    # Blender nodes
    from .animation import UniRigApplyAnimation
    from .skeleton_io import (
        UniRigLoadRiggedMesh,
        UniRigPreviewRiggedMesh,
        UniRigExportPosedFBX,
        UniRigViewRigging,
        UniRigDebugSkeleton,
        UniRigCompareSkeletons,
    )
    from .rest_pose_node import UniRigExtractRestPose
    from .orientation_check import UniRigOrientationCheck

    isolated_mappings = {
        # GPU/ML nodes
        "UniRigLoadModel": UniRigLoadModel,
        "UniRigAutoRig": UniRigAutoRig,
        "UniRigExtractSkeletonNew": UniRigExtractSkeletonNew,
        "UniRigApplySkinningMLNew": UniRigApplySkinningMLNew,
        "MIALoadModel": MIALoadModel,
        "MIAAutoRig": MIAAutoRig,
        # Blender nodes
        "UniRigApplyAnimation": UniRigApplyAnimation,
        "UniRigLoadRiggedMesh": UniRigLoadRiggedMesh,
        "UniRigPreviewRiggedMesh": UniRigPreviewRiggedMesh,
        "UniRigExportPosedFBX": UniRigExportPosedFBX,
        "UniRigViewRigging": UniRigViewRigging,
        "UniRigDebugSkeleton": UniRigDebugSkeleton,
        "UniRigCompareSkeletons": UniRigCompareSkeletons,
        "UniRigExtractRestPose": UniRigExtractRestPose,
        "UniRigOrientationCheck": UniRigOrientationCheck,
    }

    isolated_display = {
        # GPU/ML nodes
        "UniRigLoadModel": "UniRig: Load Model",
        "UniRigAutoRig": "UniRig: Auto Rig",
        "UniRigExtractSkeletonNew": "UniRig: Extract Skeleton",
        "UniRigApplySkinningMLNew": "UniRig: Apply Skinning ML",
        "MIALoadModel": "MIA: Load Model",
        "MIAAutoRig": "MIA: Auto Rig",
        # Blender nodes
        "UniRigApplyAnimation": "UniRig: Apply Animation",
        "UniRigLoadRiggedMesh": "UniRig: Load Rigged Mesh",
        "UniRigPreviewRiggedMesh": "UniRig: Preview Rigged Mesh",
        "UniRigExportPosedFBX": "UniRig: Export Posed FBX",
        "UniRigViewRigging": "UniRig: View Rigging",
        "UniRigDebugSkeleton": "UniRig: Debug Skeleton",
        "UniRigCompareSkeletons": "UniRig: Compare Skeletons",
        "UniRigExtractRestPose": "UniRig: Extract Rest Pose",
        "UniRigOrientationCheck": "UniRig: Orientation Check",
    }

    # Wrap all isolated nodes (comfy-env.toml is in this directory)
    wrapped = wrap_isolated_nodes(isolated_mappings, nodes_dir)
    NODE_CLASS_MAPPINGS.update(wrapped)
    NODE_DISPLAY_NAME_MAPPINGS.update(isolated_display)
    print(f"[UniRig] Isolated nodes loaded ({len(isolated_mappings)} nodes)")

except ImportError as e:
    print(f"[UniRig] comfy-env not installed, isolated nodes disabled: {e}")

print(f"[UniRig] Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
