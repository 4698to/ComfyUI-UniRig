"""
UniRig ComfyUI nodes package.
"""

from .base import (
    NODE_DIR,
    LIB_DIR,
    UNIRIG_PATH,
    UNIRIG_MODELS_DIR,
    BLENDER_EXE,
    BLENDER_PARSE_SKELETON,
    BLENDER_EXTRACT_MESH_INFO,
)

from .model_loaders import UniRigLoadModel
from .auto_rig import UniRigAutoRig
from .skeleton_io import (
    UniRigLoadRiggedMesh,
    UniRigPreviewRiggedMesh,
    UniRigExportPosedFBX,
    UniRigViewRigging,
    UniRigDebugSkeleton,
    UniRigCompareSkeletons,
)
from .mesh_io import UniRigLoadMesh, UniRigSaveMesh
from .animation import UniRigApplyAnimation

# MIA (Make-It-Animatable) nodes
from .mia_model_loader import MIALoadModel
from .mia_auto_rig import MIAAutoRig

# Utility nodes
from .orientation_check import UniRigOrientationCheck

NODE_CLASS_MAPPINGS = {
    "UniRigLoadModel": UniRigLoadModel,
    "UniRigAutoRig": UniRigAutoRig,
    "UniRigLoadRiggedMesh": UniRigLoadRiggedMesh,
    "UniRigPreviewRiggedMesh": UniRigPreviewRiggedMesh,
    "UniRigExportPosedFBX": UniRigExportPosedFBX,
    "UniRigViewRigging": UniRigViewRigging,
    "UniRigDebugSkeleton": UniRigDebugSkeleton,
    "UniRigCompareSkeletons": UniRigCompareSkeletons,
    "UniRigLoadMesh": UniRigLoadMesh,
    "UniRigSaveMesh": UniRigSaveMesh,
    "UniRigApplyAnimation": UniRigApplyAnimation,
    # MIA nodes
    "MIALoadModel": MIALoadModel,
    "MIAAutoRig": MIAAutoRig,
    # Utility nodes
    "UniRigOrientationCheck": UniRigOrientationCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniRigLoadModel": "UniRig: Load Model",
    "UniRigAutoRig": "UniRig: Auto Rig",
    "UniRigLoadRiggedMesh": "UniRig: Load Rigged Mesh",
    "UniRigPreviewRiggedMesh": "UniRig: Preview Rigged Mesh",
    "UniRigExportPosedFBX": "UniRig: Export Posed FBX",
    "UniRigViewRigging": "UniRig: View Rigging",
    "UniRigDebugSkeleton": "UniRig: Debug Skeleton",
    "UniRigCompareSkeletons": "UniRig: Compare Skeletons",
    "UniRigLoadMesh": "UniRig: Load Mesh",
    "UniRigSaveMesh": "UniRig: Save Mesh",
    "UniRigApplyAnimation": "UniRig: Apply Animation",
    # MIA nodes
    "MIALoadModel": "MIA: Load Model",
    "MIAAutoRig": "MIA: Auto Rig",
    # Utility nodes
    "UniRigOrientationCheck": "UniRig: Orientation Check",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
