"""UniRig Nodes."""

import pathlib
import comfy_sparse_attn
from comfy_sparse_attn import setup_link
_PKG = pathlib.Path(comfy_sparse_attn.__file__).parent
setup_link(_PKG / "sparse.py",           "sparse.py")
setup_link(_PKG / "ops_sparse.py",       "ops_sparse.py")
setup_link(_PKG / "attention_sparse.py", "attention_sparse.py")
del pathlib, comfy_sparse_attn, setup_link, _PKG

from .mesh_io import UniRigLoadMesh, UniRigSaveMesh, on_custom_loaded as mesh_io_on_custom_loaded
from .load_model import UniRigLoadModel, MIALoadModel
from .auto_rig import UniRigAutoRig
from .skeleton_extraction import UniRigExtractSkeletonNew
from .skinning import UniRigApplySkinningMLNew
from .mia_auto_rig import MIAAutoRig
from .animation import UniRigApplyAnimation
from .skeleton_io import (
    UniRigLoadRiggedMesh,
    UniRigPreviewRiggedMesh,
    UniRigExportPosedFBX,
    UniRigViewRigging,
    UniRigDebugSkeleton,
    UniRigCompareSkeletons,
)
from .orientation_check import UniRigOrientationCheck

NODE_CLASS_MAPPINGS = {
    "UniRigLoadMesh": UniRigLoadMesh,
    "UniRigSaveMesh": UniRigSaveMesh,
    "UniRigLoadModel": UniRigLoadModel,
    "UniRigAutoRig": UniRigAutoRig,
    "UniRigExtractSkeletonNew": UniRigExtractSkeletonNew,
    "UniRigApplySkinningMLNew": UniRigApplySkinningMLNew,
    "MIALoadModel": MIALoadModel,
    "MIAAutoRig": MIAAutoRig,
    "UniRigApplyAnimation": UniRigApplyAnimation,
    "UniRigLoadRiggedMesh": UniRigLoadRiggedMesh,
    "UniRigPreviewRiggedMesh": UniRigPreviewRiggedMesh,
    "UniRigExportPosedFBX": UniRigExportPosedFBX,
    "UniRigViewRigging": UniRigViewRigging,
    "UniRigDebugSkeleton": UniRigDebugSkeleton,
    "UniRigCompareSkeletons": UniRigCompareSkeletons,
    "UniRigOrientationCheck": UniRigOrientationCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniRigLoadMesh": "UniRig: Load Mesh",
    "UniRigSaveMesh": "UniRig: Save Mesh",
    "UniRigLoadModel": "UniRig: Load Model",
    "UniRigAutoRig": "UniRig: Auto Rig",
    "UniRigExtractSkeletonNew": "UniRig: Extract Skeleton",
    "UniRigApplySkinningMLNew": "UniRig: Apply Skinning ML",
    "MIALoadModel": "MIA: Load Model",
    "MIAAutoRig": "MIA: Auto Rig",
    "UniRigApplyAnimation": "UniRig: Apply Animation",
    "UniRigLoadRiggedMesh": "UniRig: Load Rigged Mesh",
    "UniRigPreviewRiggedMesh": "UniRig: Preview Rigged Mesh",
    "UniRigExportPosedFBX": "UniRig: Export Posed FBX",
    "UniRigViewRigging": "UniRig: View Rigging",
    "UniRigDebugSkeleton": "UniRig: Debug Skeleton",
    "UniRigCompareSkeletons": "UniRig: Compare Skeletons",
    "UniRigOrientationCheck": "UniRig: Orientation Check",
}


def on_custom_loaded(app):
    mesh_io_on_custom_loaded(app)
