"""
Blender nodes (isolated) - FBX/animation operations using bpy.
Runs in isolated Python 3.11 environment with bpy.
"""

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

NODE_CLASS_MAPPINGS = {
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

NODE_DISPLAY_NAME_MAPPINGS = {
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

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
