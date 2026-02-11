"""
GPU nodes (isolated) - ML inference using CUDA and bpy.
Runs in isolated Python 3.11 environment with CUDA packages.
"""

from .model_loaders import UniRigLoadModel
from .auto_rig import UniRigAutoRig
from .skeleton_extraction import UniRigExtractSkeletonNew
from .skinning import UniRigApplySkinningMLNew
from .mia_model_loader import MIALoadModel
from .mia_auto_rig import MIAAutoRig

NODE_CLASS_MAPPINGS = {
    "UniRigLoadModel": UniRigLoadModel,
    "UniRigAutoRig": UniRigAutoRig,
    "UniRigExtractSkeletonNew": UniRigExtractSkeletonNew,
    "UniRigApplySkinningMLNew": UniRigApplySkinningMLNew,
    "MIALoadModel": MIALoadModel,
    "MIAAutoRig": MIAAutoRig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniRigLoadModel": "UniRig: Load Model",
    "UniRigAutoRig": "UniRig: Auto Rig",
    "UniRigExtractSkeletonNew": "UniRig: Extract Skeleton",
    "UniRigApplySkinningMLNew": "UniRig: Apply Skinning ML",
    "MIALoadModel": "MIA: Load Model",
    "MIAAutoRig": "MIA: Auto Rig",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
