"""
UniRig ComfyUI nodes package.

Organized by dependency environment:
- main/    : Non-isolated nodes (trimesh, numpy)
- blender/ : Isolated nodes (Python 3.11 + bpy)
- gpu/     : Isolated nodes (Python 3.11 + CUDA + bpy)
"""

from pathlib import Path

# Export base module utilities for backwards compatibility
from .base import (
    NODE_DIR,
    LIB_DIR,
    UNIRIG_PATH,
    UNIRIG_MODELS_DIR,
    BLENDER_PARSE_SKELETON,
    BLENDER_EXTRACT_MESH_INFO,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ==============================================================================
# Main nodes (no isolation needed)
# ==============================================================================
from .main import NODE_CLASS_MAPPINGS as main_mappings
from .main import NODE_DISPLAY_NAME_MAPPINGS as main_display
NODE_CLASS_MAPPINGS.update(main_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(main_display)
print(f"[UniRig] Main nodes loaded ({len(main_mappings)} nodes)")

# ==============================================================================
# Isolated nodes - wrapped for subprocess execution
# ==============================================================================
try:
    from comfy_env import wrap_isolated_nodes

    nodes_dir = Path(__file__).parent

    # Blender nodes (isolated - needs Python 3.11 for bpy)
    try:
        from .blender import NODE_CLASS_MAPPINGS as blender_mappings
        from .blender import NODE_DISPLAY_NAME_MAPPINGS as blender_display
        blender_wrapped = wrap_isolated_nodes(blender_mappings, nodes_dir / "blender")
        NODE_CLASS_MAPPINGS.update(blender_wrapped)
        NODE_DISPLAY_NAME_MAPPINGS.update(blender_display)
        print(f"[UniRig] Blender nodes loaded ({len(blender_mappings)} nodes, isolated)")
    except ImportError as e:
        print(f"[UniRig] Blender nodes not available: {e}")

    # GPU nodes (isolated - needs Python 3.11 + CUDA + bpy)
    try:
        from .gpu import NODE_CLASS_MAPPINGS as gpu_mappings
        from .gpu import NODE_DISPLAY_NAME_MAPPINGS as gpu_display
        gpu_wrapped = wrap_isolated_nodes(gpu_mappings, nodes_dir / "gpu")
        NODE_CLASS_MAPPINGS.update(gpu_wrapped)
        NODE_DISPLAY_NAME_MAPPINGS.update(gpu_display)
        print(f"[UniRig] GPU nodes loaded ({len(gpu_mappings)} nodes, isolated)")
    except ImportError as e:
        print(f"[UniRig] GPU nodes not available: {e}")

except ImportError:
    print("[UniRig] comfy-env not installed, isolated nodes disabled")
    print("[UniRig] Install with: pip install comfy-env")

print(f"[UniRig] Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
