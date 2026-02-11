"""
Main nodes (non-isolated) - basic mesh I/O operations.
"""

from .mesh_io import UniRigLoadMesh, UniRigSaveMesh

NODE_CLASS_MAPPINGS = {
    "UniRigLoadMesh": UniRigLoadMesh,
    "UniRigSaveMesh": UniRigSaveMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniRigLoadMesh": "UniRig: Load Mesh",
    "UniRigSaveMesh": "UniRig: Save Mesh",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
