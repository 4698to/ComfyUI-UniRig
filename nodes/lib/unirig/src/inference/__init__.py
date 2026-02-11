# UniRig inference module

# Direct preprocessing (bpy as Python module)
try:
    from .direct_preprocess import preprocess_mesh
except ImportError:
    # bpy not available - fallback to subprocess will be used
    preprocess_mesh = None
