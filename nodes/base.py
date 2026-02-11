"""
Base setup and shared utilities for UniRig nodes.

Handles path configuration, Blender setup, and HuggingFace cache management.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import base64
from io import BytesIO

import folder_paths

# Try to import PIL for texture handling
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[UniRig] Warning: PIL not available, texture preview will be limited")


# Get paths relative to this file
NODE_DIR = Path(__file__).parent.parent.absolute()  # Go up from nodes/ to ComfyUI-UniRig/
LIB_DIR = Path(__file__).parent / "lib"  # lib is now inside nodes/
UNIRIG_PATH = str(LIB_DIR / "unirig")
BLENDER_PARSE_SKELETON = str(LIB_DIR / "blender_parse_skeleton.py")
BLENDER_EXTRACT_MESH_INFO = str(LIB_DIR / "blender_extract_mesh_info.py")

# Set up UniRig models directory in ComfyUI's models folder
# Only contains skeleton.safetensors and skin.safetensors - no HuggingFace cache
UNIRIG_MODELS_DIR = Path(folder_paths.models_dir) / "unirig"
UNIRIG_MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ['UNIRIG_MODELS_DIR'] = str(UNIRIG_MODELS_DIR)

print(f"[UniRig] Models directory: {UNIRIG_MODELS_DIR}")

# Find Blender executable
# Detection priority:
# 1. BLENDER_PATH environment variable (explicit override)
# 2. System PATH ('blender' command)
# 3. Local tools/blender/ directory (installed by comfy-env)
# 4. Print instructions (no auto-download)

import shutil

BLENDER_EXE = None

# 1. Check explicit override first (BLENDER_PATH)
if os.environ.get('BLENDER_PATH'):
    blender_path = os.environ.get('BLENDER_PATH')
    if os.path.isfile(blender_path):
        BLENDER_EXE = blender_path
        print(f"[UniRig] Using Blender from BLENDER_PATH: {BLENDER_EXE}")
    else:
        print(f"[UniRig] Warning: BLENDER_PATH set but file not found: {blender_path}")

# 2. Check system PATH and ComfyUI/tools/ via comfy_env
if BLENDER_EXE is None:
    try:
        from comfy_env.tools import find_blender
        # Blender is installed to ComfyUI/tools/blender/ (shared across all nodes)
        COMFYUI_ROOT = NODE_DIR.parent.parent  # custom_nodes/../.. = ComfyUI/
        blender_exe = find_blender(COMFYUI_ROOT / "tools" / "blender")
        if blender_exe:
            BLENDER_EXE = str(blender_exe)
            print(f"[UniRig] Found Blender: {BLENDER_EXE}")
    except ImportError:
        # Fallback if comfy_env not available
        blender_in_path = shutil.which('blender')
        if blender_in_path:
            BLENDER_EXE = blender_in_path
            print(f"[UniRig] Found Blender in PATH: {BLENDER_EXE}")

# 3. Print instructions if not found
if BLENDER_EXE is None:
    print("[UniRig] WARNING: Blender not found!")
    print("[UniRig] Skeleton extraction nodes require Blender 4.2+")
    print("[UniRig] Options:")
    print("[UniRig]   1. Install Blender and add to PATH")
    print("[UniRig]   2. Set BLENDER_PATH environment variable")
    print("[UniRig]   3. Run: python install.py")

# Set environment variable for subprocesses
if BLENDER_EXE:
    os.environ['BLENDER_EXE'] = BLENDER_EXE

# Add local UniRig to path
if UNIRIG_PATH not in sys.path:
    sys.path.insert(0, UNIRIG_PATH)


def decode_texture_to_comfy_image(texture_data_base64: str):
    """
    Decode base64 texture to ComfyUI IMAGE format (torch tensor).

    Args:
        texture_data_base64: Base64-encoded image data

    Returns:
        tuple: (torch tensor [1, H, W, 3], width, height) or (None, 0, 0)
    """
    if not texture_data_base64 or not HAS_PIL:
        return None, 0, 0

    try:
        # Decode base64
        image_data = base64.b64decode(texture_data_base64)
        pil_image = PILImage.open(BytesIO(image_data))

        # Convert to RGB if necessary
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Convert to numpy array
        img_array = np.array(pil_image).astype(np.float32) / 255.0

        # Convert to torch tensor [1, H, W, 3] for ComfyUI
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        return img_tensor, pil_image.width, pil_image.height

    except Exception as e:
        print(f"[UniRig] Error decoding texture: {e}")
        return None, 0, 0


def create_placeholder_texture(width: int = 256, height: int = 256, text: str = "No Texture"):
    """
    Create a placeholder image when no texture is available.

    Args:
        width: Image width
        height: Image height
        text: Text to display (not currently rendered, just for reference)

    Returns:
        torch.Tensor: Placeholder image tensor [1, H, W, 3]
    """
    try:
        # Create a gray image with text
        img_array = np.full((height, width, 3), 0.3, dtype=np.float32)

        # Add a simple pattern to indicate placeholder
        # Create a grid pattern
        for i in range(0, height, 32):
            img_array[i:i+2, :, :] = 0.4
        for j in range(0, width, 32):
            img_array[:, j:j+2, :] = 0.4

        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        return img_tensor

    except Exception as e:
        print(f"[UniRig] Error creating placeholder: {e}")
        # Return minimal gray image
        return torch.full((1, 64, 64, 3), 0.3)


def normalize_skeleton(vertices: np.ndarray) -> tuple:
    """
    Normalize skeleton vertices to [-1, 1] range.

    Args:
        vertices: Array of vertex positions

    Returns:
        tuple: (normalized_vertices, normalization_params)
            normalization_params contains 'center' and 'scale' for denormalization
    """
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    center = (min_coords + max_coords) / 2
    vertices_centered = vertices - center
    scale = (max_coords - min_coords).max() / 2

    if scale > 0:
        vertices_normalized = vertices_centered / scale
    else:
        vertices_normalized = vertices_centered

    normalization_params = {
        'center': center,
        'scale': scale,
        'min_coords': min_coords,
        'max_coords': max_coords
    }

    return vertices_normalized, normalization_params


