"""
Direct inference for UniRig models - no subprocess, no Lightning Trainer.

This module provides simple Python functions to run inference directly,
keeping models loaded in memory for fast repeated inference.

Uses ComfyUI ModelPatcher for proper memory management.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import logging
import comfy.model_management
import comfy.model_patcher
import comfy.utils

log = logging.getLogger("unirig")
# Shared model cache from load_model.py (single source of truth)
from ...load_model import _MODEL_CACHE


def sample_mesh_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_samples: int = 2048,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample points uniformly on mesh surface using barycentric coordinates.

    Args:
        vertices: Mesh vertices (V, 3)
        faces: Face indices (F, 3)
        num_samples: Number of points to sample
        seed: Random seed

    Returns:
        Tuple of (sampled_points, sampled_normals) each (num_samples, 3)
    """
    rng = np.random.RandomState(seed)

    # Compute face areas for weighted sampling
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Cross product gives 2x area
    cross = np.cross(v1 - v0, v2 - v0)
    face_areas = np.linalg.norm(cross, axis=1) / 2

    # Normalize face normals
    face_normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-8)

    # Sample faces weighted by area
    probs = face_areas / face_areas.sum()
    face_indices = rng.choice(len(faces), size=num_samples, p=probs)

    # Sample barycentric coordinates
    r1 = rng.random(num_samples)
    r2 = rng.random(num_samples)

    # Ensure point is inside triangle
    sqrt_r1 = np.sqrt(r1)
    u = 1 - sqrt_r1
    v = sqrt_r1 * (1 - r2)
    w = sqrt_r1 * r2

    # Get sampled points
    sampled_v0 = vertices[faces[face_indices, 0]]
    sampled_v1 = vertices[faces[face_indices, 1]]
    sampled_v2 = vertices[faces[face_indices, 2]]

    points = (
        u[:, None] * sampled_v0 +
        v[:, None] * sampled_v1 +
        w[:, None] * sampled_v2
    )

    # Get normals for sampled faces
    normals = face_normals[face_indices]

    return points.astype(np.float32), normals.astype(np.float32)


def normalize_vertices(vertices: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize vertices to [-1, 1] range.

    Args:
        vertices: Input vertices (N, 3)

    Returns:
        Tuple of (normalized_vertices, normalization_params)
    """
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    center = (min_coords + max_coords) / 2
    scale = (max_coords - min_coords).max() / 2

    if scale > 0:
        normalized = (vertices - center) / scale
    else:
        normalized = vertices - center

    params = {
        'center': center,
        'scale': scale,
        'min_coords': min_coords,
        'max_coords': max_coords,
    }

    return normalized.astype(np.float32), params


def _get_device():
    """Get the best available device."""
    return comfy.model_management.get_torch_device()


def _load_skeleton_model(checkpoint_path: str, dtype=None, attn_backend: str = "auto"):
    """
    Load the skeleton (AR) model directly.

    Uses meta device for zero-memory init, comfy.utils for safe loading.

    Args:
        checkpoint_path: Path to skeleton.safetensors
        dtype: Model precision (torch.bfloat16, torch.float16, etc.)
        attn_backend: Attention backend ('auto', 'flash_attn', 'sdpa')

    Returns:
        (model, tokenizer) tuple - model on CPU with requested dtype
    """
    import os
    from pathlib import Path
    from box import Box
    import yaml
    from .tokenizer_parse import get_tokenizer
    from .tokenizer_spec import TokenizerConfig
    from .model_parse import get_model
    from . import unirig_ar

    # Override attention backend before model creation
    if attn_backend == "sdpa":
        unirig_ar.FLASH_ATTN_AVAILABLE = False
        log.info("Attention backend: sdpa (flash_attn disabled)")
    elif attn_backend == "flash_attn":
        if not unirig_ar.FLASH_ATTN_AVAILABLE:
            raise RuntimeError("flash_attn requested but not installed")
        log.info("Attention backend: flash_attn")
    else:
        log.info("Attention backend: auto (flash_attn=%s)", unirig_ar.FLASH_ATTN_AVAILABLE)

    unirig_path = Path(__file__).parent
    original_cwd = os.getcwd()
    os.chdir(unirig_path)

    try:
        config_dir = unirig_path / 'configs'

        with open(config_dir / 'model' / 'unirig_ar_350m_1024_81920_float32.yaml') as f:
            model_config = Box(yaml.safe_load(f))

        with open(config_dir / 'tokenizer' / 'tokenizer_parts_articulationxl_256.yaml') as f:
            tokenizer_config = Box(yaml.safe_load(f))

        tokenizer = get_tokenizer(config=TokenizerConfig.parse(config=tokenizer_config))

        # Build model on meta device (zero memory, no random init)
        with torch.device("meta"):
            model = get_model(tokenizer=tokenizer, **model_config)
    finally:
        os.chdir(original_cwd)

    # Load weights with comfy.utils (safe, memory-efficient)
    log.info("Loading skeleton weights from %s", checkpoint_path)
    state_dict = comfy.utils.load_torch_file(checkpoint_path)

    # Remove 'model.' prefix if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[6:] if k.startswith('model.') else k
        cleaned_state_dict[new_key] = v

    # assign=True for meta device -> direct weight assignment
    model.load_state_dict(cleaned_state_dict, strict=True, assign=True)
    model.eval()

    if dtype is not None:
        model = model.to(dtype=dtype)
        log.info("Skeleton model dtype: %s", dtype)

    log.info("Skeleton model ready (on CPU, will be moved to GPU by ModelPatcher)")
    return model, tokenizer


def _load_skin_model(checkpoint_path: str, dtype=None, attn_backend: str = "auto"):
    """
    Load the skinning model directly.

    Uses meta device for zero-memory init, comfy.utils for safe loading.

    Args:
        checkpoint_path: Path to skin.safetensors
        dtype: Model precision (torch.bfloat16, torch.float16, etc.)
        attn_backend: Attention backend ('auto', 'flash_attn', 'sdpa')

    Returns:
        Loaded model on CPU with requested dtype
    """
    import os
    from pathlib import Path
    from box import Box
    import yaml
    from .model_parse import get_model

    log.info("Skin attention backend: ComfyUI optimized_attention")

    unirig_path = Path(__file__).parent
    original_cwd = os.getcwd()
    os.chdir(unirig_path)

    try:
        config_dir = unirig_path / 'configs'

        with open(config_dir / 'model' / 'unirig_skin.yaml') as f:
            model_config = Box(yaml.safe_load(f))

        # Build model on meta device (zero memory, no random init)
        with torch.device("meta"):
            model = get_model(tokenizer=None, **model_config)
    finally:
        os.chdir(original_cwd)

    # Load weights with comfy.utils (safe, memory-efficient)
    log.info("Loading skin weights from %s", checkpoint_path)
    state_dict = comfy.utils.load_torch_file(checkpoint_path)

    # Remove 'model.' prefix if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[6:] if k.startswith('model.') else k
        cleaned_state_dict[new_key] = v

    # assign=True for meta device -> direct weight assignment
    model.load_state_dict(cleaned_state_dict, strict=True, assign=True)

    model.eval()

    if dtype is not None:
        model = model.to(dtype=dtype)
        log.info("Skin model dtype: %s", dtype)

    log.info("Skin model ready (on CPU, will be moved to GPU by ModelPatcher)")
    return model


def get_skeleton_model(
    checkpoint_path: str,
    dtype=None,
    attn_backend: str = "auto",
    force_reload: bool = False,
):
    """Get or load cached skeleton model wrapped in ModelPatcher.

    Returns:
        (ModelPatcher, tokenizer) tuple
    """
    cache_key = f"skeleton:{checkpoint_path}"
    if cache_key not in _MODEL_CACHE or force_reload:
        model, tokenizer = _load_skeleton_model(
            checkpoint_path, dtype=dtype, attn_backend=attn_backend
        )
        load_device = comfy.model_management.get_torch_device()
        offload_device = torch.device("cpu")
        patcher = comfy.model_patcher.ModelPatcher(
            model, load_device=load_device, offload_device=offload_device
        )
        _MODEL_CACHE[cache_key] = (patcher, tokenizer)
        log.info("Skeleton model cached with ModelPatcher (load=%s, offload=%s)", load_device, offload_device)
    return _MODEL_CACHE[cache_key]


def get_skin_model(
    checkpoint_path: str,
    dtype=None,
    attn_backend: str = "auto",
    force_reload: bool = False,
):
    """Get or load cached skin model wrapped in ModelPatcher.

    Returns:
        ModelPatcher instance
    """
    cache_key = f"skin:{checkpoint_path}"
    if cache_key not in _MODEL_CACHE or force_reload:
        model = _load_skin_model(
            checkpoint_path, dtype=dtype, attn_backend=attn_backend
        )
        load_device = comfy.model_management.get_torch_device()
        offload_device = torch.device("cpu")
        patcher = comfy.model_patcher.ModelPatcher(
            model, load_device=load_device, offload_device=offload_device
        )
        _MODEL_CACHE[cache_key] = patcher
        log.info("Skin model cached with ModelPatcher (load=%s, offload=%s)", load_device, offload_device)
    return _MODEL_CACHE[cache_key]


@torch.no_grad()
def predict_skeleton(
    vertices: np.ndarray,
    normals: np.ndarray,
    checkpoint_path: str,
    cls: str = "articulationxl",
    max_new_tokens: int = 2048,
    dtype=None,
    attn_backend: str = "auto",
    **generate_kwargs,
) -> Dict[str, np.ndarray]:
    """
    Predict skeleton from mesh vertices and normals.

    Uses ComfyUI ModelPatcher for GPU memory management.

    Args:
        vertices: Mesh vertices (N, 3), should be normalized to [-1, 1]
        normals: Vertex normals (N, 3)
        checkpoint_path: Path to skeleton.safetensors
        cls: Class name for conditioning (default: "articulationxl")
        max_new_tokens: Maximum tokens to generate
        dtype: Model precision (torch.bfloat16, torch.float16, etc.)
        attn_backend: Attention backend ('auto', 'flash_attn', 'sdpa')
        **generate_kwargs: Additional kwargs for generation (num_beams, etc.)

    Returns:
        Dict with 'joints', 'parents', 'names' etc from detokenization
    """
    patcher, tokenizer = get_skeleton_model(
        checkpoint_path, dtype=dtype, attn_backend=attn_backend
    )

    # Let ComfyUI manage GPU memory
    comfy.model_management.load_models_gpu([patcher])
    device = patcher.load_device

    # Convert to tensors on the model's device
    vertices_t = torch.from_numpy(vertices).float().to(device)
    normals_t = torch.from_numpy(normals).float().to(device)

    # Default generation kwargs (matching ar_inference_articulationxl.yaml)
    default_kwargs = {
        'max_new_tokens': max_new_tokens,
        'num_beams': 15,
        'do_sample': True,
        'top_k': 5,
        'top_p': 0.95,
        'repetition_penalty': 3.0,
        'temperature': 1.5,
    }
    default_kwargs.update(generate_kwargs)

    # Run generation
    model = patcher.model
    result = model.generate(
        vertices=vertices_t,
        normals=normals_t,
        cls=cls,
        **default_kwargs,
    )

    # Convert result to dict of numpy arrays
    output = {
        'joints': np.array(result.joints) if result.joints is not None else None,
        'parents': np.array(result.parents) if result.parents is not None else None,
        'names': result.names if hasattr(result, 'names') else None,
    }

    # Add any other attributes from DetokenizeOutput
    for attr in ['tails', 'bone_names', 'root_idx']:
        if hasattr(result, attr):
            val = getattr(result, attr)
            if val is not None:
                output[attr] = np.array(val) if isinstance(val, (list, tuple)) else val

    return output


@torch.no_grad()
def predict_skinning(
    vertices: np.ndarray,
    normals: np.ndarray,
    joints: np.ndarray,
    parents: np.ndarray,
    checkpoint_path: str,
    faces: Optional[np.ndarray] = None,
    tails: Optional[np.ndarray] = None,
    voxel_grid_size: int = 196,
    dtype=None,
    attn_backend: str = "auto",
) -> np.ndarray:
    """
    Predict skinning weights from mesh and skeleton.

    Uses ComfyUI ModelPatcher for GPU memory management.

    Args:
        vertices: Mesh vertices (N, 3)
        normals: Vertex normals (N, 3)
        joints: Joint positions (J, 3)
        parents: Parent indices (J,)
        checkpoint_path: Path to skin.safetensors
        faces: Mesh faces (F, 3) - required for voxel_skin computation
        tails: Bone tail positions (J, 3) - if None, computed from joints
        voxel_grid_size: Grid size for voxel_skin (default 196)
        dtype: Model precision (torch.bfloat16, torch.float16, etc.)
        attn_backend: Attention backend ('auto', 'flash_attn', 'sdpa')

    Returns:
        Skinning weights (N, J)
    """
    patcher = get_skin_model(
        checkpoint_path, dtype=dtype, attn_backend=attn_backend
    )

    # Let ComfyUI manage GPU memory
    comfy.model_management.load_models_gpu([patcher])
    device = patcher.load_device

    num_joints = len(joints)
    num_vertices = len(vertices)

    # Compute tails if not provided (use joint positions as tails)
    if tails is None:
        # Default: tails = joints (bone has zero length, point bone)
        tails = joints.copy()

    # Compute voxel_skin if faces provided
    if faces is not None:
        # Import voxelization functions
        from .data_vertex_group import voxelization, voxel_skin

        log.info("Computing voxel_skin (grid=%s)...", voxel_grid_size)

        # Voxelize the mesh
        grid_coords = voxelization(
            vertices=vertices,
            faces=faces,
            grid=voxel_grid_size,
            backend='trimesh',
        )

        # Compute voxel skin weights
        voxel_skin_weights = voxel_skin(
            grid=voxel_grid_size,
            grid_coords=grid_coords,
            joints=joints,
            vertices=vertices,
            faces=faces,
            alpha=0.5,
            link_dis=0.00001,
            grid_query=7,
            vertex_query=1,
            grid_weight=3.0,
        )
        voxel_skin_weights = np.nan_to_num(voxel_skin_weights, nan=0., posinf=0., neginf=0.)
        log.info("voxel_skin shape: %s", voxel_skin_weights.shape)
    else:
        # Fallback: zero voxel_skin
        voxel_skin_weights = np.zeros((num_joints, num_vertices), dtype=np.float32)

    # Prepare batch
    # Note: offset is for PTv3 sparse convolutions - indicates where each batch element ends
    # For a single batch with N vertices, offset = [N]
    batch = {
        'vertices': torch.from_numpy(vertices).float().unsqueeze(0).to(device),
        'normals': torch.from_numpy(normals).float().unsqueeze(0).to(device),
        'joints': torch.from_numpy(joints).float().unsqueeze(0).to(device),
        'tails': torch.from_numpy(tails).float().unsqueeze(0).to(device),
        'voxel_skin': torch.from_numpy(voxel_skin_weights).float().unsqueeze(0).to(device),
        'parents': torch.from_numpy(parents).long().unsqueeze(0).to(device),
        'num_bones': torch.tensor([num_joints], dtype=torch.long, device=device),
        'offset': torch.tensor([num_vertices], dtype=torch.long, device=device),  # PTv3 batch offset for single sample
        'path': ['direct_inference'],
    }

    # Run prediction
    model = patcher.model
    result = model.predict_step(batch)

    # Extract skin weights - result is a list of tensors, one per batch
    skin_weights = result[0]  # Get first (only) batch item
    if isinstance(skin_weights, torch.Tensor):
        skin_weights = skin_weights.cpu().numpy()

    return skin_weights


@torch.no_grad()
def predict_skeleton_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    skeleton_checkpoint: str,
    num_samples: int = 2048,
    cls: str = "articulationxl",
    max_new_tokens: int = 2048,
    seed: int = 42,
    dtype=None,
    attn_backend: str = "auto",
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    High-level function: predict skeleton from raw mesh.

    Handles all preprocessing (normalization, sampling) internally.

    Args:
        vertices: Raw mesh vertices (V, 3)
        faces: Face indices (F, 3)
        skeleton_checkpoint: Path to skeleton.safetensors
        num_samples: Number of surface points to sample
        cls: Class name for conditioning
        max_new_tokens: Maximum tokens to generate
        seed: Random seed for sampling
        dtype: Model precision (torch.bfloat16, torch.float16, etc.)
        attn_backend: Attention backend ('auto', 'flash_attn', 'sdpa')

    Returns:
        Tuple of (skeleton_dict, normalization_params)
        - skeleton_dict: Dict with 'joints', 'parents', 'names', etc.
        - normalization_params: Dict with 'center', 'scale' for denormalization
    """
    # 1. Normalize mesh vertices
    norm_vertices, norm_params = normalize_vertices(vertices)

    # 2. Sample surface points
    sampled_points, sampled_normals = sample_mesh_surface(
        norm_vertices,
        faces,
        num_samples=num_samples,
        seed=seed,
    )

    # 3. Run skeleton prediction
    skeleton = predict_skeleton(
        vertices=sampled_points,
        normals=sampled_normals,
        checkpoint_path=skeleton_checkpoint,
        cls=cls,
        max_new_tokens=max_new_tokens,
        dtype=dtype,
        attn_backend=attn_backend,
    )

    return skeleton, norm_params


@torch.no_grad()
def rig_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    skeleton_checkpoint: str,
    skin_checkpoint: str,
    num_samples: int = 2048,
    cls: str = "articulationxl",
    seed: int = 42,
    dtype=None,
    attn_backend: str = "auto",
) -> Dict[str, Any]:
    """
    Complete rigging pipeline: mesh -> skeleton -> skinning weights.

    Args:
        vertices: Raw mesh vertices (V, 3)
        faces: Face indices (F, 3)
        skeleton_checkpoint: Path to skeleton.safetensors
        skin_checkpoint: Path to skin.safetensors
        num_samples: Number of surface points to sample
        cls: Class name for conditioning
        seed: Random seed
        dtype: Model precision (torch.bfloat16, torch.float16, etc.)
        attn_backend: Attention backend ('auto', 'flash_attn', 'sdpa')

    Returns:
        Dict with:
            - 'joints': Joint positions (J, 3) in normalized space
            - 'parents': Parent indices (J,)
            - 'names': Joint names
            - 'skin_weights': Skinning weights (num_samples, J)
            - 'sampled_vertices': Sampled surface points (num_samples, 3)
            - 'sampled_normals': Sampled normals (num_samples, 3)
            - 'normalization': Dict with center/scale for denormalization
    """
    # 1. Normalize mesh vertices
    norm_vertices, norm_params = normalize_vertices(vertices)

    # 2. Sample surface points
    sampled_points, sampled_normals = sample_mesh_surface(
        norm_vertices,
        faces,
        num_samples=num_samples,
        seed=seed,
    )

    # 3. Predict skeleton
    log.info("Predicting skeleton...")
    skeleton = predict_skeleton(
        vertices=sampled_points,
        normals=sampled_normals,
        checkpoint_path=skeleton_checkpoint,
        cls=cls,
        dtype=dtype,
        attn_backend=attn_backend,
    )

    if skeleton['joints'] is None:
        raise RuntimeError("Skeleton prediction failed - no joints generated")

    # 4. Predict skinning weights
    log.info("Predicting skinning weights...")
    skin_weights = predict_skinning(
        vertices=sampled_points,
        normals=sampled_normals,
        joints=skeleton['joints'],
        parents=skeleton['parents'],
        checkpoint_path=skin_checkpoint,
        dtype=dtype,
        attn_backend=attn_backend,
    )

    return {
        'joints': skeleton['joints'],
        'parents': skeleton['parents'],
        'names': skeleton.get('names'),
        'tails': skeleton.get('tails'),
        'skin_weights': skin_weights,
        'sampled_vertices': sampled_points,
        'sampled_normals': sampled_normals,
        'normalization': norm_params,
    }
