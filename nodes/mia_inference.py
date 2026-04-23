"""
Make-It-Animatable inference wrapper.

Provides functions to load MIA models and run inference for humanoid rigging.
Uses vendored MIA code from lib/mia/ for model loading (no bpy dependency).

IMPORTANT: All heavy imports (bpy, numpy, torch, trimesh) are lazy-loaded inside
functions to ensure torch_cluster (via mia/model.py) loads BEFORE bpy initializes
its bundled libraries. This avoids a segfault caused by library conflicts.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING
import logging
import comfy.model_management

log = logging.getLogger("unirig")


def _emit_visible_log(message: str, *args) -> None:
    """
    Emit log messages both through logger and stdout so worker console always shows them.
    """
    log.info(message, *args)
    try:
        text = message % args if args else message
    except Exception:
        text = f"{message} {args}"
    print(f"[MIA] {text}")


# Type hints only - not imported at runtime
if TYPE_CHECKING:
    import numpy as np
    import torch
    import trimesh

# Lazy bpy availability check (don't import at module level!)
_HAS_BPY: Optional[bool] = None


def _check_bpy_available() -> bool:
    """Lazily check if bpy is available. Called only when needed."""
    global _HAS_BPY
    if _HAS_BPY is None:
        try:
            import bpy  # noqa: F401
            _HAS_BPY = True
        except ImportError:
            _HAS_BPY = False
    return _HAS_BPY

# Get paths relative to this file
UTILS_DIR = Path(__file__).parent.absolute()
NODE_DIR = UTILS_DIR.parent

# MIA models directory: ComfyUI/models/mia/
# Supports override via MIA_MODELS_PATH environment variable
try:
    import folder_paths
    _COMFY_MODELS_DIR = Path(folder_paths.models_dir)
except ImportError:
    # Fallback if not running in ComfyUI context
    _COMFY_MODELS_DIR = NODE_DIR.parent.parent / "models"

if os.environ.get('MIA_MODELS_PATH'):
    MIA_MODELS_DIR = Path(os.environ['MIA_MODELS_PATH'])
else:
    MIA_MODELS_DIR = _COMFY_MODELS_DIR / "mia"

# Required model files
MIA_MODEL_FILES = [
    "bw.pth",
    "bw_normal.pth",
    "joints.pth",
    "joints_coarse.pth",
    "pose.pth",
]

# Shared model cache from load_model.py (single source of truth)
from .load_model import _MODEL_CACHE


def ensure_mia_models() -> bool:
    """
    Ensure MIA model files are downloaded.
    Downloads from HuggingFace if not present.

    Returns:
        True if all models are available, False otherwise.
    """
    missing = [m for m in MIA_MODEL_FILES if not (MIA_MODELS_DIR / m).exists()]

    if not missing:
        return True

    log.info("Downloading missing models: %s", missing)

    try:
        from huggingface_hub import hf_hub_download
        import tempfile

        MIA_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        for model_file in missing:
            log.info("Downloading %s...", model_file)
            target_path = MIA_MODELS_DIR / model_file
            with tempfile.TemporaryDirectory(dir=str(MIA_MODELS_DIR)) as tmp_dir:
                hf_hub_download(
                    repo_id="jasongzy/Make-It-Animatable",
                    filename=f"output/best/new/{model_file}",
                    local_dir=tmp_dir,
                    local_dir_use_symlinks=False,
                )
                downloaded = Path(tmp_dir) / "output" / "best" / "new" / model_file
                downloaded.rename(target_path)

        log.info("All models downloaded to %s", MIA_MODELS_DIR)
        return True

    except Exception as e:
        log.error("Error downloading models: %s", e)
        return False


def _wrap_model_patcher(model, load_device, offload_device):
    """Wrap a model in ComfyUI ModelPatcher for VRAM management."""
    import comfy.model_patcher
    return comfy.model_patcher.ModelPatcher(
        model, load_device=load_device, offload_device=offload_device
    )


def load_mia_models(dtype: str = "fp32") -> str:
    """
    Load all MIA models wrapped in ComfyUI ModelPatcher for VRAM management.

    Args:
        dtype: Model precision - "bf16", "fp16", or "fp32".

    Returns:
        Cache key string (models stay in worker, can't be pickled to host).
    """
    import torch  # Lazy import - loads torch_cluster via mia/ BEFORE bpy

    cache_key = f"mia_models_dtype={dtype}"

    if cache_key in _MODEL_CACHE:
        log.info("Using cached models")
        return cache_key  # Return key, not models

    # Ensure models are downloaded
    if not ensure_mia_models():
        raise RuntimeError("Failed to download MIA models")

    load_device = comfy.model_management.get_torch_device()
    offload_device = torch.device("cpu")
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(dtype, torch.float32)
    log.info("Loading MIA models (dtype=%s, load_device=%s, offload_device=%s)...", torch_dtype, load_device, offload_device)

    # Import vendored MIA modules
    from .mia import PCAE, JOINTS_NUM, KINEMATIC_TREE

    N = 32768  # Number of points to sample
    hands_resample_ratio = 0.5
    geo_resample_ratio = 0.0
    hierarchical_ratio = hands_resample_ratio + geo_resample_ratio

    def _load_and_wrap(name, model_cls_kwargs, checkpoint):
        """Load a model, cast dtype, wrap in ModelPatcher."""
        log.info("Loading %s...", name)
        model = PCAE(**model_cls_kwargs)
        model.load(str(MIA_MODELS_DIR / checkpoint))
        model.to(dtype=torch_dtype).eval()
        # Verify all params+buffers are target dtype
        mismatched = [(n, p.dtype) for n, p in model.named_parameters() if p.dtype != torch_dtype]
        mismatched += [(n, b.dtype) for n, b in model.named_buffers() if b.dtype != torch_dtype]
        if mismatched:
            log.warning("  %s has %d dtype mismatches after .to(): %s",
                        name, len(mismatched), mismatched[:5])
        patcher = _wrap_model_patcher(model, load_device, offload_device)
        log.info("  %s: ModelPatcher(load=%s, offload=%s, dtype=%s)", name, load_device, offload_device, torch_dtype)
        return patcher

    # Load all models on CPU with target dtype, wrapped in ModelPatcher
    patcher_coarse = _load_and_wrap("joints_coarse", dict(
        N=N, input_normal=False, deterministic=True, output_dim=JOINTS_NUM,
        predict_bw=False, predict_joints=True, predict_joints_tail=True,
    ), "joints_coarse.pth")

    patcher_bw = _load_and_wrap("bw", dict(
        N=N, input_normal=False, input_attention=False, deterministic=True,
        hierarchical_ratio=hierarchical_ratio,
    ), "bw.pth")

    patcher_bw_normal = _load_and_wrap("bw_normal", dict(
        N=N, input_normal=True, input_attention=True, deterministic=True,
        hierarchical_ratio=hierarchical_ratio,
    ), "bw_normal.pth")

    patcher_joints = _load_and_wrap("joints", dict(
        N=N, input_normal=False, deterministic=True, hierarchical_ratio=hierarchical_ratio,
        output_dim=JOINTS_NUM, kinematic_tree=KINEMATIC_TREE,
        predict_bw=False, predict_joints=True, predict_joints_tail=True,
        joints_attn_causal=True,
    ), "joints.pth")

    patcher_pose = _load_and_wrap("pose", dict(
        N=N, input_normal=False, deterministic=True, hierarchical_ratio=hierarchical_ratio,
        output_dim=JOINTS_NUM, kinematic_tree=KINEMATIC_TREE,
        predict_bw=False, predict_pose_trans=True, pose_mode="ortho6d",
        pose_input_joints=True, pose_attn_causal=True,
    ), "pose.pth")

    models = {
        "backend": "make_it_animatable",
        "patcher_coarse": patcher_coarse,
        "patcher_bw": patcher_bw,
        "patcher_bw_normal": patcher_bw_normal,
        "patcher_joints": patcher_joints,
        "patcher_pose": patcher_pose,
        "dtype": torch_dtype,
        "N": N,
        "hands_resample_ratio": hands_resample_ratio,
        "geo_resample_ratio": geo_resample_ratio,
    }

    _MODEL_CACHE[cache_key] = models
    log.info("All MIA models loaded and wrapped in ModelPatcher")

    return cache_key  # Return key, not models (models can't be pickled to host)


def get_cached_models(cache_key: str) -> Dict[str, Any]:
    """Get models from cache by key."""
    if cache_key not in _MODEL_CACHE:
        raise RuntimeError(f"Models not loaded: {cache_key}")
    return _MODEL_CACHE[cache_key]


def run_mia_inference(
    mesh: "trimesh.Trimesh",
    models: Dict[str, Any],
    output_path: str,
    no_fingers: bool = True,
    use_normal: bool = False,
    reset_to_rest: bool = True,
    normalize_ground: bool = True,
    add_root_bone: bool = True,
) -> str:
    """
    Run Make-It-Animatable inference on a mesh.

    Args:
        mesh: Input trimesh object.
        models: Loaded MIA models from load_mia_models().
        output_path: Path for output FBX file.
        no_fingers: If True, merge finger weights to hand (for models without separate fingers).
        use_normal: If True, use normals for better weights when limbs are close.
        reset_to_rest: If True, transform output to T-pose rest position.
        normalize_ground: If True, align lowest vertex to ground plane (Y=0).
        add_root_bone: If True, add a synthetic Root bone at origin before export.

    Returns:
        Path to output FBX file.
    """
    import numpy as np  # Lazy import
    import folder_paths  # Lazy import

    # Use vendored MIA pipeline
    from .mia.pipeline import prepare_input, preprocess, infer, bw_post_process
    from .mia import BONES_IDX_DICT, KINEMATIC_TREE

    N = models["N"]

    # Let ComfyUI manage GPU memory for all models
    patchers = [
        models["patcher_coarse"],
        models["patcher_bw"],
        models["patcher_bw_normal"],
        models["patcher_joints"],
        models["patcher_pose"],
    ]
    comfy.model_management.load_models_gpu(patchers)
    device = patchers[0].load_device
    dtype = models["dtype"]

    log.info("Starting MIA inference (device=%s, dtype=%s)...", device, dtype)
    _emit_visible_log(
        "Options: no_fingers=%s, use_normal=%s, reset_to_rest=%s, normalize_ground=%s, add_root_bone=%s",
        no_fingers, use_normal, reset_to_rest, normalize_ground, add_root_bone
    )
    log.info("Input mesh: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))
    # Snapshot original input scale before any pipeline stage may mutate mesh in-place.
    original_bounds = mesh.bounds.copy()
    original_input_height = float(max(
        original_bounds[1][1] - original_bounds[0][1],
        original_bounds[1][2] - original_bounds[0][2],
    ))
    _emit_visible_log("Original input height snapshot: %.6f", original_input_height)

    # Prepare input
    log.info("Step 1/4: Preparing input (N=%d)...", N)
    data = prepare_input(
        mesh,
        N=N,
        hands_resample_ratio=models["hands_resample_ratio"],
        geo_resample_ratio=models["geo_resample_ratio"],
        get_normals=use_normal,
    )

    # Preprocess (normalize, coarse joint localization)
    log.info("Step 2/4: Preprocessing (model_coarse, dtype=%s)...", dtype)
    data = preprocess(
        data,
        model_coarse=models["patcher_coarse"].model,
        device=device,
        dtype=dtype,
        hands_resample_ratio=models["hands_resample_ratio"],
        geo_resample_ratio=models["geo_resample_ratio"],
        N=N,
    )

    # Run main inference
    log.info("Step 3/4: Running inference (model_bw, model_joints, model_pose, dtype=%s)...", dtype)
    data = infer(
        data,
        model_bw=models["patcher_bw"].model,
        model_bw_normal=models["patcher_bw_normal"].model,
        model_joints=models["patcher_joints"].model,
        model_pose=models["patcher_pose"].model,
        device=device,
        dtype=dtype,
        use_normal=use_normal,
    )

    # Post-process blend weights
    log.info("Step 4/4: Post-processing blend weights...")
    joints = data.joints
    head_idx = BONES_IDX_DICT["mixamorig:Head"]
    head_y = joints[..., head_idx, 4]  # tail y position (index 3:6 is tail)
    above_head_mask = data.verts[..., 1] >= head_y

    bw = bw_post_process(
        data.bw,
        bones_idx_dict=BONES_IDX_DICT,
        above_head_mask=above_head_mask,
        no_fingers=no_fingers,
    )

    # Prepare output data for Blender export
    joints_np = data.joints.squeeze(0).float().numpy()

    log.info("reset_to_rest=%s, data.pose is None: %s", reset_to_rest, data.pose is None)
    if data.pose is not None:
        log.info("Pose shape: %s", data.pose.shape)
        pose_debug_path = os.path.join(folder_paths.get_temp_directory(), "mia_pose_debug.npy")
        np.save(pose_debug_path, data.pose.squeeze(0).float().numpy())
        log.debug("Saved pose data to %s", pose_debug_path)

    output_data = {
        "mesh": data.mesh,
        "original_visual": data.original_visual,
        "gs": None,
        "joints": joints_np[..., :3],
        "joints_tail": joints_np[..., 3:] if joints_np.shape[-1] > 3 else None,
        "bw": bw.squeeze(0).float().numpy(),
        "pose": data.pose.squeeze(0).float().numpy() if reset_to_rest and data.pose is not None else None,
        "bones_idx_dict": BONES_IDX_DICT,
        "parent_indices": KINEMATIC_TREE.parent_indices,
        "pose_ignore_list": [],
        "input_mesh_height": original_input_height,
        "source_mesh_path": str(mesh.metadata.get("file_path", "")) if hasattr(mesh, "metadata") else "",
    }

    # Export to FBX using MIA's Blender integration
    log.info("Exporting to FBX...")
    _export_mia_fbx(
        output_data,
        output_path,
        no_fingers,
        reset_to_rest,
        normalize_ground,
        add_root_bone,
    )

    log.info("Inference complete: %s", output_path)
    return output_path


def _normalize_mia_export_space(
    input_meshes: list,
    joints: "np.ndarray",
    joints_tail: Optional["np.ndarray"],
    normalize_ground: bool,
):
    """
    Normalize MIA export space to keep feet on ground.
    """
    import numpy as np  # Lazy import

    mesh_min_y = None
    mesh_max_y = None
    for mesh_obj in input_meshes:
        verts = np.array([v.co for v in mesh_obj.data.vertices], dtype=np.float32)
        if len(verts) == 0:
            continue
        cur_min_y = float(verts[:, 1].min())
        cur_max_y = float(verts[:, 1].max())
        mesh_min_y = cur_min_y if mesh_min_y is None else min(mesh_min_y, cur_min_y)
        mesh_max_y = cur_max_y if mesh_max_y is None else max(mesh_max_y, cur_max_y)

    total_y_offset = 0.0
    ground_offset = 0.0

    if normalize_ground:
        if mesh_min_y is not None:
            ground_offset = -mesh_min_y
            total_y_offset += ground_offset

    if abs(total_y_offset) > 1e-8:
        for mesh_obj in input_meshes:
            mesh_data = mesh_obj.data
            for v in mesh_data.vertices:
                v.co[1] += total_y_offset
            mesh_data.update()
        joints[:, 1] += total_y_offset
        if joints_tail is not None:
            joints_tail[:, 1] += total_y_offset

    return joints, joints_tail


def _match_input_scale(
    input_meshes: list,
    joints: "np.ndarray",
    joints_tail: Optional["np.ndarray"],
    target_mesh_height: float,
):
    """
    Uniformly scale mesh and joints to match the input mesh height (Y axis).
    """
    import numpy as np  # Lazy import

    if target_mesh_height <= 1e-8:
        return joints, joints_tail

    cur_min = None
    cur_max = None
    for mesh_obj in input_meshes:
        verts = np.array([v.co for v in mesh_obj.data.vertices], dtype=np.float32)
        if len(verts) == 0:
            continue
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        cur_min = vmin if cur_min is None else np.minimum(cur_min, vmin)
        cur_max = vmax if cur_max is None else np.maximum(cur_max, vmax)

    if cur_min is None or cur_max is None:
        return joints, joints_tail

    # In this stage data is usually Y-up, but keeping max(Y, Z) is safer
    # for assets that may still carry a different up-axis convention.
    current_mesh_height = float(max(cur_max[1] - cur_min[1], cur_max[2] - cur_min[2]))
    if current_mesh_height <= 1e-8:
        return joints, joints_tail

    scale_factor = target_mesh_height / current_mesh_height
    if abs(scale_factor - 1.0) < 1e-6:
        return joints, joints_tail

    for mesh_obj in input_meshes:
        mesh_data = mesh_obj.data
        for v in mesh_data.vertices:
            v.co *= scale_factor
        mesh_data.update()

    joints *= scale_factor
    if joints_tail is not None:
        joints_tail *= scale_factor

    _emit_visible_log(
        "Applied MIA output scale correction: current_height=%.6f, target_height=%.6f, factor=%.6f",
        current_mesh_height, target_mesh_height, scale_factor
    )
    return joints, joints_tail


def _export_mia_fbx_direct(
    data: Dict[str, Any],
    output_path: str,
    remove_fingers: bool,
    reset_to_rest: bool,
    normalize_ground: bool,
    add_root_bone: bool,
    template_path: Path,
) -> None:
    """
    Export MIA results to FBX using bpy directly (inlined, no imports needed).
    """
    import tempfile
    import numpy as np  # Lazy import
    import bpy  # Lazy import - only imported here AFTER torch_cluster loaded
    from mathutils import Vector, Matrix

    armature_name = "Armature"
    if add_root_bone:
        armature_name = "Armature_Root"

    mesh = data["mesh"]
    joints = data["joints"]
    joints_tail = data.get("joints_tail")
    bw = data["bw"]
    pose = data.get("pose")
    bones_idx_dict = dict(data["bones_idx_dict"])
    input_mesh_height = float(data.get("input_mesh_height", 0.0))
    source_mesh_path = str(data.get("source_mesh_path") or "")

    log.debug("Mesh to export: %d verts, %d faces, visual=%s",
              len(mesh.vertices), len(mesh.faces),
              type(mesh.visual).__name__ if hasattr(mesh, 'visual') else 'none')
    parent_indices = data.get("parent_indices")

    def _save_blender_uv_ppm_debug(mesh_obj, out_path, size=1024):
        """Save Blender loop-UV wireframe as PPM for debugging."""
        mesh_data = mesh_obj.data
        if not mesh_data.uv_layers:
            _emit_visible_log("Skip Blender UV debug (no uv layers): %s", out_path)
            return

        uv_layer = mesh_data.uv_layers.active or mesh_data.uv_layers[0]
        img = np.full((size, size, 3), 255, dtype=np.uint8)

        def draw_line(x0, y0, x1, y1):
            dx = abs(x1 - x0)
            sx = 1 if x0 < x1 else -1
            dy = -abs(y1 - y0)
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            while True:
                if 0 <= x0 < size and 0 <= y0 < size:
                    img[y0, x0] = (0, 0, 0)
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x0 += sx
                if e2 <= dx:
                    err += dx
                    y0 += sy

        for poly in mesh_data.polygons:
            loops = poly.loop_indices
            n = len(loops)
            for i in range(n):
                la = loops[i]
                lb = loops[(i + 1) % n]
                uva = uv_layer.data[la].uv
                uvb = uv_layer.data[lb].uv
                x0 = int(max(0, min(size - 1, round(float(uva.x) * (size - 1)))))
                y0 = int(max(0, min(size - 1, round((1.0 - float(uva.y)) * (size - 1)))))
                x1 = int(max(0, min(size - 1, round(float(uvb.x) * (size - 1)))))
                y1 = int(max(0, min(size - 1, round((1.0 - float(uvb.y)) * (size - 1)))))
                draw_line(x0, y0, x1, y1)

        with open(out_path, "wb") as f:
            f.write(f"P6\n{size} {size}\n255\n".encode("ascii"))
            f.write(img.tobytes())
        _emit_visible_log("Saved Blender UV debug: %s", out_path)

    def _save_blender_uv_ppm_debug_obj_style(mesh_obj, out_path, size=1024):
        """
        Save Blender UV using OBJ-style vt/face-vt algorithm.
        This mirrors _save_obj_uv_ppm_debug semantics for A/B comparison.
        """
        mesh_data = mesh_obj.data
        if not mesh_data.uv_layers:
            _emit_visible_log("Skip Blender OBJ-style UV debug (no uv layers): %s", out_path)
            return

        uv_layer = mesh_data.uv_layers.active or mesh_data.uv_layers[0]
        vt_list = []
        face_vt_indices = []
        vt_map = {}

        for poly in mesh_data.polygons:
            poly_vt = []
            for li in poly.loop_indices:
                uv = uv_layer.data[li].uv
                # Quantize key to avoid tiny float noise producing duplicate vt entries.
                key = (round(float(uv.x), 7), round(float(uv.y), 7))
                if key not in vt_map:
                    vt_map[key] = len(vt_list)
                    vt_list.append(key)
                poly_vt.append(vt_map[key])
            if len(poly_vt) >= 2:
                face_vt_indices.append(poly_vt)

        if not vt_list or not face_vt_indices:
            _emit_visible_log("Skip Blender OBJ-style UV debug (empty vt/f-vt): %s", out_path)
            return

        uv = np.asarray(vt_list, dtype=np.float64)
        img = np.full((size, size, 3), 255, dtype=np.uint8)

        def draw_line(x0, y0, x1, y1):
            dx = abs(x1 - x0)
            sx = 1 if x0 < x1 else -1
            dy = -abs(y1 - y0)
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            while True:
                if 0 <= x0 < size and 0 <= y0 < size:
                    img[y0, x0] = (0, 0, 0)
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x0 += sx
                if e2 <= dx:
                    err += dx
                    y0 += sy

        for poly in face_vt_indices:
            n = len(poly)
            for i in range(n):
                a = poly[i]
                b = poly[(i + 1) % n]
                u0, v0 = float(uv[a, 0]), float(uv[a, 1])
                u1, v1 = float(uv[b, 0]), float(uv[b, 1])
                x0 = int(max(0, min(size - 1, round(u0 * (size - 1)))))
                y0 = int(max(0, min(size - 1, round((1.0 - v0) * (size - 1)))))
                x1 = int(max(0, min(size - 1, round(u1 * (size - 1)))))
                y1 = int(max(0, min(size - 1, round((1.0 - v1) * (size - 1)))))
                draw_line(x0, y0, x1, y1)

        with open(out_path, "wb") as f:
            f.write(f"P6\n{size} {size}\n255\n".encode("ascii"))
            f.write(img.tobytes())
        _emit_visible_log("Saved Blender OBJ-style UV debug: %s", out_path)

    def _load_obj_face_corner_uvs(obj_path):
        """
        Load OBJ vt values and expand to face-corner uv list in file face order.
        Returns list[(u, v)] aligned to concatenated polygon loop order.
        """
        vt_list = []
        corner_uvs = []
        with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("vt "):
                    parts = line.split()
                    if len(parts) >= 3:
                        vt_list.append((float(parts[1]), float(parts[2])))
                elif line.startswith("f "):
                    tokens = line.split()[1:]
                    poly = []
                    for t in tokens:
                        subs = t.split("/")
                        if len(subs) >= 2 and subs[1] != "":
                            vt_idx = int(subs[1])
                            if vt_idx < 0:
                                vt_idx = len(vt_list) + vt_idx + 1
                            vt_zero = vt_idx - 1
                            if 0 <= vt_zero < len(vt_list):
                                poly.append(vt_list[vt_zero])
                    if len(poly) >= 3:
                        corner_uvs.extend(poly)
        return corner_uvs

    def _load_mesh_face_corner_uvs_from_memory(src_mesh):
        """
        Build face-corner UV list from in-memory trimesh using mesh.faces -> mesh.visual.uv.
        Returns None when memory UVs are unavailable.
        """
        if (
            src_mesh is None
            or not hasattr(src_mesh, "faces")
            or src_mesh.faces is None
            or len(src_mesh.faces) == 0
            or not hasattr(src_mesh, "visual")
            or src_mesh.visual is None
            or not hasattr(src_mesh.visual, "uv")
            or src_mesh.visual.uv is None
            or len(src_mesh.visual.uv) == 0
        ):
            return None

        uv = np.asarray(src_mesh.visual.uv, dtype=np.float64)
        corner_uvs = []
        for face in np.asarray(src_mesh.faces):
            poly = []
            for vid in face:
                vid = int(vid)
                if vid < 0 or vid >= len(uv):
                    return None
                poly.append((float(uv[vid, 0]), float(uv[vid, 1])))
            if len(poly) >= 3:
                corner_uvs.extend(poly)
        return corner_uvs if corner_uvs else None

    def _copy_uvs_to_blender_meshes(meshes, src_mesh, obj_path):
        """
        Copy UVs onto Blender meshes (loop UV), trying in-memory mesh first,
        then falling back to source OBJ vt/f mapping.
        """
        try:
            total_loops = sum(len(m.data.loops) for m in meshes)
            corner_uvs = None #_load_mesh_face_corner_uvs_from_memory(src_mesh)
            source_tag = "memory-mesh"
            if not corner_uvs:
                if not obj_path or not os.path.isfile(obj_path) or not obj_path.lower().endswith(".obj"):
                    _emit_visible_log("Skip UV copy: memory UV unavailable and source obj invalid")
                    return False
                corner_uvs = _load_obj_face_corner_uvs(obj_path)
                source_tag = "source-obj"

            if not corner_uvs:
                _emit_visible_log("Skip UV copy: no usable corner UVs from memory/OBJ")
                return False

            if total_loops != len(corner_uvs):
                _emit_visible_log(
                    "Skip UV copy (%s): loop mismatch (blender=%d, source=%d)",
                    source_tag,
                    total_loops,
                    len(corner_uvs),
                )
                return False

            cursor = 0
            for mesh_obj in meshes:
                mesh_data = mesh_obj.data
                if not mesh_data.uv_layers:
                    mesh_data.uv_layers.new(name="UVMap")
                uv_layer = mesh_data.uv_layers.active or mesh_data.uv_layers[0]
                loop_count = len(mesh_data.loops)
                seg = corner_uvs[cursor:cursor + loop_count]
                if len(seg) != loop_count:
                    _emit_visible_log("Skip OBJ->Blender UV copy: segment length mismatch for %s", mesh_obj.name)
                    return False
                for i in range(loop_count):
                    uv_layer.data[i].uv = seg[i]
                mesh_data.update()
                cursor += loop_count

            _emit_visible_log("Copied UVs to Blender meshes successfully (source=%s).", source_tag)
            return True
        except Exception as e:
            _emit_visible_log("Failed UV copy to Blender meshes: %s", e)
            return False

    # Restore original visual (textures/materials) before export
    original_visual = data.get("original_visual")
    if original_visual is not None:
        mesh.visual = original_visual
        log.debug("Restored original visual: %s", type(original_visual).__name__)
    else:
        log.warning("No original_visual to restore")

    # Export-chain UV sanity check using OBJ-native vt/f indexing.
    # if source_mesh_path and source_mesh_path.lower().endswith(".obj") and os.path.isfile(source_mesh_path):
    #     try:
    #         from .mesh_io import _save_obj_uv_ppm_debug
    #         uv_ppm_path = output_path.rsplit(".", 1)[0] + "_uv_export_chain_obj_native.ppm"
    #         _save_obj_uv_ppm_debug(source_mesh_path, uv_ppm_path)
    #         _emit_visible_log("Saved export-chain OBJ-native UV debug: %s", uv_ppm_path)
    #     except Exception as e:
    #         _emit_visible_log("Failed export-chain OBJ-native UV debug: %s", e)

    # Export processed mesh to temp GLB for import into Blender
    temp_mesh_file = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
    mesh_path = temp_mesh_file.name
    temp_mesh_file.close()

    mesh.export(mesh_path)
    log.debug("Exported temp GLB: %s (%d bytes)", mesh_path, os.path.getsize(mesh_path))

    try:
        log.debug("Weights: %s, Joints: %s, Bones: %d", bw.shape, joints.shape, len(bones_idx_dict))

        # Reset scene and load template
        bpy.ops.wm.read_factory_settings(use_empty=True)
        old_objs = set(bpy.context.scene.objects)
        bpy.ops.import_scene.fbx(filepath=str(template_path))
        template_objs = list(set(bpy.context.scene.objects) - old_objs)

        # Find armature
        armature = None
        for obj in template_objs:
            if obj.type == "ARMATURE":
                armature = obj
                break
        if armature is None:
            raise RuntimeError("No armature found in template!")

        log.info("Loaded template armature: %s", armature.name)
        armature.name = armature_name

        # Capture template bone orientations (including z_axis for align_roll)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='EDIT')
        template_bone_data = {}
        for bone in armature.data.edit_bones:
            template_bone_data[bone.name] = {
                'roll': bone.roll,
                'z_axis': tuple(bone.z_axis),  # Capture Z-axis for align_roll
            }
        bpy.ops.object.mode_set(mode='OBJECT')

        # Clear pose transforms
        armature.animation_data_clear()
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.select_all(action="SELECT")
        bpy.ops.pose.transforms_clear()
        bpy.ops.object.mode_set(mode='OBJECT')

        # Load input mesh from temp GLB
        old_objs = set(bpy.context.scene.objects)
        bpy.ops.import_scene.gltf(filepath=mesh_path)
        new_objs = set(bpy.context.scene.objects) - old_objs
        input_meshes = [obj for obj in new_objs if obj.type == "MESH"]

        if not input_meshes:
            raise RuntimeError("No mesh found in input!")
        
        _emit_visible_log( f"Loaded -> {mesh_path}" )

        log.debug("Loaded %d mesh(es) from GLB", len(input_meshes))
        _copy_uvs_to_blender_meshes(input_meshes, mesh, source_mesh_path)
        # uv_debug_base = output_path.rsplit(".", 1)[0]
        # for idx, mesh_obj in enumerate(input_meshes):
        #     _save_blender_uv_ppm_debug(
        #         mesh_obj,
        #         f"{uv_debug_base}_uv_after_blender_import_{idx:02d}.ppm",
        #     )
        #     _save_blender_uv_ppm_debug_obj_style(
        #         mesh_obj,
        #         f"{uv_debug_base}_uv_after_blender_import_obj_style_{idx:02d}.ppm",
        #     )

        # Remove template meshes
        for obj in template_objs:
            if obj.type == "MESH":
                bpy.data.objects.remove(obj, do_unlink=True)

        # Remove finger bones if requested
        if remove_fingers:
            finger_prefixes = [
                "LeftHandThumb", "LeftHandIndex", "LeftHandMiddle", "LeftHandRing", "LeftHandPinky",
                "RightHandThumb", "RightHandIndex", "RightHandMiddle", "RightHandRing", "RightHandPinky",
                "mixamorig:LeftHandThumb", "mixamorig:LeftHandIndex", "mixamorig:LeftHandMiddle",
                "mixamorig:LeftHandRing", "mixamorig:LeftHandPinky",
                "mixamorig:RightHandThumb", "mixamorig:RightHandIndex", "mixamorig:RightHandMiddle",
                "mixamorig:RightHandRing", "mixamorig:RightHandPinky",
            ]
            bpy.context.view_layer.objects.active = armature
            bpy.ops.object.mode_set(mode='EDIT')
            bones_to_remove = []
            for bone in armature.data.edit_bones:
                for prefix in finger_prefixes:
                    if bone.name.startswith(prefix):
                        bones_to_remove.append(bone.name)
                        break
            for bone_name in bones_to_remove:
                bone = armature.data.edit_bones.get(bone_name)
                if bone:
                    armature.data.edit_bones.remove(bone)
                if bone_name in bones_idx_dict:
                    del bones_idx_dict[bone_name]
            bpy.ops.object.mode_set(mode='OBJECT')
            log.debug("Removed %d finger bones", len(bones_to_remove))

        # Save armature's world matrix and get scaling factor
        matrix_world = armature.matrix_world.copy()
        scaling = matrix_world.to_scale()[0]
        effective_scaling = 1.0
        log.debug("Effective scaling: %s", effective_scaling)

        # Reset armature to identity
        armature.matrix_world.identity()
        bpy.context.view_layer.update()

        # Bake object transforms first, so further vertex edits don't drift object origins.
        for mesh_obj in input_meshes:
            bpy.context.view_layer.objects.active = mesh_obj
            mesh_obj.select_set(True)
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            mesh_obj.select_set(False)

        # Transform mesh vertices: Y-Z swap and divide by scaling compensation.
        for mesh_obj in input_meshes:
            mesh_data = mesh_obj.data
            verts = np.array([v.co for v in mesh_data.vertices])
            new_y = verts[:, 2].copy()
            new_z = -verts[:, 1].copy()
            verts[:, 1] = new_y
            verts[:, 2] = new_z
            verts = verts / effective_scaling
            for i, v in enumerate(mesh_data.vertices):
                v.co = verts[i]
            mesh_data.update()

        # Set bones with the same scaling compensation used for mesh.
        joints_normalized = joints / effective_scaling
        joints_tail_normalized = joints_tail / effective_scaling if joints_tail is not None else None
        joints_normalized, joints_tail_normalized = _normalize_mia_export_space(
            input_meshes=input_meshes,
            joints=joints_normalized,
            joints_tail=joints_tail_normalized,
            normalize_ground=normalize_ground,
        )
        joints_normalized, joints_tail_normalized = _match_input_scale(
            input_meshes=input_meshes,
            joints=joints_normalized,
            joints_tail=joints_tail_normalized,
            target_mesh_height=input_mesh_height,
        )
        # Re-apply placement normalization after scaling.
        joints_normalized, joints_tail_normalized = _normalize_mia_export_space(
            input_meshes=input_meshes,
            joints=joints_normalized,
            joints_tail=joints_tail_normalized,
            normalize_ground=normalize_ground,
        )

        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='EDIT')

        # Update bone positions and apply template roll for Mixamo compatibility
        for bone in armature.data.edit_bones:
            bone.use_connect = False
            if bone.name in bones_idx_dict:
                idx = bones_idx_dict[bone.name]
                bone.head = Vector(joints_normalized[idx])
                if joints_tail_normalized is not None:
                    bone.tail = Vector(joints_tail_normalized[idx])
                # Apply template roll for Mixamo-compatible twist axis
                if bone.name in template_bone_data:
                    bone.roll = template_bone_data[bone.name]['roll']

        # Remove end bones not in prediction dict
        bones_to_remove = [b.name for b in armature.data.edit_bones if b.name not in bones_idx_dict]
        for bone_name in bones_to_remove:
            bone = armature.data.edit_bones.get(bone_name)
            if bone:
                armature.data.edit_bones.remove(bone)

        # Add a synthetic Root bone and parent root-level bones to it.
        # if add_root_bone:
        #     edit_bones = armature.data.edit_bones
        #     root_name = "Root"
        #     if edit_bones.get(root_name) is None:
        #         root_level_bones = [b for b in edit_bones if b.parent is None]
        #         if root_level_bones:
        #             hips_bone = edit_bones.get("mixamorig:Hips")
        #             main_root = hips_bone if hips_bone is not None else root_level_bones[0]
        #             root_bone = edit_bones.new(root_name)
        #             root_head = main_root.head.copy()
        #             root_head.x = 0.0
        #             root_head.y = 0.0
        #             root_head.z = 0.0
        #             root_tail = root_head.copy()
        #             root_tail.y += 0.05
        #             root_bone.head = root_head
        #             root_bone.tail = root_tail
        #             root_bone.use_connect = False

        #             for bone in root_level_bones:
        #                 if bone.name != root_name:
        #                     bone.parent = root_bone
        #                     bone.use_connect = False
        #             log.info("Added synthetic root bone '%s' with %d children", root_name, len(root_level_bones))

        bpy.ops.object.mode_set(mode='OBJECT')

        # Parent mesh to armature
        for mesh_obj in input_meshes:
            mesh_obj.parent = armature

        # Apply weights
        vertices_num = [len(m.data.vertices) for m in input_meshes]
        weights_list = np.split(bw, np.cumsum(vertices_num)[:-1])

        for mesh_obj, mesh_bw in zip(input_meshes, weights_list):
            mesh_data = mesh_obj.data
            mesh_obj.vertex_groups.clear()
            for bone_name, bone_index in bones_idx_dict.items():
                group = mesh_obj.vertex_groups.new(name=bone_name)
                for v in mesh_data.vertices:
                    v_w = mesh_bw[v.index, bone_index]
                    if v_w > 1e-3:
                        group.add([v.index], float(v_w), "REPLACE")
            mesh_data.update()

        # Add armature modifier
        for mesh_obj in input_meshes:
            mod = mesh_obj.modifiers.new(name=armature_name, type='ARMATURE')
            mod.object = armature
            mod.use_vertex_groups = True

        # Restore armature matrix with unit scale to keep exported size consistent.
        loc, rot, _ = matrix_world.decompose()
        armature.matrix_world = Matrix.LocRotScale(loc, rot, (1.0, 1.0, 1.0))
        bpy.context.view_layer.update()

        # Apply pose-to-rest if needed (imports helper functions from mia_export)
        if pose is not None and reset_to_rest and parent_indices is not None:
            log.info("Applying pose-to-rest transformation...")
            _apply_pose_to_rest_inline(armature, pose, bones_idx_dict, parent_indices, input_meshes, joints_normalized, template_bone_data)

        # Fix image filepaths and pack for FBX embedding
        # FBX exporter needs proper filepaths with filenames, not just directories
        log.debug("Fixing image filepaths for FBX export...")
        fbm_dir = output_path.rsplit('.', 1)[0] + '.fbm'
        os.makedirs(fbm_dir, exist_ok=True)

        for img in bpy.data.images:
            if img.size[0] > 0 and img.size[1] > 0:  # Valid image
                # Create a proper filepath with filename
                img_filename = f"{img.name}.png"
                img_filepath = os.path.join(fbm_dir, img_filename)

                # Save the image to disk first (FBX exporter needs this)
                old_filepath = img.filepath
                img.filepath_raw = img_filepath
                img.file_format = 'PNG'
                img.save()
                log.debug("Saved texture: %s", img_filepath)

                # Now pack it
                if img.packed_file is None:
                    try:
                        img.pack()
                        log.debug("Packed: %s", img.name)
                    except Exception as e:
                        log.debug("Failed to pack %s: %s", img.name, e)

        # Export FBX
        bpy.context.view_layer.update()
        # Force deterministic unit behavior to avoid 100x scale drift across DCC tools.
        scene_units = bpy.context.scene.unit_settings
        scene_units.system = 'METRIC'
        scene_units.scale_length = 0.01 #1.0
        _emit_visible_log(
            "FBX pre-export units: system=%s scale_length=%.6f armature_scale=(%.6f, %.6f, %.6f)",
            scene_units.system,
            scene_units.scale_length,
            armature.scale.x,
            armature.scale.y,
            armature.scale.z,
        )
        scene_blend_path = output_path.rsplit('.', 1)[0] + '.blend'
        bpy.ops.wm.save_as_mainfile(
            filepath=scene_blend_path,
            check_existing=False,
            copy=True,
        )
        _emit_visible_log("Saved debug Blender scene: %s", scene_blend_path)
        bpy.ops.export_scene.fbx(
            filepath=output_path,
            use_selection=False,
            object_types={'ARMATURE', 'MESH'},
            add_leaf_bones=False,
            bake_anim=False,
            path_mode='COPY',
            embed_textures=True,
            global_scale=1.0,
            apply_unit_scale=True,
            apply_scale_options='FBX_SCALE_UNITS',
            # axis_forward='-Z',
            # axis_up='Y',
        )
        log.info("Exported to: %s", output_path)

        # Also export GLB (better texture support for preview tools)
        glb_path = output_path.rsplit('.', 1)[0] + '.glb'
        bpy.ops.export_scene.gltf(
            filepath=glb_path,
            export_format='GLB',
            export_texcoords=True,
            export_normals=True,
            export_materials='EXPORT',
            export_image_format='AUTO',
        )
        log.debug("Also exported GLB: %s", glb_path)

    finally:
        # Clean up temp mesh file
        if os.path.exists(mesh_path):
            os.remove(mesh_path)


def _apply_pose_to_rest_inline(armature_obj, pose, bones_idx_dict, parent_indices, input_meshes, mia_joints, template_bone_data=None):
    """Apply MIA's pose prediction to transform skeleton from input pose to T-pose rest (inlined)."""
    import numpy as np  # Lazy import
    import bpy  # Lazy import
    from mathutils import Matrix

    def ortho6d_to_matrix(ortho6d):
        x_raw = ortho6d[:3]
        y_raw = ortho6d[3:6]
        x = x_raw / (np.linalg.norm(x_raw) + 1e-8)
        z = np.cross(x, y_raw)
        z = z / (np.linalg.norm(z) + 1e-8)
        y = np.cross(z, x)
        return np.column_stack([x, y, z])

    def get_rotation_about_point(rotation, point):
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = point - rotation @ point
        return transform

    joints = mia_joints
    K = pose.shape[0]

    # Convert ortho6d to rotation matrices
    rot_matrices = np.zeros((K, 3, 3))
    for i in range(K):
        rot_matrices[i] = ortho6d_to_matrix(pose[i])

    # Initialize transforms
    pose_global = np.zeros((K, 4, 4))
    for i in range(K):
        pose_global[i] = get_rotation_about_point(rot_matrices[i], joints[i])

    # Propagate through kinematic chain
    posed_joints = joints.copy()
    for i in range(1, K):
        parent_idx = parent_indices[i]
        parent_matrix = pose_global[parent_idx]
        posed_joints[i] = parent_matrix[:3, :3] @ joints[i] + parent_matrix[:3, 3]
        matrix = get_rotation_about_point(rot_matrices[i], joints[i])
        matrix[:3, 3] += posed_joints[i] - joints[i]
        pose_global[i] = matrix

    pose_global[0] = np.eye(4)

    # Finger prefixes to skip
    finger_prefixes = [
        "LeftHandThumb", "LeftHandIndex", "LeftHandMiddle", "LeftHandRing", "LeftHandPinky",
        "RightHandThumb", "RightHandIndex", "RightHandMiddle", "RightHandRing", "RightHandPinky",
        "mixamorig:LeftHandThumb", "mixamorig:LeftHandIndex", "mixamorig:LeftHandMiddle",
        "mixamorig:LeftHandRing", "mixamorig:LeftHandPinky",
        "mixamorig:RightHandThumb", "mixamorig:RightHandIndex", "mixamorig:RightHandMiddle",
        "mixamorig:RightHandRing", "mixamorig:RightHandPinky",
    ]

    def is_finger_bone(name):
        return any(name.startswith(p) for p in finger_prefixes)

    # Apply transforms in pose mode
    bpy.ops.object.mode_set(mode='POSE')
    for bone_name, idx in bones_idx_dict.items():
        pbone = armature_obj.pose.bones.get(bone_name)
        if pbone is None or is_finger_bone(bone_name):
            continue
        pose_matrix = Matrix(pose_global[idx].tolist())
        pbone.matrix = pose_matrix @ pbone.bone.matrix_local
        bpy.context.view_layer.update()

    # Clear bone locations (match original MIA behavior from app_blender.py:124)
    for bone_name in bones_idx_dict:
        pbone = armature_obj.pose.bones.get(bone_name)
        if pbone:
            pbone.location = (0, 0, 0)
    bpy.context.view_layer.update()

    bpy.ops.object.mode_set(mode='OBJECT')

    # Apply posed armature as new rest pose
    for mesh_obj in input_meshes:
        bpy.context.view_layer.objects.active = mesh_obj
        for mod in mesh_obj.modifiers:
            if mod.type == 'ARMATURE':
                bpy.ops.object.modifier_apply(modifier=mod.name)
                break

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Re-apply template bone orientations after armature_apply
    # (armature_apply recalculates rolls based on new bone directions, losing template rolls)
    # Use align_roll() with template Z-axis for correct orientation regardless of bone direction
    if template_bone_data:
        from mathutils import Vector
        bpy.ops.object.mode_set(mode='EDIT')
        roll_count = 0
        for bone in armature_obj.data.edit_bones:
            if bone.name in template_bone_data:
                template_data = template_bone_data[bone.name]
                if 'z_axis' in template_data:
                    # Use align_roll with template Z-axis for correct orientation
                    bone.align_roll(Vector(template_data['z_axis']))
                else:
                    # Fallback to direct roll if z_axis not available
                    bone.roll = template_data['roll']
                roll_count += 1
        bpy.ops.object.mode_set(mode='OBJECT')
        log.info("Re-applied template bone orientations to %s bones after pose-to-rest", roll_count)

    # Re-add armature modifier
    for mesh_obj in input_meshes:
        mod = mesh_obj.modifiers.new(name="Armature_Root", type='ARMATURE')
        mod.object = armature_obj
        mod.use_vertex_groups = True

    # Clear remaining pose transforms
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='SELECT')
    bpy.ops.pose.transforms_clear()
    bpy.ops.object.mode_set(mode='OBJECT')

    log.info("Skeleton transformed to rest pose")


def _export_mia_fbx(
    data: Dict[str, Any],
    output_path: str,
    remove_fingers: bool,
    reset_to_rest: bool,
    normalize_ground: bool,
    add_root_bone: bool,
) -> None:
    """
    Export MIA results to FBX using bpy directly.

    Uses bpy (Blender as Python module) which is available in the isolated _env_unirig.
    Falls back to subprocess method if bpy is not available.
    """
    # Get template path - use UniRig's bundled Mixamo template
    ASSETS_DIR = NODE_DIR / "assets"  # ComfyUI-UniRig/assets
    template_path = ASSETS_DIR / "animation_characters" / "mixamo.fbx"
    if not template_path.exists():
        raise FileNotFoundError(f"No Mixamo template found. Expected at: {template_path}")

    if not _check_bpy_available():
        raise RuntimeError(
            "bpy is not available. MIA FBX export requires bpy.\n"
            "Ensure you are running in the unirig isolated environment."
        )

    log.info("Exporting FBX via bpy...")
    _export_mia_fbx_direct(
        data,
        output_path,
        remove_fingers,
        reset_to_rest,
        normalize_ground,
        add_root_bone,
        template_path,
    )

    if not os.path.exists(output_path):
        raise RuntimeError(f"Export completed but output file not created: {output_path}")
