"""
Make-It-Animatable inference wrapper.

Provides functions to load MIA models and run inference for humanoid rigging.
Uses vendored MIA code from lib/mia/ for model loading (no bpy dependency).
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import MagicMock

# Try to import bpy directly (works in isolated _env_unirig with Blender 4.2 as Python module)
# Only mock if bpy is truly unavailable (inference-only mode without Blender)
try:
    import bpy
    _HAS_BPY = True
except ImportError:
    from unittest.mock import MagicMock
    sys.modules['bpy'] = MagicMock()
    _HAS_BPY = False

import numpy as np
import torch
import trimesh

# Get paths relative to this file
UTILS_DIR = Path(__file__).parent.absolute()
NODE_DIR = UTILS_DIR.parent
LIB_DIR = UTILS_DIR  # mia/ is in nodes_gpu/ (same directory as this file)

# MIA models directory (downloaded from HuggingFace)
# Stored in ComfyUI's models folder: ComfyUI/models/mia/
# HuggingFace downloads to: {local_dir}/output/best/new/
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
    MIA_MODELS_DIR = _COMFY_MODELS_DIR / "mia" / "output" / "best" / "new"

# MIA_PATH is the local_dir for HuggingFace downloads
MIA_PATH = _COMFY_MODELS_DIR / "mia"

# Required model files
MIA_MODEL_FILES = [
    "bw.pth",
    "bw_normal.pth",
    "joints.pth",
    "joints_coarse.pth",
    "pose.pth",
]

# Global cache for loaded models
_MIA_MODEL_CACHE: Dict[str, Any] = {}


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

    print(f"[MIA] Downloading missing models: {missing}")

    try:
        from huggingface_hub import hf_hub_download

        MIA_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        for model_file in missing:
            print(f"[MIA] Downloading {model_file}...")
            hf_hub_download(
                repo_id="jasongzy/Make-It-Animatable",
                filename=f"output/best/new/{model_file}",
                local_dir=str(MIA_PATH),
                local_dir_use_symlinks=False,
            )

        print(f"[MIA] All models downloaded successfully")
        return True

    except Exception as e:
        print(f"[MIA] Error downloading models: {e}")
        return False


def load_mia_models(cache_to_gpu: bool = True) -> str:
    """
    Load all MIA models into memory.

    Args:
        cache_to_gpu: If True, keep models on GPU for faster inference.

    Returns:
        Cache key string (models stay in worker, can't be pickled to host).
    """
    global _MIA_MODEL_CACHE

    cache_key = f"mia_models_gpu={cache_to_gpu}"

    if cache_key in _MIA_MODEL_CACHE:
        print(f"[MIA] Using cached models")
        return cache_key  # Return key, not models

    # Ensure models are downloaded
    if not ensure_mia_models():
        raise RuntimeError("Failed to download MIA models")

    device = torch.device("cuda" if torch.cuda.is_available() and cache_to_gpu else "cpu")
    print(f"[MIA] Loading models to {device}...")

    # Import vendored MIA modules from lib/mia (no bpy dependency)
    if str(LIB_DIR) not in sys.path:
        sys.path.insert(0, str(LIB_DIR))
    from mia import PCAE, JOINTS_NUM, KINEMATIC_TREE

    N = 32768  # Number of points to sample
    hands_resample_ratio = 0.5
    geo_resample_ratio = 0.0
    hierarchical_ratio = hands_resample_ratio + geo_resample_ratio

    # Load coarse joints model (for preprocessing)
    print(f"[MIA] Loading joints_coarse model...")
    model_coarse = PCAE(
        N=N,
        input_normal=False,
        deterministic=True,
        output_dim=JOINTS_NUM,
        predict_bw=False,
        predict_joints=True,
        predict_joints_tail=True,
    )
    model_coarse.load(str(MIA_MODELS_DIR / "joints_coarse.pth")).to(device).eval()

    # Load blend weights model
    print(f"[MIA] Loading bw model...")
    model_bw = PCAE(
        N=N,
        input_normal=False,
        input_attention=False,
        deterministic=True,
        hierarchical_ratio=hierarchical_ratio,
    )
    model_bw.load(str(MIA_MODELS_DIR / "bw.pth")).to(device).eval()

    # Load blend weights with normals model
    print(f"[MIA] Loading bw_normal model...")
    model_bw_normal = PCAE(
        N=N,
        input_normal=True,
        input_attention=True,  # Checkpoint trained with attention
        deterministic=True,
        hierarchical_ratio=hierarchical_ratio,
    )
    model_bw_normal.load(str(MIA_MODELS_DIR / "bw_normal.pth")).to(device).eval()

    # Load joints model
    print(f"[MIA] Loading joints model...")
    model_joints = PCAE(
        N=N,
        input_normal=False,
        deterministic=True,
        hierarchical_ratio=hierarchical_ratio,
        output_dim=JOINTS_NUM,
        kinematic_tree=KINEMATIC_TREE,
        predict_bw=False,
        predict_joints=True,
        predict_joints_tail=True,
        joints_attn_causal=True,  # Match original MIA config
    )
    model_joints.load(str(MIA_MODELS_DIR / "joints.pth")).to(device).eval()

    # Load pose model
    print(f"[MIA] Loading pose model...")
    model_pose = PCAE(
        N=N,
        input_normal=False,
        deterministic=True,
        hierarchical_ratio=hierarchical_ratio,
        output_dim=JOINTS_NUM,
        kinematic_tree=KINEMATIC_TREE,
        predict_bw=False,
        predict_pose_trans=True,
        pose_mode="ortho6d",  # Match original MIA config
        pose_input_joints=True,
        pose_attn_causal=True,  # Match original MIA config
    )
    model_pose.load(str(MIA_MODELS_DIR / "pose.pth")).to(device).eval()

    models = {
        "backend": "make_it_animatable",
        "model_coarse": model_coarse,
        "model_bw": model_bw,
        "model_bw_normal": model_bw_normal,
        "model_joints": model_joints,
        "model_pose": model_pose,
        "device": device,
        "cache_to_gpu": cache_to_gpu,
        "N": N,
        "hands_resample_ratio": hands_resample_ratio,
        "geo_resample_ratio": geo_resample_ratio,
    }

    _MIA_MODEL_CACHE[cache_key] = models
    print(f"[MIA] All models loaded successfully")

    return cache_key  # Return key, not models (models can't be pickled to host)


def get_cached_models(cache_key: str) -> Dict[str, Any]:
    """Get models from cache by key."""
    if cache_key not in _MIA_MODEL_CACHE:
        raise RuntimeError(f"Models not loaded: {cache_key}")
    return _MIA_MODEL_CACHE[cache_key]


def run_mia_inference(
    mesh: trimesh.Trimesh,
    models: Dict[str, Any],
    output_path: str,
    no_fingers: bool = True,
    use_normal: bool = False,
    reset_to_rest: bool = True,
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

    Returns:
        Path to output FBX file.
    """
    # Use vendored pipeline from lib/mia
    if str(LIB_DIR) not in sys.path:
        sys.path.insert(0, str(LIB_DIR))
    from mia.pipeline import prepare_input, preprocess, infer, bw_post_process
    from mia import BONES_IDX_DICT, KINEMATIC_TREE

    device = models["device"]
    N = models["N"]

    print(f"[MIA] Starting inference...")
    print(f"[MIA] Options: no_fingers={no_fingers}, use_normal={use_normal}, reset_to_rest={reset_to_rest}")

    # Debug: Check input mesh visual before any processing
    print(f"[MIA] Input mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    if hasattr(mesh, 'visual'):
        print(f"[MIA] Input mesh visual type: {type(mesh.visual).__name__}")
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            print(f"[MIA]   Has UV coords: {mesh.visual.uv.shape}")
        if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
            mat = mesh.visual.material
            print(f"[MIA]   Material type: {type(mat).__name__}")
            if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                print(f"[MIA]   Has baseColorTexture!")
            if hasattr(mat, 'image') and mat.image is not None:
                print(f"[MIA]   Has image texture!")
    else:
        print(f"[MIA] WARNING: Input mesh has no visual attribute!")

    # Prepare input
    print(f"[MIA] Preparing input...")
    data = prepare_input(
        mesh,
        N=N,
        hands_resample_ratio=models["hands_resample_ratio"],
        geo_resample_ratio=models["geo_resample_ratio"],
        get_normals=use_normal,
    )

    # Debug: Check data.mesh visual after prepare_input
    if hasattr(data, 'mesh') and data.mesh is not None:
        print(f"[MIA] After prepare_input: data.mesh has {len(data.mesh.vertices)} vertices")
        if hasattr(data.mesh, 'visual'):
            print(f"[MIA]   data.mesh visual type: {type(data.mesh.visual).__name__}")
            if hasattr(data.mesh.visual, 'uv') and data.mesh.visual.uv is not None:
                print(f"[MIA]   Has UV coords: {data.mesh.visual.uv.shape}")
            if hasattr(data.mesh.visual, 'material') and data.mesh.visual.material is not None:
                print(f"[MIA]   Has material: {type(data.mesh.visual.material).__name__}")
        else:
            print(f"[MIA]   WARNING: data.mesh has no visual attribute!")
    else:
        print(f"[MIA] WARNING: data.mesh is None or missing!")

    # Preprocess (normalize, coarse joint localization)
    print(f"[MIA] Preprocessing...")
    data = preprocess(
        data,
        model_coarse=models["model_coarse"],
        device=device,
        hands_resample_ratio=models["hands_resample_ratio"],
        geo_resample_ratio=models["geo_resample_ratio"],
        N=N,
    )

    # Run main inference
    print(f"[MIA] Running model inference...")
    data = infer(
        data,
        model_bw=models["model_bw"],
        model_bw_normal=models["model_bw_normal"],
        model_joints=models["model_joints"],
        model_pose=models["model_pose"],
        device=device,
        use_normal=use_normal,
    )

    # Post-process blend weights
    print(f"[MIA] Post-processing...")
    # Get head position for above-head mask
    joints = data.joints
    head_idx = BONES_IDX_DICT[f"mixamorig:Head"]
    head_y = joints[..., head_idx, 4]  # tail y position (index 3:6 is tail)
    above_head_mask = data.verts[..., 1] >= head_y

    bw = bw_post_process(
        data.bw,
        bones_idx_dict=BONES_IDX_DICT,
        above_head_mask=above_head_mask,
        no_fingers=no_fingers,
    )

    # Prepare output data for Blender export
    joints_np = data.joints.squeeze(0).numpy()

    # Debug: check pose data availability
    print(f"[MIA] reset_to_rest={reset_to_rest}, data.pose is None: {data.pose is None}")
    if data.pose is not None:
        print(f"[MIA] Pose shape: {data.pose.shape}")
        # Save pose to known location for debugging
        pose_debug_path = os.path.join(folder_paths.get_temp_directory(), "mia_pose_debug.npy")
        np.save(pose_debug_path, data.pose.squeeze(0).numpy())
        print(f"[MIA] Saved pose data to {pose_debug_path}")

    output_data = {
        "mesh": data.mesh,
        "gs": None,
        "joints": joints_np[..., :3],
        "joints_tail": joints_np[..., 3:] if joints_np.shape[-1] > 3 else None,
        "bw": bw.squeeze(0).numpy(),
        "pose": data.pose.squeeze(0).numpy() if reset_to_rest and data.pose is not None else None,
        "bones_idx_dict": BONES_IDX_DICT,
        "parent_indices": KINEMATIC_TREE.parent_indices,  # For kinematic chain
        "pose_ignore_list": [],
    }

    # Export to FBX using MIA's Blender integration
    print(f"[MIA] Exporting to FBX...")
    _export_mia_fbx(output_data, output_path, no_fingers, reset_to_rest)

    print(f"[MIA] Inference complete: {output_path}")
    return output_path


def _export_mia_fbx_direct(
    data: Dict[str, Any],
    output_path: str,
    remove_fingers: bool,
    reset_to_rest: bool,
    template_path: Path,
) -> None:
    """
    Export MIA results to FBX using bpy directly (inlined, no imports needed).
    """
    import tempfile
    from mathutils import Vector, Matrix

    mesh = data["mesh"]
    joints = data["joints"]
    joints_tail = data.get("joints_tail")
    bw = data["bw"]
    pose = data.get("pose")
    bones_idx_dict = dict(data["bones_idx_dict"])

    # Debug: Check mesh visual before export
    print(f"[MIA Export] Mesh to export: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    if hasattr(mesh, 'visual'):
        print(f"[MIA Export] Mesh visual type: {type(mesh.visual).__name__}")
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            print(f"[MIA Export]   Has UV coords: {mesh.visual.uv.shape}")
        if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
            mat = mesh.visual.material
            print(f"[MIA Export]   Material type: {type(mat).__name__}")
            if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                print(f"[MIA Export]   Has baseColorTexture!")
            if hasattr(mat, 'image') and mat.image is not None:
                print(f"[MIA Export]   Has image texture!")
    else:
        print(f"[MIA Export] WARNING: Mesh has no visual attribute!")
    parent_indices = data.get("parent_indices")

    # Export processed mesh to temp GLB for import into Blender
    # Textures are preserved because mesh.visual is now intact (fix in mesh_io.py)
    temp_mesh_file = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
    mesh_path = temp_mesh_file.name
    temp_mesh_file.close()

    # Debug: Check what we're exporting
    print(f"[MIA Export] About to export mesh to: {mesh_path}")
    print(f"[MIA Export] Mesh has visual: {hasattr(mesh, 'visual')}")
    if hasattr(mesh, 'visual') and mesh.visual is not None:
        visual = mesh.visual
        print(f"[MIA Export] Visual kind: {visual.kind if hasattr(visual, 'kind') else 'unknown'}")
        if hasattr(visual, 'uv') and visual.uv is not None:
            print(f"[MIA Export] Has UV: shape={visual.uv.shape}")
        if hasattr(visual, 'material') and visual.material is not None:
            mat = visual.material
            print(f"[MIA Export] Material: {type(mat).__name__}")
            # Check for PBRMaterial attributes
            for attr in ['baseColorTexture', 'image', 'baseColorFactor']:
                if hasattr(mat, attr):
                    val = getattr(mat, attr)
                    if val is not None:
                        print(f"[MIA Export]   {attr}: {type(val).__name__}")

    mesh.export(mesh_path)
    print(f"[MIA Export] Exported GLB size: {os.path.getsize(mesh_path)} bytes")

    try:
        print(f"[MIA Export] Weights: {bw.shape}, Joints: {joints.shape}, Bones: {len(bones_idx_dict)}")

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

        print(f"[MIA Export] Loaded template armature: {armature.name}")

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

        # Debug: Check materials after import
        for obj in input_meshes:
            print(f"[MIA Export] Imported mesh '{obj.name}': {len(obj.data.vertices)} verts")
            if obj.data.materials:
                print(f"[MIA Export]   Has {len(obj.data.materials)} material(s)")
                for i, mat in enumerate(obj.data.materials):
                    if mat:
                        print(f"[MIA Export]   Material[{i}]: {mat.name}")
                        if mat.use_nodes and mat.node_tree:
                            for node in mat.node_tree.nodes:
                                if node.type == 'TEX_IMAGE' and node.image:
                                    print(f"[MIA Export]     Texture: {node.image.name} ({node.image.size[0]}x{node.image.size[1]})")
            else:
                print(f"[MIA Export]   No materials!")

        if not input_meshes:
            raise RuntimeError("No mesh found in input!")

        print(f"[MIA Export] Loaded {len(input_meshes)} mesh(es)")

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
            print(f"[MIA Export] Removed {len(bones_to_remove)} finger bones")

        # Save armature's world matrix and get scaling factor
        matrix_world = armature.matrix_world.copy()
        scaling = matrix_world.to_scale()[0]

        # Reset armature to identity
        armature.matrix_world.identity()
        bpy.context.view_layer.update()

        # Transform mesh vertices: Y-Z swap and divide by scaling
        for mesh_obj in input_meshes:
            mesh_data = mesh_obj.data
            verts = np.array([v.co for v in mesh_data.vertices])
            new_y = verts[:, 2].copy()
            new_z = -verts[:, 1].copy()
            verts[:, 1] = new_y
            verts[:, 2] = new_z
            verts = verts / scaling
            for i, v in enumerate(mesh_data.vertices):
                v.co = verts[i]
            mesh_data.update()

        # Set bones with joints/scaling
        joints_normalized = joints / scaling
        joints_tail_normalized = joints_tail / scaling if joints_tail is not None else None

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
            mod = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
            mod.object = armature
            mod.use_vertex_groups = True

        # Restore armature matrix
        armature.matrix_world = matrix_world
        bpy.context.view_layer.update()

        # Apply pose-to-rest if needed (imports helper functions from mia_export)
        if pose is not None and reset_to_rest and parent_indices is not None:
            print(f"[MIA Export] Applying pose-to-rest transformation...")
            _apply_pose_to_rest_inline(armature, pose, bones_idx_dict, parent_indices, input_meshes, joints_normalized, template_bone_data)

        # Debug: Check materials before FBX export
        print(f"[MIA Export] Pre-FBX export material check:")
        for mesh_obj in input_meshes:
            print(f"[MIA Export]   Mesh '{mesh_obj.name}': {len(mesh_obj.data.vertices)} verts")
            if mesh_obj.data.materials:
                print(f"[MIA Export]     Has {len(mesh_obj.data.materials)} material(s)")
                for i, mat in enumerate(mesh_obj.data.materials):
                    if mat:
                        print(f"[MIA Export]     Material[{i}]: {mat.name}")
                        if mat.use_nodes and mat.node_tree:
                            for node in mat.node_tree.nodes:
                                if node.type == 'TEX_IMAGE' and node.image:
                                    img = node.image
                                    print(f"[MIA Export]       Texture: {img.name} ({img.size[0]}x{img.size[1]}) packed={img.packed_file is not None}")
            else:
                print(f"[MIA Export]     WARNING: No materials!")

        # Fix image filepaths and pack for FBX embedding
        # FBX exporter needs proper filepaths with filenames, not just directories
        print(f"[MIA Export] Fixing image filepaths for FBX export...")
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
                print(f"[MIA Export]   Saved texture: {img_filepath}")

                # Now pack it
                if img.packed_file is None:
                    try:
                        img.pack()
                        print(f"[MIA Export]   Packed: {img.name}")
                    except Exception as e:
                        print(f"[MIA Export]   Failed to pack {img.name}: {e}")

        # Export FBX
        bpy.context.view_layer.update()
        bpy.ops.export_scene.fbx(
            filepath=output_path,
            use_selection=False,
            object_types={'ARMATURE', 'MESH'},
            add_leaf_bones=False,
            bake_anim=False,
            path_mode='COPY',
            embed_textures=True,
        )
        print(f"[MIA Export] Exported to: {output_path}")

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
        print(f"[MIA Export] Also exported GLB: {glb_path}")

        # Debug: Check output file size
        if os.path.exists(output_path):
            fbx_size = os.path.getsize(output_path)
            print(f"[MIA Export] FBX file size: {fbx_size} bytes")

    finally:
        # Clean up temp mesh file
        if os.path.exists(mesh_path):
            os.remove(mesh_path)


def _apply_pose_to_rest_inline(armature_obj, pose, bones_idx_dict, parent_indices, input_meshes, mia_joints, template_bone_data=None):
    """Apply MIA's pose prediction to transform skeleton from input pose to T-pose rest (inlined)."""
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
        print(f"[MIA Export] Re-applied template bone orientations to {roll_count} bones after pose-to-rest")

    # Re-add armature modifier
    for mesh_obj in input_meshes:
        mod = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
        mod.object = armature_obj
        mod.use_vertex_groups = True

    # Clear remaining pose transforms
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='SELECT')
    bpy.ops.pose.transforms_clear()
    bpy.ops.object.mode_set(mode='OBJECT')

    print(f"[MIA Export] Skeleton transformed to rest pose")


def _export_mia_fbx(
    data: Dict[str, Any],
    output_path: str,
    remove_fingers: bool,
    reset_to_rest: bool,
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
        # Fallback to MIA template if available
        mia_template = MIA_PATH / "data/Mixamo/character/Ch14_nonPBR.fbx"
        if mia_template.exists():
            template_path = mia_template
        else:
            raise FileNotFoundError(f"No Mixamo template found. Expected at: {template_path}")

    if _HAS_BPY:
        # Use bpy directly (preferred - no subprocess needed)
        print("[MIA] Using bpy directly for FBX export...")
        _export_mia_fbx_direct(data, output_path, remove_fingers, reset_to_rest, template_path)

        # Verify output was created
        if not os.path.exists(output_path):
            raise RuntimeError(f"Export completed but output file not created: {output_path}")
    else:
        # Fallback: use subprocess with Blender executable
        print("[MIA] bpy not available, falling back to subprocess method...")
        _export_mia_fbx_subprocess(data, output_path, remove_fingers, reset_to_rest, template_path)


def _export_mia_fbx_subprocess(
    data: Dict[str, Any],
    output_path: str,
    remove_fingers: bool,
    reset_to_rest: bool,
    template_path: Path,
) -> None:
    """
    Fallback: Export MIA results to FBX using Blender subprocess.

    NOTE: Blender subprocess export is no longer supported.
    This function now raises an error directing users to use the bpy-based export.
    """
    raise RuntimeError(
        "Blender subprocess export is no longer supported.\n"
        "Please ensure bpy is available in your environment.\n"
        "The MIA nodes require running in the unirig isolated environment with bpy."
    )


def clear_mia_cache():
    """Clear the MIA model cache."""
    global _MIA_MODEL_CACHE
    _MIA_MODEL_CACHE.clear()
    print("[MIA] Model cache cleared")
