"""
Blender script for MIA (Make-It-Animatable) FBX export.
Applies MIA-predicted joints and weights to a mesh and exports to FBX.

Usage: blender --background --python mia_export.py -- --input_path <json> --output_path <fbx> --template_path <fbx> [options]
"""

import bpy
import sys
import os
import json
import math
import numpy as np
from mathutils import Vector, Matrix, Quaternion, Euler


def ortho6d_to_matrix(ortho6d):
    """
    Convert ortho6d rotation representation to 3x3 rotation matrix.

    Args:
        ortho6d: (6,) array - first 3 values are x axis, next 3 are y axis hint

    Returns:
        (3, 3) rotation matrix as numpy array
    """
    x_raw = ortho6d[:3]
    y_raw = ortho6d[3:6]

    # Normalize x
    x = x_raw / (np.linalg.norm(x_raw) + 1e-8)

    # z = cross(x, y_raw), then normalize
    z = np.cross(x, y_raw)
    z = z / (np.linalg.norm(z) + 1e-8)

    # y = cross(z, x) - already normalized since z and x are orthonormal
    y = np.cross(z, x)

    return np.column_stack([x, y, z])


def get_rotation_about_point(rotation, point):
    """
    Create a 4x4 transform matrix that rotates about a given point.

    Args:
        rotation: (3, 3) rotation matrix
        point: (3,) point to rotate about

    Returns:
        (4, 4) transform matrix
    """
    # T = translate to origin, R = rotate, T^-1 = translate back
    # Result: point stays fixed, rotation applied around it
    transform = np.eye(4)
    transform[:3, :3] = rotation
    # translation = point - rotation @ point = (I - R) @ point
    transform[:3, 3] = point - rotation @ point
    return transform


def pose_rot_to_global(pose_rot, joints, parent_indices):
    """
    Convert per-bone local rotations to global 4x4 transforms.
    Propagates transforms through kinematic chain.

    This matches the original MIA implementation from blender_utils.py (lines 897-903):
    - Each bone rotates about its ORIGINAL joint position (not the posed position)
    - A translation offset is added to account for parent movement
    - Transforms are computed by propagating parent effects through the chain

    Args:
        pose_rot: (K, 6) ortho6d rotations per bone
        joints: (K, 3) joint positions
        parent_indices: list of parent indices (-1 for root)

    Returns:
        pose_global: (K, 4, 4) global transforms per bone
    """
    K = pose_rot.shape[0]

    # Convert ortho6d to rotation matrices
    # MIA predicts rotations FROM T-pose TO input pose (forward direction)
    rot_matrices = np.zeros((K, 3, 3))
    for i in range(K):
        rot_matrices[i] = ortho6d_to_matrix(pose_rot[i])

    # Initialize transforms - each bone rotates about its own joint initially
    pose_global = np.zeros((K, 4, 4))
    for i in range(K):
        pose_global[i] = get_rotation_about_point(rot_matrices[i], joints[i])

    # Track posed joint positions (where joints end up after parent transforms)
    posed_joints = joints.copy()

    # Propagate through kinematic chain
    # For each child bone, compute its posed joint position based on parent's transform,
    # rotate about the ORIGINAL joint position, then add translation offset
    for i in range(1, K):
        parent_idx = parent_indices[i]
        parent_matrix = pose_global[parent_idx]

        # Compute where this joint ends up after parent's transform
        posed_joints[i] = parent_matrix[:3, :3] @ joints[i] + parent_matrix[:3, 3]

        # Rotate about ORIGINAL joint position (matches original MIA)
        matrix = get_rotation_about_point(rot_matrices[i], joints[i])

        # Add translation offset to account for parent movement
        matrix[:3, 3] += posed_joints[i] - joints[i]

        pose_global[i] = matrix

    return pose_global


def apply_pose_to_rest(armature_obj, pose, bones_idx_dict, parent_indices, input_meshes, mia_joints):
    """
    Apply MIA's pose prediction to transform skeleton from input pose to T-pose rest.
    Uses the original MIA approach with kinematic chain propagation.

    Args:
        armature_obj: Blender armature object
        pose: (num_bones, 6) array of ortho6d rotations per bone
        bones_idx_dict: Mapping from bone names to indices
        parent_indices: List of parent bone indices (-1 for root)
        input_meshes: List of mesh objects to transform along with skeleton
        mia_joints: (num_bones, 3) array of MIA-predicted joint positions (normalized)
    """
    if pose is None:
        print("[MIA Export] No pose data - skipping pose-to-rest transformation")
        return

    if parent_indices is None:
        print("[MIA Export] No kinematic tree - skipping pose-to-rest transformation")
        return

    print(f"[MIA Export] Applying pose-to-rest transformation...")
    print(f"[MIA Export] Using MIA joints for pose computation (shape: {mia_joints.shape})")

    # Use MIA-predicted joints for pose computation (these match what the pose was predicted for)
    joints = mia_joints

    # Convert ortho6d to global 4x4 transforms using kinematic chain
    pose_global = pose_rot_to_global(pose, joints, parent_indices)

    # Set root bone to identity (no global movement)
    pose_global[0] = np.eye(4)

    print(f"[MIA Export] Computed global transforms for {len(pose_global)} bones")

    # === DEBUG: Print sample transforms ===
    print(f"[MIA Export DEBUG] Sample pose transforms (T-pose to input pose, forward):")
    for bone_name in ["mixamorig:Hips", "mixamorig:Spine", "mixamorig:Head"]:
        if bone_name in bones_idx_dict:
            idx = bones_idx_dict[bone_name]
            rot = pose_global[idx][:3, :3]
            trans = pose_global[idx][:3, 3]
            print(f"  {bone_name} (idx={idx}):")
            print(f"    Rotation diag: [{rot[0,0]:.4f}, {rot[1,1]:.4f}, {rot[2,2]:.4f}]")
            print(f"    Translation: [{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]")

    # Define finger bone prefixes to skip (fingers often have problematic transforms)
    finger_prefixes = [
        "LeftHandThumb", "LeftHandIndex", "LeftHandMiddle", "LeftHandRing", "LeftHandPinky",
        "RightHandThumb", "RightHandIndex", "RightHandMiddle", "RightHandRing", "RightHandPinky",
        "mixamorig:LeftHandThumb", "mixamorig:LeftHandIndex", "mixamorig:LeftHandMiddle",
        "mixamorig:LeftHandRing", "mixamorig:LeftHandPinky",
        "mixamorig:RightHandThumb", "mixamorig:RightHandIndex", "mixamorig:RightHandMiddle",
        "mixamorig:RightHandRing", "mixamorig:RightHandPinky",
    ]

    def is_finger_bone(name):
        for prefix in finger_prefixes:
            if name.startswith(prefix):
                return True
        return False

    # Apply transforms in pose mode
    bpy.ops.object.mode_set(mode='POSE')

    applied_count = 0
    skipped_fingers = 0
    for bone_name, idx in bones_idx_dict.items():
        pbone = armature_obj.pose.bones.get(bone_name)
        if pbone is None:
            continue

        # Skip finger bones - they often have problematic transforms
        # and T-pose for fingers is less important than body
        if is_finger_bone(bone_name):
            skipped_fingers += 1
            continue

        # Get the global transform for this bone
        pose_matrix = Matrix(pose_global[idx].tolist())

        # Apply: bone.matrix = pose_matrix @ bone.bone.matrix_local
        # This sets the bone's global pose
        pbone.matrix = pose_matrix @ pbone.bone.matrix_local
        bpy.context.view_layer.update()
        applied_count += 1

    print(f"[MIA Export] Skipped {skipped_fingers} finger bones (keeping original pose)")

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[MIA Export] Applied global transforms to {applied_count} bones")

    # Update view layer
    bpy.context.view_layer.update()

    # === DEBUG: Print bone positions after pose transform ===
    print(f"[MIA Export DEBUG] Bone positions AFTER pose transform:")
    bpy.ops.object.mode_set(mode='POSE')
    for bone_name in ["mixamorig:Hips", "mixamorig:Spine", "mixamorig:Head", "mixamorig:LeftArm", "mixamorig:RightArm"]:
        pbone = armature_obj.pose.bones.get(bone_name)
        if pbone:
            head_world = armature_obj.matrix_world @ pbone.head
            print(f"  {bone_name}: head=({head_world.x:.4f}, {head_world.y:.4f}, {head_world.z:.4f})")
    bpy.ops.object.mode_set(mode='OBJECT')

    # Apply the posed armature as new rest pose
    for mesh_obj in input_meshes:
        bpy.context.view_layer.objects.active = mesh_obj
        for mod in mesh_obj.modifiers:
            if mod.type == 'ARMATURE':
                bpy.ops.object.modifier_apply(modifier=mod.name)
                break

    # Apply current pose as rest pose
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode='OBJECT')

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

    # === DEBUG: Print final bone positions in rest pose ===
    print(f"[MIA Export DEBUG] Final bone positions (rest pose):")
    bpy.ops.object.mode_set(mode='EDIT')
    for bone_name in ["mixamorig:Hips", "mixamorig:Spine", "mixamorig:Head", "mixamorig:LeftArm", "mixamorig:RightArm"]:
        bone = armature_obj.data.edit_bones.get(bone_name)
        if bone:
            head_world = armature_obj.matrix_world @ bone.head
            print(f"  {bone_name}: head=({head_world.x:.4f}, {head_world.y:.4f}, {head_world.z:.4f})")
    bpy.ops.object.mode_set(mode='OBJECT')

    print(f"[MIA Export] Skeleton transformed to rest pose (using forward rotations, armature modifier inverts)")


def parse_args():
    """Parse command line arguments after '--'."""
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []

    args = {
        "input_path": None,
        "output_path": None,
        "template_path": None,
        "remove_fingers": False,
        "reset_to_rest": False,
    }

    i = 0
    while i < len(argv):
        if argv[i] == "--input_path" and i + 1 < len(argv):
            args["input_path"] = argv[i + 1]
            i += 2
        elif argv[i] == "--output_path" and i + 1 < len(argv):
            args["output_path"] = argv[i + 1]
            i += 2
        elif argv[i] == "--template_path" and i + 1 < len(argv):
            args["template_path"] = argv[i + 1]
            i += 2
        elif argv[i] == "--remove_fingers":
            args["remove_fingers"] = True
            i += 1
        elif argv[i] == "--reset_to_rest":
            args["reset_to_rest"] = True
            i += 1
        else:
            i += 1

    return args


def reset_scene():
    """Clear all objects from the scene."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def load_fbx(filepath):
    """Load FBX file and return imported objects."""
    old_objs = set(bpy.context.scene.objects)
    bpy.ops.import_scene.fbx(filepath=filepath)
    new_objs = set(bpy.context.scene.objects) - old_objs
    return list(new_objs)


def get_armature(objects):
    """Find armature object in a list of objects."""
    for obj in objects:
        if obj.type == "ARMATURE":
            return obj
    return None


def get_meshes(objects):
    """Find all mesh objects in a list of objects."""
    return [obj for obj in objects if obj.type == "MESH"]


def get_template_bone_data(armature_obj):
    """
    Capture bone orientations from template armature before modification.
    Returns dict mapping bone name to roll value.
    """
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    bone_data = {}
    for bone in armature_obj.data.edit_bones:
        bone_data[bone.name] = {
            'roll': bone.roll,
            'head': tuple(bone.head),
            'tail': tuple(bone.tail),
            'matrix': [list(row) for row in bone.matrix],
        }

    bpy.ops.object.mode_set(mode='OBJECT')
    return bone_data


# DEPRECATED: compute_scale_transform - This approach was wrong.
# The original MIA uses matrix_world manipulation instead.
# def compute_scale_transform(template_bone_data, mia_joints, bones_idx_dict):
#     """
#     Compute scale and offset to transform MIA joints to template scale.
#     ...
#     """
#     pass


# DEPRECATED: transform_joints_to_template_space - This approach was wrong.
# The original MIA uses matrix_world manipulation instead.
# def transform_joints_to_template_space(joints, joints_tail, scale, offset):
#     """
#     Transform MIA joints from normalized space to template scale.
#     """
#     pass


def apply_template_orientations(armature_obj, template_bone_data, bones_idx_dict):
    """
    Apply template bone rolls to MIA skeleton.
    This ensures bone orientations match Mixamo template for animation compatibility.
    """
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    applied_count = 0
    for bone in armature_obj.data.edit_bones:
        if bone.name in template_bone_data and bone.name in bones_idx_dict:
            template_data = template_bone_data[bone.name]
            # Set the bone roll to match template
            bone.roll = template_data['roll']
            applied_count += 1

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[MIA Export] Applied template orientations to {applied_count} bones")


def set_rest_bones(armature_obj, head, tail, bones_idx_dict, template_bone_data=None, reset_as_rest=False):
    """Update bone positions in the armature."""
    import bmesh

    if reset_as_rest:
        # Apply current pose as rest pose
        mesh_list = []
        for obj in armature_obj.children:
            if obj.type != "MESH":
                continue
            mesh_list.append(obj)
            bpy.context.view_layer.objects.active = obj
            for mod in obj.modifiers:
                if mod.type == 'ARMATURE':
                    bpy.ops.object.modifier_apply(modifier=mod.name)
                    break
            obj.select_set(True)
            bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
            obj.select_set(False)

        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.armature_apply(selected=False)
        bpy.ops.object.mode_set(mode='OBJECT')

        # Re-parent meshes
        for mesh_obj in mesh_list:
            mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.parent_set(type='ARMATURE')
        bpy.ops.object.select_all(action='DESELECT')

    # Update bone positions in edit mode
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    # First pass: update bones that are in the prediction dict
    # Use MIA's positions directly - this preserves correct skin weights relationship
    for bone in armature_obj.data.edit_bones:
        bone.use_connect = False
        if bone.name in bones_idx_dict:
            idx = bones_idx_dict[bone.name]

            # Set both head and tail from MIA (preserves correct weight mapping)
            bone.head = Vector(head[idx])
            if tail is not None:
                bone.tail = Vector(tail[idx])

            # Apply template roll for consistent twist axis
            if template_bone_data and bone.name in template_bone_data:
                bone.roll = template_bone_data[bone.name]['roll']

    # Second pass: remove end/leaf bones not in MIA's prediction dict
    # These are bones like HeadTop_End, *Thumb4, *Index4, etc.
    # They have no skinning weights, so removing them doesn't affect deformation
    # Animation retargeting should still work with the 52 functional bones
    bones_to_remove = []
    for bone in armature_obj.data.edit_bones:
        if bone.name not in bones_idx_dict:
            bones_to_remove.append(bone.name)

    for bone_name in bones_to_remove:
        bone = armature_obj.data.edit_bones.get(bone_name)
        if bone:
            armature_obj.data.edit_bones.remove(bone)

    if bones_to_remove:
        print(f"[MIA Export] Removed {len(bones_to_remove)} end bones: {bones_to_remove[:5]}...")

    bpy.ops.object.mode_set(mode='OBJECT')

    if reset_as_rest:
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.pose.transforms_clear()
        bpy.ops.object.mode_set(mode='OBJECT')

    return armature_obj


def set_weights(mesh_obj_list, weights, bones_idx_dict):
    """Apply bone weights to mesh objects."""
    if not mesh_obj_list:
        return

    # Calculate vertex counts per mesh
    vertices_num = [len(mesh_obj.data.vertices) for mesh_obj in mesh_obj_list]
    total_verts = sum(vertices_num)

    if total_verts != weights.shape[0]:
        print(f"[MIA Export] Warning: vertex count mismatch: {total_verts} vs {weights.shape[0]}")
        return

    # Split weights per mesh
    weights_list = np.split(weights, np.cumsum(vertices_num)[:-1])

    for mesh_obj, bw in zip(mesh_obj_list, weights_list):
        mesh_data = mesh_obj.data
        mesh_obj.vertex_groups.clear()

        for bone_name, bone_index in bones_idx_dict.items():
            group = mesh_obj.vertex_groups.new(name=bone_name)
            for v in mesh_data.vertices:
                v_w = bw[v.index, bone_index]
                if v_w > 1e-3:
                    group.add([v.index], float(v_w), "REPLACE")

        mesh_data.update()

    return mesh_obj_list


def remove_finger_bones(armature_obj, bones_idx_dict):
    """Remove finger bones from armature and update bones_idx_dict."""
    finger_prefixes = [
        "LeftHandThumb", "LeftHandIndex", "LeftHandMiddle", "LeftHandRing", "LeftHandPinky",
        "RightHandThumb", "RightHandIndex", "RightHandMiddle", "RightHandRing", "RightHandPinky",
        "mixamorig:LeftHandThumb", "mixamorig:LeftHandIndex", "mixamorig:LeftHandMiddle",
        "mixamorig:LeftHandRing", "mixamorig:LeftHandPinky",
        "mixamorig:RightHandThumb", "mixamorig:RightHandIndex", "mixamorig:RightHandMiddle",
        "mixamorig:RightHandRing", "mixamorig:RightHandPinky",
    ]

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    bones_to_remove = []
    for bone in armature_obj.data.edit_bones:
        for prefix in finger_prefixes:
            if bone.name.startswith(prefix):
                bones_to_remove.append(bone.name)
                break

    for bone_name in bones_to_remove:
        bone = armature_obj.data.edit_bones.get(bone_name)
        if bone:
            armature_obj.data.edit_bones.remove(bone)
        if bone_name in bones_idx_dict:
            del bones_idx_dict[bone_name]

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[MIA Export] Removed {len(bones_to_remove)} finger bones")
    return armature_obj


def export_fbx(output_path):
    """Export scene to FBX."""
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


def main():
    args = parse_args()

    if not args["input_path"] or not args["output_path"] or not args["template_path"]:
        print("Usage: blender --background --python mia_export.py -- --input_path <json> --output_path <fbx> --template_path <fbx>")
        sys.exit(1)

    print(f"[MIA Export] Input: {args['input_path']}")
    print(f"[MIA Export] Output: {args['output_path']}")
    print(f"[MIA Export] Template: {args['template_path']}")
    print(f"[MIA Export] Remove fingers: {args['remove_fingers']}")
    print(f"[MIA Export] Reset to rest: {args['reset_to_rest']}")

    # Load input data
    with open(args["input_path"], 'r') as f:
        data = json.load(f)

    # Load binary arrays
    bw_shape = data["bw_shape"]
    joints_shape = data["joints_shape"]

    bw = np.fromfile(data["bw_path"], dtype=np.float32).reshape(bw_shape)
    joints = np.fromfile(data["joints_path"], dtype=np.float32).reshape(joints_shape)

    joints_tail = None
    if "joints_tail_path" in data:
        joints_tail_shape = data["joints_tail_shape"]
        joints_tail = np.fromfile(data["joints_tail_path"], dtype=np.float32).reshape(joints_tail_shape)

    pose = None
    if "pose_path" in data:
        pose_shape = data["pose_shape"]
        pose = np.fromfile(data["pose_path"], dtype=np.float32).reshape(pose_shape)
        print(f"[MIA Export] Loaded pose data: {pose.shape}")

    parent_indices = data.get("parent_indices")
    if parent_indices:
        print(f"[MIA Export] Loaded kinematic tree: {len(parent_indices)} bones")

    bones_idx_dict = data["bones_idx_dict"]
    mesh_path = data["mesh_path"]

    print(f"[MIA Export] Loaded weights: {bw.shape}")
    print(f"[MIA Export] Loaded joints: {joints.shape}")
    print(f"[MIA Export] Bones: {len(bones_idx_dict)}")

    # Reset scene and load template
    reset_scene()
    template_objs = load_fbx(args["template_path"])
    armature = get_armature(template_objs)

    if armature is None:
        print("[MIA Export] ERROR: No armature found in template!")
        sys.exit(1)

    print(f"[MIA Export] Loaded template armature: {armature.name}")
    print(f"[MIA Export] Template rotation: {[r for r in armature.rotation_euler]}")

    # Keep template's rotation (90° X) - this transforms Y-up local to Z-up world
    # MIA joints are Y-up, so they match the template's local space directly

    # Capture template bone orientations (in local/Y-up space)
    template_bone_data = get_template_bone_data(armature)
    print(f"[MIA Export] Captured orientations for {len(template_bone_data)} template bones")

    # Clear any pose transforms
    armature.animation_data_clear()
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.pose.transforms_clear()
    bpy.ops.object.mode_set(mode='OBJECT')

    # Load input mesh
    old_objs = set(bpy.context.scene.objects)

    if mesh_path.endswith(".glb") or mesh_path.endswith(".gltf"):
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    elif mesh_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=mesh_path)
    elif mesh_path.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=mesh_path)
    else:
        print(f"[MIA Export] ERROR: Unsupported mesh format: {mesh_path}")
        sys.exit(1)

    new_objs = set(bpy.context.scene.objects) - old_objs
    input_meshes = get_meshes(list(new_objs))

    if not input_meshes:
        print("[MIA Export] ERROR: No mesh found in input!")
        sys.exit(1)

    print(f"[MIA Export] Loaded {len(input_meshes)} mesh(es) from input")

    # Remove template meshes (we use input mesh instead)
    template_meshes = get_meshes(template_objs)
    for mesh in template_meshes:
        bpy.data.objects.remove(mesh, do_unlink=True)

    # Remove finger bones if requested
    if args["remove_fingers"]:
        remove_finger_bones(armature, bones_idx_dict)

    # === DEBUG: Create output directory for intermediate files ===
    debug_dir = os.path.dirname(args["output_path"])
    debug_prefix = os.path.splitext(os.path.basename(args["output_path"]))[0]

    # === DEBUG: Print input mesh stats ===
    print(f"\n{'='*60}")
    print(f"[MIA Export DEBUG] INPUT MESH STATS (before transformation)")
    print(f"{'='*60}")
    for i, mesh_obj in enumerate(input_meshes):
        mesh_data = mesh_obj.data
        verts = np.array([v.co for v in mesh_data.vertices])
        print(f"  Mesh {i} '{mesh_obj.name}':")
        print(f"    Vertices: {len(verts)}")
        print(f"    Bounds X: [{verts[:, 0].min():.4f}, {verts[:, 0].max():.4f}]")
        print(f"    Bounds Y: [{verts[:, 1].min():.4f}, {verts[:, 1].max():.4f}]")
        print(f"    Bounds Z: [{verts[:, 2].min():.4f}, {verts[:, 2].max():.4f}]")
        print(f"    Center: ({verts.mean(axis=0)[0]:.4f}, {verts.mean(axis=0)[1]:.4f}, {verts.mean(axis=0)[2]:.4f})")

    # === DEBUG: Print input joints stats ===
    print(f"\n{'='*60}")
    print(f"[MIA Export DEBUG] INPUT JOINTS STATS (MIA output)")
    print(f"{'='*60}")
    print(f"  Joints shape: {joints.shape}")
    print(f"  Joints bounds X: [{joints[:, 0].min():.4f}, {joints[:, 0].max():.4f}]")
    print(f"  Joints bounds Y: [{joints[:, 1].min():.4f}, {joints[:, 1].max():.4f}]")
    print(f"  Joints bounds Z: [{joints[:, 2].min():.4f}, {joints[:, 2].max():.4f}]")
    hips_idx = bones_idx_dict.get("mixamorig:Hips")
    head_idx = bones_idx_dict.get("mixamorig:Head")
    if hips_idx is not None and head_idx is not None:
        print(f"  Hips position: ({joints[hips_idx, 0]:.4f}, {joints[hips_idx, 1]:.4f}, {joints[hips_idx, 2]:.4f})")
        print(f"  Head position: ({joints[head_idx, 0]:.4f}, {joints[head_idx, 1]:.4f}, {joints[head_idx, 2]:.4f})")

    # === DEBUG: Print armature info ===
    print(f"\n{'='*60}")
    print(f"[MIA Export DEBUG] ARMATURE INFO")
    print(f"{'='*60}")
    print(f"  Armature name: {armature.name}")
    print(f"  Rotation euler: {[r for r in armature.rotation_euler]}")
    print(f"  Rotation (degrees): {[r * 180 / 3.14159 for r in armature.rotation_euler]}")
    print(f"  Matrix world:\n{armature.matrix_world}")

    # === ORIGINAL MIA APPROACH ===
    # 1. Save armature's world matrix and get scaling factor
    matrix_world = armature.matrix_world.copy()
    scaling = matrix_world.to_scale()[0]
    print(f"\n[MIA Export] Step 1: Saved matrix_world, scaling = {scaling}")
    print(f"  Matrix world rotation (from to_euler): {[r * 180 / 3.14159 for r in matrix_world.to_euler()]}")

    # 2. Reset armature to identity (work in Y-up local space)
    armature.matrix_world.identity()
    bpy.context.view_layer.update()
    print(f"[MIA Export] Step 2: Reset armature.matrix_world to identity")

    # === DEBUG: Save mesh BEFORE transformation ===
    debug_mesh_before = os.path.join(debug_dir, f"{debug_prefix}_debug_mesh_BEFORE_transform.obj")
    bpy.ops.object.select_all(action='DESELECT')
    for mesh_obj in input_meshes:
        mesh_obj.select_set(True)
    bpy.ops.wm.obj_export(filepath=debug_mesh_before, export_selected_objects=True)
    bpy.ops.object.select_all(action='DESELECT')
    print(f"[MIA Export DEBUG] Saved mesh BEFORE transform: {debug_mesh_before}")

    # 3. Transform mesh vertices: Y-Z swap and divide by scaling
    print(f"\n[MIA Export] Step 3: Transforming mesh vertices (Y-Z swap, /scaling)")
    for mesh_obj in input_meshes:
        mesh_data = mesh_obj.data
        verts = np.array([v.co for v in mesh_data.vertices])

        print(f"  Mesh '{mesh_obj.name}' BEFORE transform:")
        print(f"    Sample vert[0]: ({verts[0, 0]:.4f}, {verts[0, 1]:.4f}, {verts[0, 2]:.4f})")

        # Y-Z swap: matches original MIA
        # verts[:, 1], verts[:, 2] = verts[:, 2].copy(), -verts[:, 1].copy()
        new_y = verts[:, 2].copy()
        new_z = -verts[:, 1].copy()
        verts[:, 1] = new_y
        verts[:, 2] = new_z
        verts = verts / scaling

        print(f"  Mesh '{mesh_obj.name}' AFTER transform:")
        print(f"    Sample vert[0]: ({verts[0, 0]:.4f}, {verts[0, 1]:.4f}, {verts[0, 2]:.4f})")
        print(f"    New bounds X: [{verts[:, 0].min():.4f}, {verts[:, 0].max():.4f}]")
        print(f"    New bounds Y: [{verts[:, 1].min():.4f}, {verts[:, 1].max():.4f}]")
        print(f"    New bounds Z: [{verts[:, 2].min():.4f}, {verts[:, 2].max():.4f}]")

        for i, v in enumerate(mesh_data.vertices):
            v.co = verts[i]
        mesh_data.update()

    # === DEBUG: Save mesh AFTER transformation ===
    debug_mesh_after = os.path.join(debug_dir, f"{debug_prefix}_debug_mesh_AFTER_transform.obj")
    bpy.ops.object.select_all(action='DESELECT')
    for mesh_obj in input_meshes:
        mesh_obj.select_set(True)
    bpy.ops.wm.obj_export(filepath=debug_mesh_after, export_selected_objects=True)
    bpy.ops.object.select_all(action='DESELECT')
    print(f"[MIA Export DEBUG] Saved mesh AFTER transform: {debug_mesh_after}")

    # 4. Set bones with joints/scaling (now in same Y-up normalized space as mesh)
    joints_normalized = joints / scaling
    joints_tail_normalized = joints_tail / scaling if joints_tail is not None else None
    print(f"\n[MIA Export] Step 4: Setting bones with joints/scaling")
    print(f"  Joints normalized bounds Y: [{joints_normalized[:, 1].min():.4f}, {joints_normalized[:, 1].max():.4f}]")
    if hips_idx is not None:
        print(f"  Hips normalized: ({joints_normalized[hips_idx, 0]:.4f}, {joints_normalized[hips_idx, 1]:.4f}, {joints_normalized[hips_idx, 2]:.4f})")

    set_rest_bones(armature, joints_normalized, joints_tail_normalized, bones_idx_dict,
                   template_bone_data=template_bone_data, reset_as_rest=False)

    # 5. Parent mesh to armature
    print(f"\n[MIA Export] Step 5: Parenting mesh to armature")
    for mesh_obj in input_meshes:
        mesh_obj.parent = armature
        print(f"  Parented '{mesh_obj.name}' to '{armature.name}'")

    # 6. Apply weights
    print(f"\n[MIA Export] Step 6: Applying weights")
    print(f"  Weights shape: {bw.shape}")
    print(f"  Weights sum per vertex (should be ~1.0): min={bw.sum(axis=1).min():.4f}, max={bw.sum(axis=1).max():.4f}")
    set_weights(input_meshes, bw, bones_idx_dict)

    # 7. Add armature modifier
    print(f"\n[MIA Export] Step 7: Adding armature modifier")
    for mesh_obj in input_meshes:
        mod = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
        mod.object = armature
        mod.use_vertex_groups = True
        print(f"  Added armature modifier to '{mesh_obj.name}'")

    # === DEBUG: Save rigged mesh BEFORE restoring matrix ===
    debug_rigged_before = os.path.join(debug_dir, f"{debug_prefix}_debug_rigged_BEFORE_matrix_restore.fbx")
    bpy.ops.export_scene.fbx(
        filepath=debug_rigged_before,
        use_selection=False,
        object_types={'ARMATURE', 'MESH'},
        add_leaf_bones=False,
        bake_anim=False,
    )
    print(f"[MIA Export DEBUG] Saved rigged mesh BEFORE matrix restore: {debug_rigged_before}")

    # 8. Restore armature matrix (transforms everything to Z-up world)
    armature.matrix_world = matrix_world
    bpy.context.view_layer.update()
    print(f"\n[MIA Export] Step 8: Restored armature.matrix_world")
    print(f"  Final armature rotation (degrees): {[r * 180 / 3.14159 for r in armature.rotation_euler]}")

    # === DEBUG: Print final stats ===
    print(f"\n{'='*60}")
    print(f"[MIA Export DEBUG] FINAL STATS")
    print(f"{'='*60}")
    for mesh_obj in input_meshes:
        # Get world-space bounds
        bbox_corners = [mesh_obj.matrix_world @ Vector(corner) for corner in mesh_obj.bound_box]
        xs = [c.x for c in bbox_corners]
        ys = [c.y for c in bbox_corners]
        zs = [c.z for c in bbox_corners]
        print(f"  Mesh '{mesh_obj.name}' world bounds:")
        print(f"    X: [{min(xs):.4f}, {max(xs):.4f}]")
        print(f"    Y: [{min(ys):.4f}, {max(ys):.4f}]")
        print(f"    Z: [{min(zs):.4f}, {max(zs):.4f}]")

    # Apply pose-to-rest transformation if pose data is available
    if pose is not None and args["reset_to_rest"]:
        apply_pose_to_rest(armature, pose, bones_idx_dict, parent_indices, input_meshes, joints_normalized)

    # Update scene before export
    bpy.context.view_layer.update()

    # Export
    export_fbx(args["output_path"])
    print("[MIA Export] Done!")


if __name__ == "__main__":
    main()
