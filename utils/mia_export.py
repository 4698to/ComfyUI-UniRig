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


def apply_pose_to_rest(armature_obj, pose, bones_idx_dict, parent_indices, input_meshes):
    """
    Apply MIA's pose prediction to transform skeleton from input pose to T-pose rest.

    The pose data contains LOCAL rotations (relative to parent) that describe how each
    bone is rotated from rest pose to input pose. We apply the INVERSE rotations
    propagated through the kinematic chain to transform back to rest pose.

    Args:
        armature_obj: Blender armature object
        pose: (num_bones, 6) array of ortho6d local rotations
        bones_idx_dict: Mapping from bone names to indices
        parent_indices: List of parent bone indices (-1 for root)
        input_meshes: List of mesh objects to transform along with skeleton
    """
    if pose is None:
        print("[MIA Export] No pose data - skipping pose-to-rest transformation")
        return

    if parent_indices is None:
        print("[MIA Export] No kinematic tree - skipping pose-to-rest transformation")
        return

    print(f"[MIA Export] Applying pose-to-rest transformation with kinematic chain...")

    # Convert ortho6d to rotation matrices for all bones
    rot_matrices = {}
    for name, idx in bones_idx_dict.items():
        if idx < pose.shape[0]:
            rot_matrices[name] = ortho6d_to_matrix(pose[idx])

    # Apply inverse rotations in pose mode
    # The rotations are LOCAL (relative to parent), so we apply them directly
    # Blender handles the kinematic chain propagation
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    applied_count = 0
    for bone_name, rot_matrix in rot_matrices.items():
        pbone = armature_obj.pose.bones.get(bone_name)
        if pbone is None:
            print(f"[MIA Export] Warning: pose bone {bone_name} not found")
            continue

        # Convert to Blender Matrix (3x3)
        blender_rot = Matrix([
            [rot_matrix[0, 0], rot_matrix[0, 1], rot_matrix[0, 2]],
            [rot_matrix[1, 0], rot_matrix[1, 1], rot_matrix[1, 2]],
            [rot_matrix[2, 0], rot_matrix[2, 1], rot_matrix[2, 2]],
        ])

        # MIA's pose describes rotation FROM T-pose TO input pose
        # To go FROM input pose TO T-pose, we need the INVERSE rotation
        # For rotation matrices, inverse = transpose
        inv_rot = blender_rot.transposed()

        # Set the pose bone rotation in LOCAL space
        pbone.rotation_mode = 'QUATERNION'
        pbone.rotation_quaternion = inv_rot.to_quaternion()
        applied_count += 1

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[MIA Export] Applied inverse local rotations to {applied_count} bones")

    # Update the view layer to propagate pose through kinematic chain
    bpy.context.view_layer.update()

    # Apply the posed armature as new rest pose
    # First, apply armature modifier to meshes to bake the current deformation
    for mesh_obj in input_meshes:
        bpy.context.view_layer.objects.active = mesh_obj
        for mod in mesh_obj.modifiers:
            if mod.type == 'ARMATURE':
                bpy.ops.object.modifier_apply(modifier=mod.name)
                break
        # Don't clear parent - keep relationship but modifier is applied

    # Apply current pose as rest pose (bakes bone transforms into edit bones)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Re-add armature modifier to meshes (vertex groups are preserved!)
    for mesh_obj in input_meshes:
        # Add new armature modifier pointing to our armature
        mod = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
        mod.object = armature_obj
        mod.use_vertex_groups = True

    # Clear any remaining pose transforms
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='SELECT')
    bpy.ops.pose.transforms_clear()
    bpy.ops.object.mode_set(mode='OBJECT')

    print(f"[MIA Export] Skeleton transformed to rest pose")


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


def compute_scale_transform(template_bone_data, mia_joints, bones_idx_dict):
    """
    Compute scale and offset to transform MIA joints to template scale.

    MIA outputs normalized joints (roughly [-1, 1]).
    Template is in centimeter scale (e.g., Hips at Y=104).

    Returns:
        scale: Scale factor to apply to MIA joints
        offset: Offset to add after scaling
    """
    # Get reference points from template and MIA
    hips_name = "mixamorig:Hips"
    head_name = "mixamorig:Head"

    if hips_name not in template_bone_data or head_name not in template_bone_data:
        print("[MIA Export] Warning: Missing reference bones for scale computation")
        return 1.0, np.array([0.0, 0.0, 0.0])

    if hips_name not in bones_idx_dict or head_name not in bones_idx_dict:
        print("[MIA Export] Warning: Missing reference bones in MIA output")
        return 1.0, np.array([0.0, 0.0, 0.0])

    # Template positions (in local Y-up space)
    template_hips = np.array(template_bone_data[hips_name]['head'])
    template_head = np.array(template_bone_data[head_name]['head'])
    template_height = np.linalg.norm(template_head - template_hips)

    # MIA positions
    hips_idx = bones_idx_dict[hips_name]
    head_idx = bones_idx_dict[head_name]
    mia_hips = mia_joints[hips_idx]
    mia_head = mia_joints[head_idx]
    mia_height = np.linalg.norm(mia_head - mia_hips)

    if mia_height < 0.001:
        print("[MIA Export] Warning: MIA skeleton has near-zero height")
        return 1.0, np.array([0.0, 0.0, 0.0])

    # Compute scale
    scale = template_height / mia_height

    # Compute offset: position MIA hips at template hips after scaling
    offset = template_hips - mia_hips * scale

    print(f"[MIA Export] Scale transform: scale={scale:.3f}, offset=({offset[0]:.2f}, {offset[1]:.2f}, {offset[2]:.2f})")
    print(f"[MIA Export] Template hips->head height: {template_height:.2f}")
    print(f"[MIA Export] MIA hips->head height: {mia_height:.4f}")

    return scale, offset


def transform_joints_to_template_space(joints, joints_tail, scale, offset):
    """
    Transform MIA joints from normalized space to template scale.
    """
    transformed_joints = joints * scale + offset
    transformed_tail = None
    if joints_tail is not None:
        transformed_tail = joints_tail * scale + offset

    return transformed_joints, transformed_tail


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

    # Transform MIA joints from normalized space to template scale
    # MIA outputs Y-up joints in normalized [-1, 1] space
    # Template bones are in centimeter scale (Hips at Y~104)
    scale, offset = compute_scale_transform(template_bone_data, joints, bones_idx_dict)
    joints_scaled, joints_tail_scaled = transform_joints_to_template_space(joints, joints_tail, scale, offset)

    # Scale input meshes to match template scale
    # The mesh from MIA is in normalized space, same as joints
    for mesh_obj in input_meshes:
        mesh_obj.scale = (scale, scale, scale)
        mesh_obj.location = Vector(offset)
    # Apply transforms so the mesh data is in world space
    bpy.ops.object.select_all(action='DESELECT')
    for mesh_obj in input_meshes:
        mesh_obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.select_all(action='DESELECT')
    print(f"[MIA Export] Scaled mesh(es) to template scale")

    # Update bone positions with scaled joints and template rolls for animation compatibility
    # Note: Don't use reset_as_rest here - pose transformation is handled separately
    set_rest_bones(armature, joints_scaled, joints_tail_scaled, bones_idx_dict,
                   template_bone_data=template_bone_data, reset_as_rest=False)

    # Apply weights BEFORE parenting (so vertex groups exist)
    set_weights(input_meshes, bw, bones_idx_dict)

    # Parent meshes to armature and add armature modifier
    for mesh_obj in input_meshes:
        # Set parent relationship
        mesh_obj.parent = armature
        # Add armature modifier
        mod = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
        mod.object = armature
        mod.use_vertex_groups = True

    # Apply pose-to-rest transformation if pose data is available
    if pose is not None and args["reset_to_rest"]:
        apply_pose_to_rest(armature, pose, bones_idx_dict, parent_indices, input_meshes)

    # Update scene before export
    bpy.context.view_layer.update()

    # Export
    export_fbx(args["output_path"])
    print("[MIA Export] Done!")


if __name__ == "__main__":
    main()
