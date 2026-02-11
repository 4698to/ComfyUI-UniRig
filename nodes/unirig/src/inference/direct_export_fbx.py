"""
Direct FBX export using bpy as a Python module.

This module provides the same functionality as blender_export_fbx.py but as a
direct Python import, eliminating the need for subprocess calls to Blender.

Requires: bpy>=4.0.0 (installed via pip install bpy)

IMPORTANT: bpy and mathutils are imported lazily inside functions to avoid
conflicts with torch_cluster. Do NOT add module-level bpy imports.
"""

import numpy as np
from collections import defaultdict
import math
import tempfile
import base64
import os


def export_rigged_fbx(
    joints: np.ndarray,
    parents: list,
    names: list,
    output_fbx: str,
    vertices: np.ndarray = None,
    faces: np.ndarray = None,
    skin: np.ndarray = None,
    tails: np.ndarray = None,
    uv_coords: np.ndarray = None,
    uv_faces: np.ndarray = None,
    texture_data_base64: str = "",
    texture_format: str = "PNG",
    material_name: str = "Material",
    extrude_size: float = 0.03,
    add_root: bool = False,
    use_extrude_bone: bool = True,
    use_connect_unique_child: bool = True,
    extrude_from_parent: bool = True,
) -> str:
    """
    Export skeleton and optionally skinned mesh to FBX format.

    Args:
        joints: Joint positions array (N, 3)
        parents: Parent indices list (N,) - None or -1 for root
        names: Bone names list (N,)
        output_fbx: Output FBX file path
        vertices: Optional mesh vertices (V, 3)
        faces: Optional mesh faces (F, 3)
        skin: Optional skin weights (V, N)
        tails: Optional bone tail positions (N, 3)
        uv_coords: Optional UV coordinates
        uv_faces: Optional UV face indices
        texture_data_base64: Optional base64 encoded texture
        texture_format: Texture format (default: PNG)
        material_name: Material name
        extrude_size: Default bone extrusion size
        add_root: Whether to add a root bone
        use_extrude_bone: Whether to extrude bones
        use_connect_unique_child: Whether to connect unique children
        extrude_from_parent: Whether to extrude from parent direction

    Returns:
        Path to the exported FBX file
    """
    # Lazy imports to avoid torch_cluster conflict
    import bpy
    from mathutils import Vector, Matrix, Quaternion

    print(f"[Direct FBX Export] Input joints: {joints.shape}")
    print(f"[Direct FBX Export] Output: {output_fbx}")

    # Convert inputs to numpy arrays
    joints = np.array(joints, dtype=np.float32)
    if tails is not None:
        tails = np.array(tails, dtype=np.float32)
    if vertices is not None:
        vertices = np.array(vertices, dtype=np.float32)
    if faces is not None:
        faces = np.array(faces, dtype=np.int32)
    if skin is not None:
        skin = np.array(skin, dtype=np.float32)
    if uv_coords is not None and len(uv_coords) > 0:
        uv_coords = np.array(uv_coords, dtype=np.float32)
    else:
        uv_coords = None
    if uv_faces is not None and len(uv_faces) > 0:
        uv_faces = np.array(uv_faces, dtype=np.int32)
    else:
        uv_faces = None

    if texture_data_base64 and len(texture_data_base64) > 0:
        print(f"[Direct FBX Export] Found texture data: {texture_format} ({len(texture_data_base64) // 1024}KB base64)")
    else:
        print(f"[Direct FBX Export] No texture data found")

    print(f"[Direct FBX Export] Loaded skeleton with {len(joints)} joints")
    if vertices is not None:
        print(f"[Direct FBX Export] Found mesh with {len(vertices)} vertices")
    if skin is not None:
        print(f"[Direct FBX Export] Skin weights shape: {skin.shape}")

    # Clean default scene
    _clean_bpy()

    # === T-POSE CONVERSION FOR SMPL SKELETONS ===
    SMPL_JOINT_NAMES_CHECK = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee',
                        'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot',
                        'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
                        'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist']

    is_smpl_skeleton = len(names) == 22 and all(n in SMPL_JOINT_NAMES_CHECK for n in names)

    if is_smpl_skeleton:
        joints, tails, vertices = _convert_smpl_tpose(joints, tails, vertices, skin, names)

    # === MIXAMO NORMALIZATION ===
    is_mixamo_skeleton = any(n.startswith('mixamorig:') for n in names)

    if is_mixamo_skeleton and vertices is not None and skin is not None:
        joints, tails, vertices = _normalize_mixamo(joints, tails, vertices, skin, names)

    # Make collection
    collection = bpy.data.collections.new('new_collection')
    bpy.context.scene.collection.children.link(collection)

    # Make mesh if vertices provided
    if vertices is not None:
        mesh = bpy.data.meshes.new('mesh')
        if faces is None:
            faces = []
        mesh.from_pydata(vertices.tolist(), [], faces.tolist() if isinstance(faces, np.ndarray) else [])
        mesh.update()

        # Add UV coordinates if available
        if uv_coords is not None and uv_faces is not None and len(uv_coords) > 0:
            print(f"[Direct FBX Export] Adding UV coordinates: {len(uv_coords)} UVs")
            uv_layer = mesh.uv_layers.new(name='UVMap')

            for face_idx, poly in enumerate(mesh.polygons):
                if face_idx < len(uv_faces):
                    for loop_offset, loop_idx in enumerate(poly.loop_indices):
                        uv_idx = uv_faces[face_idx][loop_offset]
                        if uv_idx < len(uv_coords):
                            uv_layer.data[loop_idx].uv = uv_coords[uv_idx]

            print(f"[Direct FBX Export] UV coordinates applied")

        # Make object from mesh
        obj = bpy.data.objects.new('character', mesh)
        collection.objects.link(obj)

        # Create and apply textured material if texture data is available
        if texture_data_base64 and len(texture_data_base64) > 0:
            _apply_texture(obj, texture_data_base64, material_name)

    # Create armature
    print("[Direct FBX Export] Creating armature...")
    bpy.ops.object.armature_add(enter_editmode=True)
    armature = bpy.data.armatures.get('Armature')
    edit_bones = armature.edit_bones

    J = joints.shape[0]
    if tails is None:
        print(f"[Direct FBX Export] Tails not provided, auto-generating...")
        tails = joints.copy()
        tails[:, 2] += extrude_size

    connects = [False for _ in range(J)]
    children = defaultdict(list)
    for i in range(1, J):
        if parents[i] is not None and parents[i] != -1:
            children[parents[i]].append(i)

    if tails is not None:
        if use_extrude_bone:
            for i in range(J):
                if len(children[i]) != 1 and extrude_from_parent and i != 0:
                    if parents[i] is not None and parents[i] != -1:
                        pjoint = joints[parents[i]]
                        joint = joints[i]
                        d = joint - pjoint
                        if np.linalg.norm(d) < 0.000001:
                            d = np.array([0., 0., 1.])
                        else:
                            d = d / np.linalg.norm(d)
                        tails[i] = joint + d * extrude_size
        if use_connect_unique_child:
            for i in range(J):
                if len(children[i]) == 1:
                    child = children[i][0]
                    tails[i] = joints[child]
                if parents[i] is not None and parents[i] != -1 and len(children[parents[i]]) == 1:
                    connects[i] = True

    # Create root bone
    if add_root:
        bone_root = edit_bones.get('Bone')
        bone_root.name = 'Root'
        bone_root.tail = Vector((joints[0, 0], joints[0, 1], joints[0, 2]))
    else:
        bone_root = edit_bones.get('Bone')
        bone_root.name = names[0]
        bone_root.head = Vector((joints[0, 0], joints[0, 1], joints[0, 2]))
        bone_root.tail = Vector((joints[0, 0], joints[0, 1], joints[0, 2] + extrude_size))

    # Create bones
    for i in range(J):
        if add_root is False and i == 0:
            continue
        edit_bones = armature.edit_bones
        pname = 'Root' if parents[i] is None or parents[i] == -1 else names[parents[i]]
        _extrude_bone(edit_bones, names[i], pname, joints[i], tails[i], connects[i])

    # Update bone positions
    for i in range(J):
        bone = edit_bones.get(names[i])
        bone.head = Vector((joints[i, 0], joints[i, 1], joints[i, 2]))
        bone.tail = Vector((tails[i, 0], tails[i, 1], tails[i, 2]))

    # Fix bone orientations for Mixamo
    if is_mixamo_skeleton:
        _fix_mixamo_bone_orientations(edit_bones, names)

    # Set bone rolls for SMPL
    if is_smpl_skeleton:
        _set_smpl_bone_rolls(edit_bones, names, J)

    # Add skinning weights
    if vertices is not None and skin is not None:
        print("[Direct FBX Export] Adding skinning weights...")
        bpy.ops.object.mode_set(mode='OBJECT')
        objects = bpy.data.objects
        for o in bpy.context.selected_objects:
            o.select_set(False)
        ob = objects['character']
        arm = bpy.data.objects['Armature']
        ob.select_set(True)
        arm.select_set(True)
        bpy.ops.object.parent_set(type='ARMATURE_NAME')

        vis = [x.name for x in ob.vertex_groups]

        # Sparsify
        argsorted = np.argsort(-skin, axis=1)
        vertex_group_reweight = skin[np.arange(skin.shape[0])[..., None], argsorted]

        group_per_vertex = vertex_group_reweight.shape[-1]
        vertex_group_reweight = vertex_group_reweight / vertex_group_reweight[..., :group_per_vertex].sum(axis=1)[..., None]

        for v, w in enumerate(skin):
            for ii in range(group_per_vertex):
                i = argsorted[v, ii]
                if i >= J:
                    continue
                n = names[i]
                if n not in vis:
                    continue
                ob.vertex_groups[n].add([v], vertex_group_reweight[v, ii], 'REPLACE')

    print("[Direct FBX Export] Armature created successfully")

    # Apply Mixamo-standard object transforms
    if is_mixamo_skeleton:
        print("[Direct FBX Export] Applying Mixamo-standard object transforms...")
        bpy.ops.object.mode_set(mode='OBJECT')

        arm_obj = bpy.data.objects.get('Armature')
        if arm_obj:
            arm_obj.rotation_euler = (math.radians(90), 0, 0)
            arm_obj.scale = (0.01, 0.01, 0.01)

        bpy.context.view_layer.update()

    # Export to FBX
    print("[Direct FBX Export] Exporting to FBX...")
    os.makedirs(os.path.dirname(output_fbx) if os.path.dirname(output_fbx) else '.', exist_ok=True)

    bpy.ops.export_scene.fbx(
        filepath=output_fbx,
        check_existing=False,
        add_leaf_bones=False,
        path_mode='COPY',
        embed_textures=True,
    )
    print(f"[Direct FBX Export] Saved to: {output_fbx}")
    print("[Direct FBX Export] Done!")

    return output_fbx


def _clean_bpy():
    """Clean the Blender scene."""
    import bpy  # Lazy import

    for c in bpy.data.actions:
        bpy.data.actions.remove(c)
    for c in bpy.data.armatures:
        bpy.data.armatures.remove(c)
    for c in bpy.data.cameras:
        bpy.data.cameras.remove(c)
    for c in bpy.data.collections:
        bpy.data.collections.remove(c)
    for c in bpy.data.images:
        bpy.data.images.remove(c)
    for c in bpy.data.materials:
        bpy.data.materials.remove(c)
    for c in bpy.data.meshes:
        bpy.data.meshes.remove(c)
    for c in bpy.data.objects:
        bpy.data.objects.remove(c)
    for c in bpy.data.textures:
        bpy.data.textures.remove(c)


def _extrude_bone(edit_bones, name, parent_name, head, tail, connect):
    """Create a new bone."""
    from mathutils import Vector  # Lazy import

    bone = edit_bones.new(name)
    bone.head = Vector((head[0], head[1], head[2]))
    bone.tail = Vector((tail[0], tail[1], tail[2]))
    bone.name = name
    parent_bone = edit_bones.get(parent_name)
    bone.parent = parent_bone
    bone.use_connect = connect


def _convert_smpl_tpose(joints, tails, vertices, skin, names):
    """Convert SMPL skeleton to T-pose if needed."""
    print("[Direct FBX Export] Detected SMPL skeleton, checking T-pose...")

    l_shoulder_idx = names.index('L_Shoulder')
    l_elbow_idx = names.index('L_Elbow')
    l_wrist_idx = names.index('L_Wrist')
    r_shoulder_idx = names.index('R_Shoulder')
    r_elbow_idx = names.index('R_Elbow')
    r_wrist_idx = names.index('R_Wrist')

    l_shoulder = joints[l_shoulder_idx]
    l_elbow = joints[l_elbow_idx]
    l_wrist = joints[l_wrist_idx]
    r_shoulder = joints[r_shoulder_idx]
    r_elbow = joints[r_elbow_idx]
    r_wrist = joints[r_wrist_idx]

    # Detect lateral axis
    shoulder_diff = r_shoulder - l_shoulder
    if abs(shoulder_diff[0]) > abs(shoulder_diff[2]):
        l_tpose_dir = np.array([1.0, 0.0, 0.0])
        r_tpose_dir = np.array([-1.0, 0.0, 0.0])
    else:
        if l_shoulder[2] < r_shoulder[2]:
            l_tpose_dir = np.array([0.0, 0.0, -1.0])
            r_tpose_dir = np.array([0.0, 0.0, 1.0])
        else:
            l_tpose_dir = np.array([0.0, 0.0, 1.0])
            r_tpose_dir = np.array([0.0, 0.0, -1.0])

    # Check if already T-posed
    l_arm_vec = l_wrist - l_shoulder
    l_arm_vec_norm = l_arm_vec / (np.linalg.norm(l_arm_vec) + 1e-8)

    if abs(l_arm_vec_norm[1]) < 0.1:
        print("[Direct FBX Export] Arms already horizontal (T-pose)")
        return joints, tails, vertices

    print(f"[Direct FBX Export] Converting to T-pose...")

    # Compute arm lengths
    l_upper_len = np.linalg.norm(l_elbow - l_shoulder)
    l_lower_len = np.linalg.norm(l_wrist - l_elbow)
    r_upper_len = np.linalg.norm(r_elbow - r_shoulder)
    r_lower_len = np.linalg.norm(r_wrist - r_elbow)

    # New T-pose positions
    new_l_elbow = l_shoulder + l_tpose_dir * l_upper_len
    new_l_wrist = new_l_elbow + l_tpose_dir * l_lower_len
    new_r_elbow = r_shoulder + r_tpose_dir * r_upper_len
    new_r_wrist = new_r_elbow + r_tpose_dir * r_lower_len

    # Compute rotations
    l_arm_vec_v = Vector(l_arm_vec).normalized()
    new_l_arm_vec = Vector(new_l_wrist - l_shoulder).normalized()
    l_rotation = l_arm_vec_v.rotation_difference(new_l_arm_vec)

    r_arm_vec = r_wrist - r_shoulder
    r_arm_vec_v = Vector(r_arm_vec).normalized()
    new_r_arm_vec = Vector(new_r_wrist - r_shoulder).normalized()
    r_rotation = r_arm_vec_v.rotation_difference(new_r_arm_vec)

    # Transform vertices
    if vertices is not None and skin is not None:
        left_arm_bones = {'L_Shoulder', 'L_Elbow', 'L_Wrist'}
        right_arm_bones = {'R_Shoulder', 'R_Elbow', 'R_Wrist'}

        left_bone_indices = [names.index(b) for b in left_arm_bones if b in names]
        right_bone_indices = [names.index(b) for b in right_arm_bones if b in names]

        for v_idx in range(len(vertices)):
            left_weight = sum(skin[v_idx, idx] for idx in left_bone_indices)
            right_weight = sum(skin[v_idx, idx] for idx in right_bone_indices)

            if left_weight < 0.001 and right_weight < 0.001:
                continue

            displacement = np.zeros(3)

            if left_weight > 0.001:
                rel_pos = vertices[v_idx] - l_shoulder
                rotated = np.array(l_rotation @ Vector(rel_pos))
                displacement += (rotated - rel_pos) * left_weight

            if right_weight > 0.001:
                rel_pos = vertices[v_idx] - r_shoulder
                rotated = np.array(r_rotation @ Vector(rel_pos))
                displacement += (rotated - rel_pos) * right_weight

            vertices[v_idx] += displacement

    # Update joints
    joints[l_elbow_idx] = new_l_elbow
    joints[l_wrist_idx] = new_l_wrist
    joints[r_elbow_idx] = new_r_elbow
    joints[r_wrist_idx] = new_r_wrist

    # Update tails
    if tails is not None:
        tails[l_shoulder_idx] = new_l_elbow
        tails[r_shoulder_idx] = new_r_elbow
        tails[l_elbow_idx] = new_l_wrist
        tails[r_elbow_idx] = new_r_wrist
        wrist_tail_len = 0.05
        tails[l_wrist_idx] = new_l_wrist + l_tpose_dir * wrist_tail_len
        tails[r_wrist_idx] = new_r_wrist + r_tpose_dir * wrist_tail_len

    print("[Direct FBX Export] T-pose conversion complete")
    return joints, tails, vertices


def _normalize_mixamo(joints, tails, vertices, skin, names):
    """Normalize Mixamo skeleton for animation compatibility."""
    print("[Direct FBX Export] Normalizing Mixamo skeleton...")

    # Get key bone indices
    hips_idx = None
    head_idx = None
    l_arm_idx = None
    r_arm_idx = None
    l_forearm_idx = None
    r_forearm_idx = None
    l_hand_idx = None
    r_hand_idx = None

    for i, name in enumerate(names):
        if name == 'mixamorig:Hips':
            hips_idx = i
        elif name == 'mixamorig:Head':
            head_idx = i
        elif name == 'mixamorig:LeftArm':
            l_arm_idx = i
        elif name == 'mixamorig:RightArm':
            r_arm_idx = i
        elif name == 'mixamorig:LeftForeArm':
            l_forearm_idx = i
        elif name == 'mixamorig:RightForeArm':
            r_forearm_idx = i
        elif name == 'mixamorig:LeftHand':
            l_hand_idx = i
        elif name == 'mixamorig:RightHand':
            r_hand_idx = i

    # Orient model to Mixamo standard
    if l_arm_idx is not None and r_arm_idx is not None and hips_idx is not None:
        l_shoulder = joints[l_arm_idx]
        r_shoulder = joints[r_arm_idx]
        hips = joints[hips_idx]
        head = joints[head_idx] if head_idx is not None else joints[np.argmax(joints[:, 2])]

        lateral_vec = r_shoulder - l_shoulder
        lateral_vec = lateral_vec / (np.linalg.norm(lateral_vec) + 1e-8)

        up_vec = head - hips
        up_vec = up_vec / (np.linalg.norm(up_vec) + 1e-8)

        lateral_xy = np.array([lateral_vec[0], lateral_vec[1], 0])
        lateral_xy_len = np.linalg.norm(lateral_xy)

        if lateral_xy_len > 0.1:
            lateral_xy = lateral_xy / lateral_xy_len
            target_lateral_xy = np.array([-1.0, 0.0])

            dot = lateral_xy[0] * target_lateral_xy[0] + lateral_xy[1] * target_lateral_xy[1]
            cross_z = lateral_xy[0] * target_lateral_xy[1] - lateral_xy[1] * target_lateral_xy[0]
            z_rotation_angle = math.atan2(cross_z, dot)

            if abs(z_rotation_angle) > 0.05:
                cos_a = math.cos(z_rotation_angle)
                sin_a = math.sin(z_rotation_angle)

                def rotate_z(points):
                    rotated = np.zeros_like(points)
                    rotated[..., 0] = cos_a * points[..., 0] - sin_a * points[..., 1]
                    rotated[..., 1] = sin_a * points[..., 0] + cos_a * points[..., 1]
                    rotated[..., 2] = points[..., 2]
                    return rotated

                vertices = rotate_z(vertices)
                joints = rotate_z(joints)
                if tails is not None:
                    tails = rotate_z(tails)

    # T-pose conversion for Mixamo
    if l_arm_idx is not None and r_arm_idx is not None and l_hand_idx is not None and r_hand_idx is not None:
        l_shoulder = joints[l_arm_idx]
        r_shoulder = joints[r_arm_idx]
        l_hand = joints[l_hand_idx]
        r_hand = joints[r_hand_idx]

        l_tpose_dir = np.array([1.0, 0.0, 0.0])
        r_tpose_dir = np.array([-1.0, 0.0, 0.0])

        l_arm_vec = l_hand - l_shoulder
        l_arm_vec_norm = l_arm_vec / (np.linalg.norm(l_arm_vec) + 1e-8)
        l_arm_x_component = abs(l_arm_vec_norm[0])

        if l_arm_x_component < 0.9:
            # Need T-pose conversion - simplified version
            l_elbow = joints[l_forearm_idx] if l_forearm_idx else None
            r_elbow = joints[r_forearm_idx] if r_forearm_idx else None

            if l_elbow is not None:
                l_upper_len = np.linalg.norm(l_elbow - l_shoulder)
                l_lower_len = np.linalg.norm(l_hand - l_elbow)
            else:
                l_upper_len = np.linalg.norm(l_hand - l_shoulder) / 2
                l_lower_len = l_upper_len

            if r_elbow is not None:
                r_upper_len = np.linalg.norm(r_elbow - r_shoulder)
                r_lower_len = np.linalg.norm(r_hand - r_elbow)
            else:
                r_upper_len = np.linalg.norm(r_hand - r_shoulder) / 2
                r_lower_len = r_upper_len

            new_l_elbow = l_shoulder + l_tpose_dir * l_upper_len if l_elbow is not None else None
            new_l_hand = (new_l_elbow if new_l_elbow is not None else l_shoulder) + l_tpose_dir * l_lower_len
            new_r_elbow = r_shoulder + r_tpose_dir * r_upper_len if r_elbow is not None else None
            new_r_hand = (new_r_elbow if new_r_elbow is not None else r_shoulder) + r_tpose_dir * r_lower_len

            # Compute rotations
            l_arm_vec_v = Vector(l_arm_vec).normalized()
            new_l_arm_vec = Vector(new_l_hand - l_shoulder).normalized()

            l_rot_axis = l_arm_vec_v.cross(new_l_arm_vec)
            if l_rot_axis.length > 0.0001:
                l_rot_axis.normalize()
                l_rot_angle = math.acos(max(-1, min(1, l_arm_vec_v.dot(new_l_arm_vec))))
                l_rotation = Quaternion(l_rot_axis, l_rot_angle)
            else:
                l_rotation = Quaternion()

            r_arm_vec = r_hand - r_shoulder
            r_arm_vec_v = Vector(r_arm_vec).normalized()
            new_r_arm_vec = Vector(new_r_hand - r_shoulder).normalized()

            r_rot_axis = r_arm_vec_v.cross(new_r_arm_vec)
            if r_rot_axis.length > 0.0001:
                r_rot_axis.normalize()
                r_rot_angle = math.acos(max(-1, min(1, r_arm_vec_v.dot(new_r_arm_vec))))
                r_rotation = Quaternion(r_rot_axis, r_rot_angle)
            else:
                r_rotation = Quaternion()

            # Transform vertices
            left_arm_bones = {'mixamorig:LeftArm', 'mixamorig:LeftForeArm', 'mixamorig:LeftHand'}
            right_arm_bones = {'mixamorig:RightArm', 'mixamorig:RightForeArm', 'mixamorig:RightHand'}

            for name in names:
                if name.startswith('mixamorig:LeftHand'):
                    left_arm_bones.add(name)
                elif name.startswith('mixamorig:RightHand'):
                    right_arm_bones.add(name)

            left_bone_indices = [names.index(b) for b in left_arm_bones if b in names]
            right_bone_indices = [names.index(b) for b in right_arm_bones if b in names]

            for v_idx in range(len(vertices)):
                left_weight = sum(skin[v_idx, idx] for idx in left_bone_indices if idx < skin.shape[1])
                right_weight = sum(skin[v_idx, idx] for idx in right_bone_indices if idx < skin.shape[1])

                if left_weight < 0.001 and right_weight < 0.001:
                    continue

                displacement = np.zeros(3)

                if left_weight > 0.001:
                    rel_pos = vertices[v_idx] - l_shoulder
                    rotated = np.array(l_rotation @ Vector(rel_pos))
                    displacement += (rotated - rel_pos) * left_weight

                if right_weight > 0.001:
                    rel_pos = vertices[v_idx] - r_shoulder
                    rotated = np.array(r_rotation @ Vector(rel_pos))
                    displacement += (rotated - rel_pos) * right_weight

                vertices[v_idx] += displacement

            # Update joints
            if l_forearm_idx is not None and new_l_elbow is not None:
                joints[l_forearm_idx] = new_l_elbow
            joints[l_hand_idx] = new_l_hand

            if r_forearm_idx is not None and new_r_elbow is not None:
                joints[r_forearm_idx] = new_r_elbow
            joints[r_hand_idx] = new_r_hand

            # Update tails
            if tails is not None:
                if l_forearm_idx is not None and new_l_elbow is not None:
                    tails[l_arm_idx] = new_l_elbow
                    tails[l_forearm_idx] = new_l_hand
                else:
                    tails[l_arm_idx] = new_l_hand

                if r_forearm_idx is not None and new_r_elbow is not None:
                    tails[r_arm_idx] = new_r_elbow
                    tails[r_forearm_idx] = new_r_hand
                else:
                    tails[r_arm_idx] = new_r_hand

                hand_tail_len = 0.05
                tails[l_hand_idx] = new_l_hand + l_tpose_dir * hand_tail_len
                tails[r_hand_idx] = new_r_hand + r_tpose_dir * hand_tail_len

    # Scale to human size
    current_height = vertices[:, 2].max() - vertices[:, 2].min()
    target_height = 1.7

    if current_height > 0.01:
        scale_factor = target_height / current_height
        vertices *= scale_factor
        joints *= scale_factor
        if tails is not None:
            tails *= scale_factor

    # Position feet at ground
    mesh_min_z = vertices[:, 2].min()
    z_offset = -mesh_min_z

    vertices[:, 2] += z_offset
    joints[:, 2] += z_offset
    if tails is not None:
        tails[:, 2] += z_offset

    # Convert to Y-up
    def convert_to_yup(coords):
        result = np.zeros_like(coords)
        result[..., 0] = coords[..., 0]
        result[..., 1] = coords[..., 2]
        result[..., 2] = -coords[..., 1]
        return result

    vertices = convert_to_yup(vertices) * 100.0
    joints = convert_to_yup(joints) * 100.0
    if tails is not None:
        tails = convert_to_yup(tails) * 100.0

    print("[Direct FBX Export] Mixamo normalization complete")
    return joints, tails, vertices


def _apply_texture(obj, texture_data_base64, material_name):
    """Apply texture to mesh object."""
    import bpy  # Lazy import

    print(f"[Direct FBX Export] Creating textured material...")
    try:
        # Ensure material_name is not None
        if material_name is None:
            material_name = "Material"

        png_data = base64.b64decode(texture_data_base64)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(png_data)
            tmp_texture_path = tmp.name

        mat = bpy.data.materials.new(name=material_name)
        mat.use_nodes = True
        mat.blend_method = 'OPAQUE'
        mat.shadow_method = 'OPAQUE'

        mat.node_tree.nodes.clear()

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        img_node = nodes.new(type='ShaderNodeTexImage')
        img_node.location = (-300, 300)

        bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf_node.location = (0, 300)
        bsdf_node.inputs['Metallic'].default_value = 0.0
        bsdf_node.inputs['Roughness'].default_value = 0.8
        bsdf_node.inputs['Specular IOR Level'].default_value = 0.3
        bsdf_node.inputs['Alpha'].default_value = 1.0

        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        output_node.location = (300, 300)

        links.new(img_node.outputs['Color'], bsdf_node.inputs['Base Color'])
        links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

        blender_image = bpy.data.images.load(tmp_texture_path)
        img_node.image = blender_image
        blender_image.pack()

        obj.data.materials.append(mat)

        for poly in obj.data.polygons:
            poly.material_index = 0

        print(f"[Direct FBX Export] Textured material applied")

        try:
            os.remove(tmp_texture_path)
        except:
            pass

    except Exception as tex_err:
        print(f"[Direct FBX Export] Warning: Could not apply texture: {tex_err}")


def _fix_mixamo_bone_orientations(edit_bones, names):
    """Fix bone orientations for Mixamo compatibility."""
    from mathutils import Vector  # Lazy import

    print("[Direct FBX Export] Fixing bone orientations for Mixamo...")

    DEFAULT_BONE_LENGTH = 5.0

    MIXAMO_BONE_DIRECTIONS = {
        'mixamorig:Hips': Vector((0, 1, 0)),
        'mixamorig:Spine': Vector((0, 1, 0)),
        'mixamorig:Spine1': Vector((0, 1, 0)),
        'mixamorig:Spine2': Vector((0, 1, 0)),
        'mixamorig:Neck': Vector((0, 1, 0)),
        'mixamorig:Head': Vector((0, 1, 0)),
        'mixamorig:LeftUpLeg': Vector((0, -1, 0)),
        'mixamorig:LeftLeg': Vector((0, -1, 0)),
        'mixamorig:LeftFoot': Vector((0, -0.632, 0.775)).normalized(),
        'mixamorig:LeftToeBase': Vector((0, -0.632, 0.775)).normalized(),
        'mixamorig:RightUpLeg': Vector((0, -1, 0)),
        'mixamorig:RightLeg': Vector((0, -1, 0)),
        'mixamorig:RightFoot': Vector((0, -0.632, 0.775)).normalized(),
        'mixamorig:RightToeBase': Vector((0, -0.632, 0.775)).normalized(),
        'mixamorig:LeftShoulder': Vector((1, 0, 0)),
        'mixamorig:LeftArm': Vector((1, 0, 0)),
        'mixamorig:LeftForeArm': Vector((1, 0, 0)),
        'mixamorig:LeftHand': Vector((1, 0, 0)),
        'mixamorig:RightShoulder': Vector((-1, 0, 0)),
        'mixamorig:RightArm': Vector((-1, 0, 0)),
        'mixamorig:RightForeArm': Vector((-1, 0, 0)),
        'mixamorig:RightHand': Vector((-1, 0, 0)),
    }

    for bone_name, target_direction in MIXAMO_BONE_DIRECTIONS.items():
        bone = edit_bones.get(bone_name)
        if not bone:
            continue

        current_direction = (bone.tail - bone.head).normalized()
        dot = current_direction.dot(target_direction)

        if dot < 0.95:
            bone_length = (bone.tail - bone.head).length
            if bone_length < 0.1:
                bone_length = DEFAULT_BONE_LENGTH
            bone.tail = bone.head + target_direction * bone_length

    # Fix bone rolls
    MIXAMO_BONE_ROLLS = {
        'mixamorig:LeftShoulder': Vector((0, -1, 0)),
        'mixamorig:LeftArm': Vector((0, -1, 0)),
        'mixamorig:LeftForeArm': Vector((0, -1, 0)),
        'mixamorig:LeftHand': Vector((0, -1, 0)),
        'mixamorig:RightShoulder': Vector((0, -1, 0)),
        'mixamorig:RightArm': Vector((0, -1, 0)),
        'mixamorig:RightForeArm': Vector((0, -1, 0)),
        'mixamorig:RightHand': Vector((0, -1, 0)),
        'mixamorig:LeftUpLeg': Vector((0, 0, 1)),
        'mixamorig:LeftLeg': Vector((0, 0, 1)),
        'mixamorig:LeftFoot': Vector((0, 1, 0)),
        'mixamorig:LeftToeBase': Vector((0, 1, 0)),
        'mixamorig:RightUpLeg': Vector((0, 0, 1)),
        'mixamorig:RightLeg': Vector((0, 0, 1)),
        'mixamorig:RightFoot': Vector((0, 1, 0)),
        'mixamorig:RightToeBase': Vector((0, 1, 0)),
        'mixamorig:Hips': Vector((0, 0, 1)),
        'mixamorig:Spine': Vector((0, 0, 1)),
        'mixamorig:Spine1': Vector((0, 0, 1)),
        'mixamorig:Spine2': Vector((0, 0, 1)),
        'mixamorig:Neck': Vector((0, 0, 1)),
        'mixamorig:Head': Vector((0, 0, 1)),
    }

    for bone_name, target_z in MIXAMO_BONE_ROLLS.items():
        bone = edit_bones.get(bone_name)
        if bone:
            bone.align_roll(target_z)


def _set_smpl_bone_rolls(edit_bones, names, J):
    """Set bone rolls for SMPL compatibility."""
    print("[Direct FBX Export] Setting bone rolls for SMPL...")

    for i in range(J):
        bone = edit_bones.get(names[i])
        if bone:
            direction = (bone.tail - bone.head).normalized()
            dx, dy, dz = direction.x, direction.y, direction.z

            if abs(dx) > 0.9:
                bone.align_roll(Vector((0, 1, 0)))
            elif abs(dy) > 0.9:
                bone.align_roll(Vector((0, 0, 1)))
            elif abs(dz) > 0.9:
                bone.align_roll(Vector((0, 1, 0)))
            else:
                bone.align_roll(Vector((0, 1, 0)))
