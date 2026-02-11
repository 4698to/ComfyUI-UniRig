"""
Direct bone debug data extraction using bpy as a Python module.

This module provides bone debugging information for the UniRig View Rigging node.
Uses the same direct bpy import pattern as direct_export_fbx.py.

Requires: bpy>=4.0.0 (installed via pip install bpy)
"""

import bpy
import math
from mathutils import Vector, Matrix, Euler


def extract_bone_debug(fbx_path: str) -> dict:
    """
    Load FBX and extract comprehensive bone debug data.

    Args:
        fbx_path: Path to the FBX file to load

    Returns:
        dict with:
            - 'bones': list of bone data dicts
            - 'armature_name': name of the armature
            - 'bone_count': total number of bones
            - 'has_mesh': whether a mesh was found
    """
    print(f"[Direct Bone Debug] Loading FBX: {fbx_path}")

    # Clean scene first
    _clean_bpy()

    # Import FBX
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    # Find armature
    armature_obj = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            armature_obj = obj
            break

    if not armature_obj:
        print("[Direct Bone Debug] No armature found in FBX")
        return {
            'bones': [],
            'armature_name': None,
            'bone_count': 0,
            'has_mesh': _has_mesh(),
            'error': 'No armature found'
        }

    armature = armature_obj.data
    print(f"[Direct Bone Debug] Found armature: {armature_obj.name} with {len(armature.bones)} bones")

    # We need to enter edit mode to get accurate bone data (head, tail, roll)
    # First, make the armature object active
    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)

    # Store bone data from edit bones (has accurate head/tail/roll)
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature.edit_bones

    bones_data = []
    for edit_bone in edit_bones:
        # Get world-space positions by applying armature's world matrix
        world_matrix = armature_obj.matrix_world
        head_world = world_matrix @ edit_bone.head
        tail_world = world_matrix @ edit_bone.tail

        # Calculate length
        length = edit_bone.length

        # Get roll (in radians)
        roll = edit_bone.roll

        # Get parent name
        parent_name = edit_bone.parent.name if edit_bone.parent else None

        # Get local matrix (4x4)
        local_matrix = edit_bone.matrix.copy()

        # Store bone data
        bone_data = {
            'name': edit_bone.name,
            'head': [head_world.x, head_world.y, head_world.z],
            'tail': [tail_world.x, tail_world.y, tail_world.z],
            'head_local': [edit_bone.head.x, edit_bone.head.y, edit_bone.head.z],
            'tail_local': [edit_bone.tail.x, edit_bone.tail.y, edit_bone.tail.z],
            'length': length,
            'roll': roll,
            'roll_degrees': math.degrees(roll),
            'parent_name': parent_name,
            'matrix_local': _matrix_to_list(local_matrix),
            'connected': edit_bone.use_connect,
            'children': [child.name for child in edit_bone.children],
        }

        bones_data.append(bone_data)

    # Exit edit mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Now get pose bone data (rotations in pose space)
    pose_bones = armature_obj.pose.bones
    for bone_data in bones_data:
        pose_bone = pose_bones.get(bone_data['name'])
        if pose_bone:
            # Get local rotation as euler (XYZ)
            local_euler = pose_bone.rotation_euler if pose_bone.rotation_mode == 'XYZ' else \
                         pose_bone.rotation_quaternion.to_euler('XYZ')

            # Get world rotation
            world_matrix = armature_obj.matrix_world @ pose_bone.matrix
            world_euler = world_matrix.to_euler('XYZ')

            bone_data['local_rotation_euler'] = [local_euler.x, local_euler.y, local_euler.z]
            bone_data['local_rotation_degrees'] = [
                math.degrees(local_euler.x),
                math.degrees(local_euler.y),
                math.degrees(local_euler.z)
            ]
            bone_data['world_rotation_euler'] = [world_euler.x, world_euler.y, world_euler.z]
            bone_data['world_rotation_degrees'] = [
                math.degrees(world_euler.x),
                math.degrees(world_euler.y),
                math.degrees(world_euler.z)
            ]
            bone_data['rotation_mode'] = pose_bone.rotation_mode

    # Build hierarchy depth for each bone
    _compute_hierarchy_depth(bones_data)

    print(f"[Direct Bone Debug] Extracted data for {len(bones_data)} bones")

    return {
        'bones': bones_data,
        'armature_name': armature_obj.name,
        'bone_count': len(bones_data),
        'has_mesh': _has_mesh(),
        'armature_world_matrix': _matrix_to_list(armature_obj.matrix_world),
    }


def _clean_bpy():
    """Clean the Blender scene."""
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


def _has_mesh() -> bool:
    """Check if scene contains any mesh objects."""
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            return True
    return False


def _matrix_to_list(matrix: Matrix) -> list:
    """Convert a Blender Matrix to a flat list (row-major)."""
    return [matrix[row][col] for row in range(4) for col in range(4)]


def _compute_hierarchy_depth(bones_data: list):
    """Compute hierarchy depth for each bone."""
    # Build name -> bone_data lookup
    bone_lookup = {bone['name']: bone for bone in bones_data}

    def get_depth(bone_data, cache={}):
        name = bone_data['name']
        if name in cache:
            return cache[name]

        parent_name = bone_data.get('parent_name')
        if parent_name is None:
            cache[name] = 0
            return 0

        parent = bone_lookup.get(parent_name)
        if parent is None:
            cache[name] = 0
            return 0

        depth = get_depth(parent, cache) + 1
        cache[name] = depth
        return depth

    cache = {}
    for bone_data in bones_data:
        bone_data['hierarchy_depth'] = get_depth(bone_data, cache)
