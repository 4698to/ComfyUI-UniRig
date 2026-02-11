"""
Direct bpy extraction for rest pose skeletons.
This module is imported directly when bpy is available as a Python module.
"""

import os
import numpy as np
from typing import List


def extract_rest_pose_from_fbx(input_fbx: str, output_fbx: str) -> int:
    """
    Extract rest pose from FBX file - strips animation and exports T-pose.

    Args:
        input_fbx: Path to input FBX file
        output_fbx: Path to output FBX file

    Returns:
        Number of bones in skeleton

    Raises:
        RuntimeError: If bpy is not available or export fails
    """
    try:
        import bpy
        from mathutils import Quaternion
    except ImportError:
        raise RuntimeError(
            "bpy module not available. Make sure you're running in the unirig isolated environment."
        )

    print(f"[DirectRestPose] Extracting rest pose from: {input_fbx}")

    # Clean default scene
    _clean_bpy()

    # Import FBX
    print(f"[DirectRestPose] Importing FBX...")
    bpy.ops.import_scene.fbx(filepath=input_fbx)
    print(f"[DirectRestPose] [OK] FBX imported")

    # Find armature
    armature = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE':
            armature = obj
            break

    if not armature:
        raise RuntimeError("No armature found in FBX file")

    bone_count = len(armature.data.bones)
    print(f"[DirectRestPose] Found armature: {armature.name} with {bone_count} bones")

    # Clear all animation data
    if armature.animation_data:
        armature.animation_data_clear()
        print(f"[DirectRestPose] Cleared animation data from armature")

    # Reset all bones to rest pose
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')

    for bone in armature.pose.bones:
        # Reset transforms to rest pose
        bone.rotation_mode = 'QUATERNION'
        bone.rotation_quaternion = Quaternion((1, 0, 0, 0))
        bone.location = (0, 0, 0)
        bone.scale = (1, 1, 1)

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[DirectRestPose] Reset {bone_count} bones to rest pose")

    # Also clear animation from any mesh children
    for child in armature.children:
        if child.animation_data:
            child.animation_data_clear()

    # Export FBX
    print(f"[DirectRestPose] Exporting to: {output_fbx}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_fbx), exist_ok=True)

    bpy.ops.export_scene.fbx(
        filepath=output_fbx,
        use_selection=False,
        bake_anim=False,  # Don't bake animation - we want rest pose
        add_leaf_bones=True,
        path_mode='COPY',
        embed_textures=True,
    )

    print(f"[DirectRestPose] [OK] Exported rest pose FBX")
    return bone_count


def create_smpl_skeleton_fbx(
    joint_positions: np.ndarray,
    joint_names: List[str],
    parent_indices: List[int],
    output_path: str,
) -> int:
    """
    Create an FBX file with SMPL skeleton from joint positions.

    Args:
        joint_positions: (N, 3) array of joint positions in T-pose
        joint_names: List of N joint names
        parent_indices: List of parent indices (-1 for root)
        output_path: Path to output FBX file

    Returns:
        Number of bones created

    Raises:
        RuntimeError: If bpy is not available
    """
    try:
        import bpy
        from mathutils import Vector, Matrix
    except ImportError:
        raise RuntimeError(
            "bpy module not available. Make sure you're running in the unirig isolated environment."
        )

    print(f"[DirectRestPose] Creating SMPL skeleton with {len(joint_names)} joints")

    # Clean default scene
    _clean_bpy()

    # Create new armature
    armature_data = bpy.data.armatures.new("SMPL_Armature")
    armature_obj = bpy.data.objects.new("SMPL_Skeleton", armature_data)

    # Link to scene
    bpy.context.scene.collection.objects.link(armature_obj)
    bpy.context.view_layer.objects.active = armature_obj

    # Enter edit mode to add bones
    bpy.ops.object.mode_set(mode='EDIT')

    # Create bones
    bones_by_name = {}
    edit_bones = armature_data.edit_bones

    for i, (name, pos) in enumerate(zip(joint_names, joint_positions)):
        bone = edit_bones.new(name)
        head = Vector(pos.tolist())

        # Find children to determine tail direction
        children_indices = [j for j, p in enumerate(parent_indices) if p == i]

        if children_indices:
            # Point toward average of children
            child_positions = joint_positions[children_indices]
            avg_child = np.mean(child_positions, axis=0)
            tail = Vector(avg_child.tolist())

            # Make sure tail is not too close to head
            direction = tail - head
            if direction.length < 0.01:
                # Default to pointing up if children are too close
                tail = head + Vector((0, 0.05, 0))
        else:
            # Leaf bone - extend in parent direction or default
            parent_idx = parent_indices[i]
            if parent_idx >= 0:
                parent_pos = Vector(joint_positions[parent_idx].tolist())
                direction = head - parent_pos
                if direction.length > 0.001:
                    direction.normalize()
                    tail = head + direction * 0.05
                else:
                    tail = head + Vector((0, 0.05, 0))
            else:
                # Root with no children - point up
                tail = head + Vector((0, 0.1, 0))

        bone.head = head
        bone.tail = tail
        bones_by_name[name] = bone

    # Set parent relationships
    for i, (name, parent_idx) in enumerate(zip(joint_names, parent_indices)):
        if parent_idx >= 0:
            parent_name = joint_names[parent_idx]
            bones_by_name[name].parent = bones_by_name[parent_name]

    # Exit edit mode
    bpy.ops.object.mode_set(mode='OBJECT')

    bone_count = len(joint_names)
    print(f"[DirectRestPose] Created {bone_count} bones")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export FBX
    print(f"[DirectRestPose] Exporting to: {output_path}")
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=False,
        bake_anim=False,
        add_leaf_bones=True,
    )

    print(f"[DirectRestPose] [OK] Exported SMPL skeleton FBX")
    return bone_count


def _clean_bpy():
    """Clean the Blender scene."""
    import bpy

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    for armature in bpy.data.armatures:
        bpy.data.armatures.remove(armature)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
    for image in bpy.data.images:
        bpy.data.images.remove(image)
