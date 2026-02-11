"""
Direct animation application using bpy as a Python module.

This module provides the same functionality as blender_apply_animation.py but as a
direct Python import, eliminating the need for subprocess calls to Blender.

Requires: bpy>=4.0.0 (installed via pip install bpy)
"""

import bpy
import os
import io
import tempfile
from mathutils import Matrix, Vector, Quaternion

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def apply_mixamo_animation(model_fbx: str, animation_fbx: str, output_fbx: str) -> str:
    """
    Apply Mixamo animation to a rigged FBX model.

    Args:
        model_fbx: Path to the rigged model FBX file
        animation_fbx: Path to the Mixamo animation FBX file
        output_fbx: Path for the output animated FBX file

    Returns:
        Path to the exported FBX file
    """
    print(f"[Direct Apply Animation] Model FBX: {model_fbx}")
    print(f"[Direct Apply Animation] Animation FBX: {animation_fbx}")
    print(f"[Direct Apply Animation] Output FBX: {output_fbx}")

    # Clean the scene
    _clean_scene()

    # Step 1: Import the rigged model FIRST
    print(f"[Direct Apply Animation] Importing model...")
    try:
        bpy.ops.import_scene.fbx(filepath=model_fbx)
        print(f"[Direct Apply Animation] Model imported successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to import model: {e}")

    # Find the model's armature and meshes
    model_armature = None
    model_meshes = []
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            model_armature = obj
        elif obj.type == 'MESH':
            model_meshes.append(obj)

    if model_armature is None:
        raise RuntimeError("No armature found in model FBX")

    model_bone_names = set(bone.name for bone in model_armature.pose.bones)
    print(f"[Direct Apply Animation] Found model armature: {model_armature.name} with {len(model_bone_names)} bones")
    print(f"[Direct Apply Animation] Found {len(model_meshes)} mesh object(s)")

    # Check model bones for mixamo prefix
    model_mixamo_count, model_total = _check_mixamo_prefix(model_bone_names)
    print(f"[Direct Apply Animation] Model has {model_mixamo_count}/{model_total} bones with mixamorig: prefix")

    if model_mixamo_count == 0:
        raise RuntimeError(
            "Model does not have mixamorig: bone names!\n"
            "The model must use 'mixamo' skeleton template for Mixamo animations.\n"
            f"Model bone names: {sorted(list(model_bone_names))[:10]}..."
        )

    # Store model objects to track
    existing_objects = set(bpy.data.objects[:])

    # Step 2: Import the animation FBX
    print(f"[Direct Apply Animation] Importing animation...")
    try:
        bpy.ops.import_scene.fbx(filepath=animation_fbx)
        print(f"[Direct Apply Animation] Animation imported successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to import animation: {e}")

    # Find the animation armature
    anim_armature = None
    anim_meshes = []
    for obj in bpy.data.objects:
        if obj not in existing_objects:
            if obj.type == 'ARMATURE':
                anim_armature = obj
            elif obj.type == 'MESH':
                anim_meshes.append(obj)

    if anim_armature is None:
        raise RuntimeError("No armature found in animation FBX")

    anim_bone_names = set(bone.name for bone in anim_armature.pose.bones)
    print(f"[Direct Apply Animation] Found animation armature: {anim_armature.name} with {len(anim_bone_names)} bones")
    print(f"[Direct Apply Animation] Animation armature scale: {anim_armature.scale[:]}")

    # Check animation bones for mixamo prefix
    anim_mixamo_count, anim_total = _check_mixamo_prefix(anim_bone_names)
    print(f"[Direct Apply Animation] Animation has {anim_mixamo_count}/{anim_total} bones with mixamorig: prefix")

    if anim_mixamo_count == 0:
        raise RuntimeError(
            "Animation does not have mixamorig: bone names!\n"
            "This animation file is not compatible with Mixamo-rigged models.\n"
            f"Animation bone names: {sorted(list(anim_bone_names))[:10]}..."
        )

    # Get animation action and frame range
    anim_action = None
    if anim_armature.animation_data and anim_armature.animation_data.action:
        anim_action = anim_armature.animation_data.action
        print(f"[Direct Apply Animation] Found animation action: {anim_action.name}")
        print(f"[Direct Apply Animation] Frame range: {anim_action.frame_range[0]} - {anim_action.frame_range[1]}")
    else:
        for action in bpy.data.actions:
            if action.users > 0 or anim_action is None:
                anim_action = action
        if anim_action:
            print(f"[Direct Apply Animation] Using action from data: {anim_action.name}")
        else:
            raise RuntimeError("No animation action found")

    frame_start = int(anim_action.frame_range[0])
    frame_end = int(anim_action.frame_range[1])

    # Delete any meshes from animation (we don't need them)
    for mesh in anim_meshes:
        bpy.data.objects.remove(mesh, do_unlink=True)

    # Find matching bones
    matching_bones = model_bone_names.intersection(anim_bone_names)
    print(f"[Direct Apply Animation] Matching bones: {len(matching_bones)} of {len(model_bone_names)} model bones")

    if len(matching_bones) == 0:
        raise RuntimeError("No matching bone names found!")

    # Check if we can use direct action copy (same skeleton structure)
    use_direct_copy = (len(matching_bones) == len(model_bone_names) and
                       len(matching_bones) == len(anim_bone_names))

    if use_direct_copy:
        _apply_direct_copy(model_armature, anim_armature, anim_action, model_meshes, output_fbx)
    else:
        _apply_partial_copy(model_armature, anim_armature, anim_action, matching_bones, model_meshes, output_fbx)

    print(f"[Direct Apply Animation] Saved to: {output_fbx}")
    print("[Direct Apply Animation] Done!")

    return output_fbx


def _clean_scene():
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for c in bpy.data.armatures:
        bpy.data.armatures.remove(c)
    for c in bpy.data.meshes:
        bpy.data.meshes.remove(c)
    for c in bpy.data.materials:
        bpy.data.materials.remove(c)
    for c in bpy.data.images:
        bpy.data.images.remove(c)
    for c in bpy.data.actions:
        bpy.data.actions.remove(c)


def _check_mixamo_prefix(bone_names):
    """Check if bones have mixamorig: prefix."""
    mixamo_count = sum(1 for name in bone_names if name.startswith("mixamorig:"))
    return mixamo_count, len(bone_names)


def _apply_direct_copy(model_armature, anim_armature, anim_action, model_meshes, output_fbx):
    """Apply animation using direct action copy (identical skeletons)."""
    print(f"[Direct Apply Animation] Using DIRECT action copy (identical skeletons)...")

    # Ensure model has animation data
    if not model_armature.animation_data:
        model_armature.animation_data_create()

    # Copy the action (not just link, so we can modify it)
    new_action = anim_action.copy()
    new_action.name = f"{model_armature.name}|Animation"
    model_armature.animation_data.action = new_action

    # Handle scale difference between armatures
    anim_scale = anim_armature.scale[0]
    model_scale = model_armature.scale[0]

    if abs(anim_scale - model_scale) > 0.0001:
        scale_factor = anim_scale / model_scale
        print(f"[Direct Apply Animation] Scaling location keyframes by {scale_factor:.4f}x")
        for fc in new_action.fcurves:
            if '.location' in fc.data_path:
                for kfp in fc.keyframe_points:
                    kfp.co[1] *= scale_factor
                    kfp.handle_left[1] *= scale_factor
                    kfp.handle_right[1] *= scale_factor

    print(f"[Direct Apply Animation] Direct copy complete - {len(new_action.fcurves)} F-curves")

    # Cleanup
    bpy.data.objects.remove(anim_armature, do_unlink=True)

    # Export
    _export_fbx(model_armature, model_meshes, output_fbx)


def _apply_partial_copy(model_armature, anim_armature, anim_action, matching_bones, model_meshes, output_fbx):
    """Apply animation using partial F-curve copy (different skeleton structures)."""
    print(f"[Direct Apply Animation] Using PARTIAL action copy ({len(matching_bones)} matching bones)...")

    # Ensure model has animation data
    if not model_armature.animation_data:
        model_armature.animation_data_create()

    # Create new action for model
    new_action = bpy.data.actions.new(name=f"{model_armature.name}|Animation")
    model_armature.animation_data.action = new_action

    # Handle scale difference
    anim_scale = anim_armature.scale[0]
    model_scale = model_armature.scale[0]
    scale_factor = anim_scale / model_scale if model_scale != 0 else 1.0

    # Copy F-curves for matching bones
    copied_fcurves = 0
    for fc in anim_action.fcurves:
        # Extract bone name from data_path like 'pose.bones["mixamorig:Hips"].location'
        if 'pose.bones["' in fc.data_path:
            start = fc.data_path.find('pose.bones["') + len('pose.bones["')
            end = fc.data_path.find('"]', start)
            bone_name = fc.data_path[start:end]

            if bone_name in matching_bones:
                # Create new fcurve in our action
                new_fc = new_action.fcurves.new(data_path=fc.data_path, index=fc.array_index)

                # Copy keyframes
                for kfp in fc.keyframe_points:
                    value = kfp.co[1]
                    # Scale location values
                    if '.location' in fc.data_path and abs(scale_factor - 1.0) > 0.0001:
                        value *= scale_factor

                    new_kfp = new_fc.keyframe_points.insert(kfp.co[0], value)
                    new_kfp.interpolation = kfp.interpolation
                    new_kfp.handle_left_type = kfp.handle_left_type
                    new_kfp.handle_right_type = kfp.handle_right_type

                copied_fcurves += 1

    print(f"[Direct Apply Animation] Copied {copied_fcurves} F-curves for matching bones")

    # Cleanup
    bpy.data.objects.remove(anim_armature, do_unlink=True)

    # Export
    _export_fbx(model_armature, model_meshes, output_fbx)


def _export_fbx(model_armature, model_meshes, output_fbx):
    """Export the animated model to FBX."""
    print(f"[Direct Apply Animation] Exporting animated FBX...")
    os.makedirs(os.path.dirname(output_fbx) if os.path.dirname(output_fbx) else '.', exist_ok=True)

    # Fix material transparency - FBX embeds PNGs with alpha, causing transparency on reimport.
    # Solution: Convert images to JPEG (no alpha channel) using PIL before export.

    if HAS_PIL:
        for img in list(bpy.data.images):
            if img.has_data and img.channels == 4 and img.packed_file:
                try:
                    # Get packed PNG data and convert to RGB JPEG using PIL
                    png_data = img.packed_file.data
                    pil_img = Image.open(io.BytesIO(png_data))
                    if pil_img.mode == 'RGBA':
                        pil_img = pil_img.convert('RGB')

                    # Save as JPEG (no alpha support)
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        pil_img.save(tmp.name, 'JPEG', quality=95)
                        tmp_path = tmp.name

                    # Load new image and update texture nodes
                    new_img = bpy.data.images.load(tmp_path)
                    new_img.pack()

                    for mat in bpy.data.materials:
                        if mat and mat.use_nodes and mat.node_tree:
                            for node in mat.node_tree.nodes:
                                if node.type == 'TEX_IMAGE' and node.image == img:
                                    node.image = new_img

                    # Remove old image and rename new one
                    old_name = img.name
                    bpy.data.images.remove(img)
                    new_img.name = old_name

                    os.remove(tmp_path)
                    print(f"[Direct Apply Animation] Converted to RGB JPEG: {old_name}")
                except Exception as e:
                    print(f"[Direct Apply Animation] Warning: Could not convert {img.name}: {e}")
    else:
        print("[Direct Apply Animation] Warning: PIL not available, skipping image alpha fix")

    for mat in bpy.data.materials:
        if mat:
            mat.blend_method = 'OPAQUE'
            mat.shadow_method = 'OPAQUE'
            if mat.use_nodes and mat.node_tree:
                # Find nodes that are ONLY connected to Alpha (and remove them)
                nodes_to_remove = []
                for node in mat.node_tree.nodes:
                    if node.type == 'TEX_IMAGE':
                        # Check if this texture is only connected to Alpha
                        outputs_used = set()
                        for link in mat.node_tree.links:
                            if link.from_node == node:
                                outputs_used.add((link.to_node.type, link.to_socket.name))
                        # If only connected to Alpha on a BSDF, remove it
                        if outputs_used == {('BSDF_PRINCIPLED', 'Alpha')}:
                            nodes_to_remove.append(node)
                            print(f"[Direct Apply Animation] Removing alpha-only texture: {node.name}")

                for node in nodes_to_remove:
                    mat.node_tree.nodes.remove(node)

                # Also disconnect any remaining alpha links and set alpha to 1.0
                for node in mat.node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        alpha_links = [link for link in mat.node_tree.links
                                       if link.to_node == node and link.to_socket.name == 'Alpha']
                        for link in alpha_links:
                            mat.node_tree.links.remove(link)
                        if 'Alpha' in node.inputs:
                            node.inputs['Alpha'].default_value = 1.0

                print(f"[Direct Apply Animation] Fixed material: {mat.name} -> OPAQUE")

    bpy.ops.object.select_all(action='DESELECT')
    model_armature.select_set(True)
    for mesh in model_meshes:
        if mesh.name in bpy.data.objects:
            mesh.select_set(True)

    bpy.context.view_layer.objects.active = model_armature

    bpy.ops.export_scene.fbx(
        filepath=output_fbx,
        use_selection=True,
        check_existing=False,
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        path_mode='COPY',
        embed_textures=True,
        mesh_smooth_type='FACE',
        use_mesh_modifiers=True,
    )
