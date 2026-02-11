"""
Direct bpy export for posed FBX files.
This module is imported directly when bpy is available as a Python module.
"""

import os
import json


def export_posed_fbx(input_fbx: str, output_fbx: str, bone_transforms: dict) -> bool:
    """
    Export FBX file with custom pose applied using bpy directly.

    Args:
        input_fbx: Path to input FBX file
        output_fbx: Path to output FBX file
        bone_transforms: Dictionary of bone name -> transform dict with position/quaternion/scale

    Returns:
        True if successful

    Raises:
        RuntimeError: If bpy is not available or export fails
    """
    import bpy
    from mathutils import Vector, Quaternion

    print(f"[DirectPosedFBX] Input FBX: {input_fbx}")
    print(f"[DirectPosedFBX] Output FBX: {output_fbx}")
    print(f"[DirectPosedFBX] Transforms for {len(bone_transforms)} bones")

    # Clean default scene
    def clean_bpy():
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

    clean_bpy()

    # Import FBX
    print(f"[DirectPosedFBX] Importing FBX...")
    bpy.ops.import_scene.fbx(filepath=input_fbx)
    print(f"[DirectPosedFBX] [OK] FBX imported successfully")

    # Recreate materials from scratch to match original export encoding
    # This prevents transparency issues where Three.js interprets re-exported materials differently
    print(f"[DirectPosedFBX] Recreating materials from scratch...")

    # Extract texture data from imported materials before deleting them
    texture_images = {}
    old_material_names = {}

    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            for slot_idx, slot in enumerate(obj.material_slots):
                if slot.material and slot.material.use_nodes:
                    mat = slot.material
                    old_material_names[slot_idx] = mat.name

                    # Find texture image in material nodes
                    for node in mat.node_tree.nodes:
                        if node.type == 'TEX_IMAGE' and node.image:
                            texture_images[slot_idx] = node.image
                            print(f"[DirectPosedFBX] Found texture: {node.image.name} ({node.image.size[0]}x{node.image.size[1]})")
                            break

    # Delete all imported materials
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)
    print(f"[DirectPosedFBX] Deleted imported materials")

    # Recreate materials from scratch for each mesh object
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            # Clear existing material slots
            obj.data.materials.clear()

            # Create new material from scratch
            for slot_idx, tex_image in texture_images.items():
                mat_name = old_material_names.get(slot_idx, 'material')

                # Create material with nodes
                mat = bpy.data.materials.new(name=mat_name)
                mat.use_nodes = True

                # Clear default nodes
                mat.node_tree.nodes.clear()

                # Create nodes
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links

                # Create image texture node
                img_node = nodes.new(type='ShaderNodeTexImage')
                img_node.location = (-300, 300)
                img_node.image = tex_image

                # Create principled BSDF node
                bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf_node.location = (0, 300)

                # Set material properties
                bsdf_node.inputs['Metallic'].default_value = 0.0
                bsdf_node.inputs['Roughness'].default_value = 0.8
                bsdf_node.inputs['Specular IOR Level'].default_value = 0.3

                # Create output node
                output_node = nodes.new(type='ShaderNodeOutputMaterial')
                output_node.location = (300, 300)

                # Link nodes: Image -> BSDF -> Output
                links.new(img_node.outputs['Color'], bsdf_node.inputs['Base Color'])
                links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

                # Assign material to mesh
                obj.data.materials.append(mat)

                print(f"[DirectPosedFBX] [OK] Recreated material: {mat_name}")

    print(f"[DirectPosedFBX] [OK] Recreated {len(texture_images)} materials from scratch")

    # Find armature object
    armature_obj = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            armature_obj = obj
            break

    if armature_obj is None:
        raise RuntimeError("No armature found in FBX file")

    print(f"[DirectPosedFBX] Found armature: {armature_obj.name}")
    print(f"[DirectPosedFBX] Bones: {len(armature_obj.pose.bones)}")

    # Apply bone transforms in pose mode
    # Set active object
    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)

    # Enter pose mode
    bpy.ops.object.mode_set(mode='POSE')

    # Apply DELTA transforms from Three.js as pose offsets
    applied_count = 0
    for bone_name, transform in bone_transforms.items():
        if bone_name in armature_obj.pose.bones:
            pose_bone = armature_obj.pose.bones[bone_name]

            # Apply delta transforms as pose offsets
            if 'position' in transform:
                pos = transform['position']
                pose_bone.location = Vector((pos['x'], pos['y'], pos['z']))

            if 'quaternion' in transform:
                quat = transform['quaternion']
                pose_bone.rotation_mode = 'QUATERNION'
                pose_bone.rotation_quaternion = Quaternion((quat['w'], quat['x'], quat['y'], quat['z']))

            if 'scale' in transform:
                scale = transform['scale']
                pose_bone.scale = Vector((scale['x'], scale['y'], scale['z']))

            applied_count += 1
        else:
            print(f"[DirectPosedFBX] Warning: Bone '{bone_name}' not found in armature")

    print(f"[DirectPosedFBX] [OK] Applied transforms to {applied_count}/{len(bone_transforms)} bones")

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Export to FBX
    print("[DirectPosedFBX] Exporting to FBX...")
    os.makedirs(os.path.dirname(output_fbx) if os.path.dirname(output_fbx) else '.', exist_ok=True)

    bpy.ops.export_scene.fbx(
        filepath=output_fbx,
        check_existing=False,
        add_leaf_bones=False,
        path_mode='COPY',
        embed_textures=True,
    )

    print(f"[DirectPosedFBX] [OK] Saved to: {output_fbx}")
    print("[DirectPosedFBX] Done!")

    return True
