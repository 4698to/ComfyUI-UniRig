"""
Blender script to export FBX file with a custom pose applied.
This script loads an existing FBX, applies bone transforms, and exports the result.
Usage: blender --background --python blender_export_posed_fbx.py -- <input_fbx> <output_fbx> <transforms_json>
"""

import bpy
import sys
import os
import json
from mathutils import Vector, Quaternion, Euler

# Get arguments after '--'
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

if len(argv) < 3:
    print("Usage: blender --background --python blender_export_posed_fbx.py -- <input_fbx> <output_fbx> <transforms_json>")
    sys.exit(1)

input_fbx = argv[0]
output_fbx = argv[1]
transforms_json = argv[2]

print(f"[Blender Posed FBX Export] Input FBX: {input_fbx}")
print(f"[Blender Posed FBX Export] Output FBX: {output_fbx}")
print(f"[Blender Posed FBX Export] Transforms: {transforms_json}")

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
try:
    print(f"[Blender Posed FBX Export] Importing FBX...")
    bpy.ops.import_scene.fbx(filepath=input_fbx)
    print(f"[Blender Posed FBX Export] ✓ FBX imported successfully")

except Exception as e:
    print(f"[Blender Posed FBX Export] Failed to import FBX: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Find armature object
armature_obj = None
for obj in bpy.data.objects:
    if obj.type == 'ARMATURE':
        armature_obj = obj
        break

if armature_obj is None:
    print(f"[Blender Posed FBX Export] Error: No armature found in FBX file")
    sys.exit(1)

print(f"[Blender Posed FBX Export] Found armature: {armature_obj.name}")
print(f"[Blender Posed FBX Export] Bones: {len(armature_obj.pose.bones)}")

# Load transform data from JSON
try:
    with open(transforms_json, 'r') as f:
        bone_transforms = json.load(f)

    print(f"[Blender Posed FBX Export] Loaded DELTA transforms (pose offsets from rest) for {len(bone_transforms)} bones")

except Exception as e:
    print(f"[Blender Posed FBX Export] Failed to load transforms: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Apply bone transforms in pose mode (using world-space to local-space conversion)
try:
    # Set active object
    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)

    # Enter pose mode
    bpy.ops.object.mode_set(mode='POSE')

    # Apply DELTA transforms from Three.js as pose offsets
    # These are computed as (current - rest) in the viewer, which matches Blender's expectation
    # that pose_bone.location/rotation are OFFSETS from the rest pose

    applied_count = 0
    for bone_name, transform in bone_transforms.items():
        if bone_name in armature_obj.pose.bones:
            pose_bone = armature_obj.pose.bones[bone_name]

            # Apply delta transforms as pose offsets (this is what pose_bone.location expects)
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
            print(f"[Blender Posed FBX Export] Warning: Bone '{bone_name}' not found in armature")

    print(f"[Blender Posed FBX Export] ✓ Applied transforms to {applied_count}/{len(bone_transforms)} bones")

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

except Exception as e:
    print(f"[Blender Posed FBX Export] Failed to apply transforms: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Export to FBX
print("[Blender Posed FBX Export] Exporting to FBX...")
os.makedirs(os.path.dirname(output_fbx) if os.path.dirname(output_fbx) else '.', exist_ok=True)

try:
    bpy.ops.export_scene.fbx(
        filepath=output_fbx,
        check_existing=False,
        use_selection=False,
        apply_scale_options='FBX_SCALE_NONE',
        bake_space_transform=False,
        bake_anim=False,
        add_leaf_bones=False,
        path_mode='COPY',
        embed_textures=True,
    )

    print(f"[Blender Posed FBX Export] ✓ Saved to: {output_fbx}")
    print("[Blender Posed FBX Export] Done!")
except Exception as e:
    print(f"[Blender Posed FBX Export] Export failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
