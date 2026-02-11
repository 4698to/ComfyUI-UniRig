"""
Blender script to apply a Mixamo animation to a rigged FBX model.
Uses constraint-based retargeting with proper scale handling.
Usage: blender --background --python blender_apply_animation.py -- <model_fbx> <animation_fbx> <output_fbx>
"""

import bpy
import sys
import os
from mathutils import Matrix, Vector, Quaternion

# Get arguments after '--'
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

if len(argv) < 3:
    print("Usage: blender --background --python blender_apply_animation.py -- <model_fbx> <animation_fbx> <output_fbx>")
    sys.exit(1)

model_fbx = argv[0]
animation_fbx = argv[1]
output_fbx = argv[2]

print(f"[Blender Apply Animation] Model FBX: {model_fbx}")
print(f"[Blender Apply Animation] Animation FBX: {animation_fbx}")
print(f"[Blender Apply Animation] Output FBX: {output_fbx}")


def clean_scene():
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


def check_mixamo_prefix(bone_names):
    """Check if bones have mixamorig: prefix."""
    mixamo_count = sum(1 for name in bone_names if name.startswith("mixamorig:"))
    return mixamo_count, len(bone_names)


clean_scene()

# Step 1: Import the rigged model FIRST
print(f"[Blender Apply Animation] Importing model...")
try:
    bpy.ops.import_scene.fbx(filepath=model_fbx)
    print(f"[Blender Apply Animation] Model imported successfully")
except Exception as e:
    print(f"[Blender Apply Animation] Failed to import model: {e}")
    sys.exit(1)

# Find the model's armature and meshes
model_armature = None
model_meshes = []
for obj in bpy.data.objects:
    if obj.type == 'ARMATURE':
        model_armature = obj
    elif obj.type == 'MESH':
        model_meshes.append(obj)

if model_armature is None:
    print(f"[Blender Apply Animation] Error: No armature found in model FBX")
    sys.exit(1)

model_bone_names = set(bone.name for bone in model_armature.pose.bones)
print(f"[Blender Apply Animation] Found model armature: {model_armature.name} with {len(model_bone_names)} bones")
print(f"[Blender Apply Animation] Found {len(model_meshes)} mesh object(s)")

# Check model bones for mixamo prefix
model_mixamo_count, model_total = check_mixamo_prefix(model_bone_names)
print(f"[Blender Apply Animation] Model has {model_mixamo_count}/{model_total} bones with mixamorig: prefix")

if model_mixamo_count == 0:
    print(f"[Blender Apply Animation] ERROR: Model does not have mixamorig: bone names!")
    print(f"[Blender Apply Animation] The model must use 'mixamo' skeleton template for Mixamo animations.")
    print(f"[Blender Apply Animation] Model bone names: {sorted(model_bone_names)[:10]}...")
    sys.exit(1)

# Store model objects to track
existing_objects = set(bpy.data.objects[:])

# Step 2: Import the animation FBX
print(f"[Blender Apply Animation] Importing animation...")
try:
    bpy.ops.import_scene.fbx(filepath=animation_fbx)
    print(f"[Blender Apply Animation] Animation imported successfully")
except Exception as e:
    print(f"[Blender Apply Animation] Failed to import animation: {e}")
    sys.exit(1)

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
    print(f"[Blender Apply Animation] Error: No armature found in animation FBX")
    sys.exit(1)

anim_bone_names = set(bone.name for bone in anim_armature.pose.bones)
print(f"[Blender Apply Animation] Found animation armature: {anim_armature.name} with {len(anim_bone_names)} bones")
print(f"[Blender Apply Animation] Animation armature scale: {anim_armature.scale[:]}")

# Check animation bones for mixamo prefix
anim_mixamo_count, anim_total = check_mixamo_prefix(anim_bone_names)
print(f"[Blender Apply Animation] Animation has {anim_mixamo_count}/{anim_total} bones with mixamorig: prefix")

if anim_mixamo_count == 0:
    print(f"[Blender Apply Animation] ERROR: Animation does not have mixamorig: bone names!")
    print(f"[Blender Apply Animation] This animation file is not compatible with Mixamo-rigged models.")
    print(f"[Blender Apply Animation] Animation bone names: {sorted(anim_bone_names)[:10]}...")
    sys.exit(1)

# Get animation action and frame range
anim_action = None
if anim_armature.animation_data and anim_armature.animation_data.action:
    anim_action = anim_armature.animation_data.action
    print(f"[Blender Apply Animation] Found animation action: {anim_action.name}")
    print(f"[Blender Apply Animation] Frame range: {anim_action.frame_range[0]} - {anim_action.frame_range[1]}")
else:
    for action in bpy.data.actions:
        if action.users > 0 or anim_action is None:
            anim_action = action
    if anim_action:
        print(f"[Blender Apply Animation] Using action from data: {anim_action.name}")
    else:
        print(f"[Blender Apply Animation] Error: No animation action found")
        sys.exit(1)

frame_start = int(anim_action.frame_range[0])
frame_end = int(anim_action.frame_range[1])

# Delete any meshes from animation (we don't need them)
for mesh in anim_meshes:
    bpy.data.objects.remove(mesh, do_unlink=True)

# Find matching bones
matching_bones = model_bone_names.intersection(anim_bone_names)
print(f"[Blender Apply Animation] Matching bones: {len(matching_bones)} of {len(model_bone_names)} model bones")

if len(matching_bones) == 0:
    print(f"[Blender Apply Animation] ERROR: No matching bone names found!")
    sys.exit(1)

# Check if we can use direct action copy (same skeleton structure)
# This works when model and animation have identical bone names
use_direct_copy = (len(matching_bones) == len(model_bone_names) and
                   len(matching_bones) == len(anim_bone_names))

if use_direct_copy:
    print(f"[Blender Apply Animation] Using DIRECT action copy (identical skeletons)...")

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
        print(f"[Blender Apply Animation] Scaling location keyframes by {scale_factor:.4f}x")
        for fc in new_action.fcurves:
            if '.location' in fc.data_path:
                for kfp in fc.keyframe_points:
                    kfp.co[1] *= scale_factor
                    kfp.handle_left[1] *= scale_factor
                    kfp.handle_right[1] *= scale_factor

    print(f"[Blender Apply Animation] Direct copy complete - {len(new_action.fcurves)} F-curves")

    # Skip to cleanup
    bpy.data.objects.remove(anim_armature, do_unlink=True)

    # Export
    print(f"[Blender Apply Animation] Exporting animated FBX...")
    os.makedirs(os.path.dirname(output_fbx) if os.path.dirname(output_fbx) else '.', exist_ok=True)

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

    print(f"[Blender Apply Animation] Saved to: {output_fbx}")
    print("[Blender Apply Animation] Done!")
    sys.exit(0)

# Step 3: Partial action copy - copy only F-curves for matching bones
print(f"[Blender Apply Animation] Using PARTIAL action copy ({len(matching_bones)} matching bones)...")

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

print(f"[Blender Apply Animation] Copied {copied_fcurves} F-curves for matching bones")

# Cleanup and export
bpy.data.objects.remove(anim_armature, do_unlink=True)

print(f"[Blender Apply Animation] Exporting animated FBX...")
os.makedirs(os.path.dirname(output_fbx) if os.path.dirname(output_fbx) else '.', exist_ok=True)

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

print(f"[Blender Apply Animation] Saved to: {output_fbx}")
print("[Blender Apply Animation] Done!")
sys.exit(0)

# OLD: Constraint-based retargeting (keeping for reference but unreachable)
print(f"[Blender Apply Animation] Using constraint-based retargeting...")

# Get scale factor between armatures
anim_scale = anim_armature.scale[0]
model_scale = model_armature.scale[0]
scale_ratio = anim_scale / model_scale if model_scale != 0 else anim_scale
print(f"[Blender Apply Animation] Animation armature scale: {anim_scale}, Model scale: {model_scale}")

# IMPORTANT: Scale model armature DOWN to match animation armature scale
# This ensures world-space matrices are comparable
print(f"[Blender Apply Animation] Scaling model to match animation armature...")
model_armature.scale = anim_armature.scale
for mesh in model_meshes:
    mesh.scale = anim_armature.scale

# Position both at origin
model_armature.location = (0, 0, 0)
anim_armature.location = (0, 0, 0)

bpy.context.view_layer.update()

# Set up scene frame range
bpy.context.scene.frame_start = frame_start
bpy.context.scene.frame_end = frame_end

# Enter pose mode for model armature
bpy.context.view_layer.objects.active = model_armature
bpy.ops.object.mode_set(mode='POSE')

root_bone_name = "mixamorig:Hips"

# Step 4: Set up WORLD space constraints (bypasses rest pose differences)
print(f"[Blender Apply Animation] Setting up WORLD space constraints...")

for bone_name in matching_bones:
    model_bone = model_armature.pose.bones[bone_name]

    # Clear any existing constraints
    for c in list(model_bone.constraints):
        model_bone.constraints.remove(c)

    # Copy Rotation in WORLD space
    rot_c = model_bone.constraints.new('COPY_ROTATION')
    rot_c.name = "AnimRetarget_Rot"
    rot_c.target = anim_armature
    rot_c.subtarget = bone_name
    rot_c.target_space = 'WORLD'
    rot_c.owner_space = 'WORLD'

    # Copy Location only for root bone, also in WORLD space
    if bone_name == root_bone_name:
        loc_c = model_bone.constraints.new('COPY_LOCATION')
        loc_c.name = "AnimRetarget_Loc"
        loc_c.target = anim_armature
        loc_c.subtarget = bone_name
        loc_c.target_space = 'WORLD'
        loc_c.owner_space = 'WORLD'

print(f"[Blender Apply Animation] Added WORLD space constraints to {len(matching_bones)} bones")

bpy.ops.object.mode_set(mode='OBJECT')

# Step 5: Bake the animation
print(f"[Blender Apply Animation] Baking animation (frames {frame_start} to {frame_end})...")

bpy.context.scene.frame_start = frame_start
bpy.context.scene.frame_end = frame_end
bpy.context.scene.frame_set(frame_start)

bpy.ops.object.select_all(action='DESELECT')
model_armature.select_set(True)
bpy.context.view_layer.objects.active = model_armature

bpy.ops.nla.bake(
    frame_start=frame_start,
    frame_end=frame_end,
    only_selected=True,
    visual_keying=True,
    clear_constraints=True,
    use_current_action=False,
    bake_types={'POSE'}
)

# Debug: check result
if model_armature.animation_data and model_armature.animation_data.action:
    action = model_armature.animation_data.action
    print(f"[Blender Apply Animation] Created {len(action.fcurves)} F-curves")
    # Check a specific bone
    for fc in action.fcurves[:5]:
        if len(fc.keyframe_points) > 0:
            print(f"[Blender Apply Animation] {fc.data_path}[{fc.array_index}]: first={fc.keyframe_points[0].co[1]:.4f}, last={fc.keyframe_points[-1].co[1]:.4f}")

print(f"[Blender Apply Animation] Animation baking complete")

bpy.ops.object.mode_set(mode='OBJECT')

# Step 5: Scale model back to original size
print(f"[Blender Apply Animation] Restoring model scale...")
model_armature.scale = (1.0, 1.0, 1.0)
for mesh in model_meshes:
    mesh.scale = (1.0, 1.0, 1.0)

# Scale location keyframes to compensate for the scale change
# When armature was at 0.01 scale, local position 82 = world 0.82
# Now at 1.0 scale, we need local position 0.82 for the same world result
# So multiply by scale_ratio (0.01) = divide by 100
if model_armature.animation_data and model_armature.animation_data.action:
    action = model_armature.animation_data.action
    for fc in action.fcurves:
        if '.location' in fc.data_path:
            for kfp in fc.keyframe_points:
                kfp.co[1] *= scale_ratio
                kfp.handle_left[1] *= scale_ratio
                kfp.handle_right[1] *= scale_ratio
    print(f"[Blender Apply Animation] Scaled location keyframes by {scale_ratio:.4f}x")

bpy.context.view_layer.update()

# Step 6: Clean up
print(f"[Blender Apply Animation] Cleaning up...")
bpy.data.objects.remove(anim_armature, do_unlink=True)

# Step 7: Export
print(f"[Blender Apply Animation] Exporting animated FBX...")
os.makedirs(os.path.dirname(output_fbx) if os.path.dirname(output_fbx) else '.', exist_ok=True)

# Fix material transparency (FBX import loses OPAQUE setting)
for mat in bpy.data.materials:
    if mat:
        mat.blend_method = 'OPAQUE'
        mat.shadow_method = 'OPAQUE'

try:
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

    print(f"[Blender Apply Animation] Saved to: {output_fbx}")
    print("[Blender Apply Animation] Done!")

except Exception as e:
    print(f"[Blender Apply Animation] Export failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
