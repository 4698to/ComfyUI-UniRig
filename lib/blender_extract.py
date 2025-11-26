"""
Blender script for mesh extraction and preprocessing for UniRig.
This script is designed to be run via: blender --background --python blender_extract.py -- <args>
"""

import bpy
import sys
import os
import numpy as np
from pathlib import Path
import base64
from io import BytesIO
import struct

# Get arguments after '--'
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

if len(argv) < 2:
    print("Usage: blender --background --python blender_extract.py -- <input_file> <output_npz>")
    sys.exit(1)

input_file = argv[0]
output_npz = argv[1]
target_face_count = int(argv[2]) if len(argv) > 2 else 50000

print(f"[Blender Extract] Input: {input_file}")
print(f"[Blender Extract] Output: {output_npz}")
print(f"[Blender Extract] Target faces: {target_face_count}")

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh based on file extension
ext = Path(input_file).suffix.lower()
print(f"[Blender Extract] Loading {ext} file...")

try:
    if ext == '.obj':
        bpy.ops.wm.obj_import(filepath=input_file)
    elif ext in ['.fbx', '.FBX']:
        bpy.ops.import_scene.fbx(filepath=input_file, ignore_leaf_bones=False, use_image_search=False)
    elif ext in ['.glb', '.gltf']:
        bpy.ops.import_scene.gltf(filepath=input_file, import_pack_images=False)
    elif ext == '.dae':
        bpy.ops.wm.collada_import(filepath=input_file)
    elif ext == '.stl':
        bpy.ops.wm.stl_import(filepath=input_file)
    else:
        print(f"[Blender Extract] Unsupported format: {ext}")
        sys.exit(1)

    print(f"[Blender Extract] Import successful")

except Exception as e:
    print(f"[Blender Extract] Import failed: {e}")
    sys.exit(1)

# Get all meshes
meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

if not meshes:
    print("[Blender Extract] Error: No meshes found in file")
    sys.exit(1)

print(f"[Blender Extract] Found {len(meshes)} mesh(es)")

# Combine all meshes
if len(meshes) > 1:
    # Select all meshes
    bpy.ops.object.select_all(action='DESELECT')
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]

    # Join meshes
    bpy.ops.object.join()
    mesh_obj = bpy.context.active_object
else:
    mesh_obj = meshes[0]

print(f"[Blender Extract] Processing mesh: {mesh_obj.name}")

# Apply all transforms
bpy.ops.object.select_all(action='DESELECT')
mesh_obj.select_set(True)
bpy.context.view_layer.objects.active = mesh_obj
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# Get mesh data
mesh = mesh_obj.data

# Triangulate
print("[Blender Extract] Triangulating...")
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.quads_convert_to_tris()
bpy.ops.object.mode_set(mode='OBJECT')

# Simplify if needed
current_faces = len(mesh.polygons)
print(f"[Blender Extract] Current face count: {current_faces}")

if current_faces > target_face_count:
    print(f"[Blender Extract] Decimating to {target_face_count} faces...")

    # Add decimate modifier
    decimate_mod = mesh_obj.modifiers.new(name='Decimate', type='DECIMATE')
    decimate_mod.ratio = target_face_count / current_faces
    decimate_mod.use_collapse_triangulate = True

    # Apply modifier
    bpy.ops.object.modifier_apply(modifier=decimate_mod.name)

    print(f"[Blender Extract] Decimated to {len(mesh.polygons)} faces")

# Extract vertex and face data
vertices = np.zeros((len(mesh.vertices), 3), dtype=np.float32)
for i, v in enumerate(mesh.vertices):
    vertices[i] = v.co

faces = np.zeros((len(mesh.polygons), 3), dtype=np.int32)
for i, p in enumerate(mesh.polygons):
    if len(p.vertices) != 3:
        print(f"[Blender Extract] Warning: Non-triangular face found")
        continue
    faces[i] = [p.vertices[0], p.vertices[1], p.vertices[2]]

print(f"[Blender Extract] Extracted {len(vertices)} vertices, {len(faces)} faces")

# Calculate vertex normals (Blender 4.2+ compatible)
# Force recalculation by updating the mesh
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.object.mode_set(mode='OBJECT')

vertex_normals = np.zeros((len(vertices), 3), dtype=np.float32)
for i, v in enumerate(mesh.vertices):
    vertex_normals[i] = v.normal

print("[Blender Extract] Calculated vertex normals")

# Calculate face normals
face_normals = np.zeros((len(faces), 3), dtype=np.float32)
for i, p in enumerate(mesh.polygons):
    face_normals[i] = p.normal

print("[Blender Extract] Calculated face normals")

# Extract UV coordinates if available
uv_coords = None
uv_faces = None
if mesh.uv_layers.active:
    uv_layer = mesh.uv_layers.active.data
    # UV coordinates are stored per loop (face corner), not per vertex
    # We need to map them to face corners
    uv_coords_list = []
    uv_faces_list = []

    for i, poly in enumerate(mesh.polygons):
        face_uvs = []
        for loop_idx in poly.loop_indices:
            uv = uv_layer[loop_idx].uv
            uv_coords_list.append([uv[0], uv[1]])
            face_uvs.append(len(uv_coords_list) - 1)
        uv_faces_list.append(face_uvs)

    uv_coords = np.array(uv_coords_list, dtype=np.float32)
    uv_faces = np.array(uv_faces_list, dtype=np.int32)
    print(f"[Blender Extract] Extracted UV coordinates: {len(uv_coords)} UVs for {len(uv_faces)} faces")
else:
    print("[Blender Extract] No UV layer found")

# Extract material/texture info if available
material_name = None
texture_path = None
texture_data_base64 = ""
texture_format = ""
texture_width = 0
texture_height = 0

def extract_texture_from_image(image, max_size=2048):
    """Extract texture data from Blender image as base64 encoded PNG string.
    Uses pure Python PNG encoding without PIL dependency."""

    try:
        # Get image dimensions
        width, height = image.size
        channels = image.channels

        print(f"[Blender Extract] Extracting texture: {width}x{height}, {channels} channels")

        # Get pixel data - Blender stores as flat RGBA float array
        pixels = np.array(image.pixels[:])

        # Reshape to (height, width, channels)
        pixels = pixels.reshape((height, width, channels))

        # Convert from float [0,1] to uint8 [0,255]
        pixels = (np.clip(pixels, 0, 1) * 255).astype(np.uint8)

        # Flip vertically (Blender stores bottom-to-top)
        pixels = np.flipud(pixels)

        # Resize if too large (simple nearest-neighbor downsampling)
        if width > max_size or height > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            print(f"[Blender Extract] Resizing texture from {width}x{height} to {new_width}x{new_height}")

            # Simple nearest-neighbor resize using numpy
            row_indices = (np.arange(new_height) * height / new_height).astype(int)
            col_indices = (np.arange(new_width) * width / new_width).astype(int)
            pixels = pixels[row_indices][:, col_indices]
            width, height = new_width, new_height

        # Encode as PNG without PIL - use pure Python PNG encoder
        png_data = encode_png(pixels, width, height, channels)

        # Encode to base64
        encoded = base64.b64encode(png_data).decode('utf-8')

        print(f"[Blender Extract] Texture encoded: {len(encoded) / 1024:.1f} KB base64")
        return encoded, 'PNG', width, height

    except Exception as e:
        print(f"[Blender Extract] Error extracting texture: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0, 0


def encode_png(pixels, width, height, channels):
    """
    Encode numpy pixel array to PNG format without PIL.
    Uses minimal PNG implementation with zlib compression.
    """
    import zlib

    def png_chunk(chunk_type, data):
        """Create a PNG chunk with CRC."""
        chunk_len = struct.pack('>I', len(data))
        chunk_data = chunk_type + data
        checksum = struct.pack('>I', zlib.crc32(chunk_data) & 0xffffffff)
        return chunk_len + chunk_data + checksum

    # PNG signature
    signature = b'\x89PNG\r\n\x1a\n'

    # IHDR chunk (image header)
    if channels == 4:
        color_type = 6  # RGBA
    elif channels == 3:
        color_type = 2  # RGB
    else:
        # Convert to RGB
        if channels == 1:
            pixels = np.repeat(pixels, 3, axis=2)
            channels = 3
            color_type = 2
        else:
            raise ValueError(f"Unsupported channel count: {channels}")

    bit_depth = 8
    compression = 0
    filter_method = 0
    interlace = 0

    ihdr_data = struct.pack('>IIBBBBB', width, height, bit_depth, color_type,
                            compression, filter_method, interlace)
    ihdr_chunk = png_chunk(b'IHDR', ihdr_data)

    # IDAT chunk (compressed image data)
    # Prepare raw data with filter bytes
    raw_data = b''
    for row in pixels:
        # Filter type 0 (None) for each row
        raw_data += b'\x00'
        raw_data += row.tobytes()

    # Compress with zlib
    compressed = zlib.compress(raw_data, level=6)
    idat_chunk = png_chunk(b'IDAT', compressed)

    # IEND chunk (end of image)
    iend_chunk = png_chunk(b'IEND', b'')

    # Combine all chunks
    png_data = signature + ihdr_chunk + idat_chunk + iend_chunk

    return png_data

if mesh_obj.material_slots:
    mat = mesh_obj.material_slots[0].material
    if mat:
        material_name = mat.name
        # Try to find base color texture
        if mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    texture_path = node.image.filepath
                    print(f"[Blender Extract] Found texture node: {node.name}")
                    print(f"[Blender Extract] Texture path: {texture_path}")

                    # Extract actual image data
                    tex_base64, tex_fmt, tex_w, tex_h = extract_texture_from_image(node.image)
                    if tex_base64:
                        texture_data_base64 = tex_base64
                        texture_format = tex_fmt
                        texture_width = tex_w
                        texture_height = tex_h
                        print(f"[Blender Extract] Texture extracted successfully: {tex_w}x{tex_h} {tex_fmt}")
                    break
        print(f"[Blender Extract] Material: {material_name}")

# Save as NPZ (raw_data format expected by UniRig)
# For skeleton extraction, skeleton fields are set to None
os.makedirs(os.path.dirname(output_npz), exist_ok=True)

np.savez_compressed(
    output_npz,
    vertices=vertices.astype(np.float32),
    vertex_normals=vertex_normals.astype(np.float32),
    faces=faces.astype(np.int32),
    face_normals=face_normals.astype(np.float32),
    uv_coords=uv_coords if uv_coords is not None else np.array([], dtype=np.float32),
    uv_faces=uv_faces if uv_faces is not None else np.array([], dtype=np.int32),
    material_name=material_name if material_name else "",
    texture_path=texture_path if texture_path else "",
    texture_data_base64=texture_data_base64,
    texture_format=texture_format,
    texture_width=texture_width,
    texture_height=texture_height,
    joints=None,
    skin=None,
    parents=None,
    names=None,
    matrix_local=None,
)

print(f"[Blender Extract] Saved to: {output_npz}")
if texture_data_base64:
    print(f"[Blender Extract] Texture data included: {texture_width}x{texture_height} {texture_format}")
else:
    print(f"[Blender Extract] No texture data extracted")
print("[Blender Extract] Done!")
