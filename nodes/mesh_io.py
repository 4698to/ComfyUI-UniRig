"""
UniRig Mesh I/O Nodes - Load and save mesh files
"""

import os
import subprocess
import tempfile
import numpy as np
import trimesh
from pathlib import Path
from typing import Tuple, Optional
import logging

log = logging.getLogger("unirig")

try:
    import igl
except ImportError:
    log.warning("libigl not found, mesh IO will not be available")
    igl = None

# ComfyUI folder paths
try:
    import folder_paths
    COMFYUI_INPUT_FOLDER = folder_paths.get_input_directory()
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except Exception as e:
    # Fallback if folder_paths not available (e.g., during testing)
    COMFYUI_INPUT_FOLDER = None
    COMFYUI_OUTPUT_FOLDER = None

# Import LIB_DIR from base module
try:
    from .base import LIB_DIR
except ImportError:
    from base import LIB_DIR

def _emit_visible_log(message: str, *args) -> None:
    """
    Emit log messages both through logger and stdout so worker console always shows them.
    """
    log.info(message, *args)
    try:
        text = message % args if args else message
    except Exception:
        text = f"{message} {args}"
    print(f"[MESH_IO] {text}")

def _save_uv_ppm_debug_varying(mesh: trimesh.Trimesh, out_path: str, size: int = 1024) -> None:
    """Save UV wireframe with support for face-varying UVs."""
    try:
        if not hasattr(mesh, "visual") or mesh.visual is None:
            return
        
        faces = np.asarray(mesh.faces)
        uv = np.asarray(mesh.visual.uv, dtype=np.float64)
        
        n_faces = len(faces)
        n_verts = len(mesh.vertices)
        n_uvs = len(uv)
        
        # 判断 UV 类型
        is_face_varying = (n_uvs == n_faces * 3)
        is_per_vertex = (n_uvs == n_verts)
        
        if not is_face_varying and not is_per_vertex:
            # 尝试 reshape，可能是展平的 face-varying
            if n_uvs % 3 == 0 and n_uvs // 3 == n_faces:
                uv = uv.reshape(-1, 3, 2)
                is_face_varying = True
            else:
                _emit_visible_log("UV/Vertex count mismatch: UV=%d, Vert=%d, Face=%d",n_uvs, n_verts, n_faces)
                return
        
        img = np.full((size, size, 3), 255, dtype=np.uint8)
        
        def draw_line(x0, y0, x1, y1):
            # ... 保持原有的 Bresenham 实现 ...
            dx = abs(x1 - x0)
            sx = 1 if x0 < x1 else -1
            dy = -abs(y1 - y0)
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            while True:
                if 0 <= x0 < size and 0 <= y0 < size:
                    img[y0, x0] = (0, 0, 0)
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x0 += sx
                if e2 <= dx:
                    err += dx
                    y0 += sy
        
        # 处理 Face-Varying UV (每个三角形的三个角独立UV)
        if is_face_varying:
            _emit_visible_log("Face-Varying UV processing...")
            uv_reshaped = uv.reshape(-1, 3, 2) if uv.ndim == 2 else uv
            for face_idx, (face, face_uv) in enumerate(zip(faces, uv_reshaped)):
                # face: [v0, v1, v2], face_uv: [[u0,v0], [u1,v1], [u2,v2]]
                for i in range(3):
                    j = (i + 1) % 3
                    u0, v0 = face_uv[i]
                    u1, v1 = face_uv[j]
                    
                    x0 = int(max(0, min(size - 1, round(u0 * (size - 1)))))
                    y0 = int(max(0, min(size - 1, round((1.0 - v0) * (size - 1)))))
                    x1 = int(max(0, min(size - 1, round(u1 * (size - 1)))))
                    y1 = int(max(0, min(size - 1, round((1.0 - v1) * (size - 1)))))
                    draw_line(x0, y0, x1, y1)
        else:
            # 原有的 Per-Vertex UV 处理逻辑
            _emit_visible_log("Per-Vertex UV processing...")
            for face in faces:
                n = len(face)
                for i in range(n):
                    a = int(face[i])
                    b = int(face[(i + 1) % n])
                    if a < 0 or b < 0 or a >= len(uv) or b >= len(uv):
                        continue
                    u0, v0 = float(uv[a, 0]), float(uv[a, 1])
                    u1, v1 = float(uv[b, 0]), float(uv[b, 1])
                    x0 = int(max(0, min(size - 1, round(u0 * (size - 1)))))
                    y0 = int(max(0, min(size - 1, round((1.0 - v0) * (size - 1)))))
                    x1 = int(max(0, min(size - 1, round(u1 * (size - 1)))))
                    y1 = int(max(0, min(size - 1, round((1.0 - v1) * (size - 1)))))
                    draw_line(x0, y0, x1, y1)

        with open(out_path, "wb") as f:
            f.write(f"P6\n{size} {size}\n255\n".encode("ascii"))
            f.write(img.tobytes())
        _emit_visible_log("Saved UV PPM debug (fixed): %s", out_path)
    except Exception as e:
        _emit_visible_log("Failed to save UV PPM debug: %s", e)

def _save_uv_ppm_debug(mesh: trimesh.Trimesh, out_path: str, size: int = 1024) -> None:
    """
    Save UV wireframe debug image as PPM.
    """
    try:
        if (
            not hasattr(mesh, "visual")
            or mesh.visual is None
            or not hasattr(mesh.visual, "uv")
            or mesh.visual.uv is None
            or len(mesh.visual.uv) == 0
            or not hasattr(mesh, "faces")
            or mesh.faces is None
            or len(mesh.faces) == 0
        ):
            _emit_visible_log("Skip UV PPM debug (missing uv/faces): %s", out_path)
            return

        uv = np.asarray(mesh.visual.uv, dtype=np.float64)
        faces = np.asarray(mesh.faces)
        img = np.full((size, size, 3), 255, dtype=np.uint8)

        def draw_line(x0, y0, x1, y1):
            dx = abs(x1 - x0)
            sx = 1 if x0 < x1 else -1
            dy = -abs(y1 - y0)
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            while True:
                if 0 <= x0 < size and 0 <= y0 < size:
                    img[y0, x0] = (0, 0, 0)
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x0 += sx
                if e2 <= dx:
                    err += dx
                    y0 += sy

        for face in faces:
            n = len(face)
            for i in range(n):
                a = int(face[i])
                b = int(face[(i + 1) % n])
                if a < 0 or b < 0 or a >= len(uv) or b >= len(uv):
                    continue
                u0, v0 = float(uv[a, 0]), float(uv[a, 1])
                u1, v1 = float(uv[b, 0]), float(uv[b, 1])
                x0 = int(max(0, min(size - 1, round(u0 * (size - 1)))))
                y0 = int(max(0, min(size - 1, round((1.0 - v0) * (size - 1)))))
                x1 = int(max(0, min(size - 1, round(u1 * (size - 1)))))
                y1 = int(max(0, min(size - 1, round((1.0 - v1) * (size - 1)))))
                draw_line(x0, y0, x1, y1)

        with open(out_path, "wb") as f:
            f.write(f"P6\n{size} {size}\n255\n".encode("ascii"))
            f.write(img.tobytes())
        _emit_visible_log("Saved UV PPM debug: %s", out_path)
    except Exception as e:
        _emit_visible_log("Failed to save UV PPM debug: %s", e)


def _save_obj_uv_ppm_debug(obj_path: str, out_path: str, size: int = 1024) -> None:
    """
    Save OBJ UV wireframe using native vt/f indices (face-corner UV mapping).
    This is more accurate than vertex-index-based UV debug for seam-heavy meshes.
    """
    try:
        vt_list = []
        face_vt_indices = []
        with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("vt "):
                    parts = line.split()
                    if len(parts) >= 3:
                        vt_list.append((float(parts[1]), float(parts[2])))
                elif line.startswith("f "):
                    tokens = line.split()[1:]
                    poly_vt = []
                    for t in tokens:
                        # OBJ face token supports: v, v/vt, v//vn, v/vt/vn
                        subs = t.split("/")
                        if len(subs) >= 2 and subs[1] != "":
                            vt_idx = int(subs[1])
                            # OBJ indices are 1-based; negative means relative to tail
                            if vt_idx < 0:
                                vt_idx = len(vt_list) + vt_idx + 1
                            poly_vt.append(vt_idx - 1)
                    if len(poly_vt) >= 2:
                        face_vt_indices.append(poly_vt)

        if not vt_list or not face_vt_indices:
            _emit_visible_log("Skip OBJ UV PPM debug (missing vt/f-vt): %s", out_path)
            return

        uv = np.asarray(vt_list, dtype=np.float64)
        img = np.full((size, size, 3), 255, dtype=np.uint8)

        def draw_line(x0, y0, x1, y1):
            dx = abs(x1 - x0)
            sx = 1 if x0 < x1 else -1
            dy = -abs(y1 - y0)
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            while True:
                if 0 <= x0 < size and 0 <= y0 < size:
                    img[y0, x0] = (0, 0, 0)
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x0 += sx
                if e2 <= dx:
                    err += dx
                    y0 += sy

        for poly in face_vt_indices:
            n = len(poly)
            for i in range(n):
                a = poly[i]
                b = poly[(i + 1) % n]
                if a < 0 or b < 0 or a >= len(uv) or b >= len(uv):
                    continue
                u0, v0 = float(uv[a, 0]), float(uv[a, 1])
                u1, v1 = float(uv[b, 0]), float(uv[b, 1])
                x0 = int(max(0, min(size - 1, round(u0 * (size - 1)))))
                y0 = int(max(0, min(size - 1, round((1.0 - v0) * (size - 1)))))
                x1 = int(max(0, min(size - 1, round(u1 * (size - 1)))))
                y1 = int(max(0, min(size - 1, round((1.0 - v1) * (size - 1)))))
                draw_line(x0, y0, x1, y1)

        with open(out_path, "wb") as f:
            f.write(f"P6\n{size} {size}\n255\n".encode("ascii"))
            f.write(img.tobytes())
        _emit_visible_log("Saved OBJ-native UV PPM debug: %s", out_path)
    except Exception as e:
        _emit_visible_log("Failed to save OBJ-native UV PPM debug: %s", e)

def load_fbx_with_blender(file_path: str) -> Tuple[Optional[trimesh.Trimesh], str]:
    """
    FBX loading is no longer supported via Blender.

    Args:
        file_path: Path to FBX file

    Returns:
        Tuple of (None, error_message)
    """
    return None, (
        "FBX file format is not directly supported.\n\n"
        "Please convert your FBX to GLB/OBJ format using Blender or other software,\n"
        "then load the converted file."
    )


def load_mesh_file(file_path: str) -> Tuple[Optional[trimesh.Trimesh], str]:
    """
    Load a mesh from file.

    Ensures the returned mesh has only triangular faces and is properly processed.

    Args:
        file_path: Path to mesh file (OBJ, PLY, STL, OFF, FBX, etc.)

    Returns:
        Tuple of (mesh, error_message)
    """
    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"

    # Check file extension - FBX requires Blender (use os.path for Windows compatibility)
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    log.info("File extension detected: '%s'", ext)

    if ext == '.fbx':
        log.info("Detected FBX file, using Blender loader")
        return load_fbx_with_blender(file_path)

    try:
        log.info("Loading: %s", file_path)

        # Try to load with trimesh first (supports many formats)
        # Do NOT use force='mesh' as it can lose visual/texture data during Scene-to-mesh conversion
        # Use process=False and maintain_order=True to preserve mesh.visual (textures/materials)
        loaded = trimesh.load(file_path, process=False, maintain_order=True)

        # log.info(f"Loaded type: {type(loaded).__name__}")
        # _emit_visible_log("Loaded file: %s", file_path)
        # #if ext == ".obj":
        # obj_uv_ppm_path = os.path.splitext(file_path)[0] + "_uv_loaded_obj_native.ppm"
        # _save_obj_uv_ppm_debug(file_path, obj_uv_ppm_path)
        # #if isinstance(loaded, trimesh.Trimesh):
        # uv_ppm_path = os.path.splitext(file_path)[0] + "_uv_loaded_trimesh.ppm"
        # _save_uv_ppm_debug_varying(loaded, uv_ppm_path)

        # Debug: Check visual data immediately after load
        if isinstance(loaded, trimesh.Scene):
            log.info(f"Scene has {len(loaded.geometry)} geometries")
            for name, geom in loaded.geometry.items():
                if hasattr(geom, 'visual'):
                    log.info(f"Geometry '{name}': visual type = {type(geom.visual).__name__}")
                    if hasattr(geom.visual, 'material'):
                        mat = geom.visual.material
                        log.info(f"Material: {type(mat).__name__}")
                        if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                            log.info(f"Has baseColorTexture: {mat.baseColorTexture.shape if hasattr(mat.baseColorTexture, 'shape') else 'yes'}")
                        if hasattr(mat, 'image') and mat.image is not None:
                            log.info(f"Has image: {mat.image.size if hasattr(mat.image, 'size') else 'yes'}")
        else:
            if hasattr(loaded, 'visual'):
                log.info(f"Mesh visual type: {type(loaded.visual).__name__}")
                if hasattr(loaded.visual, 'material'):
                    mat = loaded.visual.material
                    log.info(f"Material: {type(mat).__name__}")

        # Handle case where trimesh.load returns a Scene instead of a mesh
        if isinstance(loaded, trimesh.Scene):
            log.info(f"Converting Scene to single mesh (scene has {len(loaded.geometry)} geometries)")
            # Use dump with concatenate=True to merge geometries while preserving visual data
            mesh = loaded.dump(concatenate=True)
            log.info(f"After dump(): visual type = {type(mesh.visual).__name__ if hasattr(mesh, 'visual') else 'None'}")
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
                log.info(f"After dump(): material = {type(mesh.visual.material).__name__}")
        else:
            mesh = loaded

        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return None, f"Failed to read mesh or mesh is empty: {file_path}"

        log.info(f"Initial mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Debug: Check visual after initial processing
        if hasattr(mesh, 'visual'):
            log.info(f"Visual type: {type(mesh.visual).__name__}")
            if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                log.info("Has UV coords: %s", mesh.visual.uv.shape)
            if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
                mat = mesh.visual.material
                log.info(f"Material type: {type(mat).__name__}")
                if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                    log.info("Has baseColorTexture!")
                if hasattr(mat, 'image') and mat.image is not None:
                    log.info("Has image texture!")
        else:
            log.warning("WARNING: No visual attribute on mesh!")

        # Ensure mesh is properly triangulated
        if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
            # Check if faces are triangular
            if mesh.faces.shape[1] != 3:
                log.warning("Warning: Mesh has non-triangular faces, triangulating...")
                # Use process=False to preserve mesh.visual (textures/materials)
                mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False, maintain_order=True)
                log.info(f"After triangulation: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Count before cleanup
        verts_before = len(mesh.vertices)
        faces_before = len(mesh.faces)

        # NOTE: We do NOT call mesh.merge_vertices() here as it destroys mesh.visual (textures/materials)

        # Remove duplicate and degenerate faces (trimesh 4.x compatible)
        # unique_faces() returns boolean mask of non-duplicate faces
        # nondegenerate_faces() returns boolean mask of non-degenerate faces
        unique_mask = mesh.unique_faces()
        nondegenerate_mask = mesh.nondegenerate_faces()
        valid_faces_mask = unique_mask & nondegenerate_mask
        if not valid_faces_mask.all():
            mesh.update_faces(valid_faces_mask)

        verts_after = len(mesh.vertices)
        faces_after = len(mesh.faces)

        if verts_before != verts_after or faces_before != faces_after:
            log.info("Cleanup: %s->%s vertices, %s->%s faces", verts_before, verts_after, faces_before, faces_after)
            log.info("Removed: %s duplicate vertices, %s bad faces", verts_before - verts_after, faces_before - faces_after)

        # Store file metadata
        mesh.metadata['file_path'] = file_path
        mesh.metadata['file_name'] = os.path.basename(file_path)
        mesh.metadata['file_format'] = os.path.splitext(file_path)[1].lower()
        #uv_ppm_path = os.path.splitext(file_path)[0] + "_uv_loaded_1.ppm"
        #_save_uv_ppm_debug(mesh, uv_ppm_path)

        log.info(f"Successfully loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return mesh, ""

    except Exception as e:
        log.info(f"Trimesh failed: {str(e)}, trying libigl fallback...")
        if igl is None:
            return None, (
                f"Error loading mesh: {str(e)}; "
                "Fallback unavailable: libigl is not installed or failed to import."
            )
        # Fallback to libigl
        try:
            v, f = igl.read_triangle_mesh(file_path)
            if v is None or f is None or len(v) == 0 or len(f) == 0:
                return None, f"Failed to read mesh: {file_path}"

            log.info(f"libigl loaded: {len(v)} vertices, {len(f)} faces")

            # Use process=False to preserve mesh.visual (textures/materials)
            mesh = trimesh.Trimesh(vertices=v, faces=f, process=False, maintain_order=True)

            # Count before cleanup
            verts_before = len(mesh.vertices)
            faces_before = len(mesh.faces)

            # NOTE: We do NOT call mesh.merge_vertices() here as it destroys mesh.visual (textures/materials)

            # Remove duplicate and degenerate faces (trimesh 4.x compatible)
            unique_mask = mesh.unique_faces()
            nondegenerate_mask = mesh.nondegenerate_faces()
            valid_faces_mask = unique_mask & nondegenerate_mask
            if not valid_faces_mask.all():
                mesh.update_faces(valid_faces_mask)

            verts_after = len(mesh.vertices)
            faces_after = len(mesh.faces)

            if verts_before != verts_after or faces_before != faces_after:
                log.info("Cleanup: %s->%s vertices, %s->%s faces", verts_before, verts_after, faces_before, faces_after)

            # Store metadata
            mesh.metadata['file_path'] = file_path
            mesh.metadata['file_name'] = os.path.basename(file_path)
            mesh.metadata['file_format'] = os.path.splitext(file_path)[1].lower()

            log.info(f"Successfully loaded via libigl: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            return mesh, ""
        except Exception as e2:
            log.info("Both loaders failed!")
            return None, f"Error loading mesh: {str(e)}; Fallback error: {str(e2)}"


def save_mesh_file(mesh: trimesh.Trimesh, file_path: str) -> Tuple[bool, str]:
    """
    Save a mesh to file.

    Args:
        mesh: Trimesh object
        file_path: Output file path

    Returns:
        Tuple of (success, error_message)
    """
    if not isinstance(mesh, trimesh.Trimesh):
        return False, "Input must be a trimesh.Trimesh object"

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return False, "Mesh is empty"

    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Export the mesh
        mesh.export(file_path)

        return True, ""

    except Exception as e:
        return False, f"Error saving mesh: {str(e)}"


class UniRigLoadMesh:
    """
    Load a mesh from ComfyUI input or output folder (OBJ, PLY, STL, OFF, etc.)
    Returns trimesh.Trimesh objects for mesh handling.
    """

    # Supported mesh extensions for file browser
    #SUPPORTED_EXTENSIONS = ['.obj', '.ply', '.stl', '.off', '.gltf', '.glb', '.fbx', '.dae', '.3ds', '.vtp']
    SUPPORTED_EXTENSIONS = ['.obj']


    @classmethod
    def INPUT_TYPES(cls):
        # Get list of available mesh files from input folder (default)
        mesh_files = cls.get_mesh_files_from_input()

        # If no files found, provide a default message
        if not mesh_files:
            mesh_files = ["No mesh files found"]

        return {
            "required": {
                "source_folder": (["input", "output"], {
                    "default": "input",
                    "tooltip": "Source folder to load mesh from (ComfyUI input or output directory)"
                }),
                "file_path": (mesh_files, {
                    "tooltip": "Mesh file to load. Refresh the node after changing source_folder."
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "load_mesh"
    CATEGORY = "UniRig/IO"

    @classmethod
    def get_mesh_files_from_input(cls):
        """Get list of available mesh files in input/3d and input folders."""
        mesh_files = []

        if COMFYUI_INPUT_FOLDER is not None:
            # Scan input/3d first
            input_3d = os.path.join(COMFYUI_INPUT_FOLDER, "3d")
            if os.path.exists(input_3d):
                for file in os.listdir(input_3d):
                    if any(file.lower().endswith(ext) for ext in cls.SUPPORTED_EXTENSIONS):
                        mesh_files.append(f"3d/{file}")

            # Then scan input root
            for file in os.listdir(COMFYUI_INPUT_FOLDER):
                file_path = os.path.join(COMFYUI_INPUT_FOLDER, file)
                if os.path.isfile(file_path):
                    if any(file.lower().endswith(ext) for ext in cls.SUPPORTED_EXTENSIONS):
                        mesh_files.append(file)

        return sorted(mesh_files)

    @classmethod
    def get_mesh_files_from_output(cls):
        """Get list of available mesh files in output folder."""
        mesh_files = []

        if COMFYUI_OUTPUT_FOLDER is not None and os.path.exists(COMFYUI_OUTPUT_FOLDER):
            # Scan output folder recursively
            for root, dirs, files in os.walk(COMFYUI_OUTPUT_FOLDER):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in cls.SUPPORTED_EXTENSIONS):
                        # Get relative path from output folder
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, COMFYUI_OUTPUT_FOLDER)
                        mesh_files.append(rel_path)

        return sorted(mesh_files)

    @classmethod
    def IS_CHANGED(cls, source_folder, file_path):
        """Force re-execution when file changes."""
        base_folder = COMFYUI_INPUT_FOLDER if source_folder == "input" else COMFYUI_OUTPUT_FOLDER

        if base_folder is not None:
            if source_folder == "input":
                # Check in input/3d first, then input root
                input_3d_path = os.path.join(base_folder, "3d", file_path)
                input_path = os.path.join(base_folder, file_path)

                if os.path.exists(input_3d_path):
                    return os.path.getmtime(input_3d_path)
                elif os.path.exists(input_path):
                    return os.path.getmtime(input_path)
            else:
                # Check in output folder
                full_path = os.path.join(base_folder, file_path)
                if os.path.exists(full_path):
                    return os.path.getmtime(full_path)

        return f"{source_folder}:{file_path}"

    def load_mesh(self, source_folder, file_path):
        """
        Load mesh from file.

        Looks for files in the specified source folder (input or output).

        Args:
            source_folder: "input" or "output"
            file_path: Path to mesh file (relative to source folder or absolute)

        Returns:
            tuple: (trimesh.Trimesh,)
        """
        if not file_path or file_path.strip() == "":
            raise ValueError("File path cannot be empty")

        # Try to find the file
        full_path = None
        searched_paths = []

        if source_folder == "input" and COMFYUI_INPUT_FOLDER is not None:
            # First, try in ComfyUI input/3d folder
            input_3d_path = os.path.join(COMFYUI_INPUT_FOLDER, "3d", file_path)
            searched_paths.append(input_3d_path)
            if os.path.exists(input_3d_path):
                full_path = input_3d_path
                log.info("Found mesh in input/3d folder: %s", file_path)

            # Second, try in ComfyUI input folder
            if full_path is None:
                input_path = os.path.join(COMFYUI_INPUT_FOLDER, file_path)
                searched_paths.append(input_path)
                if os.path.exists(input_path):
                    full_path = input_path
                    log.info("Found mesh in input folder: %s", file_path)

        elif source_folder == "output" and COMFYUI_OUTPUT_FOLDER is not None:
            output_path = os.path.join(COMFYUI_OUTPUT_FOLDER, file_path)
            searched_paths.append(output_path)
            if os.path.exists(output_path):
                full_path = output_path
                log.info("Found mesh in output folder: %s", file_path)

        # If not found in source folder, try as absolute path
        if full_path is None:
            searched_paths.append(file_path)
            if os.path.exists(file_path):
                full_path = file_path
                log.info("Loading from absolute path: %s", file_path)
            else:
                # Generate error message with all searched paths
                error_msg = f"File not found: '{file_path}'\nSearched in:"
                for path in searched_paths:
                    error_msg += f"\n  - {path}"
                raise ValueError(error_msg)

        # Load the mesh
        loaded_mesh, error = load_mesh_file(full_path)

        if loaded_mesh is None:
            raise ValueError(f"Failed to load mesh: {error}")

        log.info(f"Loaded: {len(loaded_mesh.vertices)} vertices, {len(loaded_mesh.faces)} faces")

        return (loaded_mesh,)


class UniRigSaveMesh:
    """
    Save a mesh to file (OBJ, PLY, STL, OFF, etc.)
    Supports all formats provided by trimesh.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "file_path": ("STRING", {
                    "default": "output.obj",
                    "multiline": False
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "save_mesh"
    CATEGORY = "UniRig/IO"
    OUTPUT_NODE = True

    def save_mesh(self, trimesh, file_path):
        """
        Save mesh to file.

        Saves to ComfyUI's output folder if path is relative, otherwise uses absolute path.

        Args:
            trimesh: trimesh.Trimesh object
            file_path: Output file path (relative to output folder or absolute)

        Returns:
            tuple: (status_message,)
        """
        if not file_path or file_path.strip() == "":
            raise ValueError("File path cannot be empty")

        # Debug: Check what we received
        log.info(f"Received mesh type: {type(trimesh)}")
        if trimesh is None:
            raise ValueError("Cannot save mesh: received None instead of a mesh object. Check that the previous node is outputting a mesh.")

        # Check if mesh has data
        try:
            vertex_count = len(trimesh.vertices) if hasattr(trimesh, 'vertices') else 0
            face_count = len(trimesh.faces) if hasattr(trimesh, 'faces') else 0
            log.info("Mesh has %s vertices, %s faces", vertex_count, face_count)

            if vertex_count == 0 or face_count == 0:
                raise ValueError(
                    f"Cannot save empty mesh (vertices: {vertex_count}, faces: {face_count}). "
                    "Check that the previous node is producing valid geometry."
                )
        except Exception as e:
            raise ValueError(f"Error checking mesh properties: {e}. Received object may not be a valid mesh.")

        # Determine full output path
        full_path = file_path

        # If path is relative and we have output folder, use it
        if not os.path.isabs(file_path) and COMFYUI_OUTPUT_FOLDER is not None:
            full_path = os.path.join(COMFYUI_OUTPUT_FOLDER, file_path)
            log.info("Saving to output folder: %s", file_path)
        else:
            log.info("Saving to: %s", file_path)

        # Save the mesh
        success, error = save_mesh_file(trimesh, full_path)

        if not success:
            raise ValueError(f"Failed to save trimesh: {error}")

        status = f"Successfully saved mesh to: {full_path}\n"
        status += f"  Vertices: {len(trimesh.vertices)}\n"
        status += f"  Faces: {len(trimesh.faces)}"

        log.info("%s", status)

        return (status,)
