"""
Skeleton I/O nodes for UniRig - Save, Load, and Preview operations.
"""

import os
import numpy as np
import time
import shutil
import pickle
import json
import folder_paths
import logging

log = logging.getLogger("unirig")

try:
    from .base import NODE_DIR
except ImportError:
    from base import NODE_DIR

# Direct bone debug extraction module (bpy as Python module)
try:
    from .unirig import direct_extract_bone_debug as _direct_bone_debug_module
except ImportError as e:
    log.debug("Direct bone debug not available: %s", e)
    _direct_bone_debug_module = None


def _get_direct_bone_debug():
    """Get the direct bone debug extraction module for in-process extraction using bpy."""
    return _direct_bone_debug_module


class UniRigLoadRiggedMesh:
    """
    Load a rigged FBX file from disk.

    Loads existing FBX files with rigging/skeleton data, allowing you to
    preview and work with pre-rigged models.
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_files = cls.get_fbx_files_from_input()
        output_files = cls.get_fbx_files_from_output()
        all_files = sorted(set(input_files + output_files)) or [""]

        return {
            "required": {
                "fbx_file": (all_files, {"file_upload": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("fbx_output_path", "info")
    FUNCTION = "load"
    CATEGORY = "unirig"

    @classmethod
    def get_fbx_files_from_input(cls):
        """Get list of available FBX files in input folder."""
        fbx_files = []
        input_dir = folder_paths.get_input_directory()

        if input_dir is not None and os.path.exists(input_dir):
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.lower().endswith('.fbx'):
                        rel_path = os.path.relpath(os.path.join(root, file), input_dir)
                        rel_path = rel_path.replace(os.sep, '/')  # Normalize to forward slashes for cross-platform
                        fbx_files.append(rel_path)

        return sorted(fbx_files)

    @classmethod
    def get_fbx_files_from_output(cls):
        """Get list of available FBX files in output folder."""
        fbx_files = []
        output_dir = folder_paths.get_output_directory()

        if output_dir is not None and os.path.exists(output_dir):
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.lower().endswith('.fbx'):
                        rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                        rel_path = rel_path.replace(os.sep, '/')  # Normalize to forward slashes for cross-platform
                        fbx_files.append(rel_path)

        return sorted(fbx_files)

    def load(self, fbx_file):
        """Load an FBX file and return its filename in output directory."""
        log.info("Loading FBX file: %s", fbx_file)

        if not fbx_file:
            raise RuntimeError("No FBX file specified. Please select or upload an FBX file.")

        # Search in both input and output directories
        input_dir = folder_paths.get_input_directory()
        output_dir = folder_paths.get_output_directory()

        input_path = os.path.join(input_dir, fbx_file)
        output_path = os.path.join(output_dir, fbx_file)

        # Find the file
        if os.path.exists(output_path):
            fbx_path = output_path
            source = "output"
        elif os.path.exists(input_path):
            fbx_path = input_path
            source = "input"
        else:
            raise RuntimeError(f"FBX file not found: {fbx_file}")

        # If loading from input, copy to output directory
        if source == "input":
            output_filename = f"loaded_{int(time.time())}_{os.path.basename(fbx_file)}"
            final_output_path = os.path.join(output_dir, output_filename)
            shutil.copy(fbx_path, final_output_path)
            log.info("Copied from input to output: %s", output_filename)
        else:
            output_filename = fbx_file
            final_output_path = output_path
            log.info("Using existing file from output: %s", output_filename)

        # Extract mesh info using bpy
        mesh_info = {}
        try:
            import bpy
            bpy.ops.wm.read_factory_settings(use_empty=True)
            bpy.ops.import_scene.fbx(filepath=fbx_path)

            mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
            total_vertices = sum(len(obj.data.vertices) for obj in mesh_objects)
            total_faces = sum(len(obj.data.polygons) for obj in mesh_objects)

            if mesh_objects:
                import mathutils
                all_verts = []
                for obj in mesh_objects:
                    for v in obj.data.vertices:
                        all_verts.append(obj.matrix_world @ v.co)
                all_verts = np.array(all_verts)
                bbox_min = all_verts.min(axis=0).tolist()
                bbox_max = all_verts.max(axis=0).tolist()
                extents = (all_verts.max(axis=0) - all_verts.min(axis=0)).tolist()
            else:
                bbox_min = bbox_max = extents = [0, 0, 0]

            mesh_info = {
                "type": "Scene" if len(mesh_objects) > 1 else "Mesh",
                "mesh_count": len(mesh_objects),
                "total_vertices": total_vertices,
                "total_faces": total_faces,
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "extents": extents,
            }
            log.info("Mesh: %s objects, %s verts, %s faces", len(mesh_objects), total_vertices, total_faces)
        except Exception as e:
            log.info("Could not parse mesh geometry: %s", e)
            mesh_info = {"type": "Unknown", "error": str(e)}

        # Extract skeleton info using bpy
        skeleton_info = {}
        try:
            armatures = [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']
            if armatures:
                arm = armatures[0]
                bone_names = [b.name for b in arm.data.bones]
                skeleton_info = {
                    "num_bones": len(bone_names),
                    "bone_names": bone_names[:10],
                    "has_skeleton": True,
                }
                log.info("Found %s bones", len(bone_names))
            else:
                skeleton_info = {"has_skeleton": False, "note": "No armature found"}
        except Exception as e:
            log.info("Could not parse skeleton: %s", e)
            skeleton_info = {"has_skeleton": "unknown", "error": str(e)}

        # Create info string
        file_size = os.path.getsize(final_output_path)
        info_lines = [
            f"File: {os.path.basename(fbx_file)}",
            f"Size: {file_size / 1024:.1f} KB",
            "",
            "Mesh Info:",
            f"  Type: {mesh_info.get('type', 'Unknown')}",
            f"  Meshes: {mesh_info.get('mesh_count', 'Unknown')}",
            f"  Vertices: {mesh_info.get('total_vertices', 'Unknown'):,}" if isinstance(mesh_info.get('total_vertices'), int) else f"  Vertices: Unknown",
            f"  Faces: {mesh_info.get('total_faces', 'Unknown'):,}" if isinstance(mesh_info.get('total_faces'), int) else f"  Faces: Unknown",
        ]

        if 'extents' in mesh_info and mesh_info['extents']:
            extents = mesh_info['extents']
            info_lines.append(f"  Mesh Size: [{extents[0]:.3f}, {extents[1]:.3f}, {extents[2]:.3f}]")

        if 'bbox_min' in mesh_info and 'bbox_max' in mesh_info:
            bbox_min = mesh_info['bbox_min']
            bbox_max = mesh_info['bbox_max']
            info_lines.append(f"  Bounding Box:")
            info_lines.append(f"    Min: [{bbox_min[0]:.3f}, {bbox_min[1]:.3f}, {bbox_min[2]:.3f}]")
            info_lines.append(f"    Max: [{bbox_max[0]:.3f}, {bbox_max[1]:.3f}, {bbox_max[2]:.3f}]")

        info_lines.append("")
        info_lines.append("Skeleton Info:")

        if skeleton_info.get("has_skeleton"):
            info_lines.append(f"  Bones: {skeleton_info.get('num_bones', 0)}")
            if skeleton_info.get("skeleton_extents"):
                extents = skeleton_info['skeleton_extents']
                info_lines.append(f"  Skeleton Size: [{extents[0]:.3f}, {extents[1]:.3f}, {extents[2]:.3f}]")
            if skeleton_info.get("bone_names"):
                sample_bones = skeleton_info['bone_names'][:5]
                info_lines.append(f"  Sample bones: {', '.join(sample_bones)}")
        else:
            info_lines.append(f"  Status: {skeleton_info.get('note', 'No skeleton detected')}")

        info_string = "\n".join(info_lines)

        log.info("Loaded successfully")
        log.info("%s", info_string)

        return (final_output_path, info_string)


class UniRigPreviewRiggedMesh:
    """
    Preview rigged mesh with interactive FBX viewer.

    Displays the rigged FBX in a Three.js viewer with skeleton visualization
    and interactive bone manipulation controls.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fbx_output_path": ("STRING", {
                    "tooltip": "FBX filename from output directory (from UniRigApplySkinningMLNew or UniRigLoadRiggedMesh)"
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "unirig"

    def preview(self, fbx_output_path):
        """Preview the rigged mesh in an interactive FBX viewer."""
        log.info("Preparing preview...")

        # FBX should already be in output directory
        output_dir = folder_paths.get_output_directory()
        fbx_path = os.path.join(output_dir, fbx_output_path)

        if not os.path.exists(fbx_path):
            raise RuntimeError(f"FBX file not found in output directory: {fbx_output_path}")

        log.info("FBX path: %s", fbx_path)

        # FBX is already in output, so viewer can access it directly
        # Assume all FBX files have skinning and skeleton
        has_skinning = True
        has_skeleton = True

        log.info("Has skinning: %s", has_skinning)
        log.info("Has skeleton: %s", has_skeleton)

        return {
            "ui": {
                "fbx_file": [fbx_output_path],
                "has_skinning": [bool(has_skinning)],
                "has_skeleton": [bool(has_skeleton)],
            }
        }


class UniRigExportPosedFBX:
    """
    Export rigged mesh with custom bone pose to FBX.

    Takes a rigged mesh and bone transform data, applies the pose,
    and exports the result as FBX using Blender.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rigged_mesh": ("RIGGED_MESH",),
                "output_filename": ("STRING", {
                    "default": "posed_export.fbx",
                    "tooltip": "Output filename for the posed FBX"
                }),
            },
            "optional": {
                "bone_transforms_json": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                    "tooltip": "JSON string containing bone transforms (name -> {position, quaternion, scale})"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "export_posed_fbx"
    CATEGORY = "unirig"
    OUTPUT_NODE = True

    def export_posed_fbx(self, rigged_mesh, output_filename, bone_transforms_json="{}"):
        """Export rigged mesh with custom pose to FBX using bpy directly."""
        log.info("Exporting posed FBX...")

        # Get original FBX path
        fbx_path = rigged_mesh.get("fbx_path")
        if not fbx_path or not os.path.exists(fbx_path):
            raise RuntimeError(f"Rigged mesh FBX not found: {fbx_path}")

        log.info("Source FBX: %s", fbx_path)

        # Parse bone transforms
        try:
            bone_transforms = json.loads(bone_transforms_json)
            log.info(f"Loaded transforms for {len(bone_transforms)} bones")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in bone_transforms_json: {e}")

        # Prepare output path
        output_dir = folder_paths.get_output_directory()
        if not output_filename.endswith('.fbx'):
            output_filename = output_filename + '.fbx'
        output_fbx_path = os.path.join(output_dir, output_filename)

        # Use direct bpy export
        try:
            from .lib.direct_export_posed_fbx import export_posed_fbx as direct_export
            direct_export(fbx_path, output_fbx_path, bone_transforms)
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import direct_export_posed_fbx: {e}\n"
                "Make sure bpy is available in your environment (unirig isolated environment)."
            )

        if not os.path.exists(output_fbx_path):
            raise RuntimeError(f"Export completed but output file not found: {output_fbx_path}")

        log.info("[OK] Successfully exported to: %s", output_fbx_path)

        return (output_fbx_path,)


class UniRigViewRigging:
    """
    View rigging debug information.

    Displays skeleton bones with names, roll/rotation values, and other
    debugging information in an interactive 3D viewer.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fbx_output_path": ("STRING", {
                    "tooltip": "FBX filename from output directory"
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "view_rigging"
    CATEGORY = "unirig"

    def view_rigging(self, fbx_output_path):
        """View rigging debug information for the FBX file."""
        log.debug("Preparing debug view...")

        # FBX should already be in output directory
        output_dir = folder_paths.get_output_directory()

        # Handle both relative paths and absolute paths
        if os.path.isabs(fbx_output_path):
            fbx_path = fbx_output_path
        else:
            fbx_path = os.path.join(output_dir, fbx_output_path)

        if not os.path.exists(fbx_path):
            raise RuntimeError(f"FBX file not found: {fbx_output_path}")

        log.info("FBX path: %s", fbx_path)

        # Extract bone debug data using direct bpy module
        bone_debug_data = None
        bone_debug_module = _get_direct_bone_debug()

        if bone_debug_module:
            try:
                bone_debug_data = bone_debug_module.extract_bone_debug(fbx_path)
                log.debug(f"Extracted debug data for {bone_debug_data.get('bone_count', 0)} bones")
            except Exception as e:
                log.warning("Warning: Could not extract bone debug data: %s", e)
                bone_debug_data = {'error': str(e), 'bones': [], 'bone_count': 0}
        else:
            log.warning("Warning: bpy module not available, bone debug data will be limited")
            bone_debug_data = {'error': 'bpy module not available', 'bones': [], 'bone_count': 0}

        # Return data for the viewer widget
        # For relative path, just use the filename for the viewer
        if os.path.isabs(fbx_output_path):
            viewer_filename = os.path.basename(fbx_output_path)
        else:
            viewer_filename = fbx_output_path

        return {
            "ui": {
                "fbx_file": [viewer_filename],
                "bone_debug_data": [json.dumps(bone_debug_data)],
            }
        }


class UniRigDebugSkeleton:
    """
    Debug skeleton visualization with bone roll/orientation analysis.

    Opens the FBX in an enhanced debug viewer with:
    - RGB axes showing local bone coordinate systems (X=red, Y=green, Z=blue/roll)
    - Bone name labels
    - Detailed bone information panel with roll angles
    - Animation playback controls
    - Bone filtering and size controls
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fbx_path": ("STRING", {
                    "tooltip": "Path to FBX file (from output directory or absolute path)"
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "debug_skeleton"
    CATEGORY = "unirig"

    def debug_skeleton(self, fbx_path):
        """Open the FBX in the debug skeleton viewer."""
        log.debug("Preparing debug skeleton view...")

        # Handle both relative paths and absolute paths
        output_dir = folder_paths.get_output_directory()

        if os.path.isabs(fbx_path):
            full_path = fbx_path
        else:
            full_path = os.path.join(output_dir, fbx_path)

        if not os.path.exists(full_path):
            raise RuntimeError(f"FBX file not found: {fbx_path}")

        log.debug("FBX path: %s", full_path)

        # For the viewer, use relative path if in output, otherwise basename
        if os.path.isabs(fbx_path):
            viewer_filename = os.path.basename(fbx_path)
        else:
            viewer_filename = fbx_path

        return {
            "ui": {
                "fbx_file": [viewer_filename],
            }
        }


class UniRigCompareSkeletons:
    """
    Compare two skeletons side-by-side with synced rotation.

    Opens two FBX files in a split-view debug viewer where:
    - Both skeletons are displayed side-by-side
    - Camera rotation and zoom are synced between views
    - Clicking a bone in one view highlights the matching bone (by name) in the other
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fbx_path_left": ("STRING", {
                    "tooltip": "Path to left skeleton FBX file (from output directory or absolute path)"
                }),
                "fbx_path_right": ("STRING", {
                    "tooltip": "Path to right skeleton FBX file (from output directory or absolute path)"
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "compare_skeletons"
    CATEGORY = "unirig"

    def compare_skeletons(self, fbx_path_left, fbx_path_right):
        """Open both FBX files in the comparison skeleton viewer."""
        log.info("Preparing skeleton comparison view...")

        output_dir = folder_paths.get_output_directory()

        # Validate left FBX path
        if os.path.isabs(fbx_path_left):
            full_path_left = fbx_path_left
        else:
            full_path_left = os.path.join(output_dir, fbx_path_left)

        if not os.path.exists(full_path_left):
            raise RuntimeError(f"Left FBX file not found: {fbx_path_left}")

        # Validate right FBX path
        if os.path.isabs(fbx_path_right):
            full_path_right = fbx_path_right
        else:
            full_path_right = os.path.join(output_dir, fbx_path_right)

        if not os.path.exists(full_path_right):
            raise RuntimeError(f"Right FBX file not found: {fbx_path_right}")

        log.info("Left FBX: %s", full_path_left)
        log.info("Right FBX: %s", full_path_right)

        # For the viewer, use relative path if in output, otherwise basename
        if os.path.isabs(fbx_path_left):
            viewer_filename_left = os.path.basename(fbx_path_left)
        else:
            viewer_filename_left = fbx_path_left

        if os.path.isabs(fbx_path_right):
            viewer_filename_right = os.path.basename(fbx_path_right)
        else:
            viewer_filename_right = fbx_path_right

        return {
            "ui": {
                "fbx_file_left": [viewer_filename_left],
                "fbx_file_right": [viewer_filename_right],
            }
        }
