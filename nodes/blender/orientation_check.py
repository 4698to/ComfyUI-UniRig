"""
UniRig Orientation Check Node - Visual verification of mesh orientation for MIA pipeline
"""

import os
# Set osmesa for headless rendering BEFORE any OpenGL imports
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import numpy as np
import trimesh
import torch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# ComfyUI folder paths
try:
    import folder_paths
except:
    folder_paths = None

# Path to bundled reference mesh (now in nodes/blender/, so go up two levels to custom_node root)
NODE_DIR = Path(__file__).parent.parent.parent
REFERENCE_MESH_PATH = NODE_DIR / "assets" / "FinalBaseMesh.obj"


def render_mesh_front_view(mesh: trimesh.Trimesh, width: int, height: int) -> np.ndarray:
    """
    Render mesh from front view (looking down -Z axis) with colors.

    For Y-up orientation (MIA expected):
    - Camera looks down -Z axis
    - Y is up in the image
    - Character should appear standing upright, facing viewer

    Returns:
        RGB image as numpy array (H, W, 3)
    """
    # Get mesh bounds for camera positioning
    bounds = mesh.bounds
    center = mesh.centroid
    size = bounds[1] - bounds[0]
    max_dim = max(size)
    distance = max_dim * 2.5

    try:
        import pyrender

        # Create a colored mesh - use vertex colors if available, otherwise default gray
        if mesh.visual.kind == 'vertex' and hasattr(mesh.visual, 'vertex_colors'):
            mesh_pr = pyrender.Mesh.from_trimesh(mesh)
        else:
            # Create a mesh with a nice default material
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.7, 0.7, 0.8, 1.0],
                metallicFactor=0.2,
                roughnessFactor=0.8
            )
            mesh_pr = pyrender.Mesh.from_trimesh(mesh, material=material)

        # Create pyrender scene with ambient light
        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[0.15, 0.15, 0.18, 1.0])
        scene.add(mesh_pr)

        # Add camera - positioned along +Z looking at -Z (front view)
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=width/height)

        # Camera pose: looking down -Z axis, Y up
        cam_pose = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, center[1]],
            [0, 0, 1, center[2] + distance],
            [0, 0, 0, 1]
        ])
        scene.add(camera, pose=cam_pose)

        # Add key light (front-right, above)
        key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
        key_pose = np.array([
            [0.866, 0, 0.5, 0],
            [0.25, 0.866, -0.433, center[1] + max_dim],
            [-0.433, 0.5, 0.75, center[2] + distance],
            [0, 0, 0, 1]
        ])
        scene.add(key_light, pose=key_pose)

        # Add fill light (front-left)
        fill_light = pyrender.DirectionalLight(color=[0.8, 0.8, 1.0], intensity=2.0)
        fill_pose = np.array([
            [0.866, 0, -0.5, 0],
            [-0.25, 0.866, -0.433, center[1]],
            [0.433, 0.5, 0.75, center[2] + distance],
            [0, 0, 0, 1]
        ])
        scene.add(fill_light, pose=fill_pose)

        # Render
        renderer = pyrender.OffscreenRenderer(width, height)
        color, _ = renderer.render(scene)
        renderer.delete()

        return color

    except Exception as e:
        print(f"[OrientationCheck] Pyrender failed: {e}, using wireframe fallback")
        return create_wireframe_visualization(mesh, width, height)


def create_wireframe_visualization(mesh: trimesh.Trimesh, width: int, height: int) -> np.ndarray:
    """
    Create a simple 2D projection visualization when 3D rendering isn't available.
    Projects mesh vertices onto XY plane (front view for Y-up).
    """
    # Project vertices to 2D (front view: X horizontal, Y vertical)
    vertices = mesh.vertices

    # Get bounds
    x_min, y_min = vertices[:, 0].min(), vertices[:, 1].min()
    x_max, y_max = vertices[:, 0].max(), vertices[:, 1].max()

    # Add padding
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * padding
    x_max += x_range * padding
    y_min -= y_range * padding
    y_max += y_range * padding

    # Create image
    img = Image.new('RGB', (width, height), (40, 40, 46))
    draw = ImageDraw.Draw(img)

    # Scale vertices to image coordinates
    def to_img_coords(x, y):
        ix = int((x - x_min) / (x_max - x_min) * (width - 1))
        iy = int((1 - (y - y_min) / (y_max - y_min)) * (height - 1))  # Flip Y
        return ix, iy

    # Draw edges (sample to avoid too many)
    edges = mesh.edges_unique
    max_edges = 5000
    if len(edges) > max_edges:
        indices = np.random.choice(len(edges), max_edges, replace=False)
        edges = edges[indices]

    for edge in edges:
        v1, v2 = vertices[edge[0]], vertices[edge[1]]
        p1 = to_img_coords(v1[0], v1[1])
        p2 = to_img_coords(v2[0], v2[1])
        draw.line([p1, p2], fill=(100, 150, 200), width=1)

    return np.array(img)


def load_reference_mesh():
    """Load the bundled reference mesh for comparison."""
    if not REFERENCE_MESH_PATH.exists():
        print(f"[OrientationCheck] Reference mesh not found: {REFERENCE_MESH_PATH}")
        return None

    try:
        ref_mesh = trimesh.load(str(REFERENCE_MESH_PATH), force='mesh')
        if isinstance(ref_mesh, trimesh.Scene):
            ref_mesh = ref_mesh.dump(concatenate=True)
        return ref_mesh
    except Exception as e:
        print(f"[OrientationCheck] Failed to load reference mesh: {e}")
        return None


def create_comparison_image(user_mesh: trimesh.Trimesh, ref_mesh: trimesh.Trimesh, max_height: int = 512) -> np.ndarray:
    """
    Create side-by-side comparison of user mesh and reference mesh.
    """
    # Calculate dimensions for user mesh
    user_dims = user_mesh.bounds[1] - user_mesh.bounds[0]
    user_aspect = user_dims[0] / user_dims[1] if user_dims[1] > 0 else 1.0
    user_width = max(100, int(max_height * user_aspect))

    # Calculate dimensions for reference mesh
    ref_dims = ref_mesh.bounds[1] - ref_mesh.bounds[0]
    ref_aspect = ref_dims[0] / ref_dims[1] if ref_dims[1] > 0 else 1.0
    ref_width = max(100, int(max_height * ref_aspect))

    # Render both meshes
    print(f"[OrientationCheck] Rendering user mesh at {user_width}x{max_height}")
    user_img = render_mesh_front_view(user_mesh, user_width, max_height)

    print(f"[OrientationCheck] Rendering reference at {ref_width}x{max_height}")
    ref_img = render_mesh_front_view(ref_mesh, ref_width, max_height)

    # Create combined image
    gap = 25
    header_height = 35
    footer_height = 45
    total_width = user_width + gap + ref_width
    total_height = max_height + header_height + footer_height

    combined = Image.new('RGB', (total_width, total_height), (30, 30, 35))
    draw = ImageDraw.Draw(combined)

    # Load fonts
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font = ImageFont.load_default()
        font_small = font

    # Determine orientation status
    axis_names = ['X', 'Y', 'Z']
    tallest_idx = np.argmax(user_dims)
    tallest_axis = axis_names[tallest_idx]
    is_correct = (tallest_axis == 'Y')

    # Draw header with labels
    # User mesh label (orange/yellow)
    user_label = "YOUR MESH"
    draw.text((user_width // 2 - 45, 10), user_label, fill=(255, 180, 50), font=font)

    # Reference label (green)
    ref_x = user_width + gap
    draw.text((ref_x + ref_width // 2 - 45, 10), "REFERENCE", fill=(100, 255, 100), font=font)

    # Paste rendered images
    combined.paste(Image.fromarray(user_img), (0, header_height))
    combined.paste(Image.fromarray(ref_img), (ref_x, header_height))

    # Draw separator line
    sep_x = user_width + gap // 2
    draw.line([(sep_x, header_height), (sep_x, header_height + max_height)], fill=(60, 60, 65), width=2)

    # Draw footer
    footer_y = header_height + max_height
    draw.rectangle([0, footer_y, total_width, total_height], fill=(25, 25, 30))

    if is_correct:
        status_text = "[OK] Orientation looks correct (Y-up, character should be facing you)"
        status_color = (100, 255, 100)
    else:
        status_text = f"[WARNING] Check orientation: {tallest_axis} is tallest (expected Y)"
        status_color = (255, 180, 50)

    draw.text((10, footer_y + 8), status_text, fill=status_color, font=font_small)
    draw.text((10, footer_y + 25), "If sideways: rotate 90 degrees around Y axis in Blender before export",
              fill=(140, 140, 140), font=font_small)

    return np.array(combined)


class UniRigOrientationCheck:
    """
    Visual orientation check for mesh before MIA rigging.

    Renders your mesh side-by-side with a reference mesh to verify orientation.
    For MIA pipeline, character should be Y-up and facing the camera.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH", {
                    "tooltip": "Input mesh to check orientation"
                }),
            },
            "optional": {
                "max_height": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "tooltip": "Maximum height of each mesh render"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("comparison",)
    FUNCTION = "check_orientation"
    CATEGORY = "UniRig/Utils"

    def check_orientation(self, mesh, max_height=512):
        """
        Render mesh alongside reference for orientation verification.

        Args:
            mesh: trimesh.Trimesh object
            max_height: Maximum height of rendered images

        Returns:
            tuple: (ComfyUI IMAGE tensor,)
        """
        print(f"[OrientationCheck] Checking mesh orientation...")
        print(f"[OrientationCheck] Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Get mesh bounds info
        bounds = mesh.bounds
        dims = bounds[1] - bounds[0]
        print(f"[OrientationCheck] Dimensions: X={dims[0]:.3f}, Y={dims[1]:.3f}, Z={dims[2]:.3f}")

        # Determine orientation
        axis_names = ['X', 'Y', 'Z']
        tallest_idx = np.argmax(dims)
        tallest_axis = axis_names[tallest_idx]
        print(f"[OrientationCheck] Tallest axis: {tallest_axis}")

        if tallest_axis == 'Y':
            print(f"[OrientationCheck] [OK] Orientation appears correct (Y-up)")
        else:
            print(f"[OrientationCheck] [WARNING] {tallest_axis} is tallest, expected Y for Y-up")

        # Load reference mesh
        ref_mesh = load_reference_mesh()

        if ref_mesh is not None:
            # Create side-by-side comparison
            image = create_comparison_image(mesh, ref_mesh, max_height)
        else:
            # Fallback: just render user mesh with overlay
            print(f"[OrientationCheck] Reference not available, rendering single view")
            user_dims = mesh.bounds[1] - mesh.bounds[0]
            aspect = user_dims[0] / user_dims[1] if user_dims[1] > 0 else 1.0
            width = max(128, int(max_height * aspect))
            image = render_mesh_front_view(mesh, width, max_height)

        # Convert to ComfyUI IMAGE format (B, H, W, C) with values 0-1
        image_tensor = torch.from_numpy(image).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        print(f"[OrientationCheck] Done. Output shape: {image_tensor.shape}")

        return (image_tensor,)
