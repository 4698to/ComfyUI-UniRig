"""
ComfyUI-UniRig Prestartup Script

Handles setup tasks before node loading:
- Copy FBX viewer files from comfy-3d-viewers package
- Copy asset files to input/3d/
- Copy animation templates to input/animation_templates/
- Copy animation characters (e.g., official Mixamo rig) to input/animation_characters/
- Create necessary directories
"""

import os
import shutil
import json
from pathlib import Path

# Get paths relative to this script
SCRIPT_DIR = Path(__file__).parent.absolute()
COMFYUI_DIR = SCRIPT_DIR.parent.parent  # custom_nodes/ComfyUI-UniRig -> custom_nodes -> ComfyUI

# Source directories
ASSETS_DIR = SCRIPT_DIR / "assets"
WORKFLOWS_DIR = SCRIPT_DIR / "workflows"

# Target directories (using relative paths from ComfyUI root)
INPUT_3D_DIR = COMFYUI_DIR / "input" / "3d"
INPUT_ANIMATION_TEMPLATES_DIR = COMFYUI_DIR / "input" / "animation_templates"
INPUT_ANIMATION_CHARACTERS_DIR = COMFYUI_DIR / "input" / "animation_characters"
USER_WORKFLOWS_DIR = COMFYUI_DIR / "user" / "default" / "workflows"

# Web directory for viewer files
WEB_DIR = SCRIPT_DIR / "web"
THREE_DIR = WEB_DIR / "three"


def copy_fbx_viewer():
    """Copy FBX viewer files from comfy-3d-viewers package."""
    try:
        from comfy_3d_viewers import get_fbx_html_path, get_fbx_bundle_path, get_fbx_node_widget_path

        # Ensure directories exist
        WEB_DIR.mkdir(parents=True, exist_ok=True)
        THREE_DIR.mkdir(parents=True, exist_ok=True)
        JS_DIR = WEB_DIR / "js"
        JS_DIR.mkdir(parents=True, exist_ok=True)

        # Copy viewer_fbx.html to web/
        src_html = get_fbx_html_path()
        dst_html = WEB_DIR / "viewer_fbx.html"
        if os.path.exists(src_html):
            # Always copy if source is newer or destination doesn't exist
            if not dst_html.exists() or os.path.getmtime(src_html) > os.path.getmtime(dst_html):
                shutil.copy2(src_html, dst_html)
                print(f"[UniRig] Copied viewer_fbx.html from comfy-3d-viewers")
            else:
                print(f"[UniRig] viewer_fbx.html is up to date")
        else:
            print(f"[UniRig] Warning: viewer_fbx.html not found in comfy-3d-viewers package")

        # Copy viewer-bundle.js to web/three/
        src_bundle = get_fbx_bundle_path()
        dst_bundle = THREE_DIR / "viewer-bundle.js"
        if os.path.exists(src_bundle):
            if not dst_bundle.exists() or os.path.getmtime(src_bundle) > os.path.getmtime(dst_bundle):
                shutil.copy2(src_bundle, dst_bundle)
                print(f"[UniRig] Copied viewer-bundle.js from comfy-3d-viewers")
            else:
                print(f"[UniRig] viewer-bundle.js is up to date")
        else:
            print(f"[UniRig] Warning: viewer-bundle.js not found in comfy-3d-viewers package")

        # Copy mesh_preview_fbx.js to web/js/
        src_widget = get_fbx_node_widget_path()
        dst_widget = JS_DIR / "mesh_preview_fbx.js"
        if os.path.exists(src_widget):
            if not dst_widget.exists() or os.path.getmtime(src_widget) > os.path.getmtime(dst_widget):
                shutil.copy2(src_widget, dst_widget)
                print(f"[UniRig] Copied mesh_preview_fbx.js from comfy-3d-viewers")
            else:
                print(f"[UniRig] mesh_preview_fbx.js is up to date")
        else:
            print(f"[UniRig] Warning: mesh_preview_fbx.js not found in comfy-3d-viewers package")

    except ImportError:
        print("[UniRig] Warning: comfy-3d-viewers package not installed, FBX viewer may not work")
        print("[UniRig] Install with: pip install comfy-3d-viewers")
    except Exception as e:
        print(f"[UniRig] Error copying FBX viewer files: {e}")
        import traceback
        traceback.print_exc()


def copy_asset_files():
    """Copy all asset files to input/3d/ directory"""
    try:
        # Create target directory
        INPUT_3D_DIR.mkdir(parents=True, exist_ok=True)

        if not ASSETS_DIR.exists():
            print(f"[UniRig] Warning: Assets directory not found at {ASSETS_DIR}")
            return

        # Copy all files from assets directory
        for asset_file in ASSETS_DIR.iterdir():
            if asset_file.is_file():
                target_file = INPUT_3D_DIR / asset_file.name
                
                if not target_file.exists():
                    shutil.copy2(str(asset_file), str(target_file))
                    print(f"[UniRig] Copied {asset_file.name} to {target_file}")
                else:
                    print(f"[UniRig] {asset_file.name} already exists at {target_file}")

    except Exception as e:
        print(f"[UniRig] Error copying asset files: {e}")
        import traceback
        traceback.print_exc()


def copy_animation_templates():
    """Copy animation templates to input/animation_templates/ directory"""
    try:
        source_dir = ASSETS_DIR / "animation_templates"

        if not source_dir.exists():
            print(f"[UniRig] Warning: Animation templates directory not found at {source_dir}")
            return

        # Copy each format subdirectory (mixamo, smpl)
        for format_dir in source_dir.iterdir():
            if format_dir.is_dir() and not format_dir.name.startswith('.'):
                target_dir = INPUT_ANIMATION_TEMPLATES_DIR / format_dir.name
                target_dir.mkdir(parents=True, exist_ok=True)

                # Copy all animation files
                for anim_file in format_dir.iterdir():
                    if anim_file.is_file() and not anim_file.name.startswith('.'):
                        target_file = target_dir / anim_file.name

                        if not target_file.exists():
                            shutil.copy2(str(anim_file), str(target_file))
                            print(f"[UniRig] Copied {format_dir.name}/{anim_file.name} to {target_file}")
                        else:
                            print(f"[UniRig] {format_dir.name}/{anim_file.name} already exists")

        print(f"[UniRig] Animation templates ready at {INPUT_ANIMATION_TEMPLATES_DIR}")

    except Exception as e:
        print(f"[UniRig] Error copying animation templates: {e}")
        import traceback
        traceback.print_exc()


def copy_animation_characters():
    """Copy animation character references (e.g., official Mixamo rig) to input/animation_characters/"""
    try:
        source_dir = ASSETS_DIR / "animation_characters"

        if not source_dir.exists():
            print(f"[UniRig] Warning: Animation characters directory not found at {source_dir}")
            return

        # Create target directory
        INPUT_ANIMATION_CHARACTERS_DIR.mkdir(parents=True, exist_ok=True)

        # Copy all character files (FBX, etc.)
        for char_file in source_dir.iterdir():
            if char_file.is_file() and not char_file.name.startswith('.'):
                target_file = INPUT_ANIMATION_CHARACTERS_DIR / char_file.name

                if not target_file.exists():
                    shutil.copy2(str(char_file), str(target_file))
                    print(f"[UniRig] Copied animation character: {char_file.name}")
                else:
                    print(f"[UniRig] Animation character {char_file.name} already exists")

        print(f"[UniRig] Animation characters ready at {INPUT_ANIMATION_CHARACTERS_DIR}")

    except Exception as e:
        print(f"[UniRig] Error copying animation characters: {e}")
        import traceback
        traceback.print_exc()


# Execute setup tasks
try:
    print("[UniRig] Running prestartup script...")
    copy_fbx_viewer()
    copy_asset_files()
    copy_animation_templates()
    copy_animation_characters()
    print("[UniRig] Prestartup script completed.")
except Exception as e:
    print(f"[UniRig] Error during prestartup: {e}")
    import traceback
    traceback.print_exc()
