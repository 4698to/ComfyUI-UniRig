"""
Installation script for ComfyUI-UniRig.

CUDA dependencies (torch-scatter, torch-cluster, spconv) are now handled by
comfyui-envmanager. Run `comfy-env install` in this directory to install them.

This script now only handles Blender installation for backwards compatibility.
"""

import sys
import os

# Ensure installer package is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def main():
    """Main installation routine - runs comfy-env install."""
    import subprocess

    print("=" * 60, flush=True)
    print("[UniRig Install] ComfyUI-UniRig Installation", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    # First, ensure comfyui-envmanager is installed
    try:
        import comfyui_envmanager
    except ImportError:
        print("Installing comfyui-envmanager...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "comfyui-envmanager"],
            check=False
        )
        print(flush=True)

    print("Installing CUDA dependencies via comfyui-envmanager...", flush=True)
    print(flush=True)

    # Run comfy-env install
    result = subprocess.run(
        [sys.executable, "-m", "comfyui_envmanager.cli", "install"],
        cwd=_SCRIPT_DIR
    )

    print(flush=True)
    if result.returncode == 0:
        print("[UniRig Install] CUDA dependencies installed successfully!")
    else:
        print("[UniRig Install] CUDA installation failed. Try running manually:")
        print("  comfy-env install")

    print()
    print("To install Blender (for FBX export):")
    print("  python blender_install.py")
    print()
    print("=" * 60)

    return result.returncode


# Legacy exports for backwards compatibility with nodes/base.py
def find_blender_executable(blender_dir):
    """Find the blender executable in the extracted directory."""
    from installer.blender import find_blender_executable as _find_blender_executable
    return _find_blender_executable(blender_dir)


def install_blender(target_dir=None):
    """Install Blender for mesh preprocessing."""
    from installer.blender import install_blender as _install_blender
    return _install_blender(target_dir)


def get_platform_info():
    """Detect current platform and architecture."""
    from installer.utils import get_platform_info as _get_platform_info
    info = _get_platform_info()
    return info["platform"], info["arch"]


if __name__ == "__main__":
    sys.exit(main())
