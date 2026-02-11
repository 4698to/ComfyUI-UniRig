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
            [sys.executable, "-m", "pip", "install", "comfyui-envmanager>=0.0.8"],
            check=False
        )
        print(flush=True)

    # Check if torch is available (required for wheel resolution)
    try:
        import torch
        print(f"Found PyTorch {torch.__version__}", flush=True)
    except ImportError:
        print("[UniRig Install] WARNING: PyTorch not found. CUDA wheels require PyTorch.", flush=True)
        print("[UniRig Install] Install PyTorch first, then run: comfy-env install", flush=True)
        return 1

    print("Installing CUDA dependencies via comfyui-envmanager...", flush=True)
    print(flush=True)

    # Run comfy-env install directly (not subprocess, to share torch import)
    try:
        # Change to script directory for config discovery
        original_cwd = os.getcwd()
        os.chdir(_SCRIPT_DIR)

        from comfyui_envmanager.cli import main as comfy_env_main
        result = comfy_env_main(["install"])

        os.chdir(original_cwd)
        returncode = result if isinstance(result, int) else 0
    except Exception as e:
        print(f"[UniRig Install] Error: {e}")
        returncode = 1

    print(flush=True)
    if returncode == 0:
        print("[UniRig Install] CUDA dependencies installed successfully!")
    else:
        print("[UniRig Install] CUDA installation failed. Try running manually:")
        print("  comfy-env install")

    print()
    print("To install Blender (for FBX export):")
    print("  python blender_install.py")
    print()
    print("=" * 60)

    return returncode


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
