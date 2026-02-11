#!/usr/bin/env python3
"""
Installation script for ComfyUI-UniRig with isolated environment.

This script sets up an isolated Python virtual environment with all dependencies
required for UniRig. The environment is completely isolated from
ComfyUI's main environment, preventing any dependency conflicts.

Uses comfy-env package for environment management.
"""

import sys
import os
import platform
import subprocess
from pathlib import Path


# =============================================================================
# VC++ Redistributable Check (Windows only)
# =============================================================================

VCREDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"


def check_vcredist_installed():
    """Check if VC++ Redistributable DLLs are actually present on the system."""
    if platform.system() != "Windows":
        return True  # Not needed on non-Windows

    required_dlls = ['vcruntime140.dll', 'msvcp140.dll']

    # Search locations in order of preference
    search_paths = []

    # 1. System directory (most reliable)
    system_root = os.environ.get('SystemRoot', r'C:\Windows')
    search_paths.append(os.path.join(system_root, 'System32'))

    # 2. Python environment directories
    if hasattr(sys, 'base_prefix'):
        search_paths.append(os.path.join(sys.base_prefix, 'Library', 'bin'))
        search_paths.append(os.path.join(sys.base_prefix, 'DLLs'))
    if hasattr(sys, 'prefix') and sys.prefix != getattr(sys, 'base_prefix', sys.prefix):
        search_paths.append(os.path.join(sys.prefix, 'Library', 'bin'))
        search_paths.append(os.path.join(sys.prefix, 'DLLs'))

    # Check each required DLL
    for dll in required_dlls:
        found = False
        for search_path in search_paths:
            dll_path = os.path.join(search_path, dll)
            if os.path.exists(dll_path):
                found = True
                break
        if not found:
            return False

    return True


def install_vcredist():
    """Download and install VC++ Redistributable with UAC elevation."""
    import urllib.request
    import tempfile

    print("[UniRig] Downloading VC++ Redistributable...")

    # Download to temp file
    temp_path = os.path.join(tempfile.gettempdir(), "vc_redist.x64.exe")
    try:
        urllib.request.urlretrieve(VCREDIST_URL, temp_path)
    except Exception as e:
        print(f"[UniRig] Failed to download VC++ Redistributable: {e}")
        print(f"[UniRig] Please download manually from: {VCREDIST_URL}")
        return False

    print("[UniRig] Installing VC++ Redistributable (UAC prompt may appear)...")

    # Run with elevation - /passive shows progress, /quiet is fully silent
    try:
        result = subprocess.run(
            [temp_path, '/install', '/passive', '/norestart'],
            capture_output=True
        )
    except Exception as e:
        print(f"[UniRig] Failed to run installer: {e}")
        print(f"[UniRig] Please run manually: {temp_path}")
        return False

    # Clean up
    try:
        os.remove(temp_path)
    except:
        pass

    if result.returncode == 0:
        print("[UniRig] VC++ Redistributable installer completed.")
    elif result.returncode == 1638:
        # 1638 = newer version already installed
        print("[UniRig] VC++ Redistributable already installed (newer version)")
    else:
        print(f"[UniRig] Installation returned code {result.returncode}")
        print(f"[UniRig] Please install manually from: {VCREDIST_URL}")
        return False

    # Verify DLLs are actually present after installation
    if check_vcredist_installed():
        print("[UniRig] VC++ Redistributable DLLs verified!")
        return True
    else:
        print("[UniRig] Installation completed but DLLs not found in expected locations.")
        print("[UniRig] You may need to restart your system or terminal.")
        return False


def ensure_vcredist():
    """Check and install VC++ Redistributable if needed (Windows only)."""
    if platform.system() != "Windows":
        return True

    if check_vcredist_installed():
        print("[UniRig] VC++ Redistributable: OK (DLLs found)")
        return True

    print("[UniRig] VC++ Redistributable DLLs not found - attempting automatic install...")

    if install_vcredist():
        return True

    # Fallback: provide clear manual instructions
    print("")
    print("=" * 70)
    print("[UniRig] MANUAL INSTALLATION REQUIRED")
    print("=" * 70)
    print("")
    print("  The automatic installation of VC++ Redistributable failed.")
    print("  This is required for PyTorch CUDA and other native extensions.")
    print("")
    print("  Please download and install manually:")
    print(f"    {VCREDIST_URL}")
    print("")
    print("  After installation, restart your terminal and try again.")
    print("=" * 70)
    print("")
    return False


# =============================================================================
# Blender Detection and Installation
# =============================================================================

def detect_blender():
    """Detect existing Blender installation.

    Returns path to Blender executable or None if not found.
    Detection priority:
    1. BLENDER_PATH env var (explicit override)
    2. System PATH ('blender' command)
    3. Platform-specific common locations
    4. Bundled lib/blender/ directory
    """
    import shutil

    # 1. Check BLENDER_PATH env var (explicit override)
    blender_path = os.environ.get('BLENDER_PATH')
    if blender_path and os.path.isfile(blender_path):
        return blender_path

    # 2. Check system PATH
    blender_in_path = shutil.which('blender')
    if blender_in_path:
        return blender_in_path

    # 3. Check platform-specific common locations
    system = platform.system()
    common_locations = []

    if system == "Windows":
        # Windows: Check Program Files
        program_files = os.environ.get('ProgramFiles', r'C:\Program Files')
        blender_foundation = Path(program_files) / "Blender Foundation"
        if blender_foundation.exists():
            # Find latest version (e.g., Blender 4.2, Blender 4.1, etc.)
            for blender_dir in sorted(blender_foundation.glob("Blender*"), reverse=True):
                exe = blender_dir / "blender.exe"
                if exe.exists():
                    common_locations.append(exe)

    elif system == "Darwin":  # macOS
        common_locations = [
            Path("/Applications/Blender.app/Contents/MacOS/Blender"),
        ]

    else:  # Linux
        common_locations = [
            Path("/usr/bin/blender"),
            Path("/usr/local/bin/blender"),
            Path("/snap/bin/blender"),
        ]

    for loc in common_locations:
        if isinstance(loc, Path) and loc.exists():
            return str(loc)

    # 4. Check bundled lib/blender/
    node_root = Path(__file__).parent.absolute()
    blender_dir = node_root / "lib" / "blender"
    if blender_dir.exists():
        from installer.blender import find_blender_executable
        blender_exe = find_blender_executable(str(blender_dir))
        if blender_exe:
            return str(blender_exe)

    return None


def ensure_blender():
    """Ensure Blender is available, installing if needed.

    Returns path to Blender executable or None if installation failed.
    Sets BLENDER_PATH env var so other nodes can detect it.
    """
    existing = detect_blender()
    if existing:
        print(f"[UniRig] Found existing Blender: {existing}")
        os.environ['BLENDER_PATH'] = existing
        return existing

    print("[UniRig] Blender not found - installing...")
    from installer.blender import install_blender
    node_root = Path(__file__).parent.absolute()
    target_dir = node_root / "lib" / "blender"
    blender_exe = install_blender(target_dir=str(target_dir))

    if blender_exe:
        # Export path so other nodes can detect it
        os.environ['BLENDER_PATH'] = blender_exe
        print(f"[UniRig] Exported BLENDER_PATH={blender_exe}")

    return blender_exe


# =============================================================================
# Main Installation
# =============================================================================

def main():
    """Main installation function."""
    print("\n" + "=" * 60)
    print("ComfyUI-UniRig Installation (Isolated Environment)")
    print("=" * 60)

    # Check VC++ Redistributable first (required for PyTorch CUDA and native extensions)
    if not ensure_vcredist():
        print("[UniRig] WARNING: VC++ Redistributable installation failed.")
        print("[UniRig] Some features may not work. Continuing anyway...")

    from comfy_env import IsolatedEnvManager, discover_config

    node_root = Path(__file__).parent.absolute()

    # Load environment config from comfy-env.toml
    config = discover_config(node_root)
    if config is None:
        print("[UniRig] ERROR: Could not find comfy-env.toml")
        return 1

    # Get the unirig isolated environment
    if "unirig" not in config.envs:
        print("[UniRig] ERROR: No 'unirig' environment defined in config")
        return 1

    env_config = config.envs["unirig"]

    print(f"[UniRig] Loaded config: {env_config.name}")
    print(f"[UniRig]   CUDA: {env_config.cuda}")
    print(f"[UniRig]   PyTorch: {env_config.pytorch_version}")
    print(f"[UniRig]   Requirements: {len(env_config.requirements)} packages")
    print(f"[UniRig]   CUDA packages: {len(env_config.no_deps_requirements)} packages")

    # Create environment manager
    def log(msg):
        print(f"[UniRig] {msg}")

    manager = IsolatedEnvManager(base_dir=node_root, log_callback=log)

    # Check if already ready
    if manager.is_ready(env_config, verify_packages=["torch", "torch_scatter"]):
        env_dir = manager.get_env_dir(env_config)
        print("[UniRig] Isolated environment already exists and is ready!")
        print(f"[UniRig] Location: {env_dir}")
        print("[UniRig] To reinstall, delete the environment directory.")
    else:
        # Setup environment
        try:
            manager.setup(env_config, verify_packages=["torch", "torch_scatter"])
        except Exception as e:
            print(f"\n[UniRig] Installation FAILED: {e}")
            print("[UniRig] Report issues at: https://github.com/PozzettiAndrea/ComfyUI-UniRig/issues")
            return 1

    # Install Blender if not detected
    print()
    print("[UniRig] Checking Blender installation...")
    blender_exe = ensure_blender()
    if blender_exe:
        print(f"[UniRig] Blender ready: {blender_exe}")
    else:
        print("[UniRig] WARNING: Blender installation failed.")
        print("[UniRig] FBX export features may not work.")

    print("\n" + "=" * 60)
    print("[UniRig] Installation completed successfully!")
    print("=" * 60)
    return 0


# =============================================================================
# Legacy exports for backwards compatibility
# =============================================================================

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
