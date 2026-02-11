"""
Blender installation for ComfyUI-UniRig mesh preprocessing.
"""

import urllib.request
import tarfile
import zipfile
import subprocess
import shutil
import sys
import os
from pathlib import Path
from typing import Optional, Tuple, List

from .utils import InstallLogger, get_platform_info
from .config import BLENDER_VERSION, BLENDER_BASE_URL

# System libraries required by Blender on Linux
LINUX_BLENDER_DEPS = [
    ("libxi6", "libXi.so.6"),
    ("libxxf86vm1", "libXxf86vm.so.1"),
    ("libxfixes3", "libXfixes.so.3"),
    ("libxrender1", "libXrender.so.1"),
    ("libgl1", "libGL.so.1"),
    ("libxkbcommon0", "libxkbcommon.so.0"),
]


def check_linux_dependencies() -> List[str]:
    """
    Check which Linux system libraries are missing for Blender.

    Returns:
        List of missing package names
    """
    import ctypes
    import ctypes.util

    missing = []
    for pkg_name, lib_name in LINUX_BLENDER_DEPS:
        # Try to find the library
        lib_path = ctypes.util.find_library(lib_name.replace(".so", "").replace("lib", ""))
        if lib_path is None:
            # Also try direct load
            try:
                ctypes.CDLL(lib_name)
            except OSError:
                missing.append(pkg_name)

    return missing


def install_linux_dependencies() -> bool:
    """
    Install required system libraries for Blender on Linux.

    Returns:
        True if all dependencies are satisfied, False otherwise
    """
    platform_info = get_platform_info()
    if platform_info["platform"] != "linux":
        return True  # Not Linux, skip

    missing = check_linux_dependencies()
    if not missing:
        InstallLogger.info("All Blender system dependencies are installed")
        return True

    InstallLogger.info(f"Missing system libraries: {', '.join(missing)}")
    InstallLogger.info("Attempting to install via apt-get...")

    try:
        # Try to install with sudo
        cmd = ["sudo", "apt-get", "install", "-y"] + missing
        InstallLogger.info(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            InstallLogger.success("System dependencies installed successfully")
            return True
        else:
            InstallLogger.warning(f"apt-get failed: {result.stderr}")
            InstallLogger.warning("You may need to install manually:")
            InstallLogger.warning(f"  sudo apt-get install -y {' '.join(missing)}")
            return False

    except subprocess.TimeoutExpired:
        InstallLogger.warning("apt-get timed out")
        return False
    except FileNotFoundError:
        InstallLogger.warning("apt-get not found (not a Debian-based system?)")
        InstallLogger.warning("Please install these packages manually:")
        InstallLogger.warning(f"  {', '.join(missing)}")
        return False
    except Exception as e:
        InstallLogger.warning(f"Could not install system dependencies: {e}")
        return False


def get_blender_download_url(platform_name: str, architecture: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Get Blender download URL for the platform.

    Args:
        platform_name: "linux", "macos", or "windows"
        architecture: "x64" or "arm64"

    Returns:
        tuple: (download_url, version, filename) or (None, None, None) if not found
    """
    version = BLENDER_VERSION

    urls = {
        ("linux", "x64"): (
            f"{BLENDER_BASE_URL}/blender-{version}-linux-x64.tar.xz",
            version,
            f"blender-{version}-linux-x64.tar.xz"
        ),
        ("macos", "x64"): (
            f"{BLENDER_BASE_URL}/blender-{version}-macos-x64.dmg",
            version,
            f"blender-{version}-macos-x64.dmg"
        ),
        ("macos", "arm64"): (
            f"{BLENDER_BASE_URL}/blender-{version}-macos-arm64.dmg",
            version,
            f"blender-{version}-macos-arm64.dmg"
        ),
        ("windows", "x64"): (
            f"{BLENDER_BASE_URL}/blender-{version}-windows-x64.zip",
            version,
            f"blender-{version}-windows-x64.zip"
        ),
    }

    key = (platform_name, architecture)
    if key in urls:
        url, ver, filename = urls[key]
        InstallLogger.info(f"Using Blender {ver} for {platform_name}-{architecture}")
        return url, ver, filename

    return None, None, None


def download_file(url: str, dest_path: str) -> bool:
    """Download file with progress."""
    InstallLogger.info(f"Downloading: {url}")
    InstallLogger.info(f"Destination: {dest_path}")

    last_printed_percent = [-1]

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        if percent >= last_printed_percent[0] + 10 or percent >= 100:
            sys.stdout.write(f"\r[UniRig Install] Progress: {percent}%")
            sys.stdout.flush()
            last_printed_percent[0] = percent

    try:
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        sys.stdout.write("\n")
        sys.stdout.flush()
        InstallLogger.info("Download complete!")
        return True
    except Exception as e:
        print(f"\n[UniRig Install] Error downloading: {e}")
        return False


def extract_archive(archive_path: str, extract_to: str) -> bool:
    """Extract tar.gz, tar.xz, zip, or handle DMG (macOS)."""
    InstallLogger.info(f"Extracting: {archive_path}")

    try:
        if archive_path.endswith(('.tar.gz', '.tar.xz', '.tar.bz2')):
            with tarfile.open(archive_path, 'r:*') as tar:
                tar.extractall(extract_to)

        elif archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)

        elif archive_path.endswith('.dmg'):
            InstallLogger.info("DMG detected - mounting disk image...")

            mount_result = subprocess.run(
                ['hdiutil', 'attach', '-nobrowse', archive_path],
                capture_output=True,
                text=True
            )

            if mount_result.returncode != 0:
                InstallLogger.error(f"Error mounting DMG: {mount_result.stderr}")
                return False

            mount_point = None
            for line in mount_result.stdout.split('\n'):
                if '/Volumes/' in line:
                    mount_point = line.split('\t')[-1].strip()
                    break

            if not mount_point:
                InstallLogger.error("Could not find mount point")
                return False

            try:
                blender_app = Path(mount_point) / "Blender.app"
                if blender_app.exists():
                    dest_app = Path(extract_to) / "Blender.app"
                    shutil.copytree(blender_app, dest_app)
                    InstallLogger.info(f"Copied Blender.app to: {dest_app}")
                else:
                    InstallLogger.error(f"Blender.app not found in {mount_point}")
                    return False
            finally:
                subprocess.run(['hdiutil', 'detach', mount_point], check=False)

        else:
            InstallLogger.error(f"Unknown archive format: {archive_path}")
            return False

        InstallLogger.info("Extraction complete!")
        return True

    except Exception as e:
        InstallLogger.error(f"Error extracting: {e}")
        return False


def find_blender_executable(blender_dir) -> Optional[Path]:
    """Find the blender executable in the extracted directory."""
    platform_info = get_platform_info()
    plat = platform_info["platform"]

    if plat == "windows":
        exe_pattern = "**/blender.exe"
    elif plat == "macos":
        exe_pattern = "**/MacOS/blender"
    else:  # linux
        exe_pattern = "**/blender"

    executables = list(Path(blender_dir).glob(exe_pattern))

    if executables:
        return executables[0]
    return None


def install_blender(target_dir: Path = None) -> Optional[str]:
    """
    Install Blender for mesh preprocessing.

    Args:
        target_dir: Optional target directory. If None, uses lib/blender under script directory.

    Returns:
        str: Path to Blender executable, or None if installation failed.

    Environment Variables:
        UNIRIG_SKIP_BLENDER_INSTALL: Set to '1' to skip Blender download.
            Assumes Blender is already installed and available in system PATH.
    """
    import os

    InstallLogger.header("Blender Installation")

    # Check if Blender installation is disabled
    if os.environ.get('UNIRIG_SKIP_BLENDER_INSTALL', '').lower() in ('1', 'true', 'yes'):
        InstallLogger.info("UNIRIG_SKIP_BLENDER_INSTALL is set - skipping Blender download")
        InstallLogger.info("Assuming Blender is available in system PATH")
        # Try to find blender in PATH
        blender_in_path = shutil.which('blender')
        if blender_in_path:
            InstallLogger.info(f"Found Blender in PATH: {blender_in_path}")
            return blender_in_path
        InstallLogger.warning("Blender not found in PATH - some features may not work")
        return None

    if target_dir is None:
        script_dir = Path(__file__).parent.parent.absolute()
        target_dir = script_dir / "lib" / "blender"
    else:
        target_dir = Path(target_dir)

    # Detect platform first (needed for system deps check)
    platform_info = get_platform_info()
    plat = platform_info["platform"]
    arch = platform_info["arch"]

    # Install system dependencies for Linux before anything else
    if plat == "linux":
        InstallLogger.info("Checking Blender system dependencies...")
        install_linux_dependencies()

    # Check if Blender already installed
    blender_exe = find_blender_executable(target_dir)
    if blender_exe and blender_exe.exists():
        InstallLogger.info("Blender already installed at:")
        InstallLogger.info(f"  {blender_exe}")
        InstallLogger.info("Skipping download.")
        return str(blender_exe)

    if not plat or not arch:
        InstallLogger.error("Could not detect platform")
        InstallLogger.info("Please install Blender manually from: https://www.blender.org/download/")
        return None

    InstallLogger.info(f"Detected platform: {plat}-{arch}")

    # Get download URL
    url, version, filename = get_blender_download_url(plat, arch)
    if not url:
        InstallLogger.error("Could not find Blender download for your platform")
        InstallLogger.info("Please install Blender manually from: https://www.blender.org/download/")
        return None

    # Create temporary download directory
    temp_dir = target_dir.parent / "_temp_blender_download"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download
        download_path = temp_dir / filename
        if not download_file(url, str(download_path)):
            return None

        # Extract
        target_dir.mkdir(parents=True, exist_ok=True)
        if not extract_archive(str(download_path), str(target_dir)):
            return None

        InstallLogger.success("Blender installation complete!")
        InstallLogger.info(f"Location: {target_dir}")

        # Find blender executable
        blender_exe = find_blender_executable(target_dir)

        if blender_exe:
            InstallLogger.info(f"Blender executable: {blender_exe}")
            return str(blender_exe)
        else:
            InstallLogger.warning("Could not find blender executable")
            return None

    except Exception as e:
        InstallLogger.error(f"Error during installation: {e}")
        return None

    finally:
        # Cleanup temp files
        if temp_dir.exists():
            InstallLogger.info("Cleaning up temporary files...")
            shutil.rmtree(temp_dir, ignore_errors=True)
