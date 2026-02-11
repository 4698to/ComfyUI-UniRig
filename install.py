#!/usr/bin/env python3
"""
Installation script for ComfyUI-UniRig.

Uses comfy-env for:
- Isolated Python environment with CUDA dependencies
- Blender installation (via [tools] section)
- MSVC runtime on Windows
"""

import sys
from pathlib import Path


def main():
    print("\n" + "=" * 60)
    print("ComfyUI-UniRig Installation")
    print("=" * 60)

    from comfy_env import install, IsolatedEnvManager, discover_config

    node_root = Path(__file__).parent.absolute()

    # Run comfy-env install
    try:
        install(config=node_root / "comfy-env.toml", mode="isolated", node_dir=node_root)
    except Exception as e:
        print(f"\n[UniRig] Installation FAILED: {e}")
        print("[UniRig] Report issues at: https://github.com/PozzettiAndrea/ComfyUI-UniRig/issues")
        return 1

    print("\n" + "=" * 60)
    print("[UniRig] Installation completed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
