import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# Get models directory from environment or use default
def _get_models_dir() -> Path:
    """Get the UniRig models directory."""
    # Check environment variable first (set by ComfyUI-UniRig)
    models_dir = os.environ.get('UNIRIG_MODELS_DIR')
    if models_dir:
        return Path(models_dir)

    # Fallback: look for ComfyUI models/unirig relative to this file
    # This file is at: .../ComfyUI-UniRig/nodes/lib/unirig/src/inference/download.py
    # Models dir is at: .../ComfyUI/models/unirig
    this_file = Path(__file__).resolve()
    comfyui_unirig = this_file.parents[5]  # Go up to ComfyUI-UniRig
    comfyui = comfyui_unirig.parents[1]  # Go up to ComfyUI
    return comfyui / "models" / "unirig"

def download(ckpt_name: str) -> str:
    """Download model checkpoint, returns path to local file."""
    MAP = {
        'experiments/skeleton/articulation-xl_quantization_256/model.ckpt': 'skeleton.safetensors',
        'experiments/skin/articulation-xl/model.ckpt': 'skin.safetensors',
        'experiments/skin/skeleton/model.ckpt': 'skin.safetensors',
    }

    try:
        if ckpt_name not in MAP:
            print(f"[UniRig] Unknown checkpoint: {ckpt_name}")
            return ckpt_name

        filename = MAP[ckpt_name]
        models_dir = _get_models_dir()
        local_path = models_dir / filename

        # Check if already exists
        if local_path.exists():
            print(f"[UniRig] Found model: {local_path}")
            return str(local_path)

        # Create directory if needed
        models_dir.mkdir(parents=True, exist_ok=True)

        # Download directly to models folder
        print(f"[UniRig] Downloading {filename} from apozz/UniRig-safetensors...")
        hf_hub_download(
            repo_id='apozz/UniRig-safetensors',
            filename=filename,
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
        )

        print(f"[UniRig] Downloaded to: {local_path}")
        return str(local_path)

    except Exception as e:
        print(f"[UniRig] Failed to download {ckpt_name}: {e}")
        return ckpt_name
