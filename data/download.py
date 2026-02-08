"""
Download KernelCo/robot_control dataset from HuggingFace
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

def download_dataset(cache_dir="./data/cache"):
    """Download the brain-robot control dataset"""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    print("Downloading KernelCo/robot_control dataset from HuggingFace...")
    
    try:
        dataset_path = snapshot_download(
            repo_id="KernelCo/robot_control",
            repo_type="dataset",
            cache_dir=str(cache_path),
            local_dir=str(cache_path / "robot_control"),
            local_dir_use_symlinks=False
        )
        print(f"✓ Dataset downloaded to: {dataset_path}")
        return dataset_path
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Login to HuggingFace: huggingface-cli login")
        print("2. Check internet connection")
        print("3. Verify dataset exists: https://huggingface.co/datasets/KernelCo/robot_control")
        raise

if __name__ == "__main__":
    download_dataset()
