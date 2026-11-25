"""
Train VoxelMorph on the longitudinal mammography training split (run from repository root).

Usage:
    python Methods/voxelmorph/train_VoxelMorph.py
    python Methods/voxelmorph/train_VoxelMorph.py --gpu 0 --epochs 250 --batch-size 10
"""

import os
import runpy
import sys


def main():
    vxm_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(vxm_dir, "..", ".."))

    os.chdir(repo_root)
    if vxm_dir not in sys.path:
        sys.path.insert(0, vxm_dir)
    os.environ.setdefault("VXM_BACKEND", "pytorch")

    train_script = os.path.join(vxm_dir, "scripts", "torch", "train.py")
    runpy.run_path(train_script, run_name="__main__")


if __name__ == "__main__":
    main()
