#!/usr/bin/env bash
#
# Reproduce the two main result tables of the paper: registration accuracy
# (rTRE), deformation regularity (NJD) and computational cost on the evaluation
# set, plus the image-similarity and mask-overlap metrics.
#
# These are Tables 3 and 4 of the published article: "Registration accuracy,
# deformation regularity, and computational cost" and "Image-similarity and
# mask-overlap metrics".
#
# Run from the repository root:
#     bash scripts/reproduce_table3.sh
#
# Every entry point below is the same one used to produce the reported numbers.
# Each writes a JSON summary and prints the rTRE breakdown to stdout; nothing is
# aggregated here, because the paper's tables are assembled from those JSON
# files (see "Reproducing the paper tables" in the README).
#
# Expected layout (the evaluation data is not distributed with this repository,
# see the "Data" section of the README):
#
#   Dataset/
#     evaluation/{INBreast,KAU-BCMD,NLBS,CSAW-CC}/<patient>/*.png
#     evaluation/metadata_evaluation_correct.csv
#     evaluation_masks/...
#     moving_landmarks.xml  fixed_landmarks_1.xml  fixed_landmarks_2.xml
#     affine_evaluation/...        # produced by Methods/Affine/run_from_scratch.py --subset evaluation
#
# Trained weights are expected under outputs/<run>/models/. Set SEED to the
# training seed that produced them.
set -euo pipefail

SEED="${SEED:-11}"
PYTHON="${PYTHON:-python}"

run () {
    echo
    echo "=================================================================="
    echo ">>> $*"
    echo "=================================================================="
    "$@"
}

# --- lower reference -------------------------------------------------------
run "$PYTHON" Methods/Unreg/inference.py

# --- external affine transforms -------------------------------------------
# This step estimates the ANTs affine transform for every evaluation pair and
# writes both the transforms (affine_transforms/*.mat) and the pre-warped images
# (Dataset/affine_evaluation/). Every method that receives an external affine
# transform reads them, so it must run first. Use --subset train once, before
# training, to produce the pre-warped training split.
run "$PYTHON" Methods/Affine/run_from_scratch.py --subset evaluation

# --- classical methods (CPU) ----------------------------------------------
# Affine baseline: scores the transforms computed above.
run "$PYTHON" Methods/Affine/run.py
# SyN: the deformable stage only, applied on top of the same external affine.
run "$PYTHON" Methods/SyN/run_from_scratch.py
run "$PYTHON" Methods/CurCoords/run.py

# --- implicit neural representation ---------------------------------------
run "$PYTHON" Methods/IDIR/run.py --seed "$SEED"

# --- learning-based comparison methods ------------------------------------
run "$PYTHON" Methods/voxelmorph/inference.py
run "$PYTHON" Methods/transmorph/inference.py

# --- MammoRegNet ------------------------------------------------------------
# The unmodified network: no external affine transform, affine decoder head.
run "$PYTHON" Methods/MammoRegNet/evaluate/inference_original.py --seed "$SEED"
# Affine + MammoRegNet: affine decoder head, cached ANTs matrix applied to
# the moving image and to the landmarks. Writes outputs/Affine_MammoRegNet_<seed>/
# and reads the row-1 checkpoint.
run "$PYTHON" Methods/MammoRegNet/evaluate/inference_original.py --seed "$SEED" --external_affine

# --- proposed method -------------------------------------------------------
run "$PYTHON" Methods/DVFF/evaluate/inference_dvff.py --seed "$SEED" --coarse_deform

echo
echo "Done. Per-method JSON summaries are under outputs/<method>/."
echo "The ablation rows are produced by scripts/reproduce_table5.sh."
