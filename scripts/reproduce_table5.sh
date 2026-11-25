#!/usr/bin/env bash
#
# Reproduce the ablation of the proposed pipeline (Table 5 of the published
# article, "Ablation of the proposed pipeline").
#
# Run from the repository root, after scripts/reproduce_table3.sh has written
# the cached ANTs transforms in affine_transforms/:
#     bash scripts/reproduce_table5.sh
#
# The seven rows of that table are configurations of the same code. Rows 1 and 2
# are the two MammoRegNet conditions already evaluated by reproduce_table3.sh and
# are repeated here for completeness.
#
#   # | Configuration                    | external affine | coarsest level | second stage | fusion train | fusion inference | rTRE
#   1 | MammoRegNet                      | no              | affine         | no           | no           | no               | 1.89
#   2 | Affine + MammoRegNet             | yes             | affine         | no           | no           | no               | 1.69
#   3 | Fully deformable decoder         | yes             | deformable     | no           | no           | no               | 1.61
#   4 | Second stage without fusion      | yes             | deformable     | yes          | no           | no               | 1.59
#   5 | Trained with fusion, fusion off  | yes             | deformable     | yes          | yes          | no               | 1.58
#   6 | Proposed                         | yes             | deformable     | yes          | yes          | yes              | 1.57
#   7 | Proposed without affine          | no              | affine         | yes          | yes          | yes              | 1.77
#
# --coarse_deform switches the external affine step and the coarsest decoder head
# together: with the flag the script applies the cached ANTs transform and builds
# a deformable head at the coarsest level, without it no external transform is
# applied and the affine head is restored. Rows 6 and 7 differ by the flag alone.
#
# Row 2 keeps the affine head and applies the cached ANTs matrix to the image
# and to the landmarks (--external_affine). It reads the row-1 checkpoint and
# writes outputs/Affine_MammoRegNet_<seed>/. The .mat files come from
# reproduce_table3.sh.
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

# Row 1: unmodified MammoRegNet, no external affine, affine decoder head.
run "$PYTHON" Methods/MammoRegNet/evaluate/inference_original.py --seed "$SEED"

# Row 2: affine head plus the external ANTs affine on the image and the landmarks.
run "$PYTHON" Methods/MammoRegNet/evaluate/inference_original.py --seed "$SEED" --external_affine

# Row 3: first-stage checkpoint, single-view, deformable coarsest head.
# Writes into outputs/MammoRegNet_deformable_<seed>/, where that checkpoint is saved.
run "$PYTHON" Methods/MammoRegNet/evaluate/inference_original.py --seed "$SEED" --coarse_deform

# Row 4: second stage trained without fusion. The checkpoint is
# outputs/DVFF_no_fusion_<seed>/models/model_249.pth, which is also
# where inference_dvff.py --coarse_deform --no_fusion reads it.
run "$PYTHON" Methods/DVFF/evaluate/inference_dvff.py --seed "$SEED" --coarse_deform --no_fusion

# Row 5: trained with fusion, fusion switched off at inference.
run "$PYTHON" Methods/DVFF/evaluate/inference_dvff_fusion_off.py --seed "$SEED" --coarse_deform

# Row 6: the complete proposed method.
run "$PYTHON" Methods/DVFF/evaluate/inference_dvff.py --seed "$SEED" --coarse_deform

# Row 7: the complete method without external affine pre-alignment. Dropping
# --coarse_deform removes the external transform and restores the affine head, so
# this row differs from row 1 in the dual-view components alone.
# Without the flag the run folder resolves to outputs/DVFF_no_affine_<seed>/, which is
# already distinct from row 6's outputs/DVFF_<seed>/.
run "$PYTHON" Methods/DVFF/evaluate/inference_dvff.py --seed "$SEED"

echo
echo "Done. Per-configuration JSON summaries are under outputs/."
