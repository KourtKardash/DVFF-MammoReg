#!/usr/bin/env bash
# Train the longitudinal risk-prediction model on CSAW-CC with a fixed
# registration backend. The registration method is the only experimental
# variable; every other setting is identical across runs.
#
# Paths come from the environment:
#   CSV_FILE_PATH        CSAW-CC metadata csv
#   DATA_ROOT_PATH       preprocessed image root
#   OUTPUT_DIR_PATH      where run folders are written
#   AFFINE_CACHE_ROOT    cached ANTs .mat transforms (see risk_prediction/scripts/cache_affine.sh)
#   REGISTRATION_CHECKPOINT   registration model weights
#   ALIGNMENT_LEVEL      images (default) or features. The paper warps the
#                        prior image before the encoder.
#
# Registration backends (see src/registration/factory.py):
#   dvff         : set REGISTRATION_CHECKPOINT, keep --ants_prewarp and --no-coarse_affine
#   mammoregnet  : set REGISTRATION_CHECKPOINT, drop --ants_prewarp and --no-coarse_affine
#   ants_affine  : set AFFINE_CACHE_ROOT, drop REGISTRATION_CHECKPOINT
#   unreg        : drop REGISTRATION_CHECKPOINT and --ants_prewarp
set -euo pipefail

CSV_FILE_PATH="${CSV_FILE_PATH:?set CSV_FILE_PATH}"
DATA_ROOT_PATH="${DATA_ROOT_PATH:?set DATA_ROOT_PATH}"
OUTPUT_DIR_PATH="${OUTPUT_DIR_PATH:?set OUTPUT_DIR_PATH}"
AFFINE_CACHE_ROOT="${AFFINE_CACHE_ROOT:?set AFFINE_CACHE_ROOT}"

REGISTRATION_METHOD="${REGISTRATION_METHOD:-dvff}"
REGISTRATION_CHECKPOINT="${REGISTRATION_CHECKPOINT:?set REGISTRATION_CHECKPOINT}"
ALIGNMENT_LEVEL="${ALIGNMENT_LEVEL:-images}"  # images | features

TRAINING_ID="${TRAINING_ID:-1}"
DATASET="${DATASET:-CSAW}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../src" || exit 1

PYTHON="${PYTHON:-python}"
if ! "$PYTHON" -c "import accelerate" 2>/dev/null; then
    echo "Error: package 'accelerate' not found. Activate the project environment first." >&2
    exit 1
fi

"$PYTHON" -m accelerate.commands.launch \
    --num_processes "$NUM_PROCESSES" \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    main_train.py \
    --csv_file "$CSV_FILE_PATH" \
    --data_root "$DATA_ROOT_PATH" \
    --path_out_dir "$OUTPUT_DIR_PATH" \
    --registration_method "$REGISTRATION_METHOD" \
    --registration_checkpoint "$REGISTRATION_CHECKPOINT" \
    --id_training "$TRAINING_ID" \
    --use_scheduler "True" \
    --batch_size 4 \
    --augmentations "True" \
    --num_workers 7 \
    --learning_rate 5e-5 \
    --weight_decay 1e-4 \
    --lr_decay 0.5 \
    --patience_lr_scheduler 5 \
    --patience 15 \
    --num_epochs 30 \
    --dataset "$DATASET" \
    --finetune_all \
    --risk_pos_weight 6.0 \
    --seed 2023 \
    --ants_prewarp \
    --affine_cache_root "$AFFINE_CACHE_ROOT" \
    --alignment_level "$ALIGNMENT_LEVEL" \
    --no-coarse_affine \
    --no-wandb
