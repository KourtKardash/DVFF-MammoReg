#!/usr/bin/env bash
# Risk-prediction inference with the proposed method (DVFF) as registration
# backend, using cached ANTs affine transforms for the prior images.
#
# Required environment (see train.sh):
#   CSV_FILE_PATH, DATA_ROOT_PATH, AFFINE_CACHE_ROOT,
#   REGISTRATION_CHECKPOINT   DVFF registration weights
#   RISK_RESULTS_DIR          training run folder containing
#                             best_model_risk_prediction_id-<TRAINING_ID>.pth
set -euo pipefail

CSV_FILE_PATH="${CSV_FILE_PATH:?set CSV_FILE_PATH}"
DATA_ROOT_PATH="${DATA_ROOT_PATH:?set DATA_ROOT_PATH}"
AFFINE_CACHE_ROOT="${AFFINE_CACHE_ROOT:?set AFFINE_CACHE_ROOT}"

REGISTRATION_METHOD="${REGISTRATION_METHOD:-dvff}"
REGISTRATION_CHECKPOINT="${REGISTRATION_CHECKPOINT:?set REGISTRATION_CHECKPOINT}"

TRAINING_ID="${TRAINING_ID:-1}"
RISK_RESULTS_DIR="${RISK_RESULTS_DIR:?set RISK_RESULTS_DIR to the training run folder}"
TEST_OUTPUT_PATH="${TEST_OUTPUT_PATH:-${RISK_RESULTS_DIR}/test_${TRAINING_ID}}"

DATASET="${DATASET:-CSAW}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
ALIGNMENT_LEVEL="${ALIGNMENT_LEVEL:-images}"  # images | features

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../src" || exit 1

PYTHON="${PYTHON:-python}"
if ! "$PYTHON" -c "import accelerate" 2>/dev/null; then
    echo "Error: package 'accelerate' not found. Activate the project environment first." >&2
    exit 1
fi

RISK_CKPT="${RISK_RESULTS_DIR}/best_model_risk_prediction_id-${TRAINING_ID}.pth"
if [[ ! -f "$RISK_CKPT" ]]; then
    echo "Error: risk checkpoint not found: ${RISK_CKPT}" >&2
    exit 1
fi

echo "[INFO] Risk checkpoint: ${RISK_CKPT}"
echo "[INFO] DVFF registration: ${REGISTRATION_CHECKPOINT}"
echo "[INFO] Affine cache: ${AFFINE_CACHE_ROOT}"
echo "[INFO] Test output: ${TEST_OUTPUT_PATH}"
echo "[INFO] Alignment level: ${ALIGNMENT_LEVEL}"

"$PYTHON" -m accelerate.commands.launch \
    --num_processes "$NUM_PROCESSES" \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    main_test.py \
    --csv_file "$CSV_FILE_PATH" \
    --data_root "$DATA_ROOT_PATH" \
    --path_out_dir "$RISK_RESULTS_DIR" \
    --path_test_folder "$TEST_OUTPUT_PATH" \
    --registration_method "$REGISTRATION_METHOD" \
    --registration_checkpoint "$REGISTRATION_CHECKPOINT" \
    --affine_cache_root "$AFFINE_CACHE_ROOT" \
    --ants_prewarp \
    --no-coarse_affine \
    --alignment_level "$ALIGNMENT_LEVEL" \
    --id_training "$TRAINING_ID" \
    --dataset "$DATASET" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --seed 2023
