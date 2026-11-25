#!/usr/bin/env bash
# Evaluate a trained risk-prediction model on the CSAW-CC test split.
# See train.sh for the meaning of the environment variables.
set -euo pipefail

CSV_FILE_PATH="${CSV_FILE_PATH:?set CSV_FILE_PATH}"
DATA_ROOT_PATH="${DATA_ROOT_PATH:?set DATA_ROOT_PATH}"
OUTPUT_DIR_PATH="${OUTPUT_DIR_PATH:?set OUTPUT_DIR_PATH to the training run folder}"
TEST_OUTPUT_PATH="${TEST_OUTPUT_PATH:?set TEST_OUTPUT_PATH}"

REGISTRATION_METHOD="${REGISTRATION_METHOD:-mammoregnet}"
REGISTRATION_CHECKPOINT="${REGISTRATION_CHECKPOINT:?set REGISTRATION_CHECKPOINT}"
ALIGNMENT_LEVEL="${ALIGNMENT_LEVEL:-images}"  # images | features

TRAINING_ID="${TRAINING_ID:-1}"
DATASET="${DATASET:-CSAW}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../src" || exit 1

accelerate launch --num_processes 1 main_test.py \
    --csv_file "$CSV_FILE_PATH" \
    --data_root "$DATA_ROOT_PATH" \
    --path_out_dir "$OUTPUT_DIR_PATH" \
    --path_test_folder "$TEST_OUTPUT_PATH" \
    --registration_method "$REGISTRATION_METHOD" \
    --registration_checkpoint "$REGISTRATION_CHECKPOINT" \
    --alignment_level "$ALIGNMENT_LEVEL" \
    --id_training "$TRAINING_ID" \
    --dataset "$DATASET" \
    --batch_size 8 \
    --num_workers 7 \
    --seed 2023
