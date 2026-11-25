#!/usr/bin/env bash
# Pre-compute ANTs affine transforms for the CSAW-CC longitudinal pairs.
#
# All paths come from the environment so that the script is identical on a
# workstation and inside a batch job. Set them before running:
#
#   CSV_FILE=... IMAGES_ROOT=... OUTPUT_ROOT=... bash cache_affine.sh
#
# To split the work across a job array, export TASK_INDEX and NUM_TASKS
# (for SLURM: TASK_INDEX=$SLURM_ARRAY_TASK_ID NUM_TASKS=$SLURM_ARRAY_TASK_COUNT).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CSV_FILE="${CSV_FILE:?set CSV_FILE to the CSAW-CC metadata csv}"
IMAGES_ROOT="${IMAGES_ROOT:?set IMAGES_ROOT to the preprocessed image root}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to the affine cache directory}"
NUM_WORKERS="${NUM_WORKERS:-8}"

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

ARRAY_ARGS=()
if [[ -n "${TASK_INDEX:-}" && -n "${NUM_TASKS:-}" ]]; then
    ARRAY_ARGS=(--task_index "$TASK_INDEX" --num_tasks "$NUM_TASKS")
fi

# Dry run first to count the work:
#   python "$SCRIPT_DIR/cache_affine_transforms.py" \
#     --csv_file "$CSV_FILE" --images_root "$IMAGES_ROOT" \
#     --output_root "$OUTPUT_ROOT" --dry_run

python "$SCRIPT_DIR/cache_affine_transforms.py" \
  --csv_file "$CSV_FILE" \
  --images_root "$IMAGES_ROOT" \
  --output_root "$OUTPUT_ROOT" \
  --splits train val test \
  --skip_existing \
  --write_manifest \
  --num_workers "$NUM_WORKERS" \
  "${ARRAY_ARGS[@]}"
