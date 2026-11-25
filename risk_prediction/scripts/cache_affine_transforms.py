#!/usr/bin/env python3
"""
Pre-compute ANTs affine transforms for CSAW longitudinal risk-prediction pairs.

Each cached .mat maps prior (moving) -> current (fixed) for one breast view:
  L_CC, L_MLO, R_CC, or R_MLO.

Only *consecutive* exam timepoints are registered (e.g. 1->2, 2->3, 3->4),
matching BreastCancerRiskDatasetCSAWCC.

Output layout (recommended):
  {output_root}/{split}/{patient_id}/{L|R}_{CC|MLO}_{prior_key}-{current_key}.mat

Example:
  affine_cache/train/6/L_CC_2009-2011.mat

Usage (local, multiprocessing):
  python cache_affine_transforms.py \\
      --csv_file metadata.csv \\
      --images_root /path/to/Dataset_masked_split \\
      --output_root /path/to/affine_cache \\
      --splits train val test \\
      --num_workers 16

Usage (SLURM array job, one task per chunk):
  python cache_affine_transforms.py ... --task_index $SLURM_ARRAY_TASK_ID --num_tasks $SLURM_ARRAY_TASK_COUNT

Requires: antspyx, pandas, pillow, numpy
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

# ANTs uses ITK threading; limit per worker when running many processes in parallel.
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")

FILENAME_RE = re.compile(
    r"^(?P<prefix>\d+)_(?P<date>\d+)_(?P<lat>[LR])_(?P<view>CC|MLO)_(?P<idx>\d+)\.(?P<ext>png|PNG)$"
)

IMAGE_SIZE = (896, 1152)  # width, height for PIL resize (matches MammoRegNet pipeline)


@dataclass(frozen=True)
class AffinePairTask:
    split: str
    patient_id: str
    laterality: str  # L or R
    view: str  # CC or MLO
    prior_key: str
    current_key: str
    moving_path: str
    fixed_path: str
    output_mat_path: str

    @property
    def task_id(self) -> str:
        return (
            f"{self.split}/{self.patient_id}/"
            f"{self.laterality}_{self.view}_{self.prior_key}-{self.current_key}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache ANTs affine .mat transforms for CSAW pairs")
    parser.add_argument("--csv_file", type=str, required=True, help="Metadata CSV with file_name, patient_id, exam_year, laterality, viewposition")
    parser.add_argument("--images_root", type=str, required=True, help="Root with train/val/test subfolders of masked PNG images")
    parser.add_argument("--output_root", type=str, required=True, help="Where to write cached .mat files")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--image_size", type=int, nargs=2, default=list(IMAGE_SIZE), metavar=("W", "H"))
    parser.add_argument("--random_seed", type=int, default=7, help="ANTs affine random seed")
    parser.add_argument("--num_workers", type=int, default=1, help="Parallel worker processes (local mode)")
    parser.add_argument("--task_index", type=int, default=None, help="SLURM array task index (0-based)")
    parser.add_argument("--num_tasks", type=int, default=None, help="Total SLURM array tasks")
    parser.add_argument("--skip_existing", action="store_true", help="Skip pairs whose .mat already exists")
    parser.add_argument("--dry_run", action="store_true", help="Only list pairs, do not run ANTs")
    parser.add_argument("--write_manifest", action="store_true", help="Write manifest.json to output_root")
    parser.add_argument(
        "--retry_seeds",
        type=int,
        nargs="*",
        default=[11, 7],
        help="Extra ANTs random seeds if the first attempt returns no transform (default: 11 42)",
    )
    parser.add_argument(
        "--verbose_ants",
        action="store_true",
        help="Print ANTs registration logs (useful for debugging failed pairs)",
    )
    parser.add_argument(
        "--min_nonzero_fraction",
        type=float,
        default=0.001,
        help="Reject pairs whose moving/fixed image has less than this fraction of nonzero pixels",
    )
    return parser.parse_args()


def laterality_to_lr(value: str) -> str | None:
    if value in ("L", "R"):
        return value
    if value == "Left":
        return "L"
    if value == "Right":
        return "R"
    return None


def load_image_array(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("L")
    if img.size != image_size:
        img = img.resize(image_size, Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.uint8)


def mat_output_path(
    output_root: Path,
    split: str,
    patient_id: str,
    laterality: str,
    view: str,
    prior_key: str,
    current_key: str,
) -> Path:
    patient_dir = output_root / split / str(patient_id)
    return patient_dir / f"{laterality}_{view}_{prior_key}-{current_key}.mat"


def discover_pairs(
    csv_file: Path,
    images_root: Path,
    splits: Iterable[str],
    output_root: Path,
    image_size: tuple[int, int],
) -> list[AffinePairTask]:
    df = pd.read_csv(csv_file, low_memory=False)
    if "file_name" not in df.columns:
        raise ValueError("CSV must contain column 'file_name'")

    csv_data = df.set_index("file_name").to_dict(orient="index")
    tasks: list[AffinePairTask] = []

    for split in splits:
        split_dir = images_root / split
        if not split_dir.is_dir():
            print(f"[WARN] Split folder missing, skipping: {split_dir}")
            continue

        # patient_id -> exam_key -> {L_CC: filename, ...}
        patient_exams: dict[str, dict[str, dict[str, str]]] = {}

        for filename in os.listdir(split_dir):
            if filename not in csv_data:
                continue
            info = csv_data[filename]
            lat = laterality_to_lr(str(info.get("laterality", "")))
            view = str(info.get("viewposition", "")).upper()
            if lat is None or view not in ("CC", "MLO"):
                continue

            patient_id = str(info["patient_id"])
            exam_key = str(info.get("exam_year", ""))
            if not exam_key:
                m = FILENAME_RE.match(filename)
                if not m:
                    continue
                exam_key = m.group("idx")

            view_key = f"{lat}_{view}"
            patient_exams.setdefault(patient_id, {}).setdefault(exam_key, {})[view_key] = filename

        for patient_id, exams in patient_exams.items():
            if len(exams) < 2:
                continue

            def sort_key(k: str):
                try:
                    return (0, int(k))
                except ValueError:
                    return (1, k)

            sorted_keys = sorted(exams.keys(), key=sort_key)
            for i in range(1, len(sorted_keys)):
                prior_key = sorted_keys[i - 1]
                current_key = sorted_keys[i]
                for view_key in ("L_CC", "L_MLO", "R_CC", "R_MLO"):
                    if view_key not in exams[prior_key] or view_key not in exams[current_key]:
                        continue
                    lat, view = view_key.split("_")
                    moving_name = exams[prior_key][view_key]
                    fixed_name = exams[current_key][view_key]
                    moving_path = split_dir / moving_name
                    fixed_path = split_dir / fixed_name

                    out_mat = mat_output_path(
                        output_root, split, patient_id, lat, view, prior_key, current_key
                    )
                    tasks.append(
                        AffinePairTask(
                            split=split,
                            patient_id=patient_id,
                            laterality=lat,
                            view=view,
                            prior_key=str(prior_key),
                            current_key=str(current_key),
                            moving_path=str(moving_path.resolve()),
                            fixed_path=str(fixed_path.resolve()),
                            output_mat_path=str(out_mat.resolve()),
                        )
                    )
    return tasks


def image_content_stats(arr: np.ndarray) -> dict:
    nonzero = int(np.count_nonzero(arr))
    total = int(arr.size)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "nonzero": nonzero,
        "nonzero_fraction": round(nonzero / max(total, 1), 6),
    }


def validate_image_pair(moving_arr: np.ndarray, fixed_arr: np.ndarray, min_nonzero_fraction: float = 0.001) -> None:
    if moving_arr.shape != fixed_arr.shape:
        raise ValueError(
            f"Shape mismatch: moving={moving_arr.shape}, fixed={fixed_arr.shape}"
        )

    moving_stats = image_content_stats(moving_arr)
    fixed_stats = image_content_stats(fixed_arr)

    if moving_stats["nonzero_fraction"] < min_nonzero_fraction:
        raise ValueError(
            "Moving image is nearly empty after loading "
            f"(nonzero_fraction={moving_stats['nonzero_fraction']}, stats={moving_stats})"
        )
    if fixed_stats["nonzero_fraction"] < min_nonzero_fraction:
        raise ValueError(
            "Fixed image is nearly empty after loading "
            f"(nonzero_fraction={fixed_stats['nonzero_fraction']}, stats={fixed_stats})"
        )

    if moving_stats["max"] == moving_stats["min"] and fixed_stats["max"] == fixed_stats["min"]:
        raise ValueError(
            "Both images are constant (degenerate intensity); ANTs cannot estimate affine"
        )


def run_affine_registration(
    moving_arr: np.ndarray,
    fixed_arr: np.ndarray,
    output_mat: Path,
    random_seed: int,
    retry_seeds: list[int] | None = None,
    verbose: bool = False,
) -> None:
    import ants

    validate_image_pair(moving_arr, fixed_arr)

    seeds = [random_seed] + [s for s in (retry_seeds or []) if s != random_seed]
    last_error: Exception | None = None

    for seed in seeds:
        fixed_ants = ants.from_numpy(fixed_arr.astype(np.uint8))
        moving_ants = ants.from_numpy(moving_arr.astype(np.uint8))

        reg = ants.registration(
            fixed=fixed_ants,
            moving=moving_ants,
            type_of_transform="Affine",
            random_seed=seed,
            verbose=verbose,
        )

        fwd = reg.get("fwdtransforms") or []
        if fwd:
            output_mat.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fwd[0], output_mat)
            return

        last_error = RuntimeError(
            "ANTs registration returned empty fwdtransforms "
            f"(random_seed={seed}, reg_keys={sorted(reg.keys())})"
        )

    moving_stats = image_content_stats(moving_arr)
    fixed_stats = image_content_stats(fixed_arr)
    raise RuntimeError(
        f"{last_error}. "
        f"moving_stats={moving_stats}, fixed_stats={fixed_stats}. "
        "Common causes: empty/corrupt masked PNG, degenerate breast mask, or ANTs failure on this pair."
    ) from last_error


def process_task(
    task: AffinePairTask,
    image_size: tuple[int, int],
    random_seed: int,
    skip_existing: bool,
    retry_seeds: list[int] | None,
    verbose_ants: bool,
    min_nonzero_fraction: float,
) -> dict:
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    output_mat = Path(task.output_mat_path)
    if skip_existing and output_mat.is_file():
        return {"task_id": task.task_id, "status": "skipped", "mat": str(output_mat)}

    try:
        moving_path = Path(task.moving_path)
        fixed_path = Path(task.fixed_path)
        if not moving_path.is_file():
            raise FileNotFoundError(f"Moving image not found: {moving_path}")
        if not fixed_path.is_file():
            raise FileNotFoundError(f"Fixed image not found: {fixed_path}")

        moving_arr = load_image_array(moving_path, image_size)
        fixed_arr = load_image_array(fixed_path, image_size)
        validate_image_pair(
            moving_arr, fixed_arr, min_nonzero_fraction=min_nonzero_fraction
        )
        run_affine_registration(
            moving_arr,
            fixed_arr,
            output_mat,
            random_seed,
            retry_seeds=retry_seeds,
            verbose=verbose_ants,
        )
        return {"task_id": task.task_id, "status": "ok", "mat": str(output_mat)}
    except Exception as exc:
        payload = {
            "task_id": task.task_id,
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "moving_path": task.moving_path,
            "fixed_path": task.fixed_path,
        }
        try:
            if Path(task.moving_path).is_file() and Path(task.fixed_path).is_file():
                payload["moving_stats"] = image_content_stats(
                    load_image_array(Path(task.moving_path), image_size)
                )
                payload["fixed_stats"] = image_content_stats(
                    load_image_array(Path(task.fixed_path), image_size)
                )
        except Exception:
            pass
        return payload


def _task_worker(args_tuple):
    return process_task(*args_tuple)


def shard_tasks(tasks: list[AffinePairTask], task_index: int | None, num_tasks: int | None) -> list[AffinePairTask]:
    if task_index is None or num_tasks is None:
        return tasks
    if task_index < 0 or task_index >= num_tasks:
        raise ValueError(f"task_index={task_index} out of range for num_tasks={num_tasks}")
    return tasks[task_index::num_tasks]


def main() -> int:
    args = parse_args()
    csv_file = Path(args.csv_file)
    images_root = Path(args.images_root)
    output_root = Path(args.output_root)
    image_size = (int(args.image_size[0]), int(args.image_size[1]))

    tasks = discover_pairs(
        csv_file=csv_file,
        images_root=images_root,
        splits=args.splits,
        output_root=output_root,
        image_size=image_size,
    )
    tasks = shard_tasks(tasks, args.task_index, args.num_tasks)

    print(f"Discovered {len(tasks)} affine pair tasks")
    if args.dry_run:
        for t in tasks[:20]:
            print(f"  {t.task_id} -> {t.output_mat_path}")
        if len(tasks) > 20:
            print(f"  ... and {len(tasks) - 20} more")
        return 0

    if args.write_manifest:
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump([asdict(t) for t in tasks], f, indent=2)
        print(f"Wrote manifest: {manifest_path}")

    results = []
    worker_args = [
        (
            task,
            image_size,
            args.random_seed,
            args.skip_existing,
            args.retry_seeds,
            args.verbose_ants,
            args.min_nonzero_fraction,
        )
        for task in tasks
    ]

    if args.num_workers <= 1:
        for wa in worker_args:
            results.append(_task_worker(wa))
            status = results[-1]["status"]
            print(f"[{status}] {results[-1]['task_id']}")
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {pool.submit(_task_worker, wa): wa[0] for wa in worker_args}
            for fut in as_completed(futures):
                task = futures[fut]
                res = fut.result()
                results.append(res)
                print(f"[{res['status']}] {task.task_id}")

    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    err = [r for r in results if r["status"] == "error"]
    print(f"Done: ok={ok}, skipped={skipped}, errors={len(err)}")
    if err:
        log_path = output_root / "errors.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            for r in err:
                f.write(json.dumps(r) + "\n")
        print(f"Error log appended to {log_path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
