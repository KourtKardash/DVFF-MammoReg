import argparse
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np


DATASET_ORDER = ("CSAW-CC", "NLBS", "KAU-BCMD", "INBreast", "Unknown")


def infer_dataset(image_name: str) -> str:
    """Map landmark image filename to source dataset."""
    stem = image_name.replace("\\", "/").rsplit("/", 1)[-1]
    if stem.startswith("p_"):
        return "NLBS"
    if stem.startswith("BC"):
        return "KAU-BCMD"
    if re.match(r"^[0-9a-f]{16}-\d+\.png$", stem, re.IGNORECASE):
        return "INBreast"
    if re.match(r"^\d+-\d+\.png$", stem):
        return "CSAW-CC"
    return "Unknown"


def parse_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = {}

    for image in root.findall("image"):
        name = image.get("name")
        width = int(image.get("width"))
        height = int(image.get("height"))
        points_dict = {}
        for points in image.findall("points"):
            attr = points.find('attribute[@name="id"]')
            if attr is None:
                continue
            try:
                point_id = int(attr.text)
            except ValueError:
                continue

            coords_str = points.get("points")
            if not coords_str:
                continue
            try:
                x, y = map(float, coords_str.split(","))
                points_dict[point_id] = (x, y)
            except (ValueError, IndexError):
                continue

        data[name] = {
            "width": width,
            "height": height,
            "points": points_dict,
        }
    return data


def compute_average_normalized_distance(img_data_left, img_data_other):
    points_left = img_data_left["points"]
    points_other = img_data_other["points"]
    common_ids = set(points_left.keys()) & set(points_other.keys())

    if not common_ids:
        return 0.0, 0

    w = img_data_other["width"]
    h = img_data_other["height"]
    diagonal = math.sqrt(w * w + h * h)

    total_dist = 0.0
    for pid in common_ids:
        x1, y1 = points_left[pid]
        x2, y2 = points_other[pid]
        dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        total_dist += dist / diagonal

    return total_dist / len(common_ids), len(common_ids)


def _summarize(scores):
    if not scores:
        return None, None, 0
    arr = np.asarray(scores, dtype=float)
    return float(np.mean(arr)), float(np.std(arr)), int(arr.size)


def calc_tre(left_file, right_file, validated_file, verbose: bool = True):
    if verbose:
        print("Calculating rTRE...")
    left_data = parse_annotations(left_file)
    right_data = parse_annotations(right_file)
    validated_data = parse_annotations(validated_file)

    all_image_names = set(left_data.keys()) & set(right_data.keys()) & set(
        validated_data.keys()
    )
    per_image_scores = []
    per_dataset_scores = defaultdict(list)

    for name in sorted(all_image_names):
        img_left = left_data[name]
        img_right = right_data[name]
        img_valid = validated_data[name]

        avg_lr, count_lr = compute_average_normalized_distance(img_left, img_right)
        avg_lv, count_lv = compute_average_normalized_distance(img_left, img_valid)

        if count_lr == 0 and count_lv == 0:
            continue
        elif count_lr == 0:
            combined = avg_lv
        elif count_lv == 0:
            combined = avg_lr
        else:
            combined = (avg_lr + avg_lv) / 2.0

        per_image_scores.append(combined)
        per_dataset_scores[infer_dataset(name)].append(combined)

    mean, std, n = _summarize(per_image_scores)
    if verbose:
        print(
            f"\nOverall ({n} images): "
            f"{mean:.6f} +- {std:.6f}  ({100 * mean:.2f}% +- {100 * std:.2f}%)"
        )
        print("\nPer dataset:")

    results = {"Overall": {"mean": mean, "std": std, "n": n}}
    for dataset in DATASET_ORDER:
        scores = per_dataset_scores.get(dataset, [])
        if not scores:
            continue
        d_mean, d_std, d_n = _summarize(scores)
        results[dataset] = {"mean": d_mean, "std": d_std, "n": d_n}
        if verbose:
            print(
                f"  {dataset:10s} ({d_n:3d} images): "
                f"{d_mean:.6f} +- {d_std:.6f}  "
                f"({100 * d_mean:.2f}% +- {100 * d_std:.2f}%)"
            )

    unknown = per_dataset_scores.get("Unknown", [])
    if unknown and verbose:
        print(f"  WARNING: {len(unknown)} images with unrecognized filename pattern")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute overall and per-dataset rTRE")
    parser.add_argument("--moving", required=True, help="moving_landmarks.xml")
    parser.add_argument("--fixed1", required=True, help="fixed_landmarks_1.xml")
    parser.add_argument("--fixed2", required=True, help="fixed_landmarks_2.xml")
    args = parser.parse_args()
    calc_tre(args.moving, args.fixed1, args.fixed2)
