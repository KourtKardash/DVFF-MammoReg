"""
Paired Wilcoxon signed-rank tests on per-patient rTRE.

This is the statistical comparison reported in the paper. The accuracy tables
report means over the 200 evaluation pairs; the tests cannot use that level,
because pairs from one patient are not independent observations. The image-level
rTRE written by each method's evaluation script is therefore first averaged over
that patient's images, and the methods are compared at the patient level. The
Holm procedure is applied inside each family, not across both. The benchmark
family is every pair among the unregistered condition, the eight comparison
methods, the proposed method, and the proposed method with fusion disabled at
inference: 11 conditions and 55 comparisons. The ablation family is every pair
among the seven configurations of the ablation table: 21 comparisons. The
evaluation set has 51 patients and 200 images.

Inputs are the per-method output folders written by the evaluation entry points
(see scripts/reproduce_table3.sh and scripts/reproduce_table5.sh): each must
contain the warped moving_landmarks.xml and the two readers' fixed_landmarks
files. Outputs are a CSV of all pairs and a Markdown report with the p-value and
median-difference matrices.

Example:
  python analysis/wilcoxon_rTRE_patients.py --outputs_root outputs --seed 11
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]

# The rTRE helpers live with the methods; any copy will do, they are identical.
sys.path.insert(0, str(REPO_ROOT / "Methods" / "MammoRegNet" / "evaluate"))
from rTRE import (  # noqa: E402
    compute_average_normalized_distance,
    parse_annotations,
)

# (table label, folder under outputs/, families). "{seed}" is filled in from --seed.
#
# "Affine+MRN" is inference_original.py --external_affine. That keeps the
# affine decoder head, applies the cached ANTs matrix to the image and to
# the landmarks, and writes outputs/Affine_MammoRegNet_<seed>/.
# "FD decoder" is the first-stage checkpoint evaluated by
# inference_original.py --coarse_deform. That script writes
# outputs/MammoRegNet_deformable_<seed>/.
# "Second stage, no fusion" is the training-matched control: the second stage
# trained without fusion blocks. Its folder is the one written by
# main_train_dvff.py --coarse_deform --no_fusion --pretrained_path ...
METHODS = [
    ("Unregistered", "Unreg", ("benchmark",)),
    ("Affine", "Affine", ("benchmark",)),
    ("SyN", "SyN", ("benchmark",)),
    ("CurCoords", "CurCoords", ("benchmark",)),
    ("IDIR", "IDIR", ("benchmark",)),
    ("VoxelMorph", "voxelmorph", ("benchmark",)),
    ("TransMorph", "transmorph", ("benchmark",)),
    ("MammoRegNet", "MammoRegNet_{seed}", ("benchmark", "ablation")),
    ("Affine+MRN", "Affine_MammoRegNet_{seed}", ("benchmark", "ablation")),
    ("FD decoder", "MammoRegNet_deformable_{seed}", ("ablation",)),
    ("Second stage, no fusion", "DVFF_no_fusion_{seed}", ("ablation",)),
    ("DVFF fusion off", "DVFF_fusion_off_{seed}", ("benchmark", "ablation")),
    ("DVFF", "DVFF_{seed}", ("benchmark", "ablation")),
    ("DVFF no affine", "DVFF_no_affine_{seed}", ("ablation",)),
]

FAMILIES = ("benchmark", "ablation")


def patient_id(image_name: str) -> str:
    stem = image_name.replace("\\", "/").rsplit("/", 1)[-1]
    return re.sub(r"-\d+\.png$", "", stem, flags=re.IGNORECASE)


def per_image_rtre_percent(folder: Path) -> dict[str, float]:
    left = parse_annotations(folder / "moving_landmarks.xml")
    right = parse_annotations(folder / "fixed_landmarks_1.xml")
    valid = parse_annotations(folder / "fixed_landmarks_2.xml")
    out = {}
    for name in set(left) & set(right) & set(valid):
        avg_lr, c_lr = compute_average_normalized_distance(left[name], right[name])
        avg_lv, c_lv = compute_average_normalized_distance(left[name], valid[name])
        if c_lr == 0 and c_lv == 0:
            continue
        if c_lr == 0:
            score = avg_lv
        elif c_lv == 0:
            score = avg_lr
        else:
            score = 0.5 * (avg_lr + avg_lv)
        out[name] = 100.0 * score
    return out


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def holm(pvalues: list[float]) -> list[float]:
    n = len(pvalues)
    order = np.argsort(pvalues)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, pvalues[idx] * (n - rank))
        running = max(running, val)
        adj[idx] = running
    return adj.tolist()


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.2e}{stars(p)}"
    return f"{p:.4f}{stars(p)}"


def md_escape(name: str) -> str:
    return name.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patient-level Wilcoxon signed-rank tests on rTRE, within each reported family."
    )
    parser.add_argument("--outputs_root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Training seed that names the per-run output folders",
    )
    args = parser.parse_args()

    labels = [m[0] for m in METHODS]
    scores = {}
    for label, folder_template, _families in METHODS:
        folder_name = folder_template.format(seed=args.seed)
        folder = args.outputs_root / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing outputs folder: {folder}")
        scores[label] = per_image_rtre_percent(folder)
        print(f"Loaded {label:20s}  n_images={len(scores[label]):3d}  ({folder_name})")

    common = set.intersection(*[set(s) for s in scores.values()])
    pat_scores = {m: defaultdict(list) for m in labels}
    for name in common:
        pid = patient_id(name)
        for m in labels:
            pat_scores[m][pid].append(scores[m][name])

    patients = sorted(set.intersection(*[set(pat_scores[m]) for m in labels]))
    X = {
        m: np.array([float(np.mean(pat_scores[m][p])) for p in patients])
        for m in labels
    }

    print(f"\nImages (common): {len(common)}")
    print(f"Patients: {len(patients)}")
    print("\nPer-patient mean rTRE (%):")
    for m in labels:
        print(f"  {m:24s}  {X[m].mean():.3f} +- {X[m].std():.3f}")

    def compare(a: str, b: str) -> dict:
        diff = X[a] - X[b]
        two = wilcoxon(
            X[a], X[b], zero_method="wilcox", alternative="two-sided", method="auto"
        )
        greater = wilcoxon(
            X[a], X[b], zero_method="wilcox", alternative="greater", method="auto"
        )
        if np.median(diff) > 0:
            better = b
        elif np.median(diff) < 0:
            better = a
        else:
            better = "tie"
        return {
            "A": a,
            "B": b,
            "median_A_minus_B": float(np.median(diff)),
            "mean_A_minus_B": float(np.mean(diff)),
            "stat": float(two.statistic),
            "p_two_sided": float(two.pvalue),
            "p_A_worse_than_B": float(greater.pvalue),
            "better_lower_rTRE": better,
        }

    rows = []
    family_labels = {}
    for family in FAMILIES:
        members = [label for label, _folder, families in METHODS if family in families]
        family_labels[family] = members
        family_rows = [compare(a, b) for a, b in itertools.combinations(members, 2)]
        for row, hp in zip(family_rows, holm([r["p_two_sided"] for r in family_rows])):
            row["family"] = family
            row["p_holm"] = float(hp)
        rows.extend(family_rows)

    out_dir = args.outputs_root
    csv_path = out_dir / "wilcoxon_rTRE_all_pairwise.csv"
    md_path = out_dir / "wilcoxon_rTRE_all_pairwise.md"

    import csv

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def matrix_md(members: list[str], values: dict, fmt) -> str:
        header = "| A \\\\ B | " + " | ".join(md_escape(m) for m in members) + " |"
        sep = "|" + "---|" * (len(members) + 1)
        lines = [header, sep]
        for a in members:
            cells = [md_escape(a)]
            for b in members:
                if a == b:
                    cells.append("—")
                else:
                    cells.append(fmt(values[(a, b)]))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def family_values(family: str):
        raw, adjusted, median = {}, {}, {}
        for r in rows:
            if r["family"] != family:
                continue
            raw[(r["A"], r["B"])] = raw[(r["B"], r["A"])] = r["p_two_sided"]
            adjusted[(r["A"], r["B"])] = adjusted[(r["B"], r["A"])] = r["p_holm"]
            median[(r["A"], r["B"])] = r["median_A_minus_B"]
            median[(r["B"], r["A"])] = -r["median_A_minus_B"]
        return raw, adjusted, median

    mean_rows = "\n".join(
        f"| {md_escape(m)} | {X[m].mean():.3f} ± {X[m].std():.3f} |" for m in labels
    )
    pair_lines = [
        "| Family | Pair | Median(A−B) | Mean(A−B) | p (two-sided) | p (Holm) | Better (lower rTRE) |",
        "|--------|------|-------------|-----------|---------------|----------|---------------------|",
    ]
    for r in rows:
        sig = "n.s." if r["p_two_sided"] >= 0.05 else r["better_lower_rTRE"]
        pair_lines.append(
            f"| {r['family']} | {md_escape(r['A'])} vs {md_escape(r['B'])} | "
            f"{r['median_A_minus_B']:+.3f} | {r['mean_A_minus_B']:+.3f} | "
            f"{r['p_two_sided']:.4e}{stars(r['p_two_sided'])} | "
            f"{r['p_holm']:.4e}{stars(r['p_holm'])} | {sig} |"
        )

    family_sections = []
    for family in FAMILIES:
        members = family_labels[family]
        raw, adjusted, median = family_values(family)
        n_pairs = len(members) * (len(members) - 1) // 2
        family_sections.append(
            f"## {family.capitalize()} family ({len(members)} conditions, {n_pairs} comparisons)\n\n"
            f"### Two-sided p-value matrix (uncorrected)\n\n"
            f"{matrix_md(members, raw, fmt_p)}\n\n"
            f"### Two-sided p-value matrix (Holm)\n\n"
            f"{matrix_md(members, adjusted, fmt_p)}\n\n"
            f"### Median difference matrix (A − B), rTRE %\n\n"
            f"{matrix_md(members, median, lambda x: f'{x:+.3f}')}"
        )

    md = f"""# Wilcoxon signed-rank tests on patient-level rTRE

- N patients: **{len(patients)}**
- N images (common to all methods): **{len(common)}**
- Aggregation: mean image rTRE (%) per patient
- Test: paired Wilcoxon signed-rank, two-sided
- Multiple comparisons: Holm correction inside each family. The benchmark family has 55 comparisons. The ablation family has 21.
- Cell (row A, column B): p-value for A vs B; stars: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001
- Median difference A−B > 0 means A has higher rTRE (worse) than B

Conditions: Affine+MRN = MammoRegNet preceded by the external ANTs affine;
FD decoder = the first-stage MammoRegNet checkpoint, evaluated single-view with a deformable coarsest head;
Second stage, no fusion = the same second stage trained without fusion blocks;
DVFF fusion off = trained with fusion, fusion disabled at inference;
DVFF = the complete proposed method;
DVFF no affine = the complete method without external affine pre-alignment.

## Patient-level mean rTRE (%)

| Method | Mean ± SD |
|--------|-----------|
{mean_rows}

{chr(10).join(family_sections)}

## Pairs

{chr(10).join(pair_lines)}
"""
    md_path.write_text(md, encoding="utf-8")

    print(f"\nWilcoxon signed-rank (paired, two-sided), {len(rows)} pairs")
    print(
        f"{'family':12s} {'pair':46s} {'median(A-B)':>11s} {'p':>12s} {'p_holm':>12s}  better"
    )
    for r in rows:
        sig = "n.s." if r["p_two_sided"] >= 0.05 else r["better_lower_rTRE"]
        print(
            f"{r['family']:12s} {r['A']+' vs '+r['B']:46s} {r['median_A_minus_B']:+11.3f} "
            f"{r['p_two_sided']:12.4e} {r['p_holm']:12.4e}  {sig}"
        )
    print(f"\nSaved {csv_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
