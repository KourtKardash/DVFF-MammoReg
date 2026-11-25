"""Patient-level bootstrap for the risk-prediction tables.

The point estimate of each metric is computed on the full test set. The 95%
interval is the 2.5 and 97.5 percentiles of 1,000 resamples of patients
(seed 2023). A drawn patient contributes every one of her samples, and a
patient drawn k times contributes them k times. The same draw is used for
every registration backend.

Paired differences use that shared draw. The two-sided p value is twice the
share of draws on the opposite side of zero from the full-test difference.
For each metric, Holm adjustment is applied to the five comparisons with the
unregistered condition. That is the adjustment reported in Supplementary
Table S3. Comparisons between already aligned backends are not part of that
family, so their Holm column is left empty.

Each input is the predictions_per_sample.csv written by one risk run. Place
them in one directory under these names:

    Unregistered   predictions_per_sample_unreg.csv
    Affine         predictions_per_sample_affine.csv
    Affine+MRN     predictions_per_sample_aff_mrn.csv
    MammoRegNet    predictions_per_sample_mrn.csv
    Fusion off     predictions_per_sample_fusion_off.csv
                   or predictions_per_sample_no_fusion.csv
    Proposed       predictions_per_sample_proposed.csv

Required columns: sample_key, patient_id, event_time_index, event_observed,
pred_y0, pred_y1, pred_y2, pred_y3, pred_y4.

Example:
    python analysis/patient_bootstrap_risk.py --predictions_dir outputs/risk
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

N_BOOT = 1000
SEED = 2023
PRED_COLS = [f"pred_y{i}" for i in range(5)]
FILES = [
    ("Unregistered", ("predictions_per_sample_unreg.csv",)),
    ("Affine", ("predictions_per_sample_affine.csv",)),
    ("Affine+MRN", ("predictions_per_sample_aff_mrn.csv",)),
    ("MammoRegNet", ("predictions_per_sample_mrn.csv",)),
    ("Fusion off", ("predictions_per_sample_fusion_off.csv", "predictions_per_sample_no_fusion.csv")),
    ("Proposed", ("predictions_per_sample_proposed.csv",)),
]
PAIRS = [
    ("Affine", "Unregistered"),
    ("Affine+MRN", "Unregistered"),
    ("MammoRegNet", "Unregistered"),
    ("Fusion off", "Unregistered"),
    ("Proposed", "Unregistered"),
    ("Proposed", "Affine+MRN"),
    ("Proposed", "Fusion off"),
    ("Affine", "Affine+MRN"),
]


def get_censoring_dist(times, event_observed) -> dict:
    """Kaplan-Meier S(t)=P(T>t), floored at 1e-6."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(event_observed, dtype=float)
    order = np.argsort(times, kind="mergesort")
    t_sorted = times[order]
    e_sorted = events[order]
    n = len(t_sorted)
    surv = 1.0
    dist = {}
    i = 0
    while i < n:
        ti = t_sorted[i]
        j = i
        while j < n and t_sorted[j] == ti:
            j += 1
        at_risk = n - i
        deaths = float(e_sorted[i:j].sum())
        if at_risk > 0:
            surv *= 1.0 - deaths / at_risk
        dist[int(ti)] = max(surv, 1e-6)
        i = j
    return dist


def _add_pairs(scores, now_idx, early_idx, t, weight):
    """Count pairs of people at now_idx against events at early_idx.

    A higher score is a lower risk. The later person is counted correct when
    that score exceeds the earlier event.
    """
    if int(now_idx.sum()) == 0 or int(early_idx.sum()) == 0:
        return 0.0, 0.0, 0.0
    diff = scores[now_idx, int(t)][:, None] - scores[early_idx, int(t)][None, :]
    return (
        weight * diff.size,
        weight * float((diff > 0).sum()),
        weight * float((diff == 0).sum()),
    )


def cindex_fast(times, predictions, observed, censoring_dist) -> float:
    """Uno IPCW C-index, with the same pairs as the training pipeline.

    An event at year k is comparable with a control censored in that same
    year. Events are compared only with events from strictly earlier years.
    """
    scores = 1.0 - np.asarray(predictions, dtype=float)
    times = np.asarray(times, dtype=float)
    observed = np.asarray(observed, dtype=float).ravel()
    num_correct = num_tied = num_pairs = 0.0
    for t in np.unique(times):
        at_t = times == t
        died_earlier = (observed == 1.0) & (times < t)
        died_now = (observed == 1.0) & at_t
        censored_now = (observed == 0.0) & at_t
        weight = 1.0 / (float(censoring_dist[int(t)]) ** 2)
        for now_idx, early_idx in (
            (at_t, died_earlier),
            (censored_now, died_now),
        ):
            pairs, correct, tied = _add_pairs(scores, now_idx, early_idx, t, weight)
            num_pairs += pairs
            num_correct += correct
            num_tied += tied
    if num_pairs == 0:
        raise ZeroDivisionError("No admissible pairs in the dataset.")
    return (num_correct + num_tied / 2.0) / num_pairs


def yearly_auc(preds: np.ndarray, times: np.ndarray, observed: np.ndarray) -> np.ndarray:
    out = np.full(5, np.nan)
    for k in range(5):
        positive = (observed == 1.0) & (times <= k)
        negative = times >= k
        include = positive | negative
        labels = positive[include]
        if labels.size == 0 or labels.min() == labels.max():
            continue
        out[k] = roc_auc_score(labels.astype(int), preds[include, k])
    return out


def two_sided_p(point: float, boot: np.ndarray) -> float:
    finite = boot[np.isfinite(boot)]
    if point > 0:
        tail = np.mean(finite < 0)
    elif point < 0:
        tail = np.mean(finite > 0)
    else:
        return 1.0
    return float(min(1.0, 2.0 * tail))


def holm(pvalues: list[float]) -> list[float]:
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, pvalues[idx] * (n - rank))
        running = max(running, val)
        adjusted[idx] = running
    return adjusted.tolist()


def resolve_file(directory: Path, candidates: tuple[str, ...]) -> Path:
    for name in candidates:
        path = directory / name
        if path.is_file():
            return path
    joined = ", ".join(candidates)
    raise FileNotFoundError(f"None of these files is in {directory}: {joined}")


def load_aligned(directory: Path):
    frames = {}
    for name, candidates in FILES:
        df = pd.read_csv(resolve_file(directory, candidates))
        df["sample_key"] = df["sample_key"].astype(str)
        df["patient_id"] = df["patient_id"].astype(str)
        frames[name] = df.drop_duplicates("sample_key", keep="first")

    keys = sorted(set.intersection(*(set(df["sample_key"]) for df in frames.values())))
    ref = frames["Unregistered"].set_index("sample_key").loc[keys]
    times = ref["event_time_index"].to_numpy(dtype=float)
    observed = ref["event_observed"].to_numpy(dtype=float)
    patients = ref["patient_id"].to_numpy()
    preds = {}
    for name, df in frames.items():
        aligned = df.set_index("sample_key").loc[keys]
        if name != "Unregistered":
            if not np.array_equal(aligned["event_time_index"].to_numpy(dtype=float), times):
                raise RuntimeError(f"event times differ for {name}")
            if not np.array_equal(aligned["event_observed"].to_numpy(dtype=float), observed):
                raise RuntimeError(f"event labels differ for {name}")
        preds[name] = aligned[PRED_COLS].to_numpy(dtype=float)
    return patients, times, observed, preds


def patient_index_lists(patients: np.ndarray):
    _uniq, inverse = np.unique(patients, return_inverse=True)
    splits = [np.flatnonzero(inverse == i) for i in range(len(_uniq))]
    return splits, _uniq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--n_boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    patients, times, observed, preds = load_aligned(args.predictions_dir)
    censoring = get_censoring_dist(times, observed)
    splits, uniq = patient_index_lists(patients)
    n_patients = len(uniq)
    names = [name for name, _candidates in FILES]

    point_c = {name: cindex_fast(times, preds[name], observed, censoring) for name in names}
    point_auc = {name: yearly_auc(preds[name], times, observed) for name in names}

    rng = np.random.default_rng(args.seed)
    boot_c = {name: np.empty(args.n_boot) for name in names}
    boot_auc = {name: np.empty((args.n_boot, 5)) for name in names}
    for draw in range(args.n_boot):
        drawn = rng.integers(0, n_patients, size=n_patients)
        idx = np.concatenate([splits[i] for i in drawn])
        t_b = times[idx]
        o_b = observed[idx]
        for name in names:
            p_b = preds[name][idx]
            boot_c[name][draw] = cindex_fast(t_b, p_b, o_b, censoring)
            boot_auc[name][draw] = yearly_auc(p_b, t_b, o_b)
        if (draw + 1) % 100 == 0:
            print(f"bootstrap {draw + 1}/{args.n_boot}", flush=True)

    out_dir = args.out_dir or args.predictions_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "patient_bootstrap_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["backend", "metric", "point", "ci_low", "ci_high"])
        for name in names:
            c_lo, c_hi = np.percentile(boot_c[name], [2.5, 97.5])
            writer.writerow([name, "C-index", point_c[name], float(c_lo), float(c_hi)])
            for k in range(5):
                finite = boot_auc[name][:, k]
                finite = finite[np.isfinite(finite)]
                lo, hi = np.percentile(finite, [2.5, 97.5])
                writer.writerow([name, f"AUC_{k + 1}", float(point_auc[name][k]), float(lo), float(hi)])

    paired_rows = []
    for left, right in PAIRS:
        series = [("C-index", point_c[left] - point_c[right], boot_c[left] - boot_c[right])]
        for k in range(5):
            series.append(
                (
                    f"AUC_{k + 1}",
                    float(point_auc[left][k] - point_auc[right][k]),
                    boot_auc[left][:, k] - boot_auc[right][:, k],
                )
            )
        for label, diff, boot in series:
            finite = boot[np.isfinite(boot)]
            lo, hi = np.percentile(finite, [2.5, 97.5])
            paired_rows.append(
                {
                    "backend_A": left,
                    "backend_B": right,
                    "metric": label,
                    "diff": diff,
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "p": two_sided_p(diff, boot),
                    "p_holm": "",
                }
            )

    for metric in ["C-index"] + [f"AUC_{k}" for k in range(1, 6)]:
        family = [
            row for row in paired_rows
            if row["metric"] == metric and row["backend_B"] == "Unregistered"
        ]
        for row, adjusted in zip(family, holm([row["p"] for row in family])):
            row["p_holm"] = adjusted

    paired_path = out_dir / "patient_bootstrap_paired.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0].keys()))
        writer.writeheader()
        writer.writerows(paired_rows)

    print(f"patients={n_patients} samples={len(times)} events={int(observed.sum())}")
    print(f"wrote {metrics_path}")
    print(f"wrote {paired_path}")


if __name__ == "__main__":
    main()
