"""Shared survival label / stratum logic for longitudinal risk datasets."""

from __future__ import annotations

import pandas as pd


def compute_survival_stratum(
    years_to_cancer,
    years_last_followup,
    n_years: int = 5,
) -> tuple[bool, int]:
    """
    Assign a sample to (event vs control) and a 1..n_years horizon bucket.

    Matches label logic in dataset_csaw.py / dataset_embed.py:
    - Cases: bucket = calendar year of event (1 = within year 1, ..., 5 = within year 5)
    - Controls: bucket = years of known follow-up after the current exam (capped at n_years)
    """
    if years_to_cancer == 0:
        years_to_cancer = 1
    if pd.isna(years_to_cancer):
        years_to_cancer = n_years + 1

    years_last_followup = int(years_last_followup)
    time_to_cancer = int(years_to_cancer) - 1
    any_cancer = time_to_cancer < n_years

    if any_cancer:
        bucket = time_to_cancer + 1
    else:
        if years_last_followup == 0:
            years_last_followup = 1
        bucket = min(years_last_followup, n_years)

    bucket = max(1, min(n_years, int(bucket)))
    return bool(any_cancer), bucket
