import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
from collections import defaultdict
import random
import numpy as np

from dataloaders.survival_labels import compute_survival_stratum
from registration.affine_cache import affine_mat_path, affine_mats_exist
from registration.affine_flow import apply_or_compute_affine
from registration.factory import normalize_registration_method


class BreastCancerRiskDatasetCSAWCC(Dataset):
    def __init__(
        self,
        csv_file,
        image_dir,
        mode,
        transforms=None,
        n_years=5,
        registration_method: str | None = None,
        affine_cache_root: str | None = None,
        ants_prewarp: bool = False,
        prewarp_in_loader: bool = False,
        compute_missing_affine: bool = False,
    ):
        self.data = pd.read_csv(csv_file, low_memory=False)
        self.csv_data = self.data.set_index('file_name').to_dict(orient='index')
        self.data_dir = image_dir
        self.transform = transforms
        self.n_years = n_years
        self.mode = mode
        self.registration_method = normalize_registration_method(
            registration_method or "mammoregnet"
        )
        self.affine_cache_root = affine_cache_root
        self.require_affine_mats = (
            self.registration_method == "ants_affine"
            or ants_prewarp
        )
        # When enabled, warp the prior images with the cached ANTs affine here in
        # the (CPU, multi-worker) dataloader instead of in the model forward pass.
        # This runs before the horizontal flip so the aligned pair stays consistent,
        # and lets the model apply only the residual (non-rigid) DL flow.
        self.prewarp_in_loader = bool(prewarp_in_loader) and self.require_affine_mats
        # Estimate a missing .mat on the fly (only possible in the loader path,
        # where both images are available). When enabled we keep samples without a
        # cached transform instead of skipping them.
        self.compute_missing_affine = bool(compute_missing_affine) and self.prewarp_in_loader
        if self.require_affine_mats and not self.affine_cache_root:
            raise ValueError(
                "affine_cache_root is required for ants_affine or when ants_prewarp is enabled"
            )
        self._max_mat_skip_attempts = 32
        self.image_data = self._load_image_data()
        self.patient_view_pairs = self._create_longitudinal_samples()
        if self.require_affine_mats and not self.compute_missing_affine:
            self._filter_samples_with_affine_mats()
        self.num_elements = len(self.patient_view_pairs)

    def _prewarp_prior(
        self, prior_tensor, current_tensor, patient_id, laterality, view, prior_year, current_year
    ):
        """Warp a prior image [1, H, W] into current space with the cached ANTs affine.

        If the .mat is missing and ``compute_missing_affine`` is on, the affine is
        estimated on the fly (and cached) using ``current_tensor`` as the fixed image.
        """
        mat = affine_mat_path(
            self.affine_cache_root,
            self.mode,
            patient_id,
            laterality,
            view,
            prior_year,
            current_year,
        )
        prior_arr = prior_tensor.squeeze(0).numpy()  # [1, H, W] -> [H, W]
        current_arr = current_tensor.squeeze(0).numpy()
        warped = apply_or_compute_affine(
            prior_arr,
            current_arr,
            str(mat),
            compute_missing=self.compute_missing_affine,
        )
        return torch.from_numpy(warped).unsqueeze(0)

    def map_density(self, value):
        mapping = {
            'A': 1,
            'B': 2,
            'C': 3,
        }
        index = mapping.get(value, -1)  # -1 for NA or unknown values
        return torch.tensor(index, dtype=torch.long)

    def _load_image_data(self):
        """
        Load image data into a nested dictionary:
        { patient_id -> { exam_year -> { view -> metadata } } }
        """
        image_data = defaultdict(lambda: defaultdict(dict))  # Nested defaultdict
        csv_data = self.csv_data
        image_dir_path = os.path.join(self.data_dir, self.mode).replace("\\", "/")
        for filename in os.listdir(image_dir_path):
            if filename in csv_data:
                file_info = csv_data[filename]
                # Extract required fields
                patient_id = str(file_info['patient_id'])
                exam_year = file_info['exam_year']
                laterality = file_info['laterality']  # Left or Right
                view = file_info['viewposition']  # CC or MLO
                years_to_cancer = file_info['years_to_cancer']
                years_last_followup = file_info['years_to_last_followup']
                density = file_info.get('density_group', file_info.get('density'))
                # Map laterality to L and R for consistency
                laterality = 'L' if laterality == 'Left' else 'R' if laterality == 'Right' else None
                # Skip if laterality is invalid
                if laterality is None:
                    print(f"Skipping file due to invalid laterality: {filename}")
                    continue
                # Group by patient_id, exam_year, and view
                key = f"{laterality}_{view}"
                image_data[patient_id][exam_year][key] = {
                    'filename': filename,
                    'years_to_cancer': years_to_cancer,
                    'years_last_followup': years_last_followup,
                    'density': density,
                }
        return image_data


    def _create_longitudinal_samples(self):
        """
        Pre-compute longitudinal samples per patient.
        """
        samples = []
        for patient_id, year_dict in self.image_data.items():
            if len(year_dict) < 2:
                continue  # Skip patients with fewer than 2 exam years
            # Sort years in ascending order
            sorted_years = sorted(year_dict.keys())
            for idx in range(1, len(sorted_years)):
                prior_year = sorted_years[idx - 1]
                current_year = sorted_years[idx]
                # Check for both CC and MLO views for the current and prior years
                for laterality in ['L', 'R']:
                    cc_view = f"{laterality}_CC"
                    mlo_view = f"{laterality}_MLO"
                    if (cc_view in year_dict[current_year] and mlo_view in year_dict[current_year] and
                            cc_view in year_dict[prior_year] and mlo_view in year_dict[prior_year]):
                        samples.append((patient_id, laterality, current_year, prior_year))
        random.shuffle(samples)
        print(f"Found {len(samples)} valid longitudinal multi-view samples in '{self.mode}' dataset.")
        return samples

    def _affine_mats_exist_for_pair(
        self,
        patient_id: str,
        laterality: str,
        current_year,
        prior_year,
    ) -> bool:
        return affine_mats_exist(
            self.affine_cache_root,
            self.mode,
            patient_id,
            laterality,
            prior_year,
            current_year,
        )

    def _filter_samples_with_affine_mats(self) -> None:
        before = len(self.patient_view_pairs)
        self.patient_view_pairs = [
            sample
            for sample in self.patient_view_pairs
            if self._affine_mats_exist_for_pair(*sample)
        ]
        skipped = before - len(self.patient_view_pairs)
        if skipped:
            print(
                f"Skipped {skipped} samples in '{self.mode}' without cached affine .mat "
                f"(CC+MLO) under {self.affine_cache_root}."
            )
        if not self.patient_view_pairs:
            raise RuntimeError(
                f"No samples left in '{self.mode}' after filtering missing affine .mat files."
            )

    def __len__(self):
        return self.num_elements

    def survival_stratum_for_index(self, idx: int) -> tuple[bool, int]:
        """Case/control flag and 1..n_years horizon bucket (for balanced batch sampling)."""
        patient_id, laterality, current_year, _prior_year = self.patient_view_pairs[idx]
        cc_view = f"{laterality}_CC"
        metadata = self.image_data[patient_id][current_year][cc_view]
        return compute_survival_stratum(
            metadata["years_to_cancer"],
            metadata["years_last_followup"],
            self.n_years,
        )

    def __getitem__(self, idx):
        """
        Fetches a complete longitudinal multi-view sample.
        Returns a dictionary containing the current and prior CC and MLO images.
        """
        if self.require_affine_mats and not self.compute_missing_affine:
            for attempt in range(self._max_mat_skip_attempts):
                if attempt == 0:
                    sample_idx = idx
                elif self.mode == "train":
                    sample_idx = random.randrange(len(self))
                else:
                    sample_idx = (idx + attempt) % len(self)
                sample = self.patient_view_pairs[sample_idx]
                if not self._affine_mats_exist_for_pair(*sample):
                    continue
                return self._load_sample(*sample)
            raise RuntimeError(
                f"Could not find a sample with valid affine .mat files after "
                f"{self._max_mat_skip_attempts} attempts (started at idx={idx})."
            )
        sample = self.patient_view_pairs[idx]
        return self._load_sample(*sample)

    def _load_sample(
        self,
        patient_id: str,
        laterality: str,
        current_year,
        prior_year,
    ):
        # Retrieve all four filenames
        cc_view = f"{laterality}_CC"
        mlo_view = f"{laterality}_MLO"
        current_cc_file = self.image_data[patient_id][current_year][cc_view]['filename']
        current_mlo_file = self.image_data[patient_id][current_year][mlo_view]['filename']
        prior_cc_file = self.image_data[patient_id][prior_year][cc_view]['filename']
        prior_mlo_file = self.image_data[patient_id][prior_year][mlo_view]['filename']

        def load_raw(filename):
            img_path = os.path.join(self.data_dir, self.mode, filename)
            img_pil = Image.open(img_path)
            img_np = np.array(img_pil)
            img_tensor = torch.from_numpy(img_np).to(torch.float32) / 255.0
            return img_tensor.unsqueeze(0)

        current_cc_raw = load_raw(current_cc_file)
        current_mlo_raw = load_raw(current_mlo_file)
        prior_cc_raw = load_raw(prior_cc_file)
        prior_mlo_raw = load_raw(prior_mlo_file)
        if self.prewarp_in_loader:
            prior_cc_raw = self._prewarp_prior(
                prior_cc_raw, current_cc_raw, patient_id, laterality, "CC", prior_year, current_year
            )
            prior_mlo_raw = self._prewarp_prior(
                prior_mlo_raw, current_mlo_raw, patient_id, laterality, "MLO", prior_year, current_year
            )

        # Stack the four views into a single [4, 1, H, W] batch and augment them
        # together. With same_on_batch=True on the augmentation pipeline this applies
        # ONE random geometric (crop/affine) + intensity transform identically to
        # current and prior (both CC and MLO), so their geometry stays aligned.
        # Augmenting each view independently (the old behaviour) desynced the
        # current/prior pair and defeated registration.
        images = torch.stack(
            [current_cc_raw, current_mlo_raw, prior_cc_raw, prior_mlo_raw]
        )
        if self.transform is not None:
            images = self.transform(images)
        if self.mode == "train" and torch.rand(1).item() < 0.5:
            images = torch.flip(images, dims=[-1])  # consistent horizontal flip
        current_image_cc = images[0]
        current_image_mlo = images[1]
        previous_image_cc = images[2]
        previous_image_mlo = images[3]

        affine_mat_cc = ""
        affine_mat_mlo = ""
        if self.affine_cache_root:
            affine_mat_cc = str(
                affine_mat_path(
                    self.affine_cache_root,
                    self.mode,
                    patient_id,
                    laterality,
                    "CC",
                    prior_year,
                    current_year,
                )
            )
            affine_mat_mlo = str(
                affine_mat_path(
                    self.affine_cache_root,
                    self.mode,
                    patient_id,
                    laterality,
                    "MLO",
                    prior_year,
                    current_year,
                )
            )

        # Fetch metadata (primarily based on the current screening date)
        current_metadata = self.image_data[patient_id][current_year][cc_view]
        years_last_followup_raw = current_metadata['years_last_followup']
        years_to_cancer_raw = current_metadata['years_to_cancer']
        years_last_followup = years_last_followup_raw
        time_to_cancer = years_to_cancer_raw
        density = self.map_density(current_metadata['density'])

        time_gap = abs(current_year - prior_year)
        if time_gap > 5:  # Limit the time gap to 5 years max
            time_gap = 5

        if time_to_cancer == 0:
            time_to_cancer = 1
        if pd.isna(time_to_cancer):
            time_to_cancer = self.n_years + 1
        years_last_followup = int(years_last_followup)
        time_to_cancer = int(time_to_cancer)
        time_to_cancer = time_to_cancer - 1
        any_cancer = time_to_cancer < self.n_years

        target = np.zeros(self.n_years, dtype=np.int8)
        if any_cancer:
            time_at_event = int(time_to_cancer)
            event_observed = 1
            target[time_at_event:] = 1
            # Observed event: the full trajectory is known (all post-event years are
            # positive), so every year is supervised. Truncating the mask here would
            # drop the post-event positives and starve the later-horizon signal.
            y_mask = np.ones(self.n_years, dtype=np.int8)
        else:
            if years_last_followup == 0:
                years_last_followup = 1
            time_at_event = int(min(years_last_followup, self.n_years) - 1)
            event_observed = 0
            # Censored control: only supervise years we actually observed cancer-free.
            y_mask = np.array(
                [1] * (time_at_event + 1) + [0] * (self.n_years - (time_at_event + 1)),
                dtype=np.int8,
            )[:self.n_years]

        # Return the data
        return {
            'current_image_cc': current_image_cc,
            'current_image_mlo': current_image_mlo,
            'previous_image_cc': previous_image_cc,
            'previous_image_mlo': previous_image_mlo,
            'current_image_id_cc': current_cc_file,
            'current_image_id_mlo': current_mlo_file,
            'previous_image_id_cc': prior_cc_file,
            'previous_image_id_mlo': prior_mlo_file,
            'time_gap': torch.tensor(time_gap, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'y_mask': torch.tensor(y_mask, dtype=torch.float32),
            'event_observed': torch.tensor(event_observed, dtype=torch.float32),
            'event_times': torch.tensor(time_at_event, dtype=torch.float32),
            'density': density,
            'patient_id': str(patient_id),
            'laterality': str(laterality),
            'prior_year': int(prior_year),
            'current_year': int(current_year),
            # Stable key for joining predictions across registration methods / folds.
            'sample_key': f"{patient_id}|{laterality}|{prior_year}|{current_year}",
            'years_to_cancer_raw': (
                float(years_to_cancer_raw)
                if not pd.isna(years_to_cancer_raw)
                else float("nan")
            ),
            'years_last_followup_raw': (
                float(years_last_followup_raw)
                if not pd.isna(years_last_followup_raw)
                else float("nan")
            ),
            'affine_mat_cc': affine_mat_cc,
            'affine_mat_mlo': affine_mat_mlo,
        }


