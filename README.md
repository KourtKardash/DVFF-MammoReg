# Dual-View Feature Fusion for Longitudinal Mammography Registration

Official code for the article

> S. Krasnova, E. Starikova, I. Naletov, A. Krylov, D. Sorokin.
> **Improving Longitudinal Mammography Alignment Through Dual-View Feature
> Fusion: Anatomical Validation and Breast Cancer Risk Prediction.**
> *Journal of Imaging Informatics in Medicine*, 2026.
> DOI: [10.1007/s10278-026-02561-3](https://doi.org/10.1007/s10278-026-02561-3)

The repository contains the proposed method, the comparison methods, the
downstream risk-prediction pipeline, and the evaluation code. If you use it,
please cite the article (see [Citation](#citation)).

## Overview

Longitudinal mammograms of the same breast differ by large global motion caused
by repositioning and recompression, and by smaller local tissue change that is
the part of clinical interest. The proposed method separates the two. Global
alignment is delegated to a classical ANTs affine registration performed outside
the network, which frees every decoder level — including the coarsest — to
predict non-rigid motion only. Because both standard projections image the same
breast under different compression geometries, CC and MLO features are exchanged
through cross-view fusion blocks at three pyramid levels, so the deformation for
one projection is estimated with access to context from the other. The fusion
blocks can be switched off at inference without retraining, which covers the
case where only one projection is available.

## Results

Landmark accuracy and breast-mask overlap on the 200-pair evaluation set (means;
see the paper for standard deviations and the remaining metrics).

| Method                | rTRE (%) ↓ | Dice ↑ |
|-----------------------|-----------:|-------:|
| Unregistered          |       4.57 | 0.9018 |
| Affine (ANTs)         |       1.86 | 0.9576 |
| SyN                   |       1.82 | 0.9655 |
| Curvilinear coordinates |     2.90 | 0.9587 |
| IDIR                  |       3.07 | 0.8965 |
| VoxelMorph            |       1.83 | 0.9789 |
| TransMorph            |       1.76 | 0.9850 |
| MammoRegNet           |       1.89 | 0.9940 |
| Affine + MammoRegNet  |       1.69 | 0.9941 |
| **Proposed**          |   **1.57** | **0.9948** |

On the downstream longitudinal risk-prediction task the proposed method gives
the highest C-index of the configurations compared, 71.59% against 67.48% for
unregistered examinations. The paired difference is 4.11 percentage points
(95% CI 0.75–7.31). The unadjusted paired test gives p = 0.022. After Holm
adjustment over the five comparisons with the unregistered condition it is not
significant (p = 0.09). The 5-year AUC rose from 70.40% to 74.97%.

## Installation

The experiments were run in Docker. There are two images, because registration
and risk prediction were trained on different PyTorch releases. Image data,
checkpoints, and outputs stay outside the image and are mounted at run time.

Registration, the comparison methods, and the analysis scripts use PyTorch
2.4.0 with CUDA 12.4 (`requirements.txt`):

```bash
docker build -t dvff-registration:cu124 .
docker run --gpus all --shm-size=8g \
    -v "$PWD/Dataset:/workspace/Dataset" \
    -v "$PWD/outputs:/workspace/outputs" \
    -v "$PWD/affine_transforms:/workspace/affine_transforms" \
    dvff-registration:cu124 \
    python Methods/DVFF/evaluate/inference_dvff.py --coarse_deform --seed 11
```

Risk prediction uses PyTorch 2.5.1 with CUDA 12.1
(`risk_prediction/LMV-Net/requirements.txt`):

```bash
docker build -f Dockerfile.risk -t dvff-risk:cu121 .
docker run --gpus all --shm-size=8g \
    -e CSV_FILE_PATH=/data/metadata.csv \
    -e DATA_ROOT_PATH=/data/images \
    -e OUTPUT_DIR_PATH=/data/outputs \
    -e AFFINE_CACHE_ROOT=/data/affine \
    -e REGISTRATION_CHECKPOINT=/data/checkpoints/model_249.pth \
    -e MIRAI_ENCODER_PATH=/data/mgh_mammo_MIRAI_Base_May20_2019.p \
    -v /path/to/data:/data \
    dvff-risk:cu121 \
    bash /opt/lmvnet/scripts/train.sh
```

`REPO_ROOT` inside that image points at the directory that contains `Methods/`.

Notes:

- Both images set `WANDB_MODE=disabled`. Pass `-e WANDB_MODE=online` and
  `-e WANDB_API_KEY=...` to log a run.
- ANTs writes large intermediate files. Set `ANTS_TMPDIR` to a fast local disk
  if the default temporary directory is small or on a network filesystem.
- A GPU is required for training. Classical ANTs methods run on CPU.

## Data

**No image data is distributed with this repository.** All four source
collections are public and must be obtained from their providers under those
providers' terms.

| Source    | License      | How to obtain |
|-----------|--------------|---------------|
| INBreast  | CC BY-NC 4.0 | On request to the collection authors, DOI [10.1016/j.acra.2011.09.014](https://doi.org/10.1016/j.acra.2011.09.014) |
| KAU-BCMD  | CC0 1.0      | DOI [10.3390/data6110111](https://doi.org/10.3390/data6110111) |
| CSAW-CC   | CC BY 4.0    | Application to the cohort holder, DOI [10.1007/s10278-019-00278-0](https://doi.org/10.1007/s10278-019-00278-0) |
| NLBS      | CC BY 4.0    | FRDR, DOI [10.20383/103.01093](https://doi.org/10.20383/103.01093); use version 3, DOI [10.20383/103.01526](https://doi.org/10.20383/103.01526) |

The pair manifests, the expert landmark annotations, and the breast masks used
here are derived materials. They contain no images and no identifying
information. They are not redistributed in this repository and are available
from the authors on reasonable request. A reader who has obtained the source collections can
reconstruct the evaluation set exactly as it was used.

The **Mirai** image encoder used by the downstream risk-prediction pipeline is a
third-party checkpoint and is **not** included here. Obtain it from the Mirai
authors (<https://github.com/yala/Mirai>) under their terms, then pass
`--mirai_encoder_path <file>` or set `MIRAI_ENCODER_PATH=<file>`.

The weights are distributed separately from the source repository and download
links move, so the file used for the results reported here is identified by its
content rather than by a location:

| | |
|---|---|
| File name | `mgh_mammo_MIRAI_Base_May20_2019.p` |
| Size | 50,651,758 bytes |
| SHA-256 | `e66261f434474686462e5c7035de37452f47e4e7cc8831086be19397383e8af3` |

Verify a downloaded copy with

```bash
sha256sum mgh_mammo_MIRAI_Base_May20_2019.p
```

The Mirai *source* is a separate artefact with its own version: the OncoNet
modules vendored under `risk_prediction/LMV-Net/src/onconet/`, which are needed
to unpickle the encoder, come from commit `3904a9ea` of
<https://github.com/yala/Mirai>. See `NOTICE` for the local modifications.

Expected on-disk layout for the registration experiments:

```
Dataset/
  evaluation/{INBreast,KAU-BCMD,NLBS,CSAW-CC}/<patient>/*.png
  evaluation/metadata_evaluation_correct.csv
  evaluation_masks/...
  moving_landmarks.xml  fixed_landmarks_1.xml  fixed_landmarks_2.xml
  train/  train_masks/                   # training split, no external affine
  affine_train/  affine_train_masks/     # same split after the ANTs affine warp
  affine_evaluation/                     # written by Methods/Affine/run_from_scratch.py --subset evaluation
```

## Usage

All commands below are run from the repository root, inside the
`dvff-registration:cu124` container unless a command says otherwise.

### Training

The proposed method is trained in two stages. The first stage is a single-view
network with a fully deformable decoder — every level predicts non-rigid motion,
including the coarsest, where the affine head is replaced — trained after the
external ANTs affine pre-alignment:

```bash
python Methods/MammoRegNet/train/main_train_mammoregnet.py --coarse_deform
```

The second stage loads that checkpoint into the dual-view architecture, freezes
the encoder, and trains the fusion blocks, the view embeddings and the decoder:

```bash
python Methods/DVFF/train/main_train_dvff.py \
    --coarse_deform \
    --pretrained_path outputs/MammoRegNet_deformable_<seed>/models/model_249.pth \
    --freeze_encoder \
    --fusion_levels 2,3,4 \
    --fusion_warmup_epochs 0
```

The number in a folder name is the `--seed` of that run. With `--coarse_deform`
the first stage writes `outputs/MammoRegNet_deformable_<seed>/models/model_249.pth`
and the second stage writes `outputs/DVFF_<seed>/models/model_249.pth`. Without
the flag the same scripts write `outputs/MammoRegNet_<seed>/` and
`outputs/DVFF_no_affine_<seed>/`. Evaluation reads those paths.

With `--coarse_deform` both stages read `Dataset/affine_train` and the masks in
`Dataset/affine_train_masks`. That split is produced once by

```bash
python Methods/Affine/run_from_scratch.py --subset train
```

The script reads images from `Dataset/train` and masks from `Dataset/train_masks`,
warps the moving image and its mask, and writes the two affine directories.
It also writes `Dataset/affine_train/metadata_train_correct.csv`. `File Path`
in that copy names the warped png files, which is the path both training
loaders look up.
Omitting `--coarse_deform` trains on `Dataset/train` with `Dataset/train_masks`.
That is the training data of ablation rows 1 and 7.

Comparison methods have their own entry points: `Methods/voxelmorph/train_VoxelMorph.py`,
`Methods/transmorph/train_TransMorph.py`, and
`Methods/MammoRegNet/train/main_train_mammoregnet.py`. The classical methods
(`Methods/Affine/`, `Methods/SyN/`, `Methods/CurCoords/`) and `Methods/IDIR/`
are optimized per pair and are not trained.

### Inference

```bash
python Methods/DVFF/evaluate/inference_dvff.py --coarse_deform --seed 11
```

To register each projection on its own with the same fusion-trained weights:

```bash
python Methods/DVFF/evaluate/inference_dvff_fusion_off.py --coarse_deform --seed 11
```

Each entry point writes a JSON summary of all metrics under `outputs/` and
prints the overall and per-collection rTRE.

For the proposed method, the `Runtime` field in that summary times the network
alone; the affine transform is read from the cache. The inference time in
Table 3 adds the ANTs affine step, 2.20 s per pair, to that figure.

The evaluation scripts load checkpoints strictly. A checkpoint that does not
match the requested configuration, for example one trained without fusion
evaluated without `--no_fusion`, or one whose coarsest head disagrees with
`--coarse_deform`, stops the run instead of leaving layers at random weights. A
single-view first-stage checkpoint can be evaluated directly with
`inference_dvff.py --no_fusion --model_path ...`.

### Ablations

The seven rows of the ablation table are configurations of the same code. Mean
rTRE on the 200 evaluation pairs is the value reported in the paper.

| # | Configuration | External affine | Coarsest decoder level | Second stage | Fusion (train) | Fusion (inference) | rTRE (%) | How to run |
|---|---------------|-----------------|------------------------|--------------|----------------|--------------------|---------:|------------|
| 1 | MammoRegNet | no | affine | no | no | no | 1.89 | `inference_original.py` |
| 2 | Affine + MammoRegNet | yes | affine | no | no | no | 1.69 | `inference_original.py --external_affine` (`outputs/Affine_MammoRegNet_<seed>/`, checkpoint `outputs/MammoRegNet_<seed>/`) |
| 3 | Fully deformable decoder | yes | deformable | no | no | no | 1.61 | `inference_original.py --coarse_deform` (`outputs/MammoRegNet_deformable_<seed>/`) |
| 4 | Second stage without fusion | yes | deformable | yes | no | no | 1.59 | train with `--coarse_deform --no_fusion --pretrained_path ... --freeze_encoder`, then `inference_dvff.py --coarse_deform --no_fusion` (`outputs/DVFF_no_fusion_<seed>/`) |
| 5 | Trained with fusion, fusion off | yes | deformable | yes | yes | no | 1.58 | `inference_dvff_fusion_off.py --coarse_deform` |
| 6 | **Proposed** | yes | deformable | yes | yes | yes | **1.57** | `inference_dvff.py --coarse_deform` |
| 7 | Proposed without affine | no | affine | yes | yes | yes | 1.77 | train `main_train_dvff.py --pretrained_path outputs/MammoRegNet_<seed>/models/model_249.pth --freeze_encoder` with no `--coarse_deform`, then `inference_dvff.py` (`outputs/DVFF_no_affine_<seed>/`) |

`--coarse_deform` switches both properties at once, because they are not
independent: the external affine transform is what makes a deformable coarsest
level possible. With the flag, the evaluation script applies the cached ANTs
transform to the moving image and to the landmarks, and builds the decoder
with a deformable head at the coarsest level. With neither flag, no external
transform is applied and the affine head is restored. `--external_affine`
applies that same cached transform to the image and to the landmarks and
leaves the affine head in place. At evaluation every row reads
`Dataset/evaluation`. Row 6 was trained on
`Dataset/affine_train` with `Dataset/affine_train_masks`. Rows 1 and 7 were
trained on `Dataset/train` with `Dataset/train_masks`.

Row 7 removes the external affine step and restores the affine decoder head, so
it differs from row 1 in the dual-view components alone. Its second stage loads
the row-1 checkpoint, freezes the encoder, and trains fusion from epoch 0:

```bash
python Methods/DVFF/train/main_train_dvff.py \
    --pretrained_path outputs/MammoRegNet_<seed>/models/model_249.pth \
    --freeze_encoder \
    --fusion_levels 2,3,4 \
    --fusion_warmup_epochs 0
```

There is no `--coarse_deform` on that command. It reads `Dataset/train` and
`Dataset/train_masks` and writes
`outputs/DVFF_no_affine_<seed>/models/model_249.pth`. Evaluation is
`inference_dvff.py` without the flag.

Row 4 is the training-matched control for row 6: the second stage is repeated
without the fusion blocks. Row 5 keeps those blocks in the checkpoint and switches them off
only at inference.

Row 2 keeps the affine decoder head and still applies the external affine.
`inference_original.py --external_affine` reads `Dataset/evaluation`, warps
the moving image with `affine_transforms/<patient>.mat`, and applies that
same matrix to the landmarks before the network displacement. Results go to
`outputs/Affine_MammoRegNet_<seed>/`. The checkpoint stays in
`outputs/MammoRegNet_<seed>/models/model_249.pth`. The `.mat` files are
written by `Methods/Affine/run_from_scratch.py --subset evaluation`.

### Breast segmentation

Breast masks are needed by the registration loaders, which multiply each image by
its mask. Part of the training set was annotated manually; the rest was completed
by a U-Net trained on those annotations, with a composite loss
`0.5·BCE + 0.5·(1 − Dice)`, a learning rate of `1e-4`, a batch size of 4, and 50
epochs.

```bash
python segmentation/train.py \
    --images_dir Dataset/train \
    --masks_dir Dataset/masks-train \
    --checkpoint segmentation/unet_mammography_segmentation.pth

python segmentation/predict_for_train.py \
    --images_dir Dataset/train \
    --masks_dir Dataset/train_masks \
    --checkpoint segmentation/unet_mammography_segmentation.pth
```

`--masks_dir` mirrors the directory layout of `--images_dir`. Predicted masks are
post-processed by a morphological close and open, hole filling, and keeping the
largest connected component, and are written at the original image resolution.

The trained U-Net weights are **not** stored in this repository. They are
available from the authors on request. The file used here is identified by its
content, so that a copy from any source can be verified:

| | |
|---|---|
| File name | `unet_mammography_segmentation.pth` |
| Size | 124,237,371 bytes |
| SHA-256 | `b3616fb298a0c3276680c1efd70d955ce794145c38d75e6eb708561849067ce0` |

```bash
sha256sum unet_mammography_segmentation.pth
# b3616fb298a0c3276680c1efd70d955ce794145c38d75e6eb708561849067ce0
```

Pass that file to `--checkpoint`. Retraining from the manual annotations with
the command above reproduces an equivalent model, so the weights are a
convenience rather than a requirement.

### Downstream risk prediction

Registration is inserted into a longitudinal risk-prediction pipeline on
CSAW-CC. The registration backend is the only experimental variable.

First cache the ANTs affine transforms for the longitudinal pairs:

```bash
CSV_FILE=... IMAGES_ROOT=... OUTPUT_ROOT=... \
    bash risk_prediction/scripts/cache_affine.sh
```

Then train and evaluate. All paths are passed through the environment; see the
header of each script for the full list.

```bash
CSV_FILE_PATH=... DATA_ROOT_PATH=... OUTPUT_DIR_PATH=... \
AFFINE_CACHE_ROOT=... REGISTRATION_CHECKPOINT=... MIRAI_ENCODER_PATH=... \
    bash risk_prediction/LMV-Net/scripts/train.sh

RISK_RESULTS_DIR=... bash risk_prediction/LMV-Net/scripts/test_dvff.sh
```

Available registration backends are `unreg`, `ants_affine`, `voxelmorph`,
`transmorph`, `mammoregnet` and `dvff` (see
`risk_prediction/LMV-Net/src/registration/factory.py`).

### Analysis

The values reported in Tables 3 and 4 are means over the 200 evaluation pairs.
Statistical comparisons cannot use that level, because pairs from one patient are
not independent observations. For the tests, the rTRE values of a patient's pairs
are averaged, and conditions are compared by a two-sided Wilcoxon signed-rank
test over the 51 patients. Holm adjustment is applied inside each family. The
benchmark family has 11 conditions and 55 comparisons. The ablation family has
the 7 configurations of the ablation table and 21 comparisons.

```bash
python analysis/wilcoxon_rTRE_patients.py --outputs_root outputs --seed 11
```

It reads the landmark files each evaluation entry point writes under
`outputs/<method>/`, so run `scripts/reproduce_table3.sh` and
`scripts/reproduce_table5.sh` first. It writes
`outputs/wilcoxon_rTRE_all_pairwise.csv` with one row per pair and a `family`
column, and `outputs/wilcoxon_rTRE_all_pairwise.md` with the patient-level means
and, for each family, the raw and Holm-adjusted p-value matrices and the
median-difference matrix.

`test_risk_prediction.py` writes `model_results.json`. The C-index and its
interval in that file are computed on individual samples by the training
pipeline, `risk_prediction/LMV-Net/src/utils/c_index.py`. Table 6 and the paired
tests in Supplementary Table S3 are not read from that file. They come from
`analysis/patient_bootstrap_risk.py`, which resamples patients. The point
estimate uses the same pairs as the pipeline. A cancer in year k and a control
censored in that same year are comparable.

```bash
python analysis/patient_bootstrap_risk.py --predictions_dir outputs/risk
```

Each registration backend is one CSV in that directory:

| Backend | File name |
|---|---|
| Unregistered | `predictions_per_sample_unreg.csv` |
| Affine | `predictions_per_sample_affine.csv` |
| Affine + MammoRegNet | `predictions_per_sample_aff_mrn.csv` |
| MammoRegNet | `predictions_per_sample_mrn.csv` |
| Fusion off | `predictions_per_sample_fusion_off.csv` |
| Proposed | `predictions_per_sample_proposed.csv` |

`predictions_per_sample_no_fusion.csv` is accepted in place of the fusion-off
file. Required columns are `sample_key`, `patient_id`, `event_time_index`,
`event_observed`, and `pred_y0` through `pred_y4`.

## Reproducing the paper tables

```bash
bash scripts/reproduce_table3.sh   # accuracy, regularity, cost; image similarity and mask overlap
bash scripts/reproduce_table5.sh   # ablation of the proposed pipeline
```

The tables produced are, by caption:

| Script | Table |
|---|---|
| `reproduce_table3.sh` | "Registration accuracy, deformation regularity, and computational cost" (Table 3) |
| `reproduce_table3.sh` | "Image-similarity and mask-overlap metrics" (Table 4) |
| `reproduce_table5.sh` | "Ablation of the proposed pipeline" (Table 5) |

Table numbers refer to the published article.

Both scripts expect the evaluation data under `Dataset/` and trained weights
under `outputs/`; see the header of each script.

## Comparison methods

| Method | Reference | Notes |
|---|---|---|
| Affine, SyN | Avants et al., ANTs | Run through `antspyx`. For SyN the ANTs affine stage is suppressed so that it starts from the same external affine transform as the other methods. |
| Curvilinear coordinates | Anatomy-driven coordinate method | Re-implemented here; no official implementation exists. |
| IDIR | Wolterink et al., implicit neural representation | Adapted to 2D. |
| VoxelMorph | Balakrishnan et al. | Adapted to 2D. |
| TransMorph | Chen et al. | Adapted to 2D. |
| MammoRegNet | Thrun et al. | Baseline specific to longitudinal mammography, evaluated with and without an external affine transform. One change to the published implementation, in the Jacobian-determinant penalty; see the `NJD` docstring in `Methods/MammoRegNet/train/losses_mammoregnet.py`. |

## Repository layout

```
Methods/
  Affine/  SyN/          ANTs affine baseline and SyN
  CurCoords/             curvilinear coordinate method
  IDIR/                  implicit neural representation, adapted to 2D
  voxelmorph/            adapted to 2D
  transmorph/            adapted to 2D
  MammoRegNet/           baseline, including the corrected Jacobian penalty
  DVFF/                  the proposed method
  Unreg/                 the unregistered reference condition
segmentation/            U-Net that completes the breast masks
risk_prediction/         downstream risk-prediction experiment on CSAW-CC
analysis/                patient-level statistics behind the reported p values
scripts/                 table reproduction scripts
sanity_checks/           offline smoke tests on synthetic data
```

## License and attribution

This repository is released under the Apache License 2.0; see `LICENSE`.

Third-party **code** is vendored under its own licenses. `NOTICE` lists each
component, its upstream project and commit, its license, and the modifications
made here; `THIRD_PARTY_LICENSES` reproduces the full license texts. In short:
VoxelMorph is Apache-2.0; TransMorph, IDIR, MammoRegNet, Mirai/OncoNet, and the
longitudinal risk-prediction pipeline are MIT.

Third-party **model weights** are a separate matter and are governed by the
terms attached at their point of download, which need not match the license on
the corresponding source repository. No third-party weights are redistributed
here. For the one checkpoint this pipeline needs, the Mirai README states that
"Mirai (the trained model) and all code are released under the MIT license";
check the terms presented to you when you obtain the file, since those are the
ones you have agreed to.

## Citation

If you use this code, the evaluation protocol, or the results in your work,
please cite:

```bibtex
@article{krasnova2026dvff,
  author  = {Krasnova, Svetlana and Starikova, Emiliya and Naletov, Ilia and
             Krylov, Andrey and Sorokin, Dmitry},
  title   = {Improving Longitudinal Mammography Alignment Through Dual-View
             Feature Fusion: Anatomical Validation and Breast Cancer Risk
             Prediction},
  journal = {Journal of Imaging Informatics in Medicine},
  year    = {2026},
  doi     = {10.1007/s10278-026-02561-3},
  url     = {https://doi.org/10.1007/s10278-026-02561-3}
}
```

## Acknowledgements

This work was supported by the Ministry of Economic Development of the Russian
Federation in accordance with the subsidy agreement (agreement identifier
000000C313925P4H0002; grant No. 139-15-2025-012).

The research was carried out using the MSU-270 supercomputer of Lomonosov
Moscow State University.
