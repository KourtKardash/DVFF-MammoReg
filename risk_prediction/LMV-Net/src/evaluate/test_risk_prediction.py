import json
import os

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm

from models import LongitudinalMultiViewRiskModel
from registration.factory import load_registration_backend
from train.train_utils import load_risk_model_weights
from utils import (
    bootstrap_auc,
    bootstrap_auc_by_density,
    bootstrap_c_index,
    bootstrap_c_index_by_density,
    create_logger,
    get_censoring_dist,
    save_model_results_to_file,
)


def to_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    else:
        return obj


def _as_str_list(values):
    return [str(v) for v in values]


def _as_float_list(values):
    out = []
    for v in values:
        if torch.is_tensor(v):
            out.append(float(v.detach().cpu().item()))
        else:
            out.append(float(v))
    return out


def _as_int_list(values):
    out = []
    for v in values:
        if torch.is_tensor(v):
            out.append(int(v.detach().cpu().item()))
        else:
            out.append(int(v))
    return out


def test_risk(
    args,
    test_loader,
    path_model,
    out_dir,
    path_logger,
    accelerator: Accelerator,
):
    """
    Evaluate the trained model on the test dataset using the Accelerate framework.
    """
    logger = create_logger(path_logger) if accelerator.is_main_process else None
    if accelerator.is_main_process:
        print("[INFO] Loading the trained models...")
        print(f"Registration method: {args.registration_method}")
        print(f"Registration checkpoint: {args.registration_checkpoint}")
        print(f"Alignment level: {args.alignment_level}")
        print(f"ANTs pre-warp: {args.ants_prewarp}")
        if args.registration_method == "dvff":
            print(f"DVFF fusion: {getattr(args, 'dvff_use_fusion', True)}")

    reg_backend = load_registration_backend(
        method=args.registration_method,
        checkpoint_path=args.registration_checkpoint,
        coarse_affine=args.coarse_affine,
        use_fusion=getattr(args, "dvff_use_fusion", True),
        compose_ants_affine=args.ants_prewarp,
        prewarp_in_loader=getattr(args, "ants_prewarp_in_loader", False),
    )

    model_risk = LongitudinalMultiViewRiskModel(
        registration_backend=reg_backend,
        max_followup=5,
        mirai_encoder_path=getattr(args, "mirai_encoder_path", None),
        alignment_level=args.alignment_level,
    )
    load_risk_model_weights(model_risk, path_model)
    model_risk.eval()

    model_risk, test_loader = accelerator.prepare(model_risk, test_loader)

    if accelerator.is_main_process:
        print("[INFO] Evaluating on test dataset...")

    all_preds, all_times, all_events, all_densities = [], [], [], []
    all_targets, all_masks = [], []
    all_patient_ids, all_sample_keys = [], []
    all_laterality, all_prior_year, all_current_year = [], [], []
    all_y2c_raw, all_ylf_raw = [], []

    model_risk.eval()

    with torch.no_grad():
        progress_bar = tqdm(
            test_loader, desc="Testing", disable=not accelerator.is_main_process
        )
        for batch in progress_bar:
            outputs = model_risk(
                batch["current_image_cc"],
                batch["previous_image_cc"],
                batch["current_image_mlo"],
                batch["previous_image_mlo"],
                batch["time_gap"],
                affine_mat_cc=list(batch["affine_mat_cc"])
                if "affine_mat_cc" in batch
                else None,
                affine_mat_mlo=list(batch["affine_mat_mlo"])
                if "affine_mat_mlo" in batch
                else None,
            )

            risk = outputs["risk_multi"]

            all_preds.append(accelerator.gather(torch.sigmoid(risk).detach()).cpu())
            all_times.append(accelerator.gather(batch["event_times"]).cpu())
            all_events.append(accelerator.gather(batch["event_observed"]).cpu())
            all_densities.append(accelerator.gather(batch["density"]).cpu())
            all_targets.append(accelerator.gather(batch["target"]).cpu())
            all_masks.append(accelerator.gather(batch["y_mask"]).cpu())

            # Metadata / IDs (strings): gather across processes for alignment.
            all_patient_ids.extend(
                gather_object(_as_str_list(batch["patient_id"]))
            )
            all_sample_keys.extend(
                gather_object(_as_str_list(batch["sample_key"]))
            )
            all_laterality.extend(
                gather_object(_as_str_list(batch["laterality"]))
            )
            all_prior_year.extend(
                gather_object(_as_int_list(batch["prior_year"]))
            )
            all_current_year.extend(
                gather_object(_as_int_list(batch["current_year"]))
            )
            all_y2c_raw.extend(
                gather_object(_as_float_list(batch["years_to_cancer_raw"]))
            )
            all_ylf_raw.extend(
                gather_object(_as_float_list(batch["years_last_followup_raw"]))
            )

    if accelerator.is_main_process:
        print("[INFO] Aggregating results and calculating metrics...")

        predictions = torch.cat(all_preds).numpy()
        event_times = torch.cat(all_times).numpy().astype(int)
        event_observed = torch.cat(all_events).numpy()
        density_categories = torch.cat(all_densities).numpy()
        targets = torch.cat(all_targets).numpy()
        y_masks = torch.cat(all_masks).numpy()

        # With DistributedSampler, tensor gather and gather_object can both
        # include padded extras; truncate metadata to prediction length.
        n = len(predictions)
        sample_meta = {
            "patient_id": all_patient_ids[:n],
            "sample_key": all_sample_keys[:n],
            "laterality": all_laterality[:n],
            "prior_year": all_prior_year[:n],
            "current_year": all_current_year[:n],
            "years_to_cancer_raw": all_y2c_raw[:n],
            "years_last_followup_raw": all_ylf_raw[:n],
        }

        censoring_dist = get_censoring_dist(event_times, event_observed)

        save_model_results_to_file(
            predictions,
            event_times,
            event_observed,
            density_categories,
            censoring_dist,
            args.path_test_folder,
            sample_meta=sample_meta,
            targets=targets,
            y_masks=y_masks,
        )

        print("[INFO] Calculating metrics...")

        mean_c_index, c_index_ci, c_index_scores = bootstrap_c_index(
            event_times,
            predictions,
            event_observed,
            censoring_dist,
        )
        np.save(os.path.join(args.path_test_folder, "cindex_scores.npy"), c_index_scores)

        auc_summary, auc_arrays = bootstrap_auc(
            event_times, predictions, event_observed
        )
        np.savez(os.path.join(args.path_test_folder, "auc_scores.npz"), **auc_arrays)

        auc_by_density = bootstrap_auc_by_density(
            event_times, predictions, event_observed, density_categories
        )
        c_index_by_density, c_index_scores_density = bootstrap_c_index_by_density(
            event_times,
            predictions,
            event_observed,
            density_categories,
            censoring_dist,
            save_json_path=args.path_test_folder,
        )
        np.save(
            os.path.join(args.path_test_folder, "cindex_scores_density.npy"),
            c_index_scores_density,
        )

        auc_formatted = {
            f"{year}": {"Mean": mean_auc, "95% CI": ci}
            for year, (mean_auc, ci) in auc_summary.items()
        }

        results = {
            "C-index": {"Mean": mean_c_index, "95% CI": c_index_ci},
            "Yearly AUCs": auc_formatted,
            "AUC by density categories": auc_by_density,
            "C index by density categories": c_index_by_density,
        }

        print("Final Test Results:")
        results_safe = to_json_safe(results)
        print(json.dumps(results_safe, indent=2))
        logger.info(f"Final Test Results: {results}")

        results_path = os.path.join(args.path_test_folder, "results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_safe, f, indent=2)

    return None
