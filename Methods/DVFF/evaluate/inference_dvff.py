import argparse
import json
import os
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import ants
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_mutual_information as mi
from skimage.metrics import structural_similarity as ssim

from Methods.DVFF.dataloaders.dataset_dvff import (
    build_dvff_groups,
    select_canonical_pair,
)
from Methods.MammoRegNet.evaluate.rTRE import calc_tre
from Methods.DVFF.models.dvff_net import DVFFNet


def process_annotations(xml_path):
    """Same landmark scaling as inference_original.py."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for image in root.findall("image"):
        width = int(image.get("width"))
        height = int(image.get("height"))
        for points in image.findall("points"):
            points_str = points.get("points")
            x, y = points_str.split(",")
            x = float(x) * 896 / width
            y = float(y) * 1152 / height
            points.set("points", f"{x:.2f},{y:.2f}")
        image.set("width", "896")
        image.set("height", "1152")

    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def apply_transform_to_landmarks(
    warp, image_name, output_xml_path, matrix_path=None
):
    """Warp fixed landmarks: deform (flow), then optional ANTs affine."""
    tree = ET.parse(output_xml_path)
    root = tree.getroot()

    for image in root.findall("image"):
        if image.get("name") != image_name:
            continue

        points_tags = image.findall("points")
        point_list = []
        index_to_tag = []
        for points_tag in points_tags:
            x, y = map(float, points_tag.get("points").split(","))
            point_list.append([int(round(y)), int(round(x))])
            index_to_tag.append(points_tag)

        points_df = pd.DataFrame(point_list, columns=["x", "y"])
        points = points_df[["x", "y"]].values

        height, width, _ = warp.shape
        y_coords, x_coords = np.arange(height), np.arange(width)
        interp_dx = RegularGridInterpolator(
            (y_coords, x_coords),
            warp[..., 1],
            method="linear",
            bounds_error=False,
            fill_value=0,
        )
        interp_dy = RegularGridInterpolator(
            (y_coords, x_coords),
            warp[..., 0],
            method="linear",
            bounds_error=False,
            fill_value=0,
        )
        dx = interp_dx(points)
        dy = interp_dy(points)
        deformed_points = points + np.column_stack((dy, dx))

        if matrix_path is not None:
            deformed_df = pd.DataFrame(deformed_points, columns=["x", "y"])
            output_points = ants.apply_transforms_to_points(
                dim=2,
                points=deformed_df,
                transformlist=[matrix_path],
            )
        else:
            output_points = pd.DataFrame(deformed_points, columns=["x", "y"])

        for idx, points_tag in enumerate(index_to_tag):
            new_x = output_points["x"].iloc[idx]
            new_y = output_points["y"].iloc[idx]
            points_tag.set("points", f"{new_y:.2f},{new_x:.2f}")

    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)

def analyze_deformation_field_2d(warp):
    u = warp[..., 0]
    v = warp[..., 1]
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    jacobian_det = (1 + du_dy) * (1 + dv_dx) - du_dx * dv_dy
    return float(100.0 * np.sum(jacobian_det < 0) / jacobian_det.size)


def dice_coefficient(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    return (2.0 * intersection) / (mask1.sum() + mask2.sum())


def load_eval_mask(image_path: str, masks_root: str = "Dataset/evaluation_masks") -> np.ndarray:
    rel_path = Path(os.path.relpath(image_path, "Dataset/evaluation"))
    mask_path = Path(masks_root) / rel_path.with_suffix(".png")
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((896, 1152), Image.Resampling.NEAREST)
    return (np.array(mask) == 255).astype(np.float32)


def prepare_masked_pair(moving_path: str, fixed_path: str):
    """Match inference_original preprocessing: resize, mask with ==255, return uint8 arrays."""
    moving_img = Image.open(moving_path).convert("L")
    fixed_img = Image.open(fixed_path).convert("L")
    moving_img = moving_img.resize((896, 1152), Image.Resampling.LANCZOS)
    fixed_img = fixed_img.resize((896, 1152), Image.Resampling.LANCZOS)

    moving_arr = np.array(moving_img)
    fixed_arr = np.array(fixed_img)
    moving_mask = load_eval_mask(moving_path)
    fixed_mask = load_eval_mask(fixed_path)

    moving_arr = moving_arr * moving_mask.astype(np.uint8)
    fixed_arr = fixed_arr * fixed_mask.astype(np.uint8)
    return moving_arr, fixed_arr, moving_mask, fixed_mask


def apply_affine_prewarp(moving_arr, fixed_arr, moving_mask, matrix_path):
    if matrix_path is None or not os.path.exists(matrix_path):
        raise FileNotFoundError(
            f"Affine transform required for coarse_deform mode: {matrix_path}"
        )

    fixed_ants = ants.from_numpy(fixed_arr.astype(np.uint8))
    moving_ants = ants.from_numpy(moving_arr.astype(np.uint8))

    warped_img = ants.apply_transforms(
        fixed=fixed_ants,
        moving=moving_ants,
        transformlist=[matrix_path],
        interpolator="linear",
    )
    moving_arr = warped_img.numpy().astype(np.uint8)

    mask_ants = ants.from_numpy(moving_mask.astype(np.float32))
    warped_mask = ants.apply_transforms(
        fixed=fixed_ants,
        moving=mask_ants,
        transformlist=[matrix_path],
        interpolator="genericLabel",
        verbose=False,
    )
    moving_mask = warped_mask.numpy().astype(np.float32)
    return moving_arr, moving_mask


def register_dvff(
    before_mlo,
    after_mlo,
    before_cc,
    after_cc,
    moving_mask_mlo,
    fixed_mask_mlo,
    moving_mask_cc,
    fixed_mask_cc,
    model,
    device,
):
    before_mlo = before_mlo / 255.0
    after_mlo = after_mlo / 255.0
    before_cc = before_cc / 255.0
    after_cc = after_cc / 255.0

    before_mlo = np.expand_dims(before_mlo, axis=-1)
    after_mlo = np.expand_dims(after_mlo, axis=-1)
    before_cc = np.expand_dims(before_cc, axis=-1)
    after_cc = np.expand_dims(after_cc, axis=-1)
    moving_mask_mlo = np.expand_dims(moving_mask_mlo, axis=-1)
    moving_mask_cc = np.expand_dims(moving_mask_cc, axis=-1)

    input_moving_mlo = (
        torch.from_numpy(before_mlo[np.newaxis]).to(device).float().permute(0, 3, 1, 2)
    )
    input_fixed_mlo = (
        torch.from_numpy(after_mlo[np.newaxis]).to(device).float().permute(0, 3, 1, 2)
    )
    input_moving_cc = (
        torch.from_numpy(before_cc[np.newaxis]).to(device).float().permute(0, 3, 1, 2)
    )
    input_fixed_cc = (
        torch.from_numpy(after_cc[np.newaxis]).to(device).float().permute(0, 3, 1, 2)
    )
    tensor_mask_mlo = (
        torch.from_numpy(moving_mask_mlo[np.newaxis]).to(device).float().permute(0, 3, 1, 2)
    )
    tensor_mask_cc = (
        torch.from_numpy(moving_mask_cc[np.newaxis]).to(device).float().permute(0, 3, 1, 2)
    )

    with torch.no_grad():
        outputs = model(
            input_fixed_mlo,
            input_fixed_cc,
            input_moving_mlo,
            input_moving_cc,
            moving_mask_mlo=tensor_mask_mlo,
            moving_mask_cc=tensor_mask_cc,
        )

    moved_mlo = outputs["warped_mlo"][0].detach().cpu().numpy().squeeze()
    moved_cc = outputs["warped_cc"][0].detach().cpu().numpy().squeeze()
    warp_mlo = (
        outputs["flow_mlo"][0].unsqueeze(0).permute(0, 2, 3, 1).cpu().numpy().squeeze(0)
    )
    warp_cc = (
        outputs["flow_cc"][0].unsqueeze(0).permute(0, 2, 3, 1).cpu().numpy().squeeze(0)
    )
    mask_warped_mlo = None
    mask_warped_cc = None
    if outputs["warped_mask_mlo"] is not None:
        mask_warped_mlo = outputs["warped_mask_mlo"][0].detach().cpu().numpy().squeeze()
    if outputs["warped_mask_cc"] is not None:
        mask_warped_cc = outputs["warped_mask_cc"][0].detach().cpu().numpy().squeeze()

    after_mlo_u8 = (after_mlo.squeeze() * 255).astype(np.uint8)
    after_cc_u8 = (after_cc.squeeze() * 255).astype(np.uint8)
    moved_mlo_u8 = (moved_mlo * 255).astype(np.uint8)
    moved_cc_u8 = (moved_cc * 255).astype(np.uint8)

    metrics_mlo = {
        "MSE": mse(after_mlo_u8, moved_mlo_u8),
        "SSIM": ssim(after_mlo_u8, moved_mlo_u8, data_range=255),
        "MI": mi(after_mlo_u8, moved_mlo_u8),
        "CC": np.corrcoef(after_mlo_u8.flatten(), moved_mlo_u8.flatten())[0, 1],
        "NJD": analyze_deformation_field_2d(warp_mlo),
    }
    metrics_cc = {
        "MSE": mse(after_cc_u8, moved_cc_u8),
        "SSIM": ssim(after_cc_u8, moved_cc_u8, data_range=255),
        "MI": mi(after_cc_u8, moved_cc_u8),
        "CC": np.corrcoef(after_cc_u8.flatten(), moved_cc_u8.flatten())[0, 1],
        "NJD": analyze_deformation_field_2d(warp_cc),
    }

    if mask_warped_mlo is not None:
        metrics_mlo["DSC"] = dice_coefficient(mask_warped_mlo, fixed_mask_mlo.squeeze())
    if mask_warped_cc is not None:
        metrics_cc["DSC"] = dice_coefficient(mask_warped_cc, fixed_mask_cc.squeeze())

    return {
        "moved_mlo": moved_mlo,
        "moved_cc": moved_cc,
        "warp_mlo": warp_mlo,
        "warp_cc": warp_cc,
        "metrics_mlo": metrics_mlo,
        "metrics_cc": metrics_cc,
    }


def landmark_image_name(image_path: str) -> str:
    """Same convention as inference_original: folder name + .png."""
    return Path(image_path).parent.name + ".png"


def affine_matrix_path(moving_image_path: str) -> str:
    """Affine matrix is named after the pair subdirectory (e.g. 127-1.mat)."""
    return f"affine_transforms/{Path(moving_image_path).parent.name}.mat"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="DVFF evaluation script."
    )
    parser.set_defaults(use_fusion=True)
    parser.add_argument(
        "--use_fusion",
        "--use-fusion",
        dest="use_fusion",
        action="store_true",
        help="Enable cross-view fusion blocks and view_embed (must match training)",
    )
    parser.add_argument(
        "--no_fusion",
        "--no-fusion",
        dest="use_fusion",
        action="store_false",
        help="Disable cross-view fusion blocks and view_embed",
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        type=str,
        help="Directory for evaluation outputs. Default follows the flags.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help=(
            "Path to model checkpoint (default: outputs/DVFFNet[_no_fusion]_{seed}"
            "[_affine]/models/model_249.pth)"
        ),
    )
    parser.add_argument("--seed", default=11, type=int)
    parser.add_argument(
        "--coarse_deform",
        action="store_true",
        help="Model was trained with deformable head at the coarsest decoder level",
    )
    return parser.parse_args()


def resolve_run_paths(args):
    if args.coarse_deform and args.use_fusion:
        run_name = f"DVFF_{args.seed}"
    elif args.coarse_deform:
        run_name = f"DVFF_no_fusion_{args.seed}"
    elif args.use_fusion:
        run_name = f"DVFF_no_affine_{args.seed}"
    else:
        run_name = f"DVFF_no_affine_no_fusion_{args.seed}"

    output_folder = args.output_folder or f"outputs/{run_name}/"
    best_model = args.model_path or os.path.join(output_folder, "models", "model_249.pth")
    return output_folder, best_model


def main():
    args = parse_arguments()
    output_folder, best_model = resolve_run_paths(args)

    os.makedirs(output_folder, exist_ok=True)
    metadata_path = "Dataset/evaluation/metadata_evaluation_correct.csv"
    subdatasets = ["INBreast", "KAU-BCMD", "NLBS", "CSAW-CC"]
    json_file = os.path.join(output_folder, "dvff.json")

    data_path = "Dataset/"
    shutil.copy2(data_path + "moving_landmarks.xml", output_folder + "moving_landmarks.xml")
    shutil.copy2(data_path + "fixed_landmarks_1.xml", output_folder + "fixed_landmarks_1.xml")
    shutil.copy2(data_path + "fixed_landmarks_2.xml", output_folder + "fixed_landmarks_2.xml")
    process_annotations(output_folder + "moving_landmarks.xml")
    process_annotations(output_folder + "fixed_landmarks_1.xml")
    process_annotations(output_folder + "fixed_landmarks_2.xml")

    metadata = pd.read_csv(metadata_path)
    groups = [
        group
        for group in build_dvff_groups(metadata)
        if group["subdataset"] in subdatasets
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"seed: {args.seed}")
    print(f"use_fusion: {args.use_fusion}")
    print(f"coarse_affine: {not args.coarse_deform}")
    print(f"output_folder: {output_folder}")
    print(f"model: {best_model}")

    ckpt = torch.load(best_model, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    fusion_levels = DVFFNet.fusion_levels_from_state_dict(state)
    print(f"fusion_levels: {fusion_levels}")

    model = DVFFNet(
        use_fusion=args.use_fusion,
        coarse_affine=not args.coarse_deform,
        fusion_levels=fusion_levels,
    )
    load_info = model.load_checkpoint_state(state)
    if load_info["ignored_fusion_keys"]:
        print(
            "Ignored fusion-only keys from checkpoint: "
            f"{len(load_info['ignored_fusion_keys'])}"
        )
    model.set_fusion_enabled(args.use_fusion)
    model = model.to(device)
    model.eval()

    mse_list = []
    ssim_list = []
    mi_list = []
    cc_list = []
    dice_list = []
    njd_list = []
    runtime_list = []

    for group in groups:
        subdataset = group["subdataset"]
        patient_id = group["patient_id"]
        laterality = group["laterality"]

        print(
            f"\n[Patient: {patient_id}, Laterality: {laterality}, Dataset: {subdataset}]"
        )

        try:
            moving_mlo_path, fixed_mlo_path = select_canonical_pair(
                group["mlo_paths"], subdataset, metadata
            )
            moving_cc_path, fixed_cc_path = select_canonical_pair(
                group["cc_paths"], subdataset, metadata
            )
        except ValueError as exc:
            print(f"Skipping {patient_id}: {exc}")
            continue

        try:
            before_mlo, after_mlo, moving_mask_mlo, fixed_mask_mlo = prepare_masked_pair(
                moving_mlo_path, fixed_mlo_path
            )
            before_cc, after_cc, moving_mask_cc, fixed_mask_cc = prepare_masked_pair(
                moving_cc_path, fixed_cc_path
            )
        except FileNotFoundError as exc:
            print(f"Skipping {patient_id}: {exc}")
            continue

        matrix_path_mlo = affine_matrix_path(moving_mlo_path)
        matrix_path_cc = affine_matrix_path(moving_cc_path)
        landmark_matrix_mlo = matrix_path_mlo if args.coarse_deform else None
        landmark_matrix_cc = matrix_path_cc if args.coarse_deform else None

        if args.coarse_deform:
            try:
                before_mlo, moving_mask_mlo = apply_affine_prewarp(
                    before_mlo, after_mlo, moving_mask_mlo, matrix_path_mlo
                )
                before_cc, moving_mask_cc = apply_affine_prewarp(
                    before_cc, after_cc, moving_mask_cc, matrix_path_cc
                )
            except FileNotFoundError as exc:
                print(f"Skipping {patient_id}: {exc}")
                continue

        output_dir = Path(output_folder) / subdataset / f"{patient_id}_{laterality}"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            t0 = time.time()
            reg = register_dvff(
                before_mlo,
                after_mlo,
                before_cc,
                after_cc,
                moving_mask_mlo,
                fixed_mask_mlo,
                moving_mask_cc,
                fixed_mask_cc,
                model,
                device,
            )
            runtime = time.time() - t0

            f1_mlo = Path(moving_mlo_path)
            f2_mlo = Path(fixed_mlo_path)
            f1_cc = Path(moving_cc_path)
            output_path_mlo = output_dir / f"{f1_mlo.stem}_{f2_mlo.stem}.png"
            output_path_cc = output_dir / f"{f1_cc.stem}_{Path(fixed_cc_path).stem}.png"

            Image.fromarray((reg["moved_mlo"] * 255).astype(np.uint8)).save(output_path_mlo)
            Image.fromarray((reg["moved_cc"] * 255).astype(np.uint8)).save(output_path_cc)

            landmark_mlo = landmark_image_name(moving_mlo_path)
            landmark_cc = landmark_image_name(moving_cc_path)
            apply_transform_to_landmarks(
                reg["warp_mlo"],
                landmark_mlo,
                output_folder + "fixed_landmarks_1.xml",
                landmark_matrix_mlo,
            )
            apply_transform_to_landmarks(
                reg["warp_mlo"],
                landmark_mlo,
                output_folder + "fixed_landmarks_2.xml",
                landmark_matrix_mlo,
            )
            apply_transform_to_landmarks(
                reg["warp_cc"],
                landmark_cc,
                output_folder + "fixed_landmarks_1.xml",
                landmark_matrix_cc,
            )
            apply_transform_to_landmarks(
                reg["warp_cc"],
                landmark_cc,
                output_folder + "fixed_landmarks_2.xml",
                landmark_matrix_cc,
            )

            combined = {}
            for key in set(reg["metrics_mlo"]) & set(reg["metrics_cc"]):
                combined[key] = (reg["metrics_mlo"][key] + reg["metrics_cc"][key]) / 2.0
            combined["Runtime"] = runtime

            mse_list.append(combined["MSE"])
            ssim_list.append(combined["SSIM"])
            mi_list.append(combined["MI"])
            cc_list.append(combined["CC"])
            dice_list.append(combined["DSC"])
            njd_list.append(combined["NJD"])
            runtime_list.append(runtime)

            results = {}
            if os.path.exists(json_file):
                with open(json_file, "r", encoding="utf-8") as f:
                    try:
                        results = json.load(f)
                    except json.JSONDecodeError:
                        results = {}

            results[str(output_path_mlo)] = {k: round(v, 6) for k, v in combined.items()}
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        except Exception as exc:
            print(f"Error processing {patient_id}: {exc}")

    if mse_list:
        results = {}
        if os.path.exists(json_file):
            with open(json_file, "r", encoding="utf-8") as f:
                try:
                    results = json.load(f)
                except json.JSONDecodeError:
                    results = {}

        results["Mean Metrics"] = {
            "MSE": round(float(np.mean(mse_list)), 6),
            "SSIM": round(float(np.mean(ssim_list)), 6),
            "MI": round(float(np.mean(mi_list)), 6),
            "CC": round(float(np.mean(cc_list)), 6),
            "DSC": round(float(np.mean(dice_list)), 6),
            "NJD": round(float(np.mean(njd_list)), 6),
            "Runtime": round(float(np.mean(runtime_list)), 6),
        }
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Mean MSE: {np.mean(mse_list):.4f} +- {np.std(mse_list):.4f}")
        print(f"Mean SSIM: {np.mean(ssim_list):.4f} +- {np.std(ssim_list):.4f}")
        print(f"Mean MI: {np.mean(mi_list):.4f} +- {np.std(mi_list):.4f}")
        print(f"Mean CC: {np.mean(cc_list):.4f} +- {np.std(cc_list):.4f}")
        print(f"Mean DSC: {np.mean(dice_list):.4f} +- {np.std(dice_list):.4f}")
        print(f"Mean NJD: {np.mean(njd_list):.10f} +- {np.std(njd_list):.10f}")
        print(f"Mean Runtime: {np.mean(runtime_list):.4f}")

    calc_tre(
        output_folder + "moving_landmarks.xml",
        output_folder + "fixed_landmarks_1.xml",
        output_folder + "fixed_landmarks_2.xml",
    )


if __name__ == "__main__":
    main()
