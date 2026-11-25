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

from Methods.MammoRegNet.dataloaders.dataset_pairs import (
    get_file_order,
)
from Methods.MammoRegNet.evaluate.rTRE import calc_tre
from Methods.DVFF.models.dvff_net import DVFFNet


def process_annotations(xml_path):
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


def apply_transform_to_landmarks(warp, image_name, output_xml_path, matrix_path=None):
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


def register_pair(before_arr, after_arr, moving_mask, fixed_mask, model, device):
    before = before_arr / 255.0
    after = after_arr / 255.0

    before = np.expand_dims(before, axis=-1)[np.newaxis, ...]
    after = np.expand_dims(after, axis=-1)[np.newaxis, ...]
    moving_mask = np.expand_dims(moving_mask, axis=-1)[np.newaxis, ...]

    input_moving = torch.from_numpy(before).to(device).float().permute(0, 3, 1, 2)
    input_fixed = torch.from_numpy(after).to(device).float().permute(0, 3, 1, 2)
    tensor_mask = torch.from_numpy(moving_mask).to(device).float().permute(0, 3, 1, 2)

    with torch.no_grad():
        outputs = model.register_pair(input_fixed, input_moving, tensor_mask)

    moved = outputs["warped"][0].detach().cpu().numpy().squeeze()
    warp = (
        outputs["flow"][0]
        .unsqueeze(0)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
        .squeeze(0)
    )
    mask_warped = None
    if outputs["warped_mask"] is not None:
        mask_warped = outputs["warped_mask"][0].detach().cpu().numpy().squeeze()

    after_u8 = (after.squeeze() * 255).astype(np.uint8)
    moved_u8 = (moved * 255).astype(np.uint8)

    metrics = {
        "MSE": mse(after_u8, moved_u8),
        "SSIM": ssim(after_u8, moved_u8, data_range=255),
        "MI": mi(after_u8, moved_u8),
        "CC": np.corrcoef(after_u8.flatten(), moved_u8.flatten())[0, 1],
        "NJD": analyze_deformation_field_2d(warp),
    }
    if mask_warped is not None:
        metrics["DSC"] = dice_coefficient(mask_warped, fixed_mask.squeeze())

    return moved, warp, mask_warped, metrics


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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "DVFF evaluation with cross-view fusion switched off at inference: "
            "the fusion-trained weights register each projection on its own."
        )
    )
    parser.add_argument("--seed", default=11, type=int)
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="Fusion-trained checkpoint (default: outputs/DVFF_{seed}/models/model_249.pth)",
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        type=str,
        help="Evaluation output directory",
    )
    parser.add_argument(
        "--coarse_deform",
        action="store_true",
        help="Model was trained with deformable head at the coarsest decoder level",
    )
    return parser.parse_args()


def resolve_run_paths(args):
    if args.coarse_deform:
        output_folder = f"outputs/DVFF_fusion_off_{args.seed}/"
        train_folder = f"outputs/DVFF_{args.seed}/"
    else:
        output_folder = f"outputs/DVFF_no_affine_fusion_off_{args.seed}/"
        train_folder = f"outputs/DVFF_no_affine_{args.seed}/"

    if args.output_folder is not None:
        output_folder = args.output_folder
    best_model = args.model_path or os.path.join(train_folder, "models", "model_249.pth")
    return output_folder, best_model


def load_fusion_trained_model(checkpoint_path, device, coarse_affine):
    model = DVFFNet(use_fusion=False, coarse_affine=coarse_affine)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    load_info = model.load_checkpoint_state(state)
    fusion_keys = load_info["ignored_fusion_keys"]
    if fusion_keys:
        print(f"Ignored fusion-only keys from checkpoint: {len(fusion_keys)}")
    else:
        print(
            "Warning: the checkpoint has no fusion weights, so it was not trained "
            "with cross-view fusion."
        )

    return model.to(device).eval()


def main():
    args = parse_arguments()
    output_folder, best_model = resolve_run_paths(args)
    coarse_affine = not args.coarse_deform

    os.makedirs(output_folder, exist_ok=True)
    base_dir = Path("Dataset/evaluation")
    masks_dir = Path("Dataset/evaluation_masks")
    subdatasets = ["INBreast", "KAU-BCMD", "NLBS", "CSAW-CC"]
    json_file = os.path.join(output_folder, "dvff_fusion_off.json")
    metadata_path = "Dataset/evaluation/metadata_evaluation_correct.csv"
    metadata = pd.read_csv(metadata_path)

    print(f"seed: {args.seed}")
    print("use_fusion (inference): False")
    print("checkpoint (fusion-trained): True")
    print(f"coarse_affine: {coarse_affine}")
    print(f"output_folder: {output_folder}")
    print(f"model: {best_model}")

    data_path = "Dataset/"
    shutil.copy2(data_path + "moving_landmarks.xml", output_folder + "moving_landmarks.xml")
    shutil.copy2(data_path + "fixed_landmarks_1.xml", output_folder + "fixed_landmarks_1.xml")
    shutil.copy2(data_path + "fixed_landmarks_2.xml", output_folder + "fixed_landmarks_2.xml")
    process_annotations(output_folder + "moving_landmarks.xml")
    process_annotations(output_folder + "fixed_landmarks_1.xml")
    process_annotations(output_folder + "fixed_landmarks_2.xml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_fusion_trained_model(best_model, device, coarse_affine)

    mse_list = []
    ssim_list = []
    mi_list = []
    cc_list = []
    dice_list = []
    njd_list = []
    runtime_list = []

    for subdataset in subdatasets:
        subdataset_path = base_dir / subdataset
        if not subdataset_path.exists():
            print(f"Folder not found: {subdataset_path}")
            continue

        print(f"\n{'=' * 50}")
        print(f"PROCESSING: {subdataset}")
        print(f"{'=' * 50}")

        for patient_folder in sorted(subdataset_path.iterdir()):
            if not patient_folder.is_dir():
                continue

            patient_id = patient_folder.name
            image_files = []
            for ext in ["*.jpg", "*.png"]:
                image_files.extend(patient_folder.glob(ext))

            if len(image_files) < 2:
                continue

            f1, f2 = get_file_order(image_files, subdataset, metadata)
            matrix_path = f"affine_transforms/{patient_id}.mat"
            landmark_matrix = matrix_path if args.coarse_deform else None

            print(f"\n[Patient: {patient_id}]")

            before_img = Image.open(f1).convert("L")
            after_img = Image.open(f2).convert("L")
            before_img = before_img.resize((896, 1152), Image.Resampling.LANCZOS)
            after_img = after_img.resize((896, 1152), Image.Resampling.LANCZOS)

            before_arr = np.array(before_img)
            after_arr = np.array(after_img)

            m1 = Image.open(masks_dir / subdataset / patient_id / f"{f1.stem}.png").convert("L")
            m2 = Image.open(masks_dir / subdataset / patient_id / f"{f2.stem}.png").convert("L")
            m1 = m1.resize((896, 1152), Image.Resampling.NEAREST)
            m2 = m2.resize((896, 1152), Image.Resampling.NEAREST)
            m1_arr = (np.array(m1) == 255).astype(np.float32)
            m2_arr = (np.array(m2) == 255).astype(np.float32)

            if args.coarse_deform:
                try:
                    before_arr, m1_arr = apply_affine_prewarp(
                        before_arr, after_arr, m1_arr, matrix_path
                    )
                except FileNotFoundError as exc:
                    print(f"Skipping {patient_id}: {exc}")
                    continue

            before_arr = before_arr * m1_arr.astype(np.uint8)
            after_arr = after_arr * m2_arr.astype(np.uint8)

            output_dir = Path(output_folder) / subdataset / patient_id
            output_dir.mkdir(parents=True, exist_ok=True)
            f1_path = output_dir / f"{f1.stem}.png"
            f2_path = output_dir / f"{f2.stem}.png"
            mask_path = output_dir / "mask.png"
            output_path = output_dir / f"{f1.stem}_{f2.stem}.png"

            Image.fromarray(before_arr).save(f1_path)
            Image.fromarray(after_arr).save(f2_path)

            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                t0 = time.time()
                moved, warp, mask_warped, metrics = register_pair(
                    before_arr,
                    after_arr,
                    m1_arr,
                    m2_arr,
                    model,
                    device,
                )
                runtime = time.time() - t0
                metrics["Runtime"] = runtime

                Image.fromarray((moved * 255).astype(np.uint8)).save(output_path)
                if mask_warped is not None:
                    Image.fromarray((mask_warped * 255).astype(np.uint8)).save(mask_path)

                image_name = patient_id + ".png"
                apply_transform_to_landmarks(
                    warp,
                    image_name,
                    output_folder + "fixed_landmarks_1.xml",
                    landmark_matrix,
                )
                apply_transform_to_landmarks(
                    warp,
                    image_name,
                    output_folder + "fixed_landmarks_2.xml",
                    landmark_matrix,
                )

                mse_list.append(metrics["MSE"])
                ssim_list.append(metrics["SSIM"])
                mi_list.append(metrics["MI"])
                cc_list.append(metrics["CC"])
                dice_list.append(metrics["DSC"])
                njd_list.append(metrics["NJD"])
                runtime_list.append(runtime)

                results = {}
                if os.path.exists(json_file):
                    with open(json_file, "r", encoding="utf-8") as f:
                        try:
                            results = json.load(f)
                        except json.JSONDecodeError:
                            results = {}

                results[str(output_path)] = {k: round(v, 6) for k, v in metrics.items()}
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
