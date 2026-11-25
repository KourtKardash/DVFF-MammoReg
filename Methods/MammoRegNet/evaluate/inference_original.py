import os
import json
import argparse

import torch
import numpy as np
import random

import xml.etree.ElementTree as ET
import pandas as pd

from scipy.interpolate import RegularGridInterpolator
import shutil
import ants

import itertools
from PIL import Image

from pathlib import Path
import time

from skimage.metrics import normalized_mutual_information as mi
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

from Methods.MammoRegNet.utils.utils import *
from Methods.MammoRegNet.train.losses_mammoregnet import  *
from Methods.MammoRegNet.models.MammoRegNet import MammoRegNet

from Methods.MammoRegNet.evaluate.rTRE import calc_tre


def get_file_order(file_names, subdataset, metadata):
    path_to_meta = {}
    for _, row in metadata.iterrows():
        file_path = row["File Path"]
        path_to_meta[file_path] = row

    path1 = str(file_names[0])
    path2 = str(file_names[1])

    meta1 = path_to_meta[path1]
    meta2 = path_to_meta[path2]
    if subdataset in ["NLBS", "KAU-BCMD"]:
        age1 = meta1["Patient age"]
        age2 = meta2["Patient age"]

        if age1 <= age2:
            return file_names[0], file_names[1]
        else:
            return file_names[1], file_names[0]

    elif subdataset in ["INBreast", "CSAW-CC"]:
        date1 = str(meta1["Study Date"])
        date2 = str(meta2["Study Date"])

        try:
            date1_val = int(date1)
            date2_val = int(date2)

            if date1_val <= date2_val:
                return file_names[0], file_names[1]
            else:
                return file_names[1], file_names[0]
        except ValueError:
            print("Warning: Could not parse dates, using default order")
            return file_names[0], file_names[1]


def analyze_deformation_field_2d(warp):
    u = warp[..., 0]  # displacement in Y (rows)
    v = warp[..., 1]  # displacement in X (columns)

    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)

    jacobian_det = (1 + du_dy) * (1 + dv_dx) - du_dx * dv_dy

    percent_folded = 100.0 * np.sum(jacobian_det < 0) / jacobian_det.size
    return float(percent_folded)


def dice_coefficient(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    size1 = mask1.sum()
    size2 = mask2.sum()
    dice = (2.0 * intersection) / (size1 + size2)
    return dice


def apply_transform_to_landmarks(
    warp, image_name, output_xml_path, matrix_path=None
):
    tree = ET.parse(output_xml_path)
    root = tree.getroot()

    for image in root.findall('image'):
        if image.get('name') == image_name:
            points_tags = image.findall('points')
            point_list = []
            index_to_tag = []
            for points_tag in points_tags:
                x, y = map(float, points_tag.get('points').split(','))
                point_list.append([int(round(y)), int(round(x))])
                index_to_tag.append(points_tag)

            points_df = pd.DataFrame(point_list, columns=['x', 'y'])
            points = points_df[['x', 'y']].values

            height, width, _ = warp.shape
            y_coords, x_coords = np.arange(height), np.arange(width)
            interp_dx = RegularGridInterpolator(
                (y_coords, x_coords),
                warp[..., 1],
                method='linear',
                bounds_error=False,
                fill_value=0,
            )
            interp_dy = RegularGridInterpolator(
                (y_coords, x_coords),
                warp[..., 0],
                method='linear',
                bounds_error=False,
                fill_value=0,
            )

            dx = interp_dx(points)
            dy = interp_dy(points)
            deformed_points = points + np.column_stack((dy, dx))

            if matrix_path is not None:
                deformed_df = pd.DataFrame(deformed_points, columns=['x', 'y'])
                transformed_points = ants.apply_transforms_to_points(
                    dim=2,
                    points=deformed_df,
                    transformlist=[matrix_path],
                )
                output_points = transformed_points
            else:
                output_points = pd.DataFrame(deformed_points, columns=['x', 'y'])

            for i, points_tag in enumerate(index_to_tag):
                new_x = output_points['x'].iloc[i]
                new_y = output_points['y'].iloc[i]
                points_tag.set('points', f"{new_y:.2f},{new_x:.2f}")

    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)


def registration(before_img, after_img, moving_mask, fixed_mask, output_path,
                 f1_path, f2_path, p_mask, best_model, output_xml1, output_xml2,
                 matrix_path=None, coarse_affine=True, apply_external_affine=False):
    t = time.time()
    image_name = f1_path.parent.name

    img1 = Image.fromarray(before_img)
    img2 = Image.fromarray(after_img)

    img1.save(f1_path)
    img2.save(f2_path)

    if apply_external_affine:
        if matrix_path is None or not os.path.exists(matrix_path):
            raise FileNotFoundError(
                f"Affine transform required when the external affine is applied: {matrix_path}"
            )

        fixed_ants = ants.from_numpy(after_img)
        moving_ants = ants.from_numpy(before_img)

        warped_img = ants.apply_transforms(
            fixed=fixed_ants,
            moving=moving_ants,
            transformlist=[matrix_path],
            interpolator='linear',
        )
        before_img = warped_img.numpy().astype(np.uint8)

        mask_ants = ants.from_numpy(moving_mask.astype(np.float32))
        warped_mask = ants.apply_transforms(
            fixed=fixed_ants,
            moving=mask_ants,
            transformlist=[matrix_path],
            interpolator='genericLabel',
            verbose=False,
        )
        moving_mask = warped_mask.numpy().astype(np.float32)

    before_img = before_img / 255.0
    after_img = after_img / 255.0

    before_img = np.expand_dims(before_img, axis=-1)
    after_img = np.expand_dims(after_img, axis=-1)
    moving_mask = np.expand_dims(moving_mask, axis=-1)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    device = 'cuda'

    model = MammoRegNet(coarse_affine=coarse_affine)
    model.load_state_dict(torch.load(best_model, map_location=torch.device(device)))
    model = model.to(device)
    model.eval()

    before_img = before_img[np.newaxis, ...]
    after_img = after_img[np.newaxis, ...]
    moving_mask = moving_mask[np.newaxis, ...]

    input_moving = torch.from_numpy(before_img).to(device).float().permute(0, 3, 1, 2)
    input_fixed = torch.from_numpy(after_img).to(device).float().permute(0, 3, 1, 2)
    tensor_mask = torch.from_numpy(moving_mask).to(device).float().permute(0, 3, 1, 2)

    with torch.no_grad():
        pred = model(input_fixed, input_moving, tensor_mask)
        moved = pred[0][0]
        mask_warped = pred[4][0]
        warp = pred[1][0]

        moved = moved.detach().cpu().numpy().squeeze().squeeze()
        mask_warped = mask_warped.detach().cpu().numpy().squeeze().squeeze()
        warp = warp.unsqueeze(0).permute(0, 2, 3, 1).cpu().numpy()

    runtime = time.time() - t

    if torch.cuda.is_available():
        peak_memory_allocated = torch.cuda.max_memory_allocated("cuda")
        peak_memory_allocated = peak_memory_allocated / 1024 / 1024 / 1024
    else:
        peak_memory_allocated = 0

    im = Image.fromarray((moved * 255).astype(np.uint8))
    im.save(output_path)
    after_img = after_img.squeeze().squeeze()

    image = Image.fromarray((mask_warped * 255).astype(np.uint8))
    image.save(p_mask)

    after_img = (after_img * 255).astype(np.uint8)
    moved = (moved * 255).astype(np.uint8)
    per = analyze_deformation_field_2d(warp.squeeze(0))

    landmark_matrix = matrix_path if apply_external_affine else None
    apply_transform_to_landmarks(
        warp.squeeze(0), image_name + '.png', output_xml1, landmark_matrix
    )
    apply_transform_to_landmarks(
        warp.squeeze(0), image_name + '.png', output_xml2, landmark_matrix
    )

    return mse(after_img, moved), ssim(after_img, moved, data_range=255),\
           mi(after_img, moved), np.corrcoef(after_img.flatten(), moved.flatten())[0, 1],\
           dice_coefficient(mask_warped, fixed_mask), per, peak_memory_allocated, runtime


def process_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for image in root.findall('image'):
        width = int(image.get('width'))
        height = int(image.get('height'))
        for points in image.findall('points'):
            points_str = points.get('points')
            x, y = points_str.split(',')
            x = float(x)
            y = float(y)
            x = x * 896 / width
            y = y * 1152 / height
            points.set('points', f"{x:.2f},{y:.2f}")
        image.set('width', str(896))
        image.set('height', str(1152))

    tree.write(xml_path, encoding='utf-8', xml_declaration=True)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="MammoRegNet evaluation script (original pipeline)."
    )
    parser.add_argument("--seed", default=11, type=int, help="Training seed used for model paths")
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help=(
            "Path to the checkpoint. Default: "
            "outputs/MammoRegNet_<seed>/models/model_249.pth, or "
            "outputs/MammoRegNet_deformable_<seed>/models/model_249.pth "
            "with --coarse_deform. --external_affine still reads the row-1 "
            "checkpoint. --output_folder does not move this path."
        ),
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        type=str,
        help=(
            "Where to write landmarks and the JSON summary. Default: the "
            "checkpoint folder, or outputs/Affine_MammoRegNet_<seed>/ with "
            "--external_affine."
        ),
    )
    parser.add_argument(
        "--coarse_deform",
        action="store_true",
        help=(
            "Deformable head at the coarsest decoder level. Also applies the "
            "cached ANTs affine to the moving image and to landmarks."
        ),
    )
    parser.add_argument(
        "--external_affine",
        action="store_true",
        help=(
            "Apply the cached ANTs affine to the moving image and to "
            "landmarks, and keep the affine decoder head. This is Affine + "
            "MammoRegNet. The checkpoint stays in outputs/MammoRegNet_<seed>/."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.coarse_deform:
        checkpoint_folder = f"outputs/MammoRegNet_deformable_{args.seed}/"
        default_output = checkpoint_folder
    elif args.external_affine:
        checkpoint_folder = f"outputs/MammoRegNet_{args.seed}/"
        default_output = f"outputs/Affine_MammoRegNet_{args.seed}/"
    else:
        checkpoint_folder = f"outputs/MammoRegNet_{args.seed}/"
        default_output = checkpoint_folder
    output_folder = args.output_folder or default_output
    if not output_folder.endswith(("/", "\\")):
        output_folder += "/"
    os.makedirs(output_folder, exist_ok=True)
    best_model = args.model_path or os.path.join(
        checkpoint_folder, "models", "model_249.pth"
    )
    base_dir = Path("Dataset/evaluation")
    masks_dir = Path("Dataset/evaluation_masks")
    subdatasets = ["INBreast", "KAU-BCMD", "NLBS", "CSAW-CC"]
    json_file = output_folder + 'mammoregnet.json'
    metadata_path = "Dataset/evaluation/metadata_evaluation_correct.csv"
    metadata = pd.read_csv(metadata_path)

    print(f"seed: {args.seed}")
    print(f"coarse_affine: {not args.coarse_deform}")
    print(f"external_affine: {args.coarse_deform or args.external_affine}")
    print(f"output_folder: {output_folder}")
    print(f"model: {best_model}")

    data_path = "Dataset/"
    shutil.copy2(data_path + 'moving_landmarks.xml', output_folder + 'moving_landmarks.xml')
    shutil.copy2(data_path + 'fixed_landmarks_1.xml', output_folder + 'fixed_landmarks_1.xml')
    shutil.copy2(data_path + 'fixed_landmarks_2.xml', output_folder + 'fixed_landmarks_2.xml')
    process_annotations(output_folder + 'moving_landmarks.xml')
    process_annotations(output_folder + 'fixed_landmarks_1.xml')
    process_annotations(output_folder + 'fixed_landmarks_2.xml')

    mse_list = []
    ssim_list = []
    mi_list = []
    cc_list = []
    dice_list = []
    njd_list = []
    vram_list = []
    runtime_list = []

    for subdataset in subdatasets:
        subdataset_path = base_dir / subdataset
        if not subdataset_path.exists():
            print(f"Folder not found: {subdataset_path}")
            continue

        print(f"\n{'='*50}")
        print(f"PROCESSING: {subdataset}")
        print(f"{'='*50}")

        for patient_folder in sorted(subdataset_path.iterdir()):
            if not patient_folder.is_dir():
                continue

            patient_id = patient_folder.name
            image_files = []
            for ext in ["*.jpg", "*.png"]:
                image_files.extend(patient_folder.glob(ext))
            
            f1, f2 = get_file_order(image_files, subdataset, metadata)
            matrix_path = f"affine_transforms/{patient_id}.mat"

            print(f"\n[Patient: {patient_id}]")

            before_img = Image.open(f1).convert('L')
            after_img = Image.open(f2).convert('L')

            before_img = before_img.resize((896, 1152), Image.Resampling.LANCZOS)
            after_img = after_img.resize((896, 1152), Image.Resampling.LANCZOS)

            before_arr = np.array(before_img)
            after_arr = np.array(after_img)

            output_dir = Path(output_folder) / subdataset / patient_id
            output_dir.mkdir(parents=True, exist_ok=True)
            f1_path = output_dir / f"{f1.stem}.png"
            f2_path = output_dir / f"{f2.stem}.png"

            m1 = Image.open(masks_dir / subdataset / patient_id / f"{f1.stem}.png").convert('L')
            m2 = Image.open(masks_dir / subdataset / patient_id / f"{f2.stem}.png").convert('L')

            m1 = m1.resize((896, 1152), Image.Resampling.NEAREST)
            m2 = m2.resize((896, 1152), Image.Resampling.NEAREST)

            m1_arr = np.array(m1)
            m2_arr = np.array(m2)
            m1_arr = (m1_arr == 255).astype(np.float32)
            m2_arr = (m2_arr == 255).astype(np.float32)

            p = output_dir / "mask.png"

            output_path = output_dir / f"{f1.stem}_{f2.stem}.png"
            before_arr = before_arr * (m1_arr).astype(np.uint8)
            after_arr = after_arr * (m2_arr).astype(np.uint8)

            try:
                mse, ssim, mi, cc, dice, njd, vram, runtime =\
                registration(before_arr, after_arr, m1_arr, m2_arr,
                             output_path, f1_path, f2_path, p, best_model,
                             output_folder + 'fixed_landmarks_1.xml',
                             output_folder + 'fixed_landmarks_2.xml',
                             matrix_path,
                             coarse_affine=not args.coarse_deform,
                             apply_external_affine=args.coarse_deform or args.external_affine)

                if os.path.exists(json_file):
                    with open(json_file, 'r', encoding='utf-8') as f:
                        try:
                            results = json.load(f)
                        except json.JSONDecodeError:
                            results = {}
                else:
                    results = {}

                results[str(output_path)] = {
                    "MSE": round(mse, 6),
                    "SSIM": round(ssim, 6),
                    "MI": round(mi, 6),
                    "CC": round(cc, 6),
                    "DSC": round(dice, 6),
                    "NJD": round(njd, 6),
                    "VRAM": round(vram, 6),
                    "Runtime": round(runtime, 6)
                }
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                mse_list.append(mse)
                ssim_list.append(ssim)
                mi_list.append(mi)
                cc_list.append(cc)
                dice_list.append(dice)
                njd_list.append(njd)
                vram_list.append(vram)
                runtime_list.append(runtime)

            except Exception as e:
                print(f"Error creating images for {patient_id}: {e}")

    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}
    else:
        results = {}

    results["Mean Metrics"] = {
                            "MSE": round(np.mean(mse_list), 6),
                            "SSIM": round(np.mean(ssim_list), 6),
                            "MI": round(np.mean(mi_list), 6),
                            "CC": round(np.mean(cc_list), 6),
                            "DSC": round(np.mean(dice_list), 6),
                            "NJD": round(np.mean(njd_list), 6),
                            "VRAM": round(np.mean(vram_list), 6),
                            "Runtime": round(np.mean(runtime_list), 6)
                        }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Mean MSE: {np.mean(mse_list):.4f} +- {np.std(mse_list):.4f}")
    print(f"Mean SSIM: {np.mean(ssim_list):.4f} +- {np.std(ssim_list):.4f}")
    print(f"Mean Mutual Information: {np.mean(mi_list):.4f} +- {np.std(mi_list):.4f}")
    print(f"Mean Correlation Coefficient: {np.mean(cc_list):.4f} +- {np.std(cc_list):.4f}")
    print(f"Mean dice: {np.mean(dice_list):.4f} +- {np.std(dice_list):.4f}")
    print(f"Mean runtime: {np.mean(runtime_list):.4f}")
    print(f"Mean njd: {np.mean(njd_list):.10f} +- {np.std(njd_list):.10f}")
    print(f"Mean vram: {np.mean(vram_list):.4f}")

    calc_tre(output_folder + 'moving_landmarks.xml',
             output_folder + 'fixed_landmarks_1.xml',
             output_folder + 'fixed_landmarks_2.xml')


if __name__ == "__main__":
    main()
