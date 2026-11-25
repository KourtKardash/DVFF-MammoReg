import os

import json
import ants
import pandas as pd
import time
import xml.etree.ElementTree as ET
import shutil
from PIL import Image
from pathlib import Path

from skimage.metrics import normalized_mutual_information as mi
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error  as mse

import numpy as np

from rTRE import calc_tre


def get_file_order(file_names, subdataset, metadata):
    path_to_meta = {}
    for _, row in metadata.iterrows():
        file_path = row['File Path']
        path_to_meta[file_path] = row
    
    path1 = str(file_names[0])
    path2 = str(file_names[1])

    meta1 = path_to_meta[path1]
    meta2 = path_to_meta[path2]
    if subdataset in ["NLBS", "KAU-BCMD"]:
        age1 = meta1['Patient age']
        age2 = meta2['Patient age']

        if age1 <= age2:
            return file_names[0], file_names[1]
        else:
            return file_names[1], file_names[0]

    elif subdataset in ["INBreast", "CSAW-CC"]:
        date1 = str(meta1['Study Date'])
        date2 = str(meta2['Study Date'])
        
        try:
            date1_val = int(date1)
            date2_val = int(date2)
            
            if date1_val <= date2_val:
                return file_names[0], file_names[1]
            else:
                return file_names[1], file_names[0]
        except ValueError:
            print(f"Warning: Could not parse dates, using default order")
            return file_names[0], file_names[1]


def dice_coefficient(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    size1 = mask1.sum()
    size2 = mask2.sum()
    dice = (2.0 * intersection) / (size1 + size2)
    return dice


def apply_transform_to_landmarks(matrix_path, image_name, output_xml_path):
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
            transformed_points = ants.apply_transforms_to_points(
                dim=2,
                points=points_df,
                transformlist=[matrix_path]
            ) 
            for i, points_tag in enumerate(index_to_tag):
                new_x = transformed_points['x'].iloc[i]
                new_y = transformed_points['y'].iloc[i]
                points_tag.set('points', f"{new_y:.2f},{new_x:.2f}")

    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)


def registration(before_img, after_img, moving_mask, fixed_mask, output_path,
                 f1_path, f2_path, p_mask, xml_path1, xml_path2, matrix_path):
    fixed_img = after_img
    moving_img = before_img

    fixed_ants = ants.from_numpy(fixed_img)
    moving_ants = ants.from_numpy(moving_img)

    t = time.time()

    warped_img = ants.apply_transforms(
        fixed=fixed_ants,
        moving=moving_ants,
        transformlist=[matrix_path],
        interpolator='linear',
    )

    warped_np = warped_img.numpy()
    warped = warped_np.astype(np.uint8)
    warped = Image.fromarray(warped)
    warped.save(output_path) 
    runtime = time.time() - t
    img1 = Image.fromarray(before_img)
    img2 = Image.fromarray(after_img)

    img1.save(f1_path)
    img2.save(f2_path)

    mask_ants = ants.from_numpy(moving_mask)
    warped_mask = ants.apply_transforms(
        fixed=fixed_ants,
        moving=mask_ants,
        transformlist=[matrix_path],
        interpolator='genericLabel',
        verbose=False
    )

    warped_mask = warped_mask.numpy()
    image = Image.fromarray((warped_mask*255).astype(np.uint8))
    image.save(p_mask)

    image_name = f1_path.parent.name
    apply_transform_to_landmarks(matrix_path, image_name + '.png', xml_path1)
    apply_transform_to_landmarks(matrix_path, image_name + '.png', xml_path2)

    warped = np.array(warped)
    return mse(after_img, warped), ssim(after_img, warped, data_range=255),\
           mi(after_img, warped), np.corrcoef(after_img.flatten(), warped.flatten())[0, 1], \
           dice_coefficient(warped_mask, fixed_mask), runtime


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


def main():
    data_path = "Dataset/"
    output_folder = "outputs/Affine/"
    os.makedirs(output_folder, exist_ok=True)
    shutil.copy2(data_path + 'moving_landmarks.xml', output_folder + 'moving_landmarks.xml')
    shutil.copy2(data_path + 'fixed_landmarks_1.xml', output_folder + 'fixed_landmarks_1.xml')
    shutil.copy2(data_path + 'fixed_landmarks_2.xml', output_folder + 'fixed_landmarks_2.xml')
    process_annotations(output_folder + 'moving_landmarks.xml')
    process_annotations(output_folder + 'fixed_landmarks_1.xml')
    process_annotations(output_folder + 'fixed_landmarks_2.xml')

    base_dir = Path("Dataset/evaluation")
    masks_dir = Path("Dataset/evaluation_masks")
    subdatasets = ["INBreast","KAU-BCMD", "NLBS", "CSAW-CC"]
    json_file = output_folder + 'Affine.json'
    metadata = pd.read_csv(data_path + 'evaluation/metadata_evaluation_correct.csv')

    mse_list = []
    ssim_list = []
    mi_list = []
    cc_list = []
    dice_list = []
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
            
            file_names = get_file_order(image_files, subdataset, metadata)
            
            matrix_path = f"affine_transforms/{patient_id}.mat"

            print(f"[Patient: {patient_id}]")
            before_img = Image.open(file_names[0]).convert('L')
            after_img = Image.open(file_names[1]).convert('L')
    
            before_img = before_img.resize((896, 1152), Image.Resampling.LANCZOS)
            after_img = after_img.resize((896, 1152), Image.Resampling.LANCZOS)

            before_arr = np.array(before_img)
            after_arr = np.array(after_img)

            output_dir = Path(output_folder) / subdataset / patient_id
            output_dir.mkdir(parents=True, exist_ok=True)
            f1_path = output_dir / f"{file_names[0].stem}.png"
            f2_path = output_dir / f"{file_names[1].stem}.png"

            m1 = Image.open(masks_dir / subdataset / patient_id / f"{file_names[0].stem}.png").convert('L')
            m2 = Image.open(masks_dir / subdataset / patient_id / f"{file_names[1].stem}.png").convert('L')

            m1 = m1.resize((896, 1152), Image.Resampling.NEAREST)
            m2 = m2.resize((896, 1152), Image.Resampling.NEAREST)

            m1_arr = np.array(m1)
            m2_arr = np.array(m2)
            m1_arr = (m1_arr == 255).astype(np.uint8)
            m2_arr = (m2_arr == 255).astype(np.uint8)

            p = output_dir / "mask.png"

            output_path = output_dir / f"{file_names[0].stem}_{file_names[1].stem}.png"

            before_arr = before_arr * m1_arr
            after_arr = after_arr * m2_arr

            try:
                mse, ssim, mi, cc, dice, runtime = registration(before_arr, after_arr,
                                                                m1_arr, m2_arr, output_path,
                                                                f1_path, f2_path, p,
                                                                output_folder + 'fixed_landmarks_1.xml',
                                                                output_folder + 'fixed_landmarks_2.xml',
                                                                str(matrix_path))

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
                    "runtime": round(runtime, 6)
                }
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                mse_list.append(mse)
                ssim_list.append(ssim)
                mi_list.append(mi)
                cc_list.append(cc)
                dice_list.append(dice)
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

    calc_tre(output_folder + 'moving_landmarks.xml',
             output_folder + 'fixed_landmarks_1.xml',
             output_folder + 'fixed_landmarks_2.xml')


if __name__ == "__main__":
    main()