import os
import tempfile
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1" 
# ANTs writes large intermediates; point TMPDIR at a fast local disk if needed.
os.environ.setdefault('TMPDIR', os.environ.get('ANTS_TMPDIR', tempfile.gettempdir()))

import ants
import shutil
import pandas as pd
from PIL import Image
from pathlib import Path
import numpy as np
import argparse


def registration(before_img, after_img, moving_mask, fixed_mask,
                 img1_path, img2_path, mask1_path, mask2_path,
                 matrix_path):
    fixed_img = after_img
    moving_img = before_img

    fixed_ants = ants.from_numpy(fixed_img)
    moving_ants = ants.from_numpy(moving_img)

    reg = ants.registration(
        fixed=fixed_ants,
        moving=moving_ants,
        type_of_transform="Affine",
        random_seed=7
    )
    shutil.copy2(reg['fwdtransforms'][0], matrix_path)

    warped_img = ants.apply_transforms(
        fixed=fixed_ants,
        moving=moving_ants,
        transformlist=reg['fwdtransforms'],
        interpolator='linear',
    )

    warped_np = warped_img.numpy()
    warped = warped_np.astype(np.uint8)
    warped = Image.fromarray(warped)
    warped.save(img1_path) 
    img2 = Image.fromarray(after_img)
    img2.save(img2_path)

    moving_mask_ants = ants.from_numpy(moving_mask)
    fixed_mask_ants = ants.from_numpy(fixed_mask)
    warped_mask = ants.apply_transforms(
        fixed=fixed_mask_ants,
        moving=moving_mask_ants,
        transformlist=reg['fwdtransforms'],
        interpolator='genericLabel',
        verbose=False
    )

    warped_mask = warped_mask.numpy()
    image = Image.fromarray((warped_mask*255).astype(np.uint8))
    image.save(mask1_path)
    image = Image.fromarray((fixed_mask*255).astype(np.uint8))
    image.save(mask2_path)


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
    

def write_split_metadata(metadata, source_csv, source_root, dest_root):
    """Write the split CSV beside the warped images.

    Both training loaders read <data_root>/metadata_train_correct.csv and
    match rows by File Path. A byte copy would still name the source split,
    so the copy points each row at the png this script writes.
    """
    rewritten = metadata.copy()
    source_root = Path(source_root)
    dest_root = Path(dest_root)

    def rewrite(value):
        relative = Path(os.path.normpath(str(value))).relative_to(source_root)
        return (dest_root / relative).with_suffix(".png").as_posix()

    rewritten["File Path"] = rewritten["File Path"].map(rewrite)
    dest_csv = dest_root / Path(source_csv).name
    rewritten.to_csv(dest_csv, index=False)
    print(f"Wrote metadata: {dest_csv}")


def main():
    parser = argparse.ArgumentParser(description='Enabling affineregistration from scratch')
    parser.add_argument('--subset', help='Choose subset: train or evaluation.')
    args = parser.parse_args()
    if args.subset == 'train':
        subset = 'train'
    elif args.subset == 'evaluation':
        subset = 'evaluation'
    else:
        print("Invalid subset")
        return
    
    output_folder_images = f"Dataset/affine_{subset}"
    output_folder_masks = f"Dataset/affine_{subset}_masks"
    os.makedirs(output_folder_images, exist_ok=True)
    os.makedirs(output_folder_masks, exist_ok=True)
    os.makedirs("affine_transforms", exist_ok=True)
    base_dir = Path(f"Dataset/{subset}")
    masks_dir = Path(f"Dataset/{subset}_masks")
    subdatasets = ["INBreast","KAU-BCMD", "NLBS", "CSAW-CC"]
    metadata_path = f"Dataset/{subset}/metadata_{subset}_correct.csv"
    metadata = pd.read_csv(metadata_path)
    write_split_metadata(metadata, metadata_path, base_dir, output_folder_images)

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

            print(f"[Patient: {patient_id}]")
            before_img = Image.open(f1).convert('L')
            after_img = Image.open(f2).convert('L')
    
            before_img = before_img.resize((896, 1152), Image.Resampling.LANCZOS)
            after_img = after_img.resize((896, 1152), Image.Resampling.LANCZOS)

            before_arr = np.array(before_img)
            after_arr = np.array(after_img)

            output_dir_images = Path(output_folder_images) / subdataset / patient_id
            output_dir_masks = Path(output_folder_masks) / subdataset / patient_id
            output_dir_images.mkdir(parents=True, exist_ok=True)
            output_dir_masks.mkdir(parents=True, exist_ok=True)

            img1_path = output_dir_images / f"{f1.stem}.png"
            img2_path = output_dir_images / f"{f2.stem}.png"

            mask1_path = output_dir_masks / f"{f1.stem}.png"
            mask2_path = output_dir_masks / f"{f2.stem}.png"

            m1 = Image.open(masks_dir / subdataset / patient_id / f"{f1.stem}.png").convert('L')
            m2 = Image.open(masks_dir / subdataset / patient_id / f"{f2.stem}.png").convert('L')

            m1 = m1.resize((896, 1152), Image.Resampling.NEAREST)
            m2 = m2.resize((896, 1152), Image.Resampling.NEAREST)

            m1_arr = np.array(m1)
            m2_arr = np.array(m2)
            m1_arr = (m1_arr == 255).astype(np.uint8)
            m2_arr = (m2_arr == 255).astype(np.uint8)

            before_arr = before_arr * m1_arr
            after_arr = after_arr * m2_arr

            try:
                registration(before_arr, after_arr,
                             m1_arr, m2_arr,
                             img1_path, img2_path,
                             mask1_path, mask2_path,
                             matrix_path)

            except Exception as e:
                print(f"Error creating images for {patient_id}: {e}")


if __name__ == "__main__":
    main()