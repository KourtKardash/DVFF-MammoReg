import os, glob
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import pandas as pd
from pathlib import Path


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

class MammoDataset(Dataset):
    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir
        self.transforms = transforms
        self.subfolders = []
        for subdir in ['INBreast', 'KAU-BCMD', 'NLBS', 'CSAW-CC']:
            subdir_path = os.path.join(root_dir, subdir)
            pair_dirs = glob.glob(os.path.join(subdir_path, '*'))
            self.subfolders.extend(pair_dirs)

        metadata_path = f"Dataset/affine_train/metadata_train_correct.csv"
        self.metadata = pd.read_csv(metadata_path)

    def __getitem__(self, index):
        subfolder = random.choice(self.subfolders)
        images = glob.glob(os.path.join(subfolder, '*.jpg')) + glob.glob(os.path.join(subfolder, '*.png'))
        moving_img_path, fixed_img_path = get_file_order(images, Path(subfolder).parent.name, self.metadata)

        fixed_img = self.load_image(fixed_img_path)
        moving_img = self.load_image(moving_img_path)

        fixed_img, moving_img = self.transforms([fixed_img, moving_img])

        fixed_img = np.ascontiguousarray(fixed_img)  # (1, H, W)
        moving_img = np.ascontiguousarray(moving_img)  # (1, H, W)

        fixed_img = torch.from_numpy(fixed_img).float().unsqueeze(0)
        moving_img = torch.from_numpy(moving_img).float().unsqueeze(0)
        
        combined_img = torch.cat((fixed_img, moving_img), dim=0)
        return combined_img 

    def load_image(self, path):
        from PIL import Image        
        img = Image.open(path).convert('L')
        img = np.array(img, dtype=np.float32)
        img = img / 255.0
        return img

    def __len__(self):
        return 1000