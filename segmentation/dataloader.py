import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch


class MammographyDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, train=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.train = train
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        image = image.resize((928, 1168), Image.Resampling.LANCZOS)
        mask = mask.resize((928, 1168), Image.Resampling.LANCZOS)

        image = np.array(image)
        mask = np.array(mask)

        image = image.astype(np.float32) / 255.0
        mask = (mask > 180).astype(np.float32) 
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.unsqueeze(0)
        else:
            image = torch.from_numpy(image).unsqueeze(0)
            mask = torch.from_numpy(mask).unsqueeze(0)
            
        return image, mask

def get_train_transform():
    return A.Compose([
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.2),
        ToTensorV2()
    ])

def get_val_transform():
    return A.Compose([
        ToTensorV2()
    ])
