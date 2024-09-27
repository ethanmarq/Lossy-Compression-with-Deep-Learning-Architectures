import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image = t(image)
            target = t(target)
        target = torch.tensor(np.array(target), dtype=torch.int64)
        
        color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        image = color_jitter(image)
        image = transforms.ToTensor()(image)
        rand_erase = transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
        image = rand_erase(image)
        return image, target

class SegmentationDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, transforms=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transforms = transforms
        self.image_mask_pairs = self._collect_image_mask_pairs()
        print(f"Matched {len(self.image_mask_pairs)} image-mask pairs.")

    def _collect_image_mask_pairs(self):
        image_mask_pairs = []
        image_dir = self.imgs_dir
        mask_dir = self.masks_dir
        
        images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpeg', '.jpg'))])
        masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpeg', '.jpg'))])

        mask_dict = {os.path.splitext(os.path.basename(mask).split('_')[1])[0]: mask for mask in masks}
        for img in images:
            key = os.path.splitext(os.path.basename(img).split('_')[1])[0]
            if key in mask_dict:
                image_mask_pairs.append((img, mask_dict[key]))
            else:
                print(f"No matching mask for image: {img}")

        return image_mask_pairs

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transforms:
            image, mask = self.transforms(image, mask)
        return image, mask