from __future__ import print_function, division
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
class CustomDatasetForDetection(Dataset):
    def __init__(self, img_folder, feature_extractor):
        self.image_paths = []
        for filename in os.listdir(img_folder):
            try:
                file_path = os.path.join(img_folder, filename)
                _ = Image.open(file_path)
                self.image_paths.append(file_path)
            except:
                continue
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_paths[idx]
        image = Image.open(img_name)

        encoding = self.feature_extractor(images=image, return_tensors="pt")

        return encoding
