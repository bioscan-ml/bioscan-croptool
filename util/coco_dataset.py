import torchvision
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms

class DetectionDataset(torchvision.datasets.CocoDetection):
    """
    A torch dataset that inherit from CocoDetection class.
    """
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(img_folder, "custom_train.json" if train else "custom_val.json")
        super(DetectionDataset, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        """
        :return: pixel_values
        """
        # read in PIL image and target in COCO format
        img, target = super(DetectionDataset, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return pixel_values, target
