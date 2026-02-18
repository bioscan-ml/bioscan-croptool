import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def collate_fn(batch):
    """
    Keep images as a list instead of stacking them.
    This allows batching images of different sizes; the feature_extractor will pad them later.
    """
    images = [item[0] for item in batch]
    image_names = [item[1] for item in batch]
    return images, image_names


class ImageFolderDataset(Dataset):
    def __init__(self, path_to_input_folder, transform=None, list_of_images=None):
        self.folder_path = path_to_input_folder
        # Only keep common image files, skip others (e.g., .json)
        valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
        if list_of_images is None:
            candidates = os.listdir(path_to_input_folder)
        else:
            candidates = list_of_images

        self.image_names = [
            name for name in candidates
            if os.path.splitext(name)[1].lower() in valid_exts
        ]

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.folder_path, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, image_name





def init_loader_with_folder_name_and_list_of_images(
    path_to_input_folder,
    batch_size,
    list_of_images=None,
    num_workers=0,
    pin_memory=False,
):
    return DataLoader(
        ImageFolderDataset(path_to_input_folder, list_of_images=list_of_images),
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )