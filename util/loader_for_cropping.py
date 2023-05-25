import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


class ImageFolderDataset(Dataset):
    def __init__(self, path_to_input_folder, transform=None, list_of_images=None):
        self.folder_path = path_to_input_folder
        if list_of_images is None:
            self.image_names = os.listdir(path_to_input_folder)
        else:
            self.image_names = list_of_images

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





def init_loader_with_folder_name_and_list_of_images(path_to_input_folder, batch_size, list_of_images = None):
    return DataLoader(ImageFolderDataset(path_to_input_folder, list_of_images=list_of_images), batch_size=batch_size)