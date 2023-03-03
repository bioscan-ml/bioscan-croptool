import argparse
import os
import sys
from os.path import dirname, abspath
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import DetrFeatureExtractor

project_dir = dirname(dirname(abspath(__file__)))
sys.path.append(project_dir)
from model.detr import load_model_from_ckpt
from util.visualize_and_process_bbox import get_bbox_from_output, scale_bbox


def crop_image(args, model, feature_extractor):
    """
    Crop and save images based on the predicted bounding boxes from the model.
    :param model: Detr model that loaded from the checkpoint.
    :param feature_extractor: A ResNet50 model as a standard image extractor.
    """
    for filename in tqdm(os.listdir(args.input_dir)):
        f = os.path.join(args.input_dir, filename)
        if os.path.isfile(f):
            try:
                image = Image.open(f)
            except:
                print("Image not found in: " + f)
                exit(1)
                # TODO: https://stackoverflow.com/questions/31751464/how-do-i-close-an-image-opened-in-pillow
            encoding = feature_extractor(images=image, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze().unsqueeze(0)
            outputs = model(pixel_values=pixel_values, pixel_mask=None)
            bbox = get_bbox_from_output(outputs, image).detach().numpy()
            bbox = np.round(bbox, 0)
            left, top, right, bottom = scale_bbox(args, bbox[0], bbox[1], bbox[2], bbox[3])
            image_size = image.size
            cropped_img = image.crop((max(left, 0), max(top, 0), min(right, image_size[0]), min(bottom, image_size[1])))

            cropped_img.save(os.path.join(args.output_dir, filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help="Folder that contains the original images.")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to the checkpoint.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Number of images in each batch.")
    parser.add_argument('--output_dir', type=str, default="cropped_image",
                        help="Folder that will contain the cropped images.")
    parser.add_argument('--crop_ratio', type=float, default=1.4,
                        help="Scale the bbox to crop larger or small area.")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

    model = load_model_from_ckpt(args)

    crop_image(args, model, feature_extractor)
