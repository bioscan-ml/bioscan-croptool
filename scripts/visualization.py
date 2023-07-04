import argparse
from PIL import Image
import os
import sys
from transformers import DetrFeatureExtractor
from crop_images import load_model_from_ckpt
from project_path import project_dir
from util.coco_dataset import DetectionDataset
from util.visualize_and_process_bbox import visualize_predictions


def visualize(args, val_dataset, model, id2label):
    for i in range(args.visualize_number):
        pixel_values, target = val_dataset[i]
        pixel_values = pixel_values.unsqueeze(0)
        outputs = model(pixel_values=pixel_values, pixel_mask=None)

        image_id = target['image_id'].item()
        image = val_dataset.coco.loadImgs(image_id)[0]

        image = Image.open(os.path.join(args.val_folder, image['file_name']))
        visualize_predictions(image, outputs, id2label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Path to the directory that contains the split data.")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to the checkpoint.")
    parser.add_argument('--visualize_number', type=int, default=5)
    args = parser.parse_args()

    model = load_model_from_ckpt(args)
    args.val_folder = os.path.join(args.data_dir, 'val')

    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

    val_dataset = DetectionDataset(img_folder=args.val_folder, feature_extractor=feature_extractor,
                                   train=False)
    categories = val_dataset.coco.cats
    id2label = {k: v['name'] for k, v in categories.items()}

    visualize(args, val_dataset, model, id2label)
