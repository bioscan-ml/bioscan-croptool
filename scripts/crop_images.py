import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import DetrFeatureExtractor
from model import Detr
from util.visualize_and_process_bbox import rescale_bboxes


def get_bbox_from_output(pred, image):
    probas = pred.logits.softmax(-1)[0, :, :-1]
    probas_ = probas.max(-1).values
    arg_max = probas_.argmax()
    probas_ = F.one_hot(arg_max, num_classes=len(probas_))
    keep = probas_ > 0.5
    bboxes_scaled = rescale_bboxes(pred.pred_boxes[0, keep].cpu(), image.size)
    return bboxes_scaled[0]


def scale_bbox(args, left, top, right, bottom):
    x_range = right - left
    y_range = bottom - top

    x_change = x_range * args.crop_ratio
    y_change = y_range * args.crop_ratio

    left = int(left - x_change / 2)
    right = int(right + x_change / 2)
    top = int(top - y_change / 2)
    bottom = int(bottom + y_change / 2)

    return left, top, right, bottom


def load_model_from_ckpt(args, device):
    # initial model and data path
    model = Detr.load_from_checkpoint(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4,
                                      checkpoint_path=args.checkpoint_path)
    model.eval()
    model.to(device)
    return model


def crop_image(args, model, device):
    # crop image
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    for filename in tqdm(os.listdir(args.input_dir)):
        f = os.path.join(args.input_dir, filename)
        if os.path.isfile(f):
            image = Image.open(f)
            encoding = feature_extractor(images=image, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze().unsqueeze(0).to(device)
            outputs = model(pixel_values=pixel_values, pixel_mask=None)
            bbox = get_bbox_from_output(outputs, image).detach().numpy()
            bbox = np.round(bbox, 0)
            left, top, right, bottom = scale_bbox(args, bbox[0], bbox[1], bbox[2], bbox[3])
            image_size = image.size
            cropped_img = image.crop((max(left, 0), max(top, 0), min(right, image_size[1]), min(bottom, image_size[0])))

            cropped_img.save(os.path.join(args.output_dir, filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help="folder that contains the original images.")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="path to the checkpoint.")
    parser.add_argument('--output_dir', type=str, default="cropped_image",
                        help="folder that will contain the cropped images.")
    parser.add_argument('--crop_ratio', type=float, default=1.1,
                        help="scale the bbox to crop larger or small area.")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_from_ckpt(args, device)

    crop_image(args, model, device)
