import argparse
import os
import sys

import torch
from PIL import Image
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DetrFeatureExtractor

from model import Detr
from util.coco_relevent import CocoDetection
from util.detr.datasets import get_coco_api_from_dataset
from util.detr.datasets.coco_eval import CocoEvaluator
from util.visualize_and_process_bbox import visualize_predictions


def collate_fn(batch):
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    pixel_values = [item[0] for item in batch]
    encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch


def create_dataloader(args):
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available, please check your CUDA and torch version.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.train_folder = os.path.join(args.data_dir, 'train')
    args.val_folder = os.path.join(args.data_dir, 'val')
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

    train_dataset = CocoDetection(img_folder=args.train_folder, feature_extractor=feature_extractor)
    val_dataset = CocoDetection(img_folder=args.val_folder, feature_extractor=feature_extractor,
                                train=False)
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=args.batch_size)
    cats = train_dataset.coco.cats
    id2label = {k: v['name'] for k, v in cats.items()}

    return train_dataloader, val_dataset, val_dataloader, device, feature_extractor, id2label


def initialize_model(args, train_dataloader, val_dataloader):
    return Detr(lr=args.learning_rate, lr_backbone=args.lr_backbone, weight_decay=args.weight_decay,
                train_dataloader=train_dataloader, val_dataloader=val_dataloader)


def initialize_trainer(args):
    return Trainer(gpus=args.gpus, max_steps=args.max_steps, gradient_clip_val=args.gradient_clip_val,
                   default_root_dir=args.output_dir)


def visualize(args, val_dataset, model, id2label):
    for i in range(args.visualize_number):
        pixel_values, target = val_dataset[i]
        pixel_values = pixel_values.unsqueeze(0).to(device)


        outputs = model(pixel_values=pixel_values, pixel_mask=None)

        image_id = target['image_id'].item()
        image = val_dataset.coco.loadImgs(image_id)[0]

        print(image['file_name'])
        image = Image.open(os.path.join(args.val_folder, image['file_name']))
        visualize_predictions(image, outputs, id2label)


def evaluation(model, val_dataset, val_dataloader, device, feature_extractor):
    base_ds = get_coco_api_from_dataset(val_dataset)  # this is actually just calling the coco attribute

    iou_types = ['bbox']
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    model.to(device)
    model.eval()
    print("Running evaluation...")
    for idx, batch in enumerate(tqdm(val_dataloader)):
        # get the inputs
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in
                  batch["labels"]]  # these are in DETR format, resized + normalized

        # forward pass
        outputs = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = feature_extractor.post_process(outputs, orig_target_sizes)  # convert outputs of model to COCO api
        res = {target['image_id'].item(): output for target, output in zip(labels, results)}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help="path to the directory that contains the split data")
    parser.add_argument('--batch_size', type=str, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=300)
    parser.add_argument('--gradient_clip_val', type=float, default=0.1)
    parser.add_argument('--output_dir', type=str, required=True, help="The path used to store the checkpoint")
    parser.add_argument('--visualize_number', type=int, default=5)

    args = parser.parse_args()

    train_dataloader, val_dataset, val_dataloader, device, feature_extractor, id2label = create_dataloader(args)

    model = initialize_model(args, train_dataloader, val_dataloader)

    trainer = initialize_trainer(args)

    trainer.fit(model)

    evaluation(model, val_dataset, val_dataloader, device, feature_extractor)

    visualize(args, val_dataset, model, id2label)
