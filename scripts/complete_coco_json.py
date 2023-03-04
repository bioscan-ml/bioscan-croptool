# Adding some missing information that are needed for the model to train to the coco (The area of each semantic mask and a flag call iscrowd)
# Note: the area is not calculated properly, here we use the area of the bounding box instead of the area of the semantic mask.
import json
import shutil
import os
import argparse

from sklearn.model_selection import train_test_split

def add_missing_information_to_coco_json(coco_annotation_dict):
    images = coco_annotation_dict['images']
    annotations = coco_annotation_dict['annotations']

    # Adding missing information to the annotation data
    for i in images:
        i['file_name'] = os.path.basename(os.path.normpath(i['toras_path']))
    for i in annotations:
        i['area'] = i['bbox'][2] * i['bbox'][3]
        i['iscrowd'] = 0
    return images, annotations


def complete_coco_json(args):
    coco_annotation_file = open(os.path.join(args.input_dir, "coco_annotations.json"))
    coco_annotation_dict = json.load(coco_annotation_file)

    if 'categories' not in coco_annotation_dict.keys() or len(coco_annotation_dict['categories']) == 0:
        # Hard coded categories
        # If the original coco file does not contain categories of the insect, hardcode one.
        categories = [
            {
                "id": 0,
                "name": "insect",
                "supercategory": "N/A"
            }
        ]
    else:
        categories = coco_annotation_dict['categories']
    images, annotations = add_missing_information_to_coco_json(coco_annotation_dict)

    processed_coco_json = {'images': images, 'categories': categories, 'annotations': annotations}
    with open(os.path.join(args.input_dir, "coco_annotations_processed.json"), "w") as f:
        json.dump(processed_coco_json, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help="path to the folder that contains the images and coco file.")
    args = parser.parse_args()

    complete_coco_json(args)

