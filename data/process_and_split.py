# Adding some missing information that are needed for the model to train to the coco (The area of each semantic mask and a flag call iscrowd)
# Note: the area is not calculated properly, here we use the area of the bounding box instead of the area of the semantic mask.
import json
import shutil
import os
import argparse

from sklearn.model_selection import train_test_split


def create_directories(args):
    args.train_folder_path = os.path.join(args.dataset_name, "train")
    args.val_folder_path = os.path.join(args.dataset_name, "val")

    # Remove the exist
    if os.path.exists(args.dataset_name):
        shutil.rmtree(args.dataset_name)

    os.makedirs(args.dataset_name, exist_ok=True)
    os.makedirs(args.train_folder_path, exist_ok=True)
    os.makedirs(args.val_folder_path, exist_ok=True)


def add_missing_information_to_coco_json(coco_annotation_dict):
    images = coco_annotation_dict['images']
    annotations = coco_annotation_dict['annotations']

    # Adding missing information to the annotation data
    for i in images:
        i['file_name'] = i['toras_path'][15:]
    for i in annotations:
        i['area'] = i['bbox'][2] * i['bbox'][3]
        i['iscrowd'] = 0
    return images, annotations


def load_data_and_complete_coco_json(args):
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
    return images, annotations, categories


def split_data_and_copy_image(args, images, annotations, categories):
    ids = []
    for i in images:
        ids.append(i['id'])

    train_ids, val_ids = train_test_split(ids, test_size=0.5)
    train_images = []
    val_images = []

    train_annotations = []
    val_annotations = []

    for i in images:
        curr_id = i['id']
        src = i['toras_path']
        if curr_id in train_ids:
            train_images.append(i)
            dst = os.path.join(args.train_folder_path, i['file_name'])
        else:
            val_images.append(i)
            dst = os.path.join(args.val_folder_path, i['file_name'])
        shutil.copyfile(src, dst)

    for i in annotations:
        curr_id = i['id']
        if curr_id in train_ids:
            train_annotations.append(i)
        else:
            val_annotations.append(i)

    # Hard coded categories
    # As coco file

    # Save annotation data.
    train_dict = {'images': train_images, 'categories': categories, 'annotations': train_annotations}

    val_dict = {'images': val_images, 'categories': categories, 'annotations': val_annotations}

    with open(os.path.join(args.train_folder_path, "custom_train.json"), "w") as f:
        json.dump(train_dict, f, indent=4)
    with open(os.path.join(args.val_folder_path, "custom_val.json"), "w") as f:
        json.dump(val_dict, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help="path to the folder that contains the images and coco file.")
    parser.add_argument('--dataset_name', type=str, required=True, help="name of the dataset")
    args = parser.parse_args()

    create_directories(args)

    images, annotations, categories = load_data_and_complete_coco_json(args)

    split_data_and_copy_image(args, images, annotations, categories)
