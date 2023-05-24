import argparse
import os
import sys
from os.path import dirname, abspath
import numpy as np
from tqdm import tqdm
from transformers import DetrFeatureExtractor
from PIL import Image, ImageDraw, ImageOps
import shutil

project_dir = dirname(dirname(abspath(__file__)))
sys.path.append(project_dir)
from model.detr import load_model_from_ckpt
from util.visualize_and_process_bbox import get_bbox_from_output, scale_bbox

"""
This is a special one time script for special purpose,  will be removed from the repo after the task is done.
"""


def unzip_tars_to_folder(path_to_tar, path_to_unzipped_folder):

    # open file
    os.system('tar -xvf ' + path_to_tar + ' --strip=7 -C ' + path_to_unzipped_folder)

    file.close()

def get_size_with_aspect_ratio(image_size, size):
    # Reference:
    # https://huggingface.co/transformers/v4.9.2/_modules/transformers/models/detr/feature_extraction_detr.html
    w, h = image_size

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)

    return ow, oh


def expand_image(args, image, size, direction):
    border_color = (args.background_color_R, args.background_color_G, args.background_color_B)
    border_image = None
    if direction == 'left':
        border_image = ImageOps.expand(image, border=(size, 0, 0, 0), fill=border_color)
    elif direction == 'top':
        border_image = ImageOps.expand(image, border=(0, size, 0, 0), fill=border_color)
    elif direction == 'right':
        border_image = ImageOps.expand(image, border=(0, 0, size, 0), fill=border_color)
    elif direction == 'bottom':
        border_image = ImageOps.expand(image, border=(0, 0, 0, size), fill=border_color)
    else:
        exit("Wrong expand direction.")

    return border_image


def rotate_image_and_bbox_if_necesscary(image, left, top, right, bottom):
    image_size = image.size
    image = image.rotate(90, expand=True)
    new_left = top
    new_right = bottom
    new_bottom = image_size[0] - left
    new_top = image_size[0] - right

    return image, new_left, new_top, new_right, new_bottom


def change_size_to_4_3(left, top, right, bottom):
    width = right - left
    height = bottom - top
    if width < height / 3 * 4:
        extend_length = height / 3 * 4 - width
        left = left - extend_length / 2
        right = right + extend_length - extend_length / 2

    elif height < width / 4 * 3:
        extend_length = width / 4 * 3 - height
        top = top - extend_length / 2
        bottom = bottom + extend_length - extend_length / 2
    return left, top, right, bottom


def crop_image(args, model, feature_extractor):
    """
    Crop and save images based on the predicted bounding boxes from the model.
    :param model: Detr model that loaded from the checkpoint.
    :param feature_extractor: A ResNet50 model as a standard image extractor.
    """

    list_of_un_cropped_images = []
    path_to_cropped_folder = os.path.join(args.local_output_dir, "cropped_" + args.current_image_folder_name)
    path_to_cropped_and_resized_folder = os.path.join(args.local_output_dir,
                                                      "cropped_resized_" + args.current_image_folder_name)

    os.makedirs(path_to_cropped_folder, exist_ok=True)
    os.makedirs(path_to_cropped_and_resized_folder, exist_ok=True)
    list_of_cropped_images = os.listdir(path_to_cropped_folder)
    list_of_cropped_and_resized_images = os.listdir(path_to_cropped_and_resized_folder)

    pbar_in_crop_image = tqdm(os.listdir(args.input_dir))
    for filename in pbar_in_crop_image:
        pbar_in_crop_image.set_description("Checking the un-cropped images.")
        name_of_cropped_image = "cropped_" + filename
        name_of_cropped_and_resized_image = "cropped_resized_" + filename
        if name_of_cropped_image not in list_of_cropped_images or name_of_cropped_and_resized_image not in list_of_cropped_and_resized_images:
            list_of_un_cropped_images.append(filename)
    pbar_in_crop_image = tqdm(list_of_un_cropped_images)
    for filename in pbar_in_crop_image:
        pbar_in_crop_image.set_description("Cropping images.")
        f = os.path.join(args.input_dir, filename)
        if os.path.isfile(f):
            try:
                image = Image.open(f)
            except:
                print("Image not found in: " + f)
                exit(1)
            encoding = feature_extractor(images=image, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze().unsqueeze(0)
            outputs = model(pixel_values=pixel_values, pixel_mask=None)
            bbox = get_bbox_from_output(outputs, image).detach().numpy()
            bbox = np.round(bbox, 0)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            if args.show_bbox:
                draw = ImageDraw.Draw(image)
                draw.rectangle((left, top, right, bottom), outline=(255, 0, 0), width=args.width_of_bbox)
            left, top, right, bottom = scale_bbox(args, left, top, right, bottom)

            image_size = image.size

            if args.fix_ratio:
                width = right - left
                height = bottom - top

                if height > width and args.rotate_image:
                    image, left, top, right, bottom = rotate_image_and_bbox_if_necesscary(image, left, top, right,
                                                                                          bottom)
                    image_size = image.size

                left, top, right, bottom = change_size_to_4_3(left, top, right, bottom)
                left = round(left)
                top = round(top)
                right = round(right)
                bottom = round(bottom)
                # if left < 0 or top < 0 or right > image_size[0] or bottom > image_size[1]:
                #     print("Please crop this image manually: " + f)

            # cropped_img = image.crop((max(left, 0), max(top, 0), min(right, image_size[0]), min(bottom, image_size[1])))

            # Check if width is smaller than the bbox

            if left < 0:
                border_size = 0 - left
                right = right - left
                left = 0
                image = expand_image(args, image, border_size, 'left')

            if top < 0:
                border_size = 0 - top
                bottom = bottom - top
                top = 0
                image = expand_image(args, image, border_size, 'top')

            if right > image.size[0]:
                border_size = right - image.size[0] + 1
                image = expand_image(args, image, border_size, 'right')

            if bottom > image.size[1]:
                border_size = bottom - image.size[1] + 1
                image = expand_image(args, image, border_size, 'bottom')

            cropped_img = image.crop((left, top, right, bottom))

            cropped_img.save(os.path.join(path_to_cropped_folder, "cropped_" + filename))
            if args.save_resized:
                new_width, new_height = get_size_with_aspect_ratio(image.size, 256)
                cropped_and_resized_img = cropped_img.resize((new_width, new_height))
                cropped_and_resized_img.save(os.path.join(path_to_cropped_and_resized_folder,
                                                          "cropped_resized_" + filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_dir', type=str, required=True,
    #                     help="Folder that contains the original images.")
    parser.add_argument('--input_txt', type=str, required=True,
                        help="Path to the txt file that contains the names of tar files you want to download to local."
                             "storage.")
    parser.add_argument('--remote_input_dir', type=str, required=True,
                        help="Path to the directory that contains the image folders you want to download to your local "
                             "storage.")
    parser.add_argument('--local_input_dir', type=str, default="local_input_dir",
                        help="Path to the directory that you want to save the un-cropped image directories and the "
                             "checkpoint")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to the checkpoint that needed to be copied to the local.")
    parser.add_argument('--local_output_dir', type=str, default="local_output_dir",
                        help="Folder that will contain the cropped images in both un-resized and resized.")
    parser.add_argument('--remote_output_dir', type=str, default="cropped_image",
                        help="Folder that will contain the cropped images in both un-resized and resized.")
    parser.add_argument('--save_resized', default=True,
                        action='store_true', help="Also save the image with shorter edge resized to 256")
    parser.add_argument('--crop_ratio', type=float, default=1.4,
                        help="Scale the bbox to crop larger or small area.")
    parser.add_argument('--show_bbox', default=False,
                        action='store_true')
    parser.add_argument('--width_of_bbox', type=int, default=3,
                        help="Define the width of the bound of bounding boxes.")
    parser.add_argument('--fix_ratio', default=False,
                        action='store_true', help='Further extent the image to make the ratio in 4:3.')
    parser.add_argument('--equal_extend', default=True,
                        action='store_true', help='Extand equal size in both height and width.')
    parser.add_argument('--rotate_image', default=False,
                        action='store_true', help='Rotate the insect to fit 4:3 naturally.')

    parser.add_argument('--background_color_R', type=int, default=204,
                        help="Define the background color's R value.")
    parser.add_argument('--background_color_G', type=int, default=218,
                        help="Define the background color's G value.")
    parser.add_argument('--background_color_B', type=int, default=243,
                        help="Define the background color's B value.")

    args = parser.parse_args()
    os.makedirs(args.local_output_dir, exist_ok=True)

    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

    model = load_model_from_ckpt(args)

    os.makedirs(args.local_input_dir, exist_ok=True)
    os.makedirs(args.local_output_dir, exist_ok=True)



    # Download the checkpoint.
    shutil.copyfile(args.checkpoint_path, os.path.join(args.local_input_dir,
                                                       "pretrained_with_IP1000_and_IW1000.ckpt"))

    with open(args.input_txt) as file:
        image_tar_names = [line.rstrip() for line in file]

    image_folder_names = []

    pbar = tqdm(image_tar_names)
    for tarfile_name in pbar:
        pbar.set_description("Copying image tars")
        folder_name = tarfile_name.replace(".tar", "")
        image_folder_names.append(folder_name)
        target_folder_path = os.path.join(args.local_input_dir, folder_name)
        if os.path.exists(target_folder_path):
            continue
        src_tar_path = os.path.join(args.remote_input_dir, tarfile_name)
        target_tar_path = os.path.join(args.local_input_dir, tarfile_name)
        if not os.path.exists(target_tar_path):
            shutil.copyfile(src_tar_path, target_tar_path)
        folder_name = tarfile_name.replace(".tar", "")
        os.makedirs(target_folder_path, exist_ok=True)
        unzip_tars_to_folder(target_tar_path, target_folder_path)
        os.remove(target_tar_path)


    pbar = tqdm(image_folder_names)
    for folder_name in pbar:
        pbar.set_description("Cropping for each folder")
        args.input_dir = os.path.join(args.local_input_dir, folder_name)
        args.current_image_folder_name = folder_name
        crop_image(args, model, feature_extractor)

    pbar = tqdm(os.listdir(args.local_output_dir))
    for folder_name in pbar:
        if os.path.exists(os.path.join(args.remote_output_dir, folder_name)):
            shutil.rmtree(os.path.join(args.remote_output_dir, folder_name))
        shutil.copytree(os.path.join(args.local_output_dir, folder_name),
                        os.path.join(args.remote_output_dir, folder_name))

    # shutil.rmtree(os.path.join(args.local_input_dir))
