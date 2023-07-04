import argparse
import numpy as np
import torch
import os
import sys
from tqdm import tqdm
from transformers import DetrFeatureExtractor
from PIL import Image, ImageDraw, ImageOps
import shutil
from project_path import project_dir
from model.detr import load_model_from_ckpt
from util.visualize_and_process_bbox import get_bbox_from_output, scale_bbox

"""
This is a special one time script for special purpose,  will be removed from the repo after the task is done.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_dir', type=str, required=True,
    #                     help="Folder that contains the original images.")
    parser.add_argument('--input_txt', type=str, required=True,
                        help="Path to the txt file that contains the names of image folder you want to download to local."
                             "storage.")
    parser.add_argument('--remote_input_dir', type=str, required=True,
                        help="Path to the directory that contains the image folders you want to download to your local "
                             "storage.")
    parser.add_argument('--local_input_dir', type=str, default="local_input_dir",
                        help="Path to the directory that you want to save the un-cropped image directories and the "
                             "checkpoint")
    parser.add_argument('--local_output_dir', type=str, default="local_output_dir",
                        help="Folder that will contain the cropped images in both un-resized and resized.")
    parser.add_argument('--remote_output_dir', type=str, default="cropped_image",
                        help="Folder that will contain the cropped images in both un-resized and resized.")

    args = parser.parse_args()
    os.makedirs(args.local_input_dir, exist_ok=True)
    os.makedirs(args.local_output_dir, exist_ok=True)

    with open(args.input_txt) as file:
        image_folder_names = [line.rstrip() for line in file]

    pbar = tqdm(image_folder_names)
    for folder_name in pbar:
        remote_input_folder = os.path.join(args.remote_input_dir, folder_name)
        input_folder_dir = os.path.join(args.local_input_dir, folder_name)
        resize_folder_dir = os.path.join(args.local_output_dir, folder_name)
        os.makedirs(resize_folder_dir, exist_ok=True)
        os.system("cp -r " + os.path.join(args.remote_input_dir, folder_name) + " " + args.local_input_dir)
        # shutil.copytree(os.path.join(args.remote_input_dir, folder_name),
        #                 input_folder_dir)


        os.system("find " + input_folder_dir + " -name '*.jpg' -exec bash -c 'convert ${0} -resize x256 " + resize_folder_dir + "/${0##*/}' {} \;")

        shutil.rmtree(os.path.join(args.local_input_dir, folder_name))

        if os.path.exists(os.path.join(args.remote_output_dir, folder_name)):
            shutil.rmtree(os.path.join(args.remote_output_dir, folder_name))
        shutil.copytree(resize_folder_dir,
                        os.path.join(args.remote_output_dir, folder_name))
