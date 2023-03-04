# As the "Toronto Annotation suite" has a storage limit of 100mb.
# Some of the image may need to be resized before using the segmentation tools.
# This script is a example.
from PIL import Image
import argparse
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help="path to the directory with images that need to be resized.")
    parser.add_argument('--output_dir', type=str, required=True, help="path to the directory where you want to save the resized image.")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for filename in os.listdir(args.input_dir):
        f = os.path.join(args.input_dir, filename)
        # checking if it is a file
        img = Image.open(f)
        img.thumbnail((4000, 4000), Image.ANTIALIAS)
        img.save(os.path.join(args.output_dir, "resized_" + filename), "JPEG")