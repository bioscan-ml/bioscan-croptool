import argparse
import os
import h5py
import io
import numpy as np
from PIL import Image
from tqdm import tqdm
"""
This is a special one time script for special purpose,  will be removed from the repo after the task is done.
"""

def save_single_image(image, image_name, hdf5):
    dataset = hdf5['bioscan_dataset']
    binary_data_io = io.BytesIO()
    image.save(binary_data_io, format='JPEG')
    # Read the binary data from the BytesIO object
    binary_data = binary_data_io.getvalue()
    binary_data_np = np.frombuffer(binary_data, dtype=np.uint8)
    dataset.create_dataset(image_name, data=binary_data_np)


def make_hdf5(path='', data_typ=''):
    with h5py.File(path, 'w') as hdf5:
        dataset = hdf5.create_group('bioscan_dataset')
        dataset.attrs['Description'] = f'BioScan Dataset: {data_typ} Images'
        dataset.attrs['Copyright Holder'] = 'CBG Photography Group'
        dataset.attrs['Copyright Institution'] = 'Centre for Biodiversity Genomics (email:CBGImaging@gmail.com)'
        dataset.attrs['Photographer'] = 'CBG Robotic Imager'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="hdf5_workspace/unzipped/cropped_256",
                        help="Folder that contains the original images.")
    parser.add_argument('--hdf5_path', type=str, default="hdf5_workspace/output_hdf5/cropped_256.hdf5",
                        help="Path to the output hdf5 file.")
    parser.add_argument('--datatype', type=str, default="cropped",
                        help="Original or cropped images")
    args = parser.parse_args()

    make_hdf5(path=args.hdf5_path, data_typ=args.datatype)
    pbar_parts = tqdm(os.listdir(args.input_dir))
    with h5py.File(args.hdf5_path, 'a') as hdf5:
        for sub_dir in pbar_parts:
            pbar_parts.set_description("Parts")
            path_to_part_folder = os.path.join(args.input_dir, sub_dir)
            pbar_files = tqdm(os.listdir(path_to_part_folder))
            for file_name in pbar_files:
                pbar_files.set_description("Current part progress")
                file_path = os.path.join(path_to_part_folder, file_name)
                image = Image.open(file_path)
                save_single_image(image, file_name, hdf5)
