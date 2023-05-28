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

def save_single_image(binary_data_np, image_name, hdf5):
    dataset = hdf5['bioscan_dataset']
    # Read the binary data from the BytesIO object
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
    parser.add_argument('--image_set', type=str, default="cropped_256",
                        help="Original or cropped images")
    args = parser.parse_args()

    make_hdf5(path=args.hdf5_path, data_typ=args.image_set)
    pbar_parts = tqdm(os.listdir(args.input_dir))
    with h5py.File(args.hdf5_path, 'a') as hdf5:
        for sub_dir in pbar_parts:
            pbar_parts.set_description("Parts")
            path_to_part_folder = os.path.join(args.input_dir, sub_dir)
            pbar_files = tqdm(os.listdir(path_to_part_folder))
            for file_name in pbar_files:
                if file_name in hdf5['bioscan_dataset'].keys():
                    with open(args.image_set + '_hdf5_writing_log.txt', 'w') as f:
                        f.write(file_name + "already exist.")
                    continue
                pbar_files.set_description("Current part progress")
                file_path = os.path.join(path_to_part_folder, file_name)
                with open(file_path, 'rb') as f:
                    image_data = f.read()
                byte_io = io.BytesIO(image_data)
                binary_data = byte_io.getvalue()
                binary_data_np = np.frombuffer(binary_data, dtype=np.uint8)
                save_single_image(binary_data_np, file_name, hdf5)
