import argparse
import os
import h5py
import io
import numpy as np
from PIL import Image
from tqdm import tqdm

if __name__ == '__main__':
    with h5py.File("hdf5_workspace/output_hdf5/cropped_256.hdf5", 'r') as hdf5_file:
        images = hdf5_file['bioscan_dataset']
        first_key = [key for key in images.keys()][0]
        if "ABCKHUA" in images.keys():
            print("File exist")
        byte_io = io.BytesIO(np.array(images[first_key]))
        image = Image.open(byte_io)
        image.show()
