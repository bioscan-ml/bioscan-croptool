import argparse
import os
import h5py
import io
import numpy as np
from PIL import Image
from tqdm import tqdm

if __name__ == '__main__':
    with h5py.File("hdf5_workspace/output_hdf5/cropped_25.hdf5", 'r') as hdf5_file:
        images = hdf5_file['bioscan_dataset']
        img = Image.open(io.BytesIO(np.array(images[images.keys[0]])))
        img.show()
