# Google Colab Notebook

`https://colab.research.google.com/drive/1QyamrJYnwBIfmFKzW6_ODSQyu4ENGklN?usp=sharing`

# Set environment
For now, you can set the environment by typing
```shell
conda create -n BioScan-croptool python=3.10
conda activate BioScan-croptool
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git

```
in the terminal. However, based on your GPU version, you may have to modify the torch version and install other packages manually in difference version.
# Data preparation
Please download the  [metadata of BioScan-1M](https://aspis.cmpt.sfu.ca/projects/bioscan/data/versions/v0.2/v0.2.1/metadata/BioScan_Insect_Dataset_metadata_2.tsv) to the 'data' directory.

You can do this by
```shell
mkdir data
cd data
wget https://aspis.cmpt.sfu.ca/projects/bioscan/data/versions/v0.2/v0.2.1/metadata/BioScan_Insect_Dataset_metadata_2.tsv
```
You may also need to download the [cropped 256 images](https://aspis.cmpt.sfu.ca/projects/bioscan/data/versions/v0.2/v0.2.1/archive/cropped_256/cropped_256.zip) and unzip it. (If you use the lab machine, you can ignore this step.)

# Sample data and test frozen CLIP
```shell
python scripts/sameple_different_level_data.py 
python scripts/initial_example.py --remote_image_dir {where you unzipped the images}
```
