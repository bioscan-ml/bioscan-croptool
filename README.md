# Set environment
For now, you can set the environment by typing
```shell
conda create -n BioScan-croptool python=3.10
conda activate BioScan-croptool
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt

```
in the terminal. However, based on your GPU version, you may have to modify the torch version and install other packages manually in difference version.
# Data preparation
Please download the annotated data and images from https://drive.google.com/file/d/1UdYd99MKRyvqirdAssV8Ds4dkfhXjtW3/view?usp=sharing. Then unzip it in `/data`.
You can split and prepare the data by
```shell
python data/process_and_split.py --input_dir data/100_Rig_Images --dataset_name data/insect
```
# Train and eval
```shell
python scripts/train.py --data_dir data/insect --output_dir insect_detection_ckpt
```
Here is a checkpoint for you to use, so you can skip this step. (https://drive.google.com/file/d/1cYyg5TTFRigSxak5EBchLEij8XXHRNfk/view?usp=sharing)
# Visualization
To visualize the predicted bounding box
```shell
python scripts/visualization.py --data_dir data/insect --checkpoint_path insect_detection_ckpt/lightning_logs/version_0/checkpoints/epoch=11-step=300.ckpt
```
# Crop image
You can put the insect images that need to be cropped in a folder (Maybe call `original_images`), then type
```shell
python scripts/crop_images.py --input_dir original_images --checkpoint_path insect_detection_ckpt/lightning_logs/version_0/checkpoints/epoch=11-step=300.ckpt
```
in the terminal.

# Acknowledgement
This repo is built upon [Fine_tuning_DetrForObjectDetection_on_custom_dataset](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb).


