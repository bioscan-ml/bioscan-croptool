# Google Colab Notebook

`https://colab.research.google.com/drive/1QyamrJYnwBIfmFKzW6_ODSQyu4ENGklN?usp=sharing`

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
Please download the annotated data and images by `wget https://aspis.cmpt.sfu.ca/projects/bioscan/BIOUG_1k_images_resized.zip`. Then unzip it in `/data`.
You can split and prepare the data by
```shell
python scripts/complete_coco_json.py --input_dir data/resized
python scripts/split_data.py --input_dir data/resized --dataset_name data/insect
```
If you want to annotate more data for the training part, you can check Toronto Annotation suite(https://aidemos.cs.toronto.edu/toras).
Note that some of the information is missing from their coco annotation file, that is why the `complete_coco_json.py` exist. (However, this scripts use boudning box area to replace the mask area, but it is not affecting the cropping tool too much).


# Train and eval
```shell
python scripts/train.py --data_dir data/insect --output_dir insect_detection_ckpt
```
Here is a checkpoint for you to use, so you can skip the training. `wget https://aspis.cmpt.sfu.ca/projects/bioscan/ckpt_for_pined_images.ckpt`
If you want to use it, you can put it into your project folder ,and change the `checkpoint_path` in following commands. (To `--checkpoint_path ckpt_for_pined_images.ckpt`)

# Evaluation
To evaluate the checkpoint:
```shell
python scripts/visualization.py --data_dir data/insect --checkpoint_path insect_detection_ckpt/lightning_logs/version_0/checkpoints/epoch=11-step=300.ckpt
```

# Visualization
To visualize the predicted bounding box
```shell
python scripts/visualization.py --data_dir data/insect --checkpoint_path insect_detection_ckpt/lightning_logs/version_0/checkpoints/epoch=11-step=300.ckpt
```
# Crop image
You can put the insect images that need to be cropped in a folder (Maybe call `original_images`), then type
```shell
python scripts/crop_images.py --input_dir original_images --checkpoint_path insect_detection_ckpt/lightning_logs/version_0/checkpoints/epoch=11-step=300.ckpt --crop_ratio 1.4
```
in the terminal.
Note that by setting  `--crop_ratio 1.4`, the cropped image is 1.4 scaled than the predicted boudning box. If you want to check the origional bounding box, you can add `--show_bbox` at the end of the command.


If you want the cropped image in 4:3 ratio, you can add `--fix_ratio` to the command. Here is an example:
```shell
python scripts/crop_images.py --input_dir Part_2 --output_dir Part_2_cropped --checkpoint_path epoch=11-step=600_trained_on_part_1.ckpt --crop_ratio 1.4 --show_bbox --fix_ratio
```


# Acknowledgement
This repo is built upon [Fine_tuning_DetrForObjectDetection_on_custom_dataset](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb).
