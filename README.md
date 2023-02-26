# Set environment
For now, you can set the environment by typing
```shell
conda create -n BioScan_croptool 
conda activate ioScan_croptool
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
conda env update --file environment.yml
```
in the terminal. However, based on your GPU version, you may have to modify the torch version and install other packages manually in difference version.
# Data preparation
Please download the annotated data and images from https://drive.google.com/file/d/1UdYd99MKRyvqirdAssV8Ds4dkfhXjtW3/view?usp=sharing. Then unzip it in `/data`.
You can split and prepare the data by
```shell
python process_and_split.py --input_dir 100_Rig_Images --dataset_name insect
```
# Train and eval
```shell
python train.py --data_dir data/insect --output_dir insect_detection_ckpt
```
Here is a checkpoint for you to use, so you can skip this step. (https://drive.google.com/file/d/1cYyg5TTFRigSxak5EBchLEij8XXHRNfk/view?usp=sharing)
# Crop image
You can put the imsect images that need to be cropped in this folder, then type
```shell
python crop_images.py --input_dir original_images --checkpoint_path insect_detection_ckpt/lightning_logs/version_0/checkpoints/epoch=11-step=300.ckpt
```
in the terminal.
