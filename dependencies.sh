#!/bin/bash

#Run this script as "bash dependencies.sh" to install all required packages.

# Install modules
pip install numpy scipy matplotlib opencv-python scikit-image Pillow bm3d
# Install pytorch
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install deepinv

pip install torchsummary 


##### To download the pretrained models, run the following commands #####
pip install gdown
FOLDER_ID="16aBKsnIKQZBIpyucz17zYT9_nneT4V0V?usp=drive_link"

gdown --folder "$FOLDER_ID"
