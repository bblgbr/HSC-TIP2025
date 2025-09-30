#!/bin/sh
pip install gdown
cd pretrained_models

# download pretrained encoder
gdown --fuzzy https://drive.google.com/file/d/1RnnBL77j_Can0dY1KOiXHvG224MxjvzC/view?usp=sharing

# download arcface pretrained model
gdown --fuzzy https://drive.google.com/file/d/1coFTz-Kkgvoc_gRT8JFzqCgeC3lAFWQp/view?usp=sharing

# download face parsing model from https://github.com/zllrunning/face-parsing.PyTorch
gdown --fuzzy https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812

# download pSp and e4e pretrained model from https://github.com/eladrich/pixel2style2pixel and https://github.com/omertov/encoder4editing
mkdir stylegan2_pth
cd stylegan2_pth
gdown --fuzzy https://drive.google.com/file/d/1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0/view?usp=sharing

gdown --fuzzy https://drive.google.com/file/d/17faPqBce2m1AQeLCLHUVXaDfxMRU2QcV/view?usp=sharing

gdown --fuzzy https://drive.google.com/file/d/1-L0ZdnQLwtdy6-A_Ccgq5uNJGTqE7qBa/view?usp=sharing
cd ..
cd ..

