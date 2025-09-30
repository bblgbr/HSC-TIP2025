import argparse
import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import yaml
from calflops import calculate_flops
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, utils

from utils.datasets import *
from utils.functions import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None
device = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='001', help='Path to the config file.')
parser.add_argument('--type', type=str, default='face', help='dataset class: face or church or car')
parser.add_argument('--arcface_model_path', type=str, default='./pretrained_models/backbone.pth', help='pretrained arcface model')
parser.add_argument('--parsing_model_path', type=str, default='./pretrained_models/79999_iter.pth', help='pretrained parsing model')
parser.add_argument('--log_path', type=str, default='./logs/', help='log file path')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint file path')
parser.add_argument('--checkpoint_noiser', type=str, default='', help='checkpoint file path')
parser.add_argument('--multigpu', type=bool, default=False, help='use multiple gpus')
parser.add_argument('--input_path', type=str, default='./test/', help='evaluation data file path')
parser.add_argument('--save_path', type=str, default='./output/image/', help='output data save path')
opts = parser.parse_args()

if opts.type == 'church':
    from nets.trainer_channel_church import *
    stylegan_model_path = './pretrained_models/stylegan2_pth/e4e_church_encode.pt'
elif opts.type == 'face':
    from nets.trainer_channel_face import *
    stylegan_model_path = './pretrained_models/stylegan2_pth/psp_ffhq_encode.pt'
elif opts.type == 'car':
    from nets.trainer_channel_car import *
    stylegan_model_path = './pretrained_models/stylegan2_pth/e4e_cars_encode.pt'

is_car = True if opts.type == 'car' else False
log_dir = os.path.join(opts.log_path, opts.config) + '/'
config = yaml.load(open('./configs/' + opts.config + '.yaml', 'r'), Loader=yaml.FullLoader)

# Initialize trainer
trainer = Trainer(config, opts)
trainer.initialize(stylegan_model_path, opts.arcface_model_path, opts.parsing_model_path)  
trainer.to(device)

state_dict = torch.load(opts.checkpoint)
trainer.load_checkpoint(opts.checkpoint)
trainer.enc.eval()

img_to_tensor = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

img_to_tensor_church = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

img_to_tensor_car = transforms.Compose([
    transforms.Resize((384, 512)),
    transforms.Pad(padding=(0, 64, 0, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# simple inference
image_dir = opts.input_path
save_dir = opts.save_path
os.makedirs(save_dir, exist_ok=True)

with torch.no_grad():
    img_list = [glob.glob1(image_dir, ext) for ext in ['*jpg','*png','*webp']]
    img_list = [item for sublist in img_list for item in sublist]
    img_list.sort()
    start = time()
    for i, img_name in enumerate(tqdm(img_list)):
        if opts.type == 'car':
            image_A = img_to_tensor_car(Image.open(image_dir + img_name)).unsqueeze(0).to(device)
        elif opts.type == 'church':
            image_A = img_to_tensor_church(Image.open(image_dir + img_name)).unsqueeze(0).to(device)
        else:
            image_A = img_to_tensor(Image.open(image_dir + img_name)).unsqueeze(0).to(device)
        
        print(img_name)
        output = trainer.compressed_test(img=image_A, return_latent=True, compressed_stage=4)
        feature = output.pop()
        latent = output.pop()
        if is_car:
            #np.save(save_dir + 'latent_code_%d.npy'%i, latent.cpu().numpy())
            utils.save_image(clip_img(output[1][:, :, 64:448, :]), save_dir + img_name)
        else:
            utils.save_image(clip_img(output[1]), save_dir + img_name)
    end = time()
    print(f'average_bpp: {trainer.test_bpp / trainer.test_num}')
    print(f'average_time: (ms) {(end - start) / len(img_list) * 1000}')


