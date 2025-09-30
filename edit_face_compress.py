import numpy as np
import argparse
import os
import torch
from torchvision import transforms
from PIL import Image
from nets.trainer_channel_face import *
import yaml
from tqdm import tqdm
device = torch.device('cuda')

def linear_interpolate(latent_code,
                       boundary,
                       start_distance=-3.0,
                       end_distance=3.0,
                       steps=10):
    assert (latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
                    len(boundary.shape) == 2 and
                    boundary.shape[1] == latent_code.shape[-1])

    linspace = np.linspace(start_distance, end_distance, steps)
    if len(latent_code.shape) == 2:
        linspace = linspace.reshape(-1, 1).astype(np.float32)
        return latent_code + linspace * boundary
    if len(latent_code.shape) == 3:
        linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
        return latent_code + linspace * boundary.reshape(1, 1, -1)
    
def edit(opts):
    n_steps = 5
    delta_on_feature = True
    step_scale = 15
    config = yaml.load(open('./configs/' + opts.config + '.yaml', 'r'), Loader=yaml.FullLoader)
    # trainer = Trainer(config, opts)
    # trainer.initialize(opts.stylegan_model_path, opts.arcface_model_path, opts.parsing_model_path) 
    # trainer.load_checkpoint(os.path.join(opts.log_path, opts.config + '/checkpoint.pth'))
    # trainer.to(device)
    trainer = Trainer(config, opts)
    trainer.initialize(opts.stylegan_model_path, opts.arcface_model_path, opts.parsing_model_path)  
    trainer.to(device)
    trainer.load_checkpoint(opts.checkpoint)
    trainer.enc.eval()

    img_to_tensor = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    os.makedirs(opts.save_path, exist_ok=True)
    output_dir = os.path.join(opts.save_path, opts.edit)
    os.makedirs(output_dir, exist_ok=True)
    boundary_dir = os.path.join(opts.boundary_dir, opts.edit + '_boundary.npy')

    with torch.no_grad():
        img_list = sorted([os.path.join(opts.input_path, img) for img in os.listdir(opts.input_path)])
        for img in tqdm(img_list):
            img_name = os.path.basename(img)
            img_A = img_to_tensor(Image.open(img)).unsqueeze(0).to(device)
            # output = trainer.test(img=img_A, return_latent=True)
            output = trainer.compressed_test(img=img_A, return_latent=True, compressed_stage=4)
            feature = output.pop() # [16 x 16 x 512]
            latent = output.pop() # [1 x 18 x 512]
            boundary = np.load(boundary_dir) # (1,9216)
            w_0 = latent.cpu().numpy().reshape(1, -1) # (1,9216)
            out = linear_interpolate(w_0, boundary, start_distance=-step_scale, end_distance=step_scale, steps=n_steps) # 经过插值得到5*9216的w
            w_0 = torch.tensor(w_0).view(1, -1, 512).to(device)
            for j in [4]:
                w_1 = torch.tensor(out[j]).view(1, -1, 512).to(device) # 最大插值项插值后的latent，还是(1,9216)
                # calculate delta feature
                _, fea_0 = trainer.StyleGAN([w_0], input_is_latent=True, return_features=True)  # 每一层的所有feature
                _, fea_1 = trainer.StyleGAN([w_1], input_is_latent=True, return_features=True)
                
                features = [None]*5 + [feature + fea_1[5] - fea_0[5]] + [None]*(17-5)
                x_1, _ = trainer.StyleGAN([w_1], input_is_latent=True, features_in=features, feature_scale=1.0) # 用插值之后的隐变量直接生成
                inv_img = np.clip(clip_img(x_1)[0].cpu().numpy()*255.,0,255).astype(np.uint8)
                inv_img = Image.fromarray(inv_img.transpose(1,2,0))
                inv_img.save(os.path.join(output_dir, img_name))
        

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='001', help='Path to the config file.')
    parser.add_argument('--pretrained_model_path', type=str, default='./pretrained_models/143_enc.pth', help='pretrained stylegan2 model')
    parser.add_argument('--stylegan_model_path', type=str, default='./pretrained_models/stylegan2_pth/psp_ffhq_encode.pt', help='pretrained stylegan2 model')
    parser.add_argument('--arcface_model_path', type=str, default='./pretrained_models/backbone.pth', help='pretrained arcface model')
    parser.add_argument('--parsing_model_path', type=str, default='./pretrained_models/79999_iter.pth', help='pretrained parsing model')
    parser.add_argument('--log_path', type=str, default='./logs/', help='log file path')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint file path')
    parser.add_argument('--checkpoint_noiser', type=str, default='', help='checkpoint file path')
    parser.add_argument('--multigpu', type=bool, default=False, help='use multiple gpus')
    parser.add_argument('--input_path', type=str, default='./data/celeba_hq_5k/', help='evaluation data file path')
    parser.add_argument('--save_path', type=str, default='./output/', help='output data save path')
    parser.add_argument('--edit', type=str, default='Smiling', help='output data save path') # ['Heavy_Makeup', 'Smiling', 'Eyeglasses']
    parser.add_argument('--boundary_dir', type=str, default='./data/edit_boundary/', help='output data save path')
    opts = parser.parse_args()
    return opts


if __name__ == '__main__':
    opts = args()
    edit(opts=opts)
    

