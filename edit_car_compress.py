import torch
import sys
from PIL import Image
import yaml
from nets.trainer_channel_car import *
from tqdm import tqdm
import argparse
device = torch.device('cuda')

def tensor2im(var):
	# var shape: (3, H, W)
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))

def edit(latents, pca, edit_directions):
    edit_latents = []
    for latent in latents:
        for pca_idx, start, end, strength in edit_directions:
            delta = get_delta(pca, latent, pca_idx, strength)
            delta_padded = torch.zeros(latent.shape).to('cuda')
            delta_padded[start:end] += delta.repeat(end - start, 1)
            edit_latents.append(latent + delta_padded)
    return torch.stack(edit_latents)


def get_delta(pca, latent, idx, strength):
    # pca: ganspace checkpoint. latent: (16, 512) w+
    w_centered = latent - pca['mean'].to('cuda')
    lat_comp = pca['comp'].to('cuda')
    lat_std = pca['std'].to('cuda')
    w_coord = torch.sum(w_centered[0].reshape(-1)*lat_comp[idx].reshape(-1)) / lat_std[idx]
    delta = (strength - w_coord)*lat_comp[idx]*lat_std[idx]
    return delta


class LatentEditor(object):
    def __init__(self, stylegan_generator, is_cars=False):
        self.generator = stylegan_generator
        self.is_cars = is_cars  # Since the cars StyleGAN output is 384x512, there is a need to crop the 512x512 output.

    def apply_ganspace(self, latent, ganspace_pca, edit_directions):
        edit_latents = edit(latent, ganspace_pca, edit_directions)
        # return self._latents_to_image(edit_latents)
        return edit_latents

    def apply_interfacegan(self, latent, direction, factor=1, factor_range=None):
        edit_latents = []
        if factor_range is not None:  # Apply a range of editing factors. for example, (-5, 5)
            for f in range(*factor_range):
                edit_latent = latent + f * direction
                edit_latents.append(edit_latent)
            edit_latents = torch.cat(edit_latents)
        else:
            edit_latents = latent + factor * direction
        return self._latents_to_image(edit_latents)

    def _latents_to_image(self, latents):
        with torch.no_grad():
            images, _ = self.generator([latents], randomize_noise=False, input_is_latent=True)
            if self.is_cars:
                images = images[:, :, 64:448, :]  # 512x512 -> 384x512
        horizontal_concat_image = torch.cat(list(images), 2)
        final_image = tensor2im(horizontal_concat_image)
        return final_image

def edit_cars(opts):
    # direction: ["Cube", "Viewpoint I", "Color", "Viewpoint II", "Grass"]
    n_steps = 5
    delta_on_feature = True
    step_scale = 15
    config = yaml.load(open('./configs/' + opts.config + '.yaml', 'r'), Loader=yaml.FullLoader)
    trainer = Trainer(config, opts)
    trainer.initialize(opts.stylegan_model_path, opts.arcface_model_path, opts.parsing_model_path)  
    trainer.to(device)
    trainer.load_checkpoint(opts.checkpoint)
    trainer.enc.eval()

    img_to_tensor_car = transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.Pad(padding=(0, 64, 0, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    os.makedirs(opts.save_path, exist_ok=True)
    output_dir = os.path.join(opts.save_path, opts.edit)
    os.makedirs(output_dir, exist_ok=True)
    
    ganspace_pca = torch.load('boundaries_ours/cars_pca.pt')
    directions = {
        "Viewpoint_I": (0, 0, 5, 2),
        "Viewpoint_II": (0, 0, 5, -2),
        "Cube": (16, 3, 6, 25),
        "Color": (22, 9, 11, -8),
        "Grass": (41, 9, 11, -18),
    }

    editor = LatentEditor(trainer.StyleGAN, is_cars=True)

    with torch.no_grad():
        img_list = sorted([os.path.join(opts.input_path, img) for img in os.listdir(opts.input_path)])
        for img in tqdm(img_list):
            img_name = os.path.basename(img)
            img_A = img_to_tensor_car(Image.open(img)).unsqueeze(0).to(device)
            output = trainer.compressed_test(img=img_A, return_latent=True, compressed_stage=4)
            # output = trainer.test(img=img_A, return_latent=True)
            feature = output.pop() # [16 x 16 x 512]
            latent = output.pop() # [1 x 18 x 512]
            w_0 = latent
            w_1 = editor.apply_ganspace(w_0, ganspace_pca, [directions[opts.edit]])[0]
            w_1 = w_1.unsqueeze(0)
            # calculate delta feature
            _, fea_0 = trainer.StyleGAN([w_0], input_is_latent=True, return_features=True)  # 每一层的所有feature
            _, fea_1 = trainer.StyleGAN([w_1], input_is_latent=True, return_features=True)
            edit_image_w, _= trainer.StyleGAN([w_1], input_is_latent=True, return_features=True)
            
            features = [None]*3 + [feature + fea_1[3] - fea_0[3]] + [None]*(17-3)
            x_1, _ = trainer.StyleGAN([w_1], input_is_latent=True, features_in=features, feature_scale=1.0) # 用插值之后的隐变量直接生成
            x_1 =  x_1[:, :, 64:448, :]
            edit_image_w =  edit_image_w[:, :, 64:448, :]
            x_1 = torch.cat([img_A[:, :, 64:448, :], x_1, edit_image_w], dim=3)
            inv_img = np.clip(clip_img(x_1)[0].cpu().numpy()*255.,0,255).astype(np.uint8)
            inv_img = Image.fromarray(inv_img.transpose(1,2,0))
            inv_img.save(os.path.join(output_dir, img_name))



def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='001', help='Path to the config file.')
    parser.add_argument('--stylegan_model_path', type=str, default='./pretrained_models/stylegan2_pth/e4e_cars_encode.pt', help='pretrained stylegan2 model')
    parser.add_argument('--arcface_model_path', type=str, default='./pretrained_models/backbone.pth', help='pretrained arcface model')
    parser.add_argument('--parsing_model_path', type=str, default='./pretrained_models/79999_iter.pth', help='pretrained parsing model')
    parser.add_argument('--log_path', type=str, default='./logs/', help='log file path')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint file path')
    parser.add_argument('--checkpoint_noiser', type=str, default='', help='checkpoint file path')
    parser.add_argument('--multigpu', type=bool, default=False, help='use multiple gpus')
    parser.add_argument('--input_path', type=str, default='./data/car/', help='evaluation data file path')
    parser.add_argument('--save_path', type=str, default='./output/', help='output data save path')
    parser.add_argument('--edit', type=str, default='Viewpoint_I', help='output data save path') # ["Cube", "Viewpoint_I", "Color", "Viewpoint_II", "Grass"]
    opts = parser.parse_args()
    return opts


if __name__ == '__main__':
    opts = args()
    edit_cars(opts=opts)
    
