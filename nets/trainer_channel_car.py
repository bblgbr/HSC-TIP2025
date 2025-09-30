import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from PIL import Image
from torch.autograd import grad
from torchvision import transforms, utils
from time import time
import face_alignment
import lpips
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../pretrained_models/')
from pretrained_models.models.stylegan2.model import Generator, get_keys

from nets.feature_style_encoder import *
from utils.functions import *
from nets.arcface.iresnet import *
from nets.face_parsing.model import BiSeNet
from utils.ranger import Ranger
from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
from nets.hsc import latent_compress_car as latent_compress, feature_compress_channel_car as feature_compress
from compressai import set_entropy_coder

class Trainer(nn.Module):
    def __init__(self, config, opts):
        super(Trainer, self).__init__()
        # Load Hyperparameters
        self.config = config
        self.device = torch.device(self.config['device'])
        self.scale = int(np.log2(config['resolution']/config['enc_resolution']))
        self.scale_mode = 'bilinear'
        self.opts = opts
        self.n_styles = 2 * int(np.log2(config['resolution'])) - 2
        self.idx_k = 5
        if 'idx_k' in self.config:
            self.idx_k = self.config['idx_k']
        if 'stylegan_version' in self.config and self.config['stylegan_version'] == 3:
            self.n_styles = 14
        # Networks
        in_channels = 256
        if 'in_c' in self.config:
            in_channels = config['in_c']
        enc_residual = False
        if 'enc_residual' in self.config:
            enc_residual = self.config['enc_residual']
        enc_residual_coeff = False
        if 'enc_residual_coeff' in self.config:
            enc_residual_coeff = self.config['enc_residual_coeff']
        resnet_layers = [4,5,6]
        if 'enc_start_layer' in self.config:
            st_l = self.config['enc_start_layer']
            resnet_layers = [st_l, st_l+1, st_l+2]
        if 'scale_mode' in self.config:
            self.scale_mode = self.config['scale_mode']
        # Load encoder
        self.stride = (self.config['fs_stride'], self.config['fs_stride'])
        self.enc = fs_encoder_v2(n_styles=self.n_styles, opts=opts, residual=enc_residual, use_coeff=enc_residual_coeff, resnet_layer=resnet_layers, stride=self.stride)
        
        ##########################
        # Other nets
        self.StyleGAN = self.init_stylegan(config)
        self.Arcface = iresnet50()
        self.parsing_net = BiSeNet(n_classes=19)
        # Optimizers
        # Latent encoder
        self.enc_params = list(self.enc.parameters()) 
        if 'freeze_iresnet' in self.config and self.config['freeze_iresnet']:
            self.enc_params =  list(self.enc.styles.parameters())
        if 'optimizer' in self.config and self.config['optimizer'] == 'ranger':
            self.enc_opt = Ranger(self.enc_params, lr=config['lr'], betas=(config['beta_1'], config['beta_2']), weight_decay=config['weight_decay'])
        else:
            self.enc_opt = torch.optim.Adam(self.enc_params, lr=config['lr'], betas=(config['beta_1'], config['beta_2']), weight_decay=config['weight_decay'])
        self.enc_scheduler = torch.optim.lr_scheduler.StepLR(self.enc_opt, step_size=config['step_size'], gamma=config['gamma'])

        self.fea_avg = None

        ##########################
        # compress net
        self.latent_compressor = latent_compress(N=192, M=192)
        self.feature_compressor = feature_compress(N=192, M=192)
        latent_conf = {
            "net": {"type": "Adam", "lr": self.config['latent_learning_rate']},
            "aux": {"type": "Adam", "lr": self.config['latent_aux_learning_rate']},
        }
        optimizer = net_aux_optimizer(self.latent_compressor, latent_conf)
        self.latent_optimizer = optimizer["net"]
        self.latent_aux_optimizer = optimizer["aux"]
        feature_conf = {
            "net": {"type": "Adam", "lr": self.config['feature_learning_rate']},
            "aux": {"type": "Adam", "lr": self.config['feature_aux_learning_rate']},
        }
        optimizer = net_aux_optimizer(self.feature_compressor, feature_conf)
        self.feature_optimizer = optimizer["net"]
        self.feature_aux_optimizer = optimizer["aux"]
        self.clip_max_norm = self.config['clip_max_norm']
        self.latent_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.latent_optimizer, "min")
        self.feature_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.feature_optimizer, "min")
        self.out_latent_criterion = 0
        self.out_feature_criterion = 0
        self.test_bpp = 0
        self.test_num = 0
        # self.latent_aux_loss = 0
        # self.feature_aux_loss = 0
        # self.sumloss = 0



    def initialize(self, stylegan_model_path, arcface_model_path, parsing_model_path):
        # load StyleGAN model
        stylegan_state_dict = torch.load(stylegan_model_path, map_location='cpu')
        self.StyleGAN.load_state_dict(get_keys(stylegan_state_dict, 'decoder'), strict=True)
        self.StyleGAN.to(self.device)
        # get StyleGAN average latent in w space and the noise inputs
        self.dlatent_avg = stylegan_state_dict['latent_avg'].to(self.device)
        self.noise_inputs = [getattr(self.StyleGAN.noises, f'noise_{i}').to(self.device) for i in range(self.StyleGAN.num_layers)]
        # load Arcface weight
        self.Arcface.load_state_dict(torch.load(self.opts.arcface_model_path))
        self.Arcface.eval()
        # load face parsing net weight
        self.parsing_net.load_state_dict(torch.load(self.opts.parsing_model_path))
        self.parsing_net.eval()
        # load lpips net weight
        self.loss_fn = lpips.LPIPS(net='alex', spatial=False)
        self.loss_fn.to(self.device)
        self.rate_loss_fn = RateDistortionLoss(lmbda=self.config['lmbda'])
        self.rate_loss_fn.to(self.device)
        self.latent_compressor.to(self.device)
        self.feature_compressor.to(self.device)
    
    def init_stylegan(self, config):
        """StyleGAN = G_main(
            truncation_psi=config['truncation_psi'], 
            resolution=config['resolution'], 
            use_noise=config['use_noise'],  
            randomize_noise=config['randomize_noise']
        )"""
        StyleGAN = Generator(config['resolution'], 512, 8) # 图像尺寸，隐空间维度，z到w的线性层层数
        return StyleGAN
    
    def mapping(self, z):
        return self.StyleGAN.get_latent(z).detach()

    def L1loss(self, input, target):
        return nn.L1Loss()(input,target)
    
    def L2loss(self, input, target):
        return nn.MSELoss()(input,target)

    def CEloss(self, x, target_age):
        return nn.CrossEntropyLoss()(x, target_age)
    
    def LPIPS(self, input, target, multi_scale=False):
        if multi_scale:
            out = 0
            for k in range(3):
                out += self.loss_fn.forward(downscale(input, k, self.scale_mode), downscale(target, k, self.scale_mode)).mean()
        else:
            out = self.loss_fn.forward(downscale(input, self.scale, self.scale_mode), downscale(target, self.scale, self.scale_mode)).mean()
        return out
    
    def IDloss(self, input, target):
        x_1 = F.interpolate(input, (112,112))
        x_2 = F.interpolate(target, (112,112))
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        if 'multi_layer_idloss' in self.config and self.config['multi_layer_idloss']:
            id_1 = self.Arcface(x_1, return_features=True)
            id_2 = self.Arcface(x_2, return_features=True)
            return sum([1 - cos(id_1[i].flatten(start_dim=1), id_2[i].flatten(start_dim=1)) for i in range(len(id_1))])
        else:
            id_1 = self.Arcface(x_1)
            id_2 = self.Arcface(x_2)
            return 1 - cos(id_1, id_2)
    
    def landmarkloss(self, input, target):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        x_1 = stylegan_to_classifier(input, out_size=(512, 512))
        x_2 = stylegan_to_classifier(target, out_size=(512,512))
        out_1 = self.parsing_net(x_1)
        out_2 = self.parsing_net(x_2)
        parsing_loss = sum([1 - cos(out_1[i].flatten(start_dim=1), out_2[i].flatten(start_dim=1)) for i in range(len(out_1))])
        return parsing_loss.mean()
        

    def feature_match(self, enc_feat, dec_feat, layer_idx=None):
        loss = []
        if layer_idx is None:
            layer_idx = [i for i in range(len(enc_feat))]
        for i in layer_idx:
            loss.append(self.L1loss(enc_feat[i], dec_feat[i]))
        return loss
    
    def encode(self, img):
        w_recon, fea = self.enc(downscale(img, self.scale, self.scale_mode)) 
        w_recon = w_recon + self.dlatent_avg
        return w_recon, fea

    def get_image(self, w=None, img=None, noise=None, zero_noise_input=True, training_mode=True):
        
        x_1, n_1 = img, noise
        if x_1 is None:
            x_1, _ = self.StyleGAN([w], input_is_latent=True, noise = n_1)
           
        w_delta = None
        fea = None
        features = None
        return_features = False
        # Reconstruction
        k = 0
        if 'use_fs_encoder' in self.config and self.config['use_fs_encoder']:
            return_features = True
            k = self.idx_k # 在第五层之后加入训练的特征
            w_recon, fea = self.enc(downscale(x_1, self.scale, self.scale_mode)) 
            w_recon = w_recon + self.dlatent_avg # 加整个网络的平均
            features = [None]*k + [fea] + [None]*(17-k)
        else:
            w_recon = self.enc(downscale(x_1, self.scale, self.scale_mode)) + self.dlatent_avg        

        # generate image
        x_1_recon, fea_recon = self.StyleGAN([w_recon], input_is_latent=True, return_features=True, features_in=features, feature_scale=min(1.0, 0.0001*self.n_iter)) # 得到重建人脸和插入的feature
        fea_recon = fea_recon[k].detach() 
        return [x_1_recon, x_1[:,:3,:,:], w_recon, w_delta, n_1, fea, fea_recon]

    def compute_loss(self, w=None, img=None, noise=None, real_img=None):
        return self.compute_loss_stylegan2(w=w, img=img, noise=noise, real_img=real_img)

    def compute_loss_stylegan2(self, w=None, img=None, noise=None, real_img=None):
        
        if img is None:
            # generate synthetic images
            if noise is None:
                noise = [torch.randn(w.size()[:1] + ee.size()[1:]).to(self.device) for ee in self.noise_inputs]
            img, _ = self.StyleGAN([w], input_is_latent=True, noise = noise)
            img = img.detach()

        if img is not None and real_img is not None:
            # concat synthetic and real data
            img = torch.cat([img, real_img], dim=0)
            noise = [torch.cat([ee, ee], dim=0) for ee in noise]
        
        out = self.get_image(w=w, img=img, noise=noise)
        x_1_recon, x_1, w_recon, w_delta, n_1, fea_1, fea_recon = out

        # Loss setting
        w_l2, w_lpips, w_id = self.config['w']['l2'], self.config['w']['lpips'], self.config['w']['id']
        b = x_1.size(0)//2
        if 'l2loss_on_real_image' in self.config and self.config['l2loss_on_real_image']:
            b = x_1.size(0)
        self.l2_loss = self.L2loss(x_1_recon[:b], x_1[:b]) if w_l2 > 0 else torch.tensor(0) # l2 loss only on synthetic data
        # LPIPS
        multiscale_lpips=False if 'multiscale_lpips' not in self.config else self.config['multiscale_lpips']
        self.lpips_loss = self.LPIPS(x_1_recon, x_1, multi_scale=multiscale_lpips).mean() if w_lpips > 0 else torch.tensor(0)
        self.id_loss = self.IDloss(x_1_recon, x_1).mean() if w_id > 0 else torch.tensor(0)
        self.landmark_loss = self.landmarkloss(x_1_recon, x_1) if self.config['w']['landmark'] > 0 else torch.tensor(0)
        
        if 'use_fs_encoder' in self.config and self.config['use_fs_encoder']:
            k = self.idx_k 
            features = [None]*k + [fea_1] + [None]*(17-k)
            x_1_recon_2, _ = self.StyleGAN([w_recon], noise=n_1, input_is_latent=True, features_in=features, feature_scale=min(1.0, 0.0001*self.n_iter))
            self.lpips_loss += self.LPIPS(x_1_recon_2, x_1, multi_scale=multiscale_lpips).mean() if w_lpips > 0 else torch.tensor(0)
            self.id_loss += self.IDloss(x_1_recon_2, x_1).mean() if w_id > 0 else torch.tensor(0)
            self.landmark_loss += self.landmarkloss(x_1_recon_2, x_1) if self.config['w']['landmark'] > 0 else torch.tensor(0)

        # downscale image
        x_1 = downscale(x_1, self.scale, self.scale_mode)
        x_1_recon = downscale(x_1_recon, self.scale, self.scale_mode)
        
        # Total loss
        w_l2, w_lpips, w_id = self.config['w']['l2'], self.config['w']['lpips'], self.config['w']['id']
        self.loss = w_l2*self.l2_loss + w_lpips*self.lpips_loss + w_id*self.id_loss
        
        if 'f_recon' in self.config['w']:
            self.feature_recon_loss = self.L2loss(fea_1, fea_recon) 
            self.loss += self.config['w']['f_recon']*self.feature_recon_loss
        if 'l1' in self.config['w'] and self.config['w']['l1']>0:
            self.l1_loss = self.L1loss(x_1_recon, x_1)
            self.loss += self.config['w']['l1']*self.l1_loss
        if 'landmark' in self.config['w']:
            self.loss += self.config['w']['landmark']*self.landmark_loss
        return self.loss

    def test(self, w=None, img=None, noise=None, zero_noise_input=True, return_latent=False, training_mode=False):        
        if 'n_iter' not in self.__dict__.keys():
            self.n_iter = 1e5
        out = self.get_image(w=w, img=img, noise=noise, training_mode=training_mode)
        x_1_recon, x_1, w_recon, w_delta, n_1, fea_1 = out[:6]
        output = [x_1, x_1_recon]
        if return_latent:
            output += [w_recon, fea_1]
        return output

    def log_loss(self, logger, n_iter, prefix='train'):
        logger.log_value(prefix + '/l2_loss', self.l2_loss.item(), n_iter + 1)
        logger.log_value(prefix + '/lpips_loss', self.lpips_loss.item(), n_iter + 1)
        logger.log_value(prefix + '/id_loss', self.id_loss.item(), n_iter + 1)
        # logger.log_value(prefix + '/total_loss', self.loss.item(), n_iter + 1)
        if 'f_recon' in self.config['w']:
            logger.log_value(prefix + '/feature_recon_loss', self.feature_recon_loss.item(), n_iter + 1)
        if 'l1' in self.config['w'] and self.config['w']['l1']>0:
            logger.log_value(prefix + '/l1_loss', self.l1_loss.item(), n_iter + 1)
        if 'landmark' in self.config['w']:
            logger.log_value(prefix + '/landmark_loss', self.landmark_loss.item(), n_iter + 1)
        if isinstance(self.out_latent_criterion, dict):
            logger.log_value(prefix + '/latent_bpp_loss', self.out_latent_criterion["real_bpp_loss"].item(), n_iter + 1)
            logger.log_value(prefix + '/latent_mse_loss', self.out_latent_criterion["mse_loss"].item(), n_iter + 1)
        if isinstance(self.out_feature_criterion, dict):
            logger.log_value(prefix + '/feature_bpp_loss', self.out_feature_criterion["real_bpp_loss"].item(), n_iter + 1)
            logger.log_value(prefix + '/feature_mse_loss', self.out_feature_criterion["mse_loss"].item(), n_iter + 1)
        
        if hasattr(self, 'feature_aux_loss'):
            logger.log_value(prefix + '/feature_aux_loss', self.feature_aux_loss.item(), n_iter + 1)
        if hasattr(self, 'latent_aux_loss'):
            logger.log_value(prefix + '/latent_aux_loss', self.latent_aux_loss.item(), n_iter + 1)
        
        if hasattr(self, 'sumloss'):
            logger.log_value(prefix + '/sum_loss', self.sumloss.item(), n_iter + 1)
        
    def save_image(self, log_dir, n_epoch, n_iter, prefix='/train/', w=None, img=None, noise=None, training_mode=True):
        return self.save_image_stylegan2(log_dir=log_dir, n_epoch=n_epoch, n_iter=n_iter, prefix=prefix, w=w, img=img, noise=noise, training_mode=training_mode)

    def save_image_stylegan2(self, log_dir, n_epoch, n_iter, prefix='/train/', w=None, img=None, noise=None, training_mode=True):
        os.makedirs(log_dir + prefix, exist_ok=True)
        with torch.no_grad():
            out = self.get_image(w=w, img=img, noise=noise, training_mode=training_mode)
            x_1_recon, x_1, w_recon, w_delta, n_1, fea_1 = out[:6]
            x_1 = downscale(x_1, self.scale, self.scale_mode)
            x_1_recon = downscale(x_1_recon, self.scale, self.scale_mode)
            out_img = torch.cat((x_1, x_1_recon), dim=3)
            #fs
            if 'use_fs_encoder' in self.config and self.config['use_fs_encoder']:
                k = self.idx_k 
                features = [None]*k + [fea_1] + [None]*(17-k)
                x_1_recon_2, _ = self.StyleGAN([w_recon], noise=n_1, input_is_latent=True, features_in=features, feature_scale=min(1.0, 0.0001*self.n_iter)) # 开始训练时权重小一点，慢慢变为1
                x_1_recon_2 = downscale(x_1_recon_2, self.scale, self.scale_mode)
                out_img = torch.cat((x_1, x_1_recon, x_1_recon_2), dim=3)
            utils.save_image(clip_img(out_img[:1]), log_dir + prefix + 'epoch_' +str(n_epoch+1) + '_iter_' + str(n_iter+1) + '_0.jpg')
            if out_img.size(0)>1:
                utils.save_image(clip_img(out_img[1:]), log_dir + prefix + 'epoch_' +str(n_epoch+1) + '_iter_' + str(n_iter+1) + '_1.jpg')
                        
    def save_model(self, log_dir):
        torch.save(self.enc.state_dict(),'{:s}/enc.pth.tar'.format(log_dir))

    def save_checkpoint(self, n_epoch, log_dir):
        checkpoint_state = {
            'n_epoch': n_epoch,
            'enc_state_dict': self.enc.state_dict(),
            'enc_opt_state_dict': self.enc_opt.state_dict(),
            'enc_scheduler_state_dict': self.enc_scheduler.state_dict(),
            'latent_compressor_state_dict': self.latent_compressor.state_dict(),
            'feature_compressor_state_dict': self.feature_compressor.state_dict(),
            'latent_compressor_opt_dict': self.latent_optimizer.state_dict(),
            'feature_compressor_opt_dict': self.feature_optimizer.state_dict(),
            'latent_compressor_lr_dict': self.latent_lr_scheduler.state_dict(),
            'feature_compressor_lr_dict': self.feature_lr_scheduler.state_dict(),

        }
        torch.save(checkpoint_state, '{:s}/checkpoint.pth'.format(log_dir))
        if (n_epoch+1)%10 == 0 :
            torch.save(checkpoint_state, '{:s}/checkpoint'.format(log_dir)+'_'+str(n_epoch+1)+'.pth')
    
    def load_model(self, log_dir):
        # self.enc.load_state_dict(torch.load('{:s}/enc.pth.tar'.format(log_dir)))
        self.enc.load_state_dict(torch.load('{:s}'.format(log_dir)))

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.enc.load_state_dict(state_dict['enc_state_dict'])
        self.enc_opt.load_state_dict(state_dict['enc_opt_state_dict'])
        self.enc_scheduler.load_state_dict(state_dict['enc_scheduler_state_dict'])
        self.latent_compressor.load_state_dict(state_dict['latent_compressor_state_dict'])
        self.feature_compressor.load_state_dict(state_dict['feature_compressor_state_dict'])
        self.latent_optimizer.load_state_dict(state_dict['latent_compressor_opt_dict'])
        self.feature_optimizer.load_state_dict(state_dict['feature_compressor_opt_dict'])
        self.latent_lr_scheduler.load_state_dict(state_dict['latent_compressor_lr_dict'])
        self.feature_lr_scheduler.load_state_dict(state_dict['feature_compressor_lr_dict'])
        return state_dict['n_epoch'] + 1

    def update(self, w=None, img=None, noise=None, real_img=None, n_iter=0):
        self.n_iter = n_iter
        self.enc_opt.zero_grad()
        self.compute_loss(w=w, img=img, noise=noise, real_img=real_img).backward()
        self.enc_opt.step()

    def update_latent(self, latent=None, n_iter=0):
        self.n_iter = n_iter
        self.latent_optimizer.zero_grad()
        self.latent_aux_optimizer.zero_grad()

        batch_size = latent.shape[0]
        original_shape = latent.shape
        latent = latent.view(batch_size, 8, 32, 32)

        out_net = self.latent_compressor(latent)

        self.out_latent_criterion = self.rate_loss_fn(out_net, latent)
        self.out_latent_criterion["loss"].backward()
        if self.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.latent_compressor.parameters(), self.clip_max_norm)
        self.latent_optimizer.step()

        self.latent_aux_loss = self.latent_compressor.aux_loss()
        self.latent_aux_loss.backward()
        self.latent_aux_optimizer.step()
        compressed_latent = out_net["x_hat"].view(original_shape)
        return compressed_latent

    def update_feature(self, feature=None, latent=None, n_iter=0):
        self.latent_compressor.eval()
        self.n_iter = n_iter
        self.feature_optimizer.zero_grad()
        self.feature_aux_optimizer.zero_grad()

        batch_size = latent.shape[0]
        original_shape = latent.shape
        latent = latent.view(batch_size, 8, 32, 32)
        out_net = self.feature_compressor(feature, latent)

        self.out_feature_criterion = self.rate_loss_fn(out_net, feature)
        self.out_feature_criterion["loss"].backward()
        if self.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.feature_compressor.parameters(), self.clip_max_norm)
        self.feature_optimizer.step()

        self.feature_aux_loss = self.feature_compressor.aux_loss()
        self.feature_aux_loss.backward()
        self.feature_aux_optimizer.step()
        compressed_feature = out_net["x_hat"]
        return compressed_feature

    def update_both(self, feature, latent, n_iter=0):
        self.n_iter = n_iter
        self.latent_optimizer.zero_grad()
        self.latent_aux_optimizer.zero_grad()
        self.feature_optimizer.zero_grad()
        self.feature_aux_optimizer.zero_grad()

        batch_size = latent.shape[0]
        original_shape = latent.shape
        latent = latent.view(batch_size, 8, 32, 32)
  
        out_net = self.latent_compressor(latent)
        self.out_latent_criterion = self.rate_loss_fn(out_net, latent)
        compressed_latent = out_net["x_hat"]

        out_net = self.feature_compressor(feature, latent)
        compressed_feature = out_net["x_hat"]

        self.out_feature_criterion = self.rate_loss_fn(out_net, feature)
        self.sumloss = self.out_latent_criterion["loss"] + self.out_feature_criterion["loss"]
        self.sumloss.backward()
        if self.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.latent_compressor.parameters(), self.clip_max_norm)

        if self.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.feature_compressor.parameters(), self.clip_max_norm)

        self.latent_optimizer.step()
        self.feature_optimizer.step()
        
        self.latent_aux_loss = self.latent_compressor.aux_loss()
        self.latent_aux_loss.backward()
        self.feature_aux_loss = self.feature_compressor.aux_loss()
        self.feature_aux_loss.backward()

        self.latent_aux_optimizer.step()
        self.feature_aux_optimizer.step()
        compressed_latent = compressed_latent.view(original_shape)

        return compressed_latent, compressed_feature
    
    def get_feature_latent(self, w=None, img=None, noise=None, real_img=None, n_iter=0):
        self.n_iter = n_iter
        if img is None:
            # generate synthetic images
            if noise is None:
                noise = [torch.randn(w.size()[:1] + ee.size()[1:]).to(self.device) for ee in self.noise_inputs]
            img, _ = self.StyleGAN([w], input_is_latent=True, noise = noise)
            img = img.detach()

        if img is not None and real_img is not None:
            # concat synthetic and real data
            img = torch.cat([img, real_img], dim=0)
            noise = [torch.cat([ee, ee], dim=0) for ee in noise]
        
        out = self.get_image(w=w, img=img, noise=noise)
        x_1_recon, x_1, w_recon, w_delta, n_1, fea_1, fea_recon = out
        return w_recon, fea_1
    
    def compute_compressed_loss_stylegan2(self, w=None, img=None, noise=None, real_img=None, compressed_stage=None):
        
        if img is None:
            # generate synthetic images
            if noise is None:
                noise = [torch.randn(w.size()[:1] + ee.size()[1:]).to(self.device) for ee in self.noise_inputs]
            img, _ = self.StyleGAN([w], input_is_latent=True, noise = noise)
            img = img.detach()

        if img is not None and real_img is not None:
            # concat synthetic and real data
            img = torch.cat([img, real_img], dim=0)
            noise = [torch.cat([ee, ee], dim=0) for ee in noise]
        
        out = self.get_image(w=w, img=img, noise=noise)
        x_1_recon, x_1, w_recon, w_delta, n_1, fea_1, fea_recon = out
        
        # debug
        # breakpoint()

        # use compressed feature and latent to calculate the loss
        batch_size = w_recon.shape[0]
        original_shape = w_recon.shape
        w_recon = w_recon.view(batch_size, 8, 32, 32) # 这里将9改为了7 因为分辨率降低了 style少了四个
        w_recon = self.latent_compressor(w_recon)["x_hat"]
        fea_1 = self.feature_compressor(fea_1, w_recon)["x_hat"] if compressed_stage > 1 else fea_1
        w_recon = w_recon.view(original_shape)

        # Loss setting
        w_l2, w_lpips, w_id = self.config['w']['l2'], self.config['w']['lpips'], self.config['w']['id']
        b = x_1.size(0)//2
        if 'l2loss_on_real_image' in self.config and self.config['l2loss_on_real_image']:
            b = x_1.size(0)
        self.l2_loss = self.L2loss(x_1_recon[:b], x_1[:b]) if w_l2 > 0 else torch.tensor(0) # l2 loss only on synthetic data
        # LPIPS
        multiscale_lpips=False if 'multiscale_lpips' not in self.config else self.config['multiscale_lpips']
        self.lpips_loss = self.LPIPS(x_1_recon, x_1, multi_scale=multiscale_lpips).mean() if w_lpips > 0 else torch.tensor(0)
        self.id_loss = self.IDloss(x_1_recon, x_1).mean() if w_id > 0 else torch.tensor(0)
        self.landmark_loss = self.landmarkloss(x_1_recon, x_1) if self.config['w']['landmark'] > 0 else torch.tensor(0)
        
        if 'use_fs_encoder' in self.config and self.config['use_fs_encoder']:
            k = self.idx_k 
            features = [None]*k + [fea_1] + [None]*(17-k)
            x_1_recon_2, _ = self.StyleGAN([w_recon], noise=n_1, input_is_latent=True, features_in=features, feature_scale=min(1.0, 0.0001*self.n_iter))
            self.lpips_loss += self.LPIPS(x_1_recon_2, x_1, multi_scale=multiscale_lpips).mean() if w_lpips > 0 else torch.tensor(0)
            self.id_loss += self.IDloss(x_1_recon_2, x_1).mean() if w_id > 0 else torch.tensor(0)
            self.landmark_loss += self.landmarkloss(x_1_recon_2, x_1) if self.config['w']['landmark'] > 0 else torch.tensor(0)

        # downscale image
        x_1 = downscale(x_1, self.scale, self.scale_mode)
        x_1_recon = downscale(x_1_recon, self.scale, self.scale_mode)
        
        # Total loss
        w_l2, w_lpips, w_id = self.config['w']['l2'], self.config['w']['lpips'], self.config['w']['id']
        self.loss = w_l2*self.l2_loss + w_lpips*self.lpips_loss + w_id*self.id_loss
        
        if 'f_recon' in self.config['w']:
            self.feature_recon_loss = self.L2loss(fea_1, fea_recon) 
            self.loss += self.config['w']['f_recon']*self.feature_recon_loss
        if 'l1' in self.config['w'] and self.config['w']['l1']>0:
            self.l1_loss = self.L1loss(x_1_recon, x_1)
            self.loss += self.config['w']['l1']*self.l1_loss
        if 'landmark' in self.config['w']:
            self.loss += self.config['w']['landmark']*self.landmark_loss
        return self.loss
    
    def save_compressed_image(self, log_dir, n_epoch, n_iter, prefix='/train/', w=None, img=None, noise=None, training_mode=True, compressed_stage=None):
        return self.save_compressed_image_stylegan2(log_dir=log_dir, n_epoch=n_epoch, n_iter=n_iter, prefix=prefix, w=w, img=img, noise=noise, training_mode=training_mode, compressed_stage=compressed_stage)

    def save_compressed_image_stylegan2(self, log_dir, n_epoch, n_iter, prefix='/train/', w=None, img=None, noise=None, training_mode=True, compressed_stage=None):
        os.makedirs(log_dir + prefix, exist_ok=True)
        with torch.no_grad():
            out = self.get_compressed_image(w=w, img=img, noise=noise, training_mode=training_mode, compressed_stage=compressed_stage)
            x_1_recon, x_1, w_recon, w_delta, n_1, fea_1 = out[:6]
            x_1 = downscale(x_1, self.scale, self.scale_mode)
            x_1_recon = downscale(x_1_recon, self.scale, self.scale_mode)
            out_img = torch.cat((x_1, x_1_recon), dim=3)
            #fs
            if 'use_fs_encoder' in self.config and self.config['use_fs_encoder']:
                k = self.idx_k 
                features = [None]*k + [fea_1] + [None]*(17-k)
                x_1_recon_2, _ = self.StyleGAN([w_recon], noise=n_1, input_is_latent=True, features_in=features, feature_scale=1) # 开始训练时权重小一点，慢慢变为1
                # x_1_recon_2, _ = self.StyleGAN([w_recon], noise=n_1, input_is_latent=True, features_in=features, feature_scale=min(1.0, 0.0001*self.n_iter)) # 开始训练时权重小一点，慢慢变为1
                x_1_recon_2 = downscale(x_1_recon_2, self.scale, self.scale_mode)
                out_img = torch.cat((x_1, x_1_recon, x_1_recon_2), dim=3)
            utils.save_image(clip_img(out_img[:1]), log_dir + prefix + 'epoch_' +str(n_epoch+1) + '_iter_' + str(n_iter+1) + '_0.jpg')
            if out_img.size(0)>1:
                utils.save_image(clip_img(out_img[1:]), log_dir + prefix + 'epoch_' +str(n_epoch+1) + '_iter_' + str(n_iter+1) + '_1.jpg')

    def forward(self, x):
        return self.compressed_test(img=x, return_latent=True, compressed_stage=4)
    
    def compressed_test(self, w=None, img=None, noise=None, zero_noise_input=True, return_latent=False, training_mode=False, compressed_stage=None):        
        if 'n_iter' not in self.__dict__.keys():
            self.n_iter = 1e5
        set_entropy_coder("ans")
        self.latent_compressor.update(force=True)
        self.feature_compressor.update(force=True)
        out = self.get_compressed_image(w=w, img=img, noise=noise, training_mode=training_mode, compressed_stage=compressed_stage)
        x_1_recon, x_1, w_recon, w_delta, n_1, fea_1 = out[:6]
        output = [x_1, x_1_recon]
        if return_latent:
            output += [w_recon, fea_1]
        return output

    def get_compressed_image(self, w=None, img=None, noise=None, zero_noise_input=True, training_mode=True, compressed_stage=None):
        x_1, n_1 = img, noise
        if x_1 is None:
            x_1, _ = self.StyleGAN([w], input_is_latent=True, noise = n_1)
           
        w_delta = None
        fea = None
        features = None
        return_features = False
        # Reconstruction
        k = 0
        if 'use_fs_encoder' in self.config and self.config['use_fs_encoder']:
            return_features = True
            k = self.idx_k # 在第五层之后加入训练的特征
            start_time = time()
            w_recon, fea = self.enc(downscale(x_1, self.scale, self.scale_mode))
            w_recon = w_recon + self.dlatent_avg # 加整个网络的平均
            batch_size = w_recon.shape[0]
            original_shape = w_recon.shape
            w_recon = w_recon.view(batch_size, 8, 32, 32)

            if compressed_stage != 4:
                w_recon_compressed = self.latent_compressor(w_recon)["x_hat"]
                w_recon = w_recon_compressed.view(original_shape)
                fea = self.feature_compressor(fea, w_recon_compressed)["x_hat"] if compressed_stage > 1 else fea
            else:
                w_recon_encode = self.latent_compressor.compress(w_recon)
                mse = nn.MSELoss()
                w_recon_compressed = self.latent_compressor.decompress(w_recon_encode["strings"], w_recon_encode["shape"])["x_hat"]
                w_recon = w_recon_compressed.view(original_shape)
                fea_encode = self.feature_compressor.compress(fea, w_recon_compressed)
                end_time = time()
                enc_time = end_time - start_time
                start_time = time()
                fea = self.feature_compressor.decompress(fea_encode["strings"], fea_encode["shape"], w_recon_compressed)["x_hat"]
                # w_recon_shape = self.latent_compressor.compress(w_recon)["shape"]
                bpp = (len(w_recon_encode["strings"][0][0]) + len(fea_encode["strings"][0][0])) * 8.0 / 512 / 384
                latent_bpp = len(w_recon_encode["strings"][0][0]) * 8.0 / 512 / 384
                feature_bpp = len(fea_encode["strings"][0][0]) * 8.0 / 512 / 384
                print(f"s_bpp: {latent_bpp:4f}, f_bpp: {feature_bpp:4f}, sum_bpp: {bpp:4f}")
                self.test_bpp += bpp
                self.test_num += 1

            features = [None]*k + [fea] + [None]*(17-k)
        else:
            w_recon = self.enc(downscale(x_1, self.scale, self.scale_mode)) + self.dlatent_avg
            batch_size = w_recon.shape[0]
            original_shape = w_recon.shape
            w_recon = w_recon.view(batch_size, 8, 32, 32)
            w_recon = self.latent_compressor(w_recon)["x_hat"].view(original_shape)

        # generate image
        # x_1_recon, fea_recon = self.StyleGAN([w_recon], input_is_latent=True, return_features=True, features_in=features, feature_scale=min(1.0, 0.0001*self.n_iter)) # 得到重建人脸和插入的feature
        x_1_recon, fea_recon = self.StyleGAN([w_recon], input_is_latent=True, return_features=True, features_in=features, feature_scale=1) # 得到重建人脸和插入的feature
        # x_1_recon, fea_recon = self.StyleGAN([w_recon], input_is_latent=True, return_features=True, feature_scale=1) # 得到重建人脸和插入的feature
        end_time = time()
        dec_time = end_time - start_time
        print(f"enc time: {enc_time}, dec time: {dec_time}")
        fea_recon = fea_recon[k].detach() 
        return [x_1_recon, x_1[:,:3,:,:], w_recon, w_delta, n_1, fea, fea_recon]

    def compress_boundary(self, boundary=None, compress_stage=1):
        original_shape = boundary.shape
        boundary = boundary.view(1, 9, 32, 32)
        if compress_stage != 4:
            boundary_recon_compressed = self.latent_compressor(boundary)["x_hat"]
            boundary_recon = boundary_recon_compressed.view(original_shape)
        else:
            boundary_recon_encode = self.latent_compressor.compress(boundary)
            boundary_recon_compressed = self.latent_compressor.decompress(boundary_recon_encode["strings"], boundary_recon_encode["shape"])["x_hat"]
            bpp = (len(boundary_recon_encode["strings"][0][0])) * 8.0 / 1024 / 1024
            boundary_recon = boundary_recon_compressed.view(original_shape)
            print(f"boundary {bpp}")
        return boundary_recon