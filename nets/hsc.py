import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
import warnings
from compressai.ans import RansEncoder, RansDecoder, BufferedRansEncoder, RansDecoder
from compressai.registry import register_model
from compressai.layers import GDN, MaskedConv2d
from compressai.entropy_models import EntropyBottleneck
from compressai.models.utils import conv, deconv
from compressai.models.google import FactorizedPrior, MeanScaleHyperprior

eps = 1e-9
class Quantizer():
    def quantize(self, inputs, quantize_type="noise"):
        if quantize_type == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs
        elif quantize_type == "ste":
            return torch.round(inputs) - inputs.detach() + inputs
        else:
            return torch.round(inputs)

class latent_compress(FactorizedPrior):
    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=N, **kwargs)

        self.entropy_bottleneck = EntropyBottleneck(M)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(9, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 9, 2),
        )

        self.N = N
        self.M = M
    
    def forward(self, x):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }
    
    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}



    
class feature_compress_channel(MeanScaleHyperprior):
    def __init__(self, N=192, M=192, slice = 8, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.g_a = nn.Sequential(
            ResidualBlockWithStride(512, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=1),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=1),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=1),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 1),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 1),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 1),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 512, 2),
        )

        # 相比于前面的方法需要更新输出的channel数
        self.latent_convert = nn.Sequential(
            nn.Conv2d(9, N, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N, N, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=1),
        )
        
        self.slice = slice 
        self.slice_size = M//self.slice #Channel size for one slice. Note that M % slice should be zero
        self.y_size_list = [(i + 1) * self.slice_size for i in range(self.slice -1)]
        self.y_size_list.append(M)    #[32, 64, 96, 128, 160, 192, 224, 256, 288, 320] if M = 320 and slice = 10
        EP_inputs = [i * self.slice_size for i in range(self.slice)]    #Input channel size for entropy parameters layer. [0, 32, 64, 96, 128, 160, 192, 224, 256, 288] if M = 320 and slice = 10
        self.EPlist = nn.ModuleList([])
        for y_size in EP_inputs:
            EP = nn.Sequential(
                conv(y_size + M, M - (N//2), stride=1, kernel_size= 3),
                nn.LeakyReLU(inplace=True),
                conv(M - (N//2), (M + N) // 4,  stride=1, kernel_size= 3),
                nn.LeakyReLU(inplace=True),
                conv((M + N) // 4, M * 2 // slice, stride=1, kernel_size=3),     
            )
            self.EPlist.append(EP)

    def forward(self, feature, latent):
        y = self.g_a(feature)  # y.shape [2, 192, 8, 8]
        latent_params = self.latent_convert(latent) # latent_params [2, 192, 8, 8]
        list_sliced_y = [] #Stores each slice of y
        for i in range(self.slice - 1):
            list_sliced_y.append(y[:,(self.slice_size * i):(self.slice_size * (i + 1)),:,:])
        list_sliced_y.append(y[:,self.slice_size * (self.slice - 1):,:,:])
        y_hat_cumul = torch.Tensor().to(y.device) #Cumulative y_hat. Stores already encoded y_hat slice
        scales_hat_list = []
        means_hat_list = []
        for i in range(self.slice):
            if i == 0:
                gaussian_params = self.EPlist[0](
                    latent_params
                )
            else:
                gaussian_params = self.EPlist[i](
                    torch.cat([latent_params, y_hat_cumul], dim = 1)
                )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            scales_hat_list.append(scales_hat)
            means_hat_list.append(means_hat)
            y_hat_sliced = self.gaussian_conditional.quantize(
               list_sliced_y[i] , "noise" if self.training else "dequantize"
            )
            y_hat_cumul = torch.cat([y_hat_cumul, y_hat_sliced], dim = 1)

        scales_all = torch.cat(scales_hat_list, dim = 1)
        means_all = torch.cat(means_hat_list, dim = 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_all, means=means_all)
        x_hat = self.g_s(y_hat_cumul)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods},
        }

    def compress(self, feature, latent):
        encoder = BufferedRansEncoder()
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        indexes_list = []
        symbols_list = []
        y_strings = []        
        y = self.g_a(feature)
        latent_params = self.latent_convert(latent)
        list_sliced_y = []
        for i in range(self.slice - 1):
            list_sliced_y.append(y[:,(self.slice_size * i):self.slice_size * (i + 1),:,:])
        list_sliced_y.append(y[:,self.slice_size * (self.slice - 1):,:,:])
        y_hat = torch.Tensor().to(feature.device)
        for i in range(self.slice):
            y_sliced = list_sliced_y[i] #size[1, M/S * i, H', W']
            if i == 0 :
                gaussian_params = self.EPlist[0](
                    latent_params
                )
            else: 
                gaussian_params = self.EPlist[i](
                    torch.cat([latent_params, y_hat], dim=1)
                )
            #gaussian_params = gaussian_params.squeeze(3).squeeze(2) #size ([1,256])
            scales_hat, means_hat = gaussian_params.chunk(2, 1) 
            indexes = self.gaussian_conditional.build_indexes(scales_hat)
            
            y_hat_sliced = self.gaussian_conditional.quantize(y_sliced, "symbols", means_hat)
            symbols_list.extend(y_hat_sliced.reshape(-1).tolist())
            indexes_list.extend(indexes.reshape(-1).tolist())
            y_hat_sliced = y_hat_sliced + means_hat
            
            y_hat = torch.cat([y_hat, y_hat_sliced], dim = 1)
           
        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        y_string = encoder.flush()
        y_strings.append(y_string)
        return {"strings": [y_strings], "shape": y.size()[-4:]}     

    def decompress(self, strings, shape, latent):
        latent_params = self.latent_convert(latent)
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        decoder = RansDecoder()
        decoder.set_stream(strings[0][0])
        y_hat = torch.Tensor().to(latent.device)
        for i in range(self.slice):
            if i == 0:
                gaussian_params = self.EPlist[0](latent_params) 
            else:
                gaussian_params = self.EPlist[i](
                    torch.cat([latent_params, y_hat], dim = 1)
                )
            scales_sliced, means_sliced = gaussian_params.chunk(2,1)
            indexes_sliced = self.gaussian_conditional.build_indexes(scales_sliced)
            y_sliced_hat = decoder.decode_stream(
                indexes_sliced.reshape(-1).tolist(), cdf, cdf_lengths, offsets
            )
            y_sliced_hat  =torch.Tensor(y_sliced_hat).reshape(scales_sliced.shape).to(scales_sliced.device)
            y_sliced_hat += means_sliced
            y_hat = torch.cat([y_hat, y_sliced_hat], dim = 1)
            
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}
        


class latent_compress_church(latent_compress):
    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=N, **kwargs)
        self.g_a = nn.Sequential(
            ResidualBlockWithStride(7, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 7, 2),
        )

class latent_compress_car(latent_compress):
    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=N, **kwargs)
        self.g_a = nn.Sequential(
            ResidualBlockWithStride(8, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 8, 2),
        )


class feature_compress_channel_car(feature_compress_channel):
    def __init__(self, N=192, M=192, slice = 8, **kwargs):
        super().__init__(N=192, M=192, slice = 8, **kwargs)
        self.latent_convert = nn.Sequential(
            nn.Conv2d(8, N, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N, N, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )



class feature_compress_channel_church(feature_compress_channel):
    def __init__(self, N=192, M=192, slice = 8, **kwargs):
        super().__init__(N=192, M=192, slice = 8, **kwargs)
        self.latent_convert = nn.Sequential(
            nn.Conv2d(7, N, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N, N, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=1),
        )