import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# sn 은 spectral norm을 뜻함

class ConvLReLU(nn.Module):
    def __init__(self, i_channels ,o_channels, kernel, stride, pad, layer_norm_bool=False, lrelu=True):
        super(ConvLReLU, self).__init__()

        if (kernel - stride) %2 == 0:
            pad_top = pad
            pad_bottom = pad
            pad_left = pad
            pad_right = pad
        else:
            pad_top = pad
            pad_bottom = kernel - stride - pad_top
            pad_left = pad
            pad_right = kernel - stride - pad_left

        self.paddings = (pad_left, pad_right, pad_top, pad_bottom, 0,0, 0,0)
        self.kernel = kernel
        self.o_channels = o_channels
        
        conv2d = nn.Conv2d(i_channels,o_channels, kernel,stride=stride)
        print('conv2d', conv2d)
        nn.init.normal_(conv2d.weight, mean=0, std=0.02)
        self.spectral_norm_conv = torch.nn.utils.spectral_norm(conv2d)
        self.layer_norm_bool = layer_norm_bool
        self.lrelu = lrelu
        self.lrelu_func = torch.nn.LeakyReLU(0.2)

    def forward(self, input):
        # x input ? (12, 256, 256, 3) 형태? 아니면 padding도, spectral_norm_conv 형태도 바뀌어야 함.
        # N C H W 로 들어와야 함.
        
        x = torch.nn.functional.pad(input, self.paddings)
        # N C H W
        # how can i decide the input size of LayerNorm...?
        x = self.spectral_norm_conv(x)
        
        if self.layer_norm_bool:
            layer_norm = torch.nn.LayerNorm(x.shape[1:]).cuda()
            x = layer_norm(x)
        if self.lrelu:
            x = self.lrelu_func(x)  
        return x


class Discriminator(nn.Module):
    def __init__(self, ch=64, n_dis=3):
        super(Discriminator, self).__init__()
        self.ch = ch
        self.n_dis = n_dis

        self.init_stage = ConvLReLU(3, 32, kernel=3, stride=1, pad=1)

        self.middle_stage = nn.Sequential(
            ConvLReLU(32, 64, kernel=3, stride=2, pad=1),
            ConvLReLU(64, 128, kernel=3, stride=1, pad=1, layer_norm_bool=True),
            ConvLReLU(128, 128, kernel=3, stride=2, pad=1),
            ConvLReLU(128, 256, kernel=3, stride=1, pad=1, layer_norm_bool=True)
        )

        self.last_stage = nn.Sequential(
            ConvLReLU(256, 512, kernel=3, stride=1, pad=1, layer_norm_bool=True),
            ConvLReLU(512, 1, kernel=3, stride=1, pad=1, layer_norm_bool=False, lrelu=False)
        )

    def forward(self, input):
        # channel = self.ch // 2
        # x = ConvLReLU(input.shape[1], channel, kernel=3, stride=1, pad=1)(input)

        # for i in range(1, self.n_dis):
        #     x = ConvLReLU(x.shape[1], channel*2, kernel=3, stride=2, pad=1)(x)
        #     x = ConvLReLU(x.shape[1], channel*4, kernel=3, stride=1, pad=1, layer_norm=True)(x)
        #     channel = channel * 2
        
        # x = ConvLReLU(x.shape[1], channel*2, kernel=3, stride=1, pad=1, layer_norm=True)(x)
        # x = ConvLReLU(x.shape[1], 1, kernel=3, stride=1, pad=1, layer_norm=False, lrelu=False)(x)
        x = self.init_stage(input)
        x = self.middle_stage(x)
        x = self.last_stage(x)

        return x