import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision

class Tensorboard():
    def __init__(self, training_rate):
        self.writer = SummaryWriter()
        self.step = 0
        self.n_steps = 10 # 10 iters씩 Data check
        self.d_steps = self.n_steps / training_rate
        
        assert self.n_steps % self.d_steps == 0, 'Check the training rate!'
        self.losses = {
            'init_con_loss' : 0.0,
            'main_con_loss' : 0.0,
            'style_loss' : 0.0,
            'real_d_loss' : 0.0,
            'gray_d_loss' : 0.0,
            'fake_d_loss' : 0.0,
            'real_blur_d_loss' : 0.0,
            'd_loss' : 0.0,
            'tv_loss' : 0.0,
            'color_loss' : 0.0,
            'g_loss' : 0.0,
        }

    def step_one(self, init_con_loss=0.0, main_con_loss=0.0, style_loss=0.0, real_d_loss=0.0, \
                gray_d_loss=0.0, fake_d_loss=0.0, real_blur_d_loss=0.0, d_loss=0.0, tv_loss=0.0, color_loss=0.0, g_loss=0.0):
        self.step += 1

        self.losses['init_con_loss'] += init_con_loss
        self.losses['main_con_loss'] += main_con_loss
        self.losses['style_loss'] += style_loss
        self.losses['real_d_loss'] += real_d_loss
        self.losses['gray_d_loss'] += gray_d_loss
        self.losses['fake_d_loss'] += fake_d_loss
        self.losses['real_blur_d_loss'] += real_blur_d_loss
        self.losses['d_loss'] += d_loss
        self.losses['tv_loss'] += tv_loss
        self.losses['color_loss'] += color_loss
        self.losses['g_loss'] += g_loss

        if self.step % self.n_steps == 0:
            keys = list(self.losses.keys())
            keys.sort()
            for key in keys:
                if self.losses[key] != 0 and 'init' in key:
                    self.writer.add_scalar('Init/'+ key, self.losses[key]/self.n_steps, self.step)
                elif self.losses[key] != 0 and 'd_' in key:
                    self.writer.add_scalar('D_loss/'+ key, self.losses[key]/self.d_steps, self.step)
                elif self.losses[key] != 0:
                    self.writer.add_scalar('G_loss/'+ key, self.losses[key]/self.n_steps, self.step)
                self.losses[key] = 0

    def reset(self):
        keys = list(self.losses.keys()).sort()
        for key in keys:
            self.losses[key] = 0.0
    
    def close(self):
        self.writer.close()