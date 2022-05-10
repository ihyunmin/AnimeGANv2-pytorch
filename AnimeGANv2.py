from calendar import c
from tools.utils import *
from glob import glob
import time
import numpy as np
from model.generator import Generator
from model.discriminator import Discriminator
from tools.data_loader import ImageDataset
from vgg.vgg import VGG
from losses import *
import os
from torch.utils.data import DataLoader

class AnimeGANv2(object) :
    def __init__(self, args):
        self.model_name = 'AnimeGANv2'
        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset

        self.epoch = args.epoch
        self.init_epoch = args.init_epoch # args.epoch // 20

        self.gan_type = args.gan_type
        self.batch_size = args.batch_size
        self.save_freq = args.save_freq

        self.init_lr = args.init_lr
        self.d_lr = args.d_lr
        self.g_lr = args.g_lr

        """ Weight """
        self.g_adv_weight = args.g_adv_weight
        self.d_adv_weight = args.d_adv_weight
        self.con_weight = args.con_weight
        self.sty_weight = args.sty_weight
        self.color_weight = args.color_weight
        self.tv_weight = args.tv_weight

        self.training_rate = args.training_rate
        self.ld = args.ld

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        """ Discriminator """
        self.n_dis = args.n_dis
        self.ch = args.ch
        self.sn = args.sn

        self.sample_dir = os.path.join(args.sample_dir)
        check_folder(self.sample_dir)

        # imageGenerator -> folder image load?
        self.real_dataset = ImageDataset('./dataset/train_photo')
        self.anime_dataset = ImageDataset('./dataset/{}'.format(self.dataset_name + '/style'))
        self.anime_smooth_dataset = ImageDataset('./dataset/{}'.format(self.dataset_name + '/smooth'))
        
        self.real_dataloader = DataLoader(
                                    dataset=self.real_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    drop_last=False
                                )
        self.anime_dataloader = DataLoader(
                                    dataset=self.anime_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    drop_last=False
                                )
        self.anime_smooth_dataloader = DataLoader(
                                    dataset=self.anime_smooth_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    drop_last=False
                                )

        self.anime_sampler = iter(self.anime_dataloader)
        self.anime_smooth_sampler = iter(self.anime_smooth_dataloader)
        self.real_sampler = iter(self.real_dataloader)
        # dataset number? why it is max?
        self.dataset_num = max(self.real_dataset.num_images, self.anime_dataset.num_images)

        check = 0
        for test in self.real_dataloader:
            check += 1
        print(check)

        # use frozen VGG19
        vgg_model_name = 'vgg19_bn'
        self.vgg = VGG.from_pretrained(vgg_model_name)
        self.vgg.cuda()
        
        for child in self.vgg.children():
            for param in child.parameters():
                param.requires_grad = False

        print()
        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# init_epoch : ", self.init_epoch)
        print("# training image size [H, W] : ", self.img_size)
        print("# g_adv_weight,d_adv_weight,con_weight,sty_weight,color_weight,tv_weight : ", self.g_adv_weight,self.d_adv_weight,self.con_weight,self.sty_weight,self.color_weight,self.tv_weight)
        print("# init_lr,g_lr,d_lr : ", self.init_lr,self.g_lr,self.d_lr)
        print(f"# training_rate G -- D: {self.training_rate} : 1" )
        print()

    ##################################################################################
    # Model
    # I will use lsgan or gan
    ##################################################################################
    def gradient_panalty(self, real, fake, scope="discriminator"):
        # if self.gan_type.__contains__('dragan') :
        #     eps = tf.random_uniform(shape=tf.shape(real), minval=0., maxval=1.)
        #     _, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
        #     x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region

        #     fake = real + 0.5 * x_std * eps

        # alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        # interpolated = real + alpha * (fake - real)

        # logit, _= self.discriminator(interpolated, reuse=True, scope=scope)

        # grad = tf.gradients(logit, interpolated)[0] # gradient of D(interpolated)
        # grad_norm = tf.norm(flatten(grad), axis=1) # l2 norm

        # GP = 0
        # # WGAN - LP
        # if self.gan_type.__contains__('lp'):
        #     GP = self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))

        # elif self.gan_type.__contains__('gp') or self.gan_type == 'dragan' :
        #     GP = self.ld * tf.reduce_mean(tf.square(grad_norm - 1.))

        # return GP
        return 0.0

    def build_model(self):

        """ Define Loss """
        if self.gan_type.__contains__('gp') or self.gan_type.__contains__('lp') or self.gan_type.__contains__('dragan') :
            self.GP = self.gradient_panalty(real=self.anime, fake=self.generated)
        else :
            self.GP = 0.0

        self.generator = Generator().cuda()
        self.discriminator = Discriminator().cuda()

    def get_images(self):
        try:    
            anime_images = next(self.anime_sampler)
        except StopIteration:
            self.anime_sampler = iter(self.anime_dataloader)
            anime_images = next(self.anime_sampler)
        try:
            anime_smooth_images = next(self.anime_smooth_sampler)
        except StopIteration:
            self.anime_smooth_sampler = iter(self.anime_smooth_dataloader)
            anime_smooth_images = next(self.anime_sampler)
        try:
            real_images = next(self.real_sampler)
        except StopIteration:
            self.real_sampler = iter(self.real_dataloader)
            real_images = next(self.real_sampler)
        
        return anime_images, anime_smooth_images, real_images

    def train(self):
        
        # restore check-point if it exits
        start_epoch = self.load(self.checkpoint_dir)
        if start_epoch != 0:
            start_epoch += 1
            print(" [*] Load SUCCESS, starting epoch is {}".format(start_epoch))
        else:
            print(" [!] Load failed..., starting epoch is {}".format(start_epoch))

        # loop for epoch
        init_mean_loss = []
        mean_loss = []
        # training times , G : D = self.training_rate : 1
        j = self.training_rate

        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.g_lr, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.d_lr, betas=(0.5, 0.999))

        for epoch in range(start_epoch, self.epoch):
            for idx in range(int(self.dataset_num / self.batch_size)):

                anime_images, anime_smooth_images, real_images = self.get_images()
                # anime_images, anime_smooth_images, real_images = self.anime_dataloader[idx], self.anime_smooth_dataloader[idx], self.real_dataloader[idx]
                
                real = real_images[0].cuda()
                anime = anime_images[0].cuda()
                anime_gray = anime_images[1].cuda()
                anime_smooth = anime_smooth_images[1].cuda()
                
                assert real.shape[1] == 3 and anime.shape[1] == 3 and anime_gray.shape[1] == 3 and anime_smooth.shape[1] == 3, \
                        'Image shape input error, shape : {}'.format(str(list(real.shape)))
                
                fake = self.generator(real)
                generated_logit = self.discriminator(fake)

                assert fake.shape[1] == 3 and generated_logit.shape[1] == 1, \
                        'Generated Image shape input error, shape : {}'.format(str(list(fake.shape)))

                if epoch < self.init_epoch :
                    # 
                    # Init phase : Train the generator with only content loss from the vgg19.
                    # 
                    start_time = time.time()
                    optimizer_G.zero_grad()

                    assert real.shape == fake.shape, 'Real and Fake have to same shape each other'
                    content_loss = con_loss(self.vgg, real, fake)
                    
                    print(content_loss, type(content_loss))
                    assert type(content_loss.item()) is float, 'con_loss must be float'
                    
                    content_loss.backward()
                    optimizer_G.step()
                    init_mean_loss.append(content_loss.item())

                    print("Epoch: %3d Step: %5d / %5d  time: %f s init_con_loss: %.8f  mean_con_loss: %.8f" % (epoch, idx,int(self.dataset_num / self.batch_size), time.time() - start_time, content_loss, np.mean(init_mean_loss)))
                    if (idx+1)%200 ==0:
                        init_mean_loss.clear()
                else :
                    start_time = time.time()

                    if j == self.training_rate:
                        # Update D
                        optimizer_D.zero_grad()
                        
                        anime_logit = self.discriminator(anime)
                        anime_gray_logit = self.discriminator(anime_gray)
                        # generated_logit = self.discriminator(fake)
                        smooth_logit = self.discriminator(anime_smooth)

                        assert anime_logit.shape[1] == 1 and anime_gray_logit.shape[1] == 1 and generated_logit.shape[1] == 1 and smooth_logit.shape[1] == 1, \
                                'Discriminator logits shape must be (B, 1, ?, ?) , shape : {}'.format(str(list(anime_logit.shape)))

                        d_loss = self.d_adv_weight * discriminator_loss(self.gan_type, anime_logit, anime_gray_logit, generated_logit, smooth_logit) + self.GP
                        print(d_loss, type(d_loss))
                        assert type(d_loss.item()) is float, 'd_loss must be float'

                        d_loss.backward()
                        optimizer_D.step()

                    # Update G

                    optimizer_G.zero_grad()

                    c_loss, s_loss = con_sty_loss(self.vgg, real, anime_gray, fake)
                    tv_loss = self.tv_weight * total_variation_loss(fake)
                    t_loss = self.con_weight * c_loss + self.sty_weight * s_loss + color_loss(real, fake) * self.color_weight + tv_loss
                    g_loss = self.g_adv_weight * generator_loss(self.gan_type, generated_logit)

                    g_total_loss =  t_loss + g_loss
                    
                    g_total_loss.backward()
                    optimizer_G.step()
                    
                    mean_loss.append([d_loss.item(), g_loss.item()])

                    if j == self.training_rate:
                        print(
                            "Epoch: %3d Step: %5d / %5d  time: %f s d_loss: %.8f, g_loss: %.8f -- mean_d_loss: %.8f, mean_g_loss: %.8f" % (
                                epoch, idx, int(self.dataset_num / self.batch_size), time.time() - start_time, d_loss, g_loss, np.mean(mean_loss, axis=0)[0],
                                np.mean(mean_loss, axis=0)[1]))
                    else:
                        print(
                            "Epoch: %3d Step: %5d / %5d time: %f s , g_loss: %.8f --  mean_g_loss: %.8f" % (
                                epoch, idx, int(self.dataset_num / self.batch_size), time.time() - start_time, g_loss, np.mean(mean_loss, axis=0)[1]))

                    if (idx + 1) % 200 == 0:
                        mean_loss.clear()

                    j = j - 1
                    if j < 1:
                        j = self.training_rate

            if (epoch + 1) >= self.init_epoch and np.mod(epoch + 1, self.save_freq) == 0:
                self.save(self.checkpoint_dir, epoch)

            if epoch >= self.init_epoch -1:
                """ Result Image """
                val_files = glob('./dataset/{}/*.*'.format('val'))
                save_path = './{}/{:03d}/'.format(self.sample_dir, epoch)
                check_folder(save_path)
                
                self.generator.eval()

                for i, sample_file in enumerate(val_files):
                    print('val: '+ str(i) + sample_file)
                    sample_image = np.asarray(load_test_data(sample_file, self.img_size))
                    assert sample_image.shape[1] == 3, 'Sample images shape is not correct'
                    
                    test_real = sample_image
                    test_image = torch.from_numpy(sample_image).cuda()
                    
                    with torch.no_grad():
                        test_generated = self.generator(test_image)
                        test_generated = test_generated.cpu().numpy()
                    print(type(test_real), type(test_generated))
                    save_images(test_real, save_path+'{:03d}_a.jpg'.format(i), None)
                    save_images(test_generated, save_path+'{:03d}_b.jpg'.format(i), None)
                self.generator.train()
                    

    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(self.model_name, self.dataset_name,
                                                          self.gan_type,
                                                          int(self.g_adv_weight), int(self.d_adv_weight),
                                                          int(self.con_weight), int(self.sty_weight),
                                                          int(self.color_weight), int(self.tv_weight))


    def save(self, checkpoint_dir, step):
        # checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        checkpoint_dir = os.path.join(checkpoint_dir)
        g_path = os.path.join(checkpoint_dir, f'generator_{step}.pth')
        d_path = os.path.join(checkpoint_dir, f'discriminator_{step}.pth')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        torch.save(self.generator.state_dict(), g_path)
        torch.save(self.discriminator.state_dict(), d_path)
        

    # check point 불러오기
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        
        pth_files = os.listdir(checkpoint_dir)
        epoch = 0
        for pth_name in pth_files:
            pth_path = os.path.join(checkpoint_dir, pth_name)
            print(pth_path)
            epoch = pth_path.split('.')[0].split('_')[1]
            if pth_name.split('_')[0] == 'generator':
                self.generator.load_state_dict(torch.load(pth_path, map_location='cuda'))
            else:
                self.discriminator.load_state_dict(torch.load(pth_path, map_location='cuda'))
                
            
        start_epoch = int(epoch)
        self.generator.train()
        self.discriminator.train()

        return start_epoch
