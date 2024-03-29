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
from tensor_board import Tensorboard

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
                                    drop_last=True
                                )
        self.anime_dataloader = DataLoader(
                                    dataset=self.anime_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    drop_last=True
                                )
        self.anime_smooth_dataloader = DataLoader(
                                    dataset=self.anime_smooth_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    drop_last=True
                                )

        self.anime_sampler = iter(self.anime_dataloader)
        self.anime_smooth_sampler = iter(self.anime_smooth_dataloader)
        self.real_sampler = iter(self.real_dataloader)
        # dataset number? why it is max?
        self.dataset_num = max(self.real_dataset.num_images, self.anime_dataset.num_images)

        # use frozen VGG19
        vgg_model_name = 'vgg19'
        self.vgg = VGG.from_pretrained(vgg_model_name)
        self.vgg.cuda()
        
        for child in self.vgg.children():
            for param in child.parameters():
                param.requires_grad = False

        self.tensorboard = Tensorboard(self.training_rate)

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
            anime_smooth_images = next(self.anime_smooth_sampler)
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
        
        test_path = os.path.join(os.getcwd(), 'test_images')
        os.makedirs(test_path, exist_ok=True)
        
        for epoch in range(start_epoch, self.epoch):
            # Class 만들어서 Loss 구할 것.
            # iters ==> 미리 정한 step으로 나누어떨어지게끔.
            iters = int(self.dataset_num / self.batch_size) - int(self.dataset_num / self.batch_size) % self.tensorboard.n_steps
            for idx in range(iters):
                anime_images, anime_smooth_images, real_images = self.get_images()
                # anime_images, anime_smooth_images, real_images = self.anime_dataloader[idx], self.anime_smooth_dataloader[idx], self.real_dataloader[idx]
                
                """
                    Data Loader에 문제 있는지 체크
                """
                # image_path = os.path.join(test_path, str(epoch) + '_' + str(idx) + '_')
                # test_real_image = np.transpose(real_images[0][0], (1,2,0))
                # test_anime_image = np.transpose(anime_images[0][0], (1,2,0))
                # test_anime_gray_image = np.transpose(anime_images[1][0], (1,2,0))
                # test_anime_smooth_image = np.transpose(anime_smooth_images[1][0], (1,2,0))

                # print(test_real_image.shape)

                # cv2.imwrite(image_path + 'real_image.jpg', cv2.cvtColor((test_real_image.numpy()+1)*255, cv2.COLOR_BGR2RGB))
                # cv2.imwrite(image_path + 'anime_image.jpg', cv2.cvtColor((test_anime_image.numpy()+1)*255, cv2.COLOR_BGR2RGB))
                # cv2.imwrite(image_path + 'anime_gray_image.jpg', cv2.cvtColor((test_anime_gray_image.numpy()+1)*255, cv2.COLOR_BGR2RGB))
                # cv2.imwrite(image_path + 'anime_smooth_image.jpg', cv2.cvtColor((test_anime_smooth_image.numpy()+1)*255, cv2.COLOR_BGR2RGB))

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
                    
                    assert type(content_loss.item()) is float, 'con_loss must be float'
                    
                    content_loss.backward()
                    optimizer_G.step()
                    init_mean_loss.append(content_loss.item())
                    
                    self.tensorboard.step_one(init_con_loss=content_loss.item())

                    print("Epoch: %3d Step: %5d / %5d  time: %f s init_con_loss: %.8f  mean_con_loss: %.8f" % (epoch, idx, iters, time.time() - start_time, content_loss, np.mean(init_mean_loss)))
                    if (idx+1)%200 ==0:
                        init_mean_loss.clear()
                else :
                    start_time = time.time()

                    d_loss, real_loss, gray_loss, fake_loss, real_blur_loss = 0.0, 0.0, 0.0, 0.0, 0.0
                    if j == self.training_rate:
                        # Update D
                        optimizer_D.zero_grad()
                        
                        fake = self.generator(real)
                        generated_logit = self.discriminator(fake)
                        anime_logit = self.discriminator(anime)
                        anime_gray_logit = self.discriminator(anime_gray)
                        smooth_logit = self.discriminator(anime_smooth)

                        assert anime_logit.shape[1] == 1 and anime_gray_logit.shape[1] == 1 and generated_logit.shape[1] == 1 and smooth_logit.shape[1] == 1, \
                                'Discriminator logits shape must be (B, 1, ?, ?) , shape : {}'.format(str(list(anime_logit.shape)))
                        
                        sum_loss, real_loss, gray_loss, fake_loss, real_blur_loss = discriminator_loss(self.gan_type, anime_logit, anime_gray_logit, generated_logit, smooth_logit)

                        d_loss = self.d_adv_weight * sum_loss + self.GP
                        # print(d_loss, type(d_loss))
                        assert type(d_loss.item()) is float, 'd_loss must be float'

                        d_loss.backward()
                        optimizer_D.step()

                    # Update G

                    optimizer_G.zero_grad()
                    
                    fake = self.generator(real)
                    generated_logit = self.discriminator(fake)
                    c_loss, s_loss = con_sty_loss(self.vgg, real, anime, fake)
                    tv_loss = self.tv_weight * total_variation_loss(fake)
                    col_loss = color_loss(real,fake)
                    t_loss = self.con_weight * c_loss + self.sty_weight * s_loss + col_loss * self.color_weight + tv_loss
                    g_loss = self.g_adv_weight * generator_loss(self.gan_type, generated_logit)

                    # print(t_loss, g_loss)
                    g_total_loss =  t_loss + g_loss
                    
                    # print(g_total_loss)
                    g_total_loss.backward()
                    optimizer_G.step()
                    
                    mean_loss.append([d_loss.item(), g_loss.item()])

                    self.tensorboard.step_one(main_con_loss=c_loss.item() * self.con_weight, style_loss=s_loss.item() * self.sty_weight, real_d_loss=real_loss.item(), gray_d_loss=gray_loss.item(), \
                                                fake_d_loss=fake_loss.item(), real_blur_d_loss=real_blur_loss.item(), d_loss=d_loss.item(), tv_loss=tv_loss.item() * self.tv_weight, \
                                                color_loss=col_loss.item() * self.color_weight, g_loss = g_loss.item())

                    if j == self.training_rate:
                        print(
                            "Epoch: %3d Step: %5d / %5d  time: %f s d_loss: %.8f, g_loss: %.8f -- mean_d_loss: %.8f, mean_g_loss: %.8f" % (
                                epoch, idx, iters, time.time() - start_time, d_loss, g_loss, np.mean(mean_loss, axis=0)[0],
                                np.mean(mean_loss, axis=0)[1]))
                    else:
                        print(
                            "Epoch: %3d Step: %5d / %5d time: %f s , g_loss: %.8f --  mean_g_loss: %.8f" % (
                                epoch, idx, iters, time.time() - start_time, g_loss, np.mean(mean_loss, axis=0)[1]))

                    if (idx + 1) % 200 == 0:
                        mean_loss.clear()
                    
                    j = j - 1
                    if j < 1:
                        j = self.training_rate

            if (epoch + 1) >= self.init_epoch and np.mod(epoch + 1, self.save_freq) == 0:
                self.save(self.checkpoint_dir, epoch)

            if epoch >= self.init_epoch -1:
                """ Result Image """
                val_files = sorted(glob('./dataset/{}/*.*'.format('val')))
                save_path = './{}/{:03d}/'.format(self.sample_dir, epoch)
                check_folder(save_path)
                
                # self.generator.eval()

                for i, sample_file in enumerate(val_files):
                    print('val '+ str(i) + 'th image : ' +sample_file)
                    sample_image = np.asarray(load_test_data(sample_file, self.img_size))
                    # print('val image : ', sample_image.shape)
                    assert sample_image.shape[1] == 3, 'Sample images shape is not correct'
                    
                    test_real = sample_image
                    test_image = torch.from_numpy(sample_image).cuda()
                    
                    with torch.no_grad():
                        test_generated = self.generator(test_image)
                        test_generated = test_generated.cpu().numpy()
                        
                    save_images(test_real, save_path+'{:03d}_a.jpg'.format(i), None)
                    save_images(test_generated, save_path+'{:03d}_b.jpg'.format(i), None)
                # self.generator.train()
        self.tensorboard.close()


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
            try:
                now_epoch = pth_path.split('.')[0].split('_')[1]
                if epoch < int(now_epoch):
                    epoch = int(now_epoch)
                    if pth_name.split('_')[0] == 'generator':
                        self.generator.load_state_dict(torch.load(pth_path, map_location='cuda'))
                    else:
                        self.discriminator.load_state_dict(torch.load(pth_path, map_location='cuda'))
            except:
                continue
                
        start_epoch = int(epoch)
        self.generator.train()
        self.discriminator.train()

        return start_epoch
