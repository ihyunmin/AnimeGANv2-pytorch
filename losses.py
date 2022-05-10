import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def con_loss(vgg, real, fake):

    # vgg.build(real)
    # real_feature_map = vgg.conv4_4_no_activation

    # vgg.build(fake)
    # fake_feature_map = vgg.conv4_4_no_activation

    real_feature_map = vgg.extract_features(real)
    fake_feature_map = vgg.extract_features(fake)
    loss = L1_loss(real_feature_map, fake_feature_map)

    return loss

def con_sty_loss(vgg, real, anime, fake):

    real_feature_map = vgg.extract_features(real)
    fake_feature_map = vgg.extract_features(fake)
    anime_feature_map = vgg.extract_features(anime[:fake_feature_map.shape[0]])

    c_loss = L1_loss(real_feature_map, fake_feature_map)
    s_loss = style_loss(anime_feature_map, fake_feature_map)

    return c_loss, s_loss

def style_loss(style, fake):
    return L1_loss(gram(style), gram(fake))

def L1_loss(real, fake):
    L1loss = nn.L1Loss()
    loss = L1loss(real, fake)
    return loss

def gram(x):
    # n, c, h, w : Batch size, number of feature maps, dimensions of feature maps
    n, c, h, w = x.shape
    features = x.view(n*c, h*w)
    G = torch.mm(features, features.t())
    return G.div(n*c*h*w)

    # shape_x = tf.shape(x)
    # b = shape_x[0]
    # c = shape_x[3]
    # x = tf.reshape(x, [b, -1, c])
    # return tf.matmul(tf.transpose(x, [0, 2, 1]), x) / tf.cast((tf.size(x) // b), tf.float32)

def total_variation_loss(inputs):
    """
    -> tv loss
    A smooth loss in fact. Like the smooth prior in MRF.
    V(y) = || y_{n+1} - y_n ||_2
    """
    loss = nn.MSELoss()
    l2loss_dh = loss(inputs[:, :, :-1, :], inputs[:, :, 1:, :])
    l2loss_dw = loss(inputs[:, :, :, :-1], inputs[:, :, :, 1:])
    size_dh = inputs.shape[2]
    size_dw = inputs.shape[3]

    return l2loss_dh / size_dh + l2loss_dw / size_dw

def color_loss(real, fake):
    real = rgb_to_yuv(real)
    fake = rgb_to_yuv(fake)

    return  L1_loss(real[:,:,:,0], fake[:,:,:,0]) + Huber_loss(real[:,:,:,1],fake[:,:,:,1]) + Huber_loss(real[:,:,:,2],fake[:,:,:,2])

def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YUV.

    .. image:: _static/img/rgb_to_yuv.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        YUV version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_yuv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    u: torch.Tensor = -0.147 * r - 0.289 * g + 0.436 * b
    v: torch.Tensor = 0.615 * r - 0.515 * g - 0.100 * b

    out: torch.Tensor = torch.stack([y, u, v], -3)

    return out


def Huber_loss(x,y):
    huber_loss = nn.HuberLoss()
    loss = huber_loss(x,y)
    return loss


def generator_loss(loss_func, fake):

    fake_loss = 0
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    valid = Variable(Tensor(fake.shape).fill_(1.0), requires_grad=False)

    if loss_func == 'lsgan' :
        mseloss = nn.MSELoss()
        fake_loss = mseloss(fake, valid)

    if loss_func == 'gan' or loss_func == 'dragan':
        bceloss = nn.BCELoss()
        fake_loss = bceloss(torch.sigmoid(fake), valid)
    
    loss = fake_loss

    return loss

def discriminator_loss(loss_func, real, gray, generated, real_blur):
    real_loss = 0
    gray_loss = 0
    fake_loss = 0
    real_blur_loss = 0

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    valid = Variable(Tensor(real.shape).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(real.shape).fill_(0.0), requires_grad=False)
    
    if loss_func == 'lsgan' :
        mseloss = nn.MSELoss()
        real_loss = mseloss(real, valid)
        gray_loss = mseloss(gray, fake)
        fake_loss = mseloss(generated, fake)
        real_blur_loss = mseloss(real_blur, fake)

    if loss_func == 'gan' or loss_func == 'dragan' :
        # Be careful about the sigmoid function.
        bceloss = nn.BCELoss()
        real_loss = bceloss(torch.sigmoid(real), valid)
        gray_loss = bceloss(torch.sigmoid(gray), fake)
        fake_loss = bceloss(torch.sigmoid(generated), fake)
        real_blur_loss = bceloss(torch.sigmoid(real_blur), fake)

    # for Hayao : 1.2, 1.2, 1.2, 0.8
    # for Paprika : 1.0, 1.0, 1.0, 0.005
    # for Shinkai: 1.7, 1.7, 1.7, 1.0

    loss = 1.7 * real_loss +  1.7 * fake_loss + 1.7 * gray_loss  +  1.0 * real_blur_loss

    return loss