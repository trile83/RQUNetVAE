import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.flatten import Unflatten
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import OrderedDict
from torch.nn import init
from skimage import exposure
from scipy.signal import fftconvolve


def rescale(image): ## function to rescale image for visualization
    map_img =  np.zeros(image.shape)
    for band in range(image.shape[0]):
        p2, p98 = np.percentile(image[band,:,:], (2, 98))
        map_img[band,:,:] = exposure.rescale_intensity(image[band,:,:], in_range=(p2, p98))
    return map_img

# deal with nan
def nan_ptp(a):
    return np.ptp(a[np.isfinite(a)])


## Function to calculate correlation
def ssim(im1, im2, window, k=(0.01, 0.03), l=255):
    """See https://ece.uwaterloo.ca/~z70wang/research/ssim/"""
    # Check if the window is smaller than the images.
    for a, b in zip(window.shape, im1.shape):
        if a > b:
            return None, None
    # Values in k must be positive according to the base implementation.
    for ki in k:
        if ki < 0:
            return None, None

    c1 = (k[0] * l) ** 2
    c2 = (k[1] * l) ** 2
    window = window/np.sum(window)

    mu1 = fftconvolve(im1, window, mode='valid')
    mu2 = fftconvolve(im2, window, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = fftconvolve(im1 * im1, window, mode='valid') - mu1_sq
    sigma2_sq = fftconvolve(im2 * im2, window, mode='valid') - mu2_sq
    sigma12 = fftconvolve(im1 * im2, window, mode='valid') - mu1_mu2

    if c1 > 0 and c2 > 0:
        num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_map = num / den
    else:
        num1 = 2 * mu1_mu2 + c1
        num2 = 2 * sigma12 + c2
        den1 = mu1_sq + mu2_sq + c1
        den2 = sigma1_sq + sigma2_sq + c2
        ssim_map = np.ones(np.shape(mu1))
        index = (den1 * den2) > 0
        ssim_map[index] = (num1[index] * num2[index]) / (den1[index] * den2[index])
        index = (den1 != 0) & (den2 == 0)
        ssim_map[index] = num1[index] / den1[index]

    mssim = ssim_map.mean()
    return mssim, ssim_map


def nrmse(im1, im2):
    a, b = im1.shape
    rmse = np.sqrt(np.sum((im2 - im1) ** 2) / float(a * b))
    max_val = max(np.max(im1), np.max(im2))
    min_val = min(np.min(im1), np.min(im2))
    return 1 - (rmse / (max_val - min_val))

# function for moving correlated frame in the skip connection
def feat_ext(ndvi, f):
    f = f.detach().cpu().numpy()
    f_new = f.reshape((f.shape[1],f.shape[2],f.shape[3]))
    h = f_new.shape[1]
    w = f_new.shape[2]
    ndvi_h, ndvi_w = ndvi.shape
    bin_size = ndvi_h // h
    ndvi_res = ndvi.reshape((h, bin_size,
                                h, bin_size, 1)).max(3).max(1)
    ndvi_res = ndvi_res.reshape((ndvi_res.shape[0],ndvi_res.shape[1]))
    #print('ndvi_res shape: ', ndvi_res.shape)

    #print('max ndvi: ', np.max(ndvi))

    index_ndvi = np.ma.where(ndvi_res < 0.1, 1, 0)

    #b = np.zeros((f.shape[1],f.shape[2]))
    b = np.zeros((f.shape))
    for i in range(f_new.shape[0]):
        feature = f_new[i,:,:].reshape((f_new.shape[1],f_new.shape[2]))
        h,w = feature.shape
        feature = 2.*(feature - np.min(feature))/nan_ptp(feature)-1
        
        nrmse_val = nrmse(ndvi_res, feature)
        #nrmse_val = nrmse(index_ndvi, feature)
        # ssim_val, ssim_map = ssim(ndvi_res, feature, window=np.zeros((11,11)), k=(0.01, 0.03), l=255)
        # print(i)
        # print('nrmse_val: ', nrmse_val)
        # print('ssim_val: ', ssim_val)

        if np.isnan(nrmse_val) or nrmse_val < 0.70001:
            b[:,i,:,:] = np.zeros((feature.shape))
            #b[:,i,:,:] = feature
        elif nrmse_val > 0.70:
            b[:,i,:,:] = feature + index_ndvi
            #b[:,i,:,:] = (feature + ndvi_res) / 2
            #b[:,i,:,:] = np.zeros((feature.shape))

        #b[:,i,:,:] = index_ndvi

    tensor = torch.tensor(b, dtype = torch.float16).cuda()

    return tensor


# Create Unet VAE 
# 3x3 convolution module for each block
def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

# decoding (up) convolution module for decoder block
def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

def conv_out(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

# Encoder Block for UNet
class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, segment = True, pooling=True, batchnorm=True, dropout=False):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.segment = segment
        self.pooling = pooling
        self.dropout = dropout
        self.batchnorm = batchnorm
        
        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.batchnormalize = nn.BatchNorm2d(out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.segment and self.batchnorm:
            x = self.batchnormalize(x) # better for segmentation
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        if self.dropout:
            x = self.drop(x)

        return x, before_pool

## Decoder Block for Unet
class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, segment=True,
                 merge_mode='concat', up_mode='bilinear'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.segment = segment
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.batchnorm = nn.BatchNorm2d(out_channels)

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
            mode=self.up_mode)

        # skip connection from decoder to encoder
        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)


    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.segment:
            from_up = self.batchnorm(from_up) # better for segmentation
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet_VAE_feat_ext(nn.Module):
    def __init__(self, num_classes, segment, ndvi, in_channels=3, depth=5, 
                 start_filts=64, up_mode='upsample', 
                 merge_mode='concat', enc_out_dim=1024, latent_dim=100):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet_VAE_feat_ext, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.segment = segment
        self.ndvi = ndvi
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        #self.tau = nn.Parameter(torch.rand(1))
        self.tau = torch.tensor((0.0))

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False
            batchnorm = True if i < depth-1 else False
            if self.segment and i > (depth-3):
                dropout = False
            else:
                dropout = False

            down_conv = DownConv(ins, outs, segment=self.segment, pooling=pooling, batchnorm=batchnorm, dropout=dropout)
            self.down_convs.append(down_conv)

        #Flatten
        self.flatten = nn.Flatten()

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks

        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, segment=self.segment, up_mode=up_mode,
                merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        # the dimension before flatten is 1024 x 16 x 16 = 262144
        self.fc1 = nn.Linear(enc_out_dim * 16 * 16, latent_dim)
        self.fc2 = nn.Linear(enc_out_dim * 16 * 16, latent_dim)
        self.fc3 = nn.Linear(latent_dim, enc_out_dim * 16 * 16)
        self.act = nn.ReLU()

        #self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)
        
    def reparameterize(self, mu, logvar): # similar to sampling class in Keras code
        std = logvar.mul(0.5).exp_()
        std = std.cuda()
        eps = torch.normal(mu, std)
        eps = eps.cuda()
        z = mu + std * eps
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
         
        # encoder pathway, save outputs for merging
        # for i, module in enumerate(self.down_convs):
        #     x, before_pool = module(x)
        #     encoder_outs.append(before_pool)

        #self.tau += 1
        #print('tau: ', self.tau)

        #print("x shape: ", x.shape)

        s_dict = {}  
        for i, module in enumerate(self.down_convs):
            x, s = module(x)
            s_dict[i] = s   

        x_encoded = self.flatten(x)

        # calculate z_mean, z_log_var
        # z, mu, logvar = self.bottleneck(x_encoded)
        # z = self.act(self.fc3(z))
        # z = torch.reshape(z, x.shape)

        # Step 3 - Riesz-Quincunx truncation for skip-connecting signals (alpha):
        s_smooth_dict = {} # new list for shrinkage tensors

        ## smoothing operations
        for i in range(len(s_dict)):
            f = s_dict[i]
        
            f_re = torch.zeros(f.shape)

            f_re = feat_ext(self.ndvi, f)

            f_re = f_re.cuda()

            #f_re = f
        
            s_smooth_dict[i] = f_re

        # decoder pathway
        # for i, module in enumerate(self.up_convs):
        #     before_pool = encoder_outs[-(i+2)]
        #     if i == 0:
        #         x = module(before_pool, z)
        #     else:
        #         x = module(before_pool, x)

        for i, module in enumerate(self.up_convs):
            s = s_smooth_dict[5-2-i]
            if i == 0:
                x = module(s, x)
            else:
                x = module(s, x)

        x = self.conv_final(x)
        x_recon = F.relu(x)
        #kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        #return x, mu, logvar, x_recon, kl_loss
        return x, x_recon