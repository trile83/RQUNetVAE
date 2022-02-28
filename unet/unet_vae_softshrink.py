import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.flatten import Unflatten
import torchvision
from PIL import Image
import numpy as np
from collections import OrderedDict
from torch.nn import init

# Custom activation function
class softShrink(nn.Module):
    def __init__(self, alpha):
        super(softShrink, self).__init__()

        self.alpha = alpha

    def forward(self, x):
        x = x.detach().cpu().numpy()
        if x.all() == 0:
            print("tensor equal zero")
            y = x
        else:
            y = (x/(np.abs(x)+0.000000000000001))*np.maximum(np.abs(x)-self.alpha, 0)

        y = torch.from_numpy(y).float().cuda()

        return y

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
def upconv2x2(in_channels, out_channels, idx, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        if idx == 0:
            return nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                conv1x1(in_channels, out_channels)
            )
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
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=1,groups=groups,stride=1),
        nn.ReLU()
    )


# Encoder Block for UNet
class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, alpha, pooling=True, dropout=False, shrink = False):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.dropout = dropout
        self.shrink = shrink

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.batchnorm = nn.BatchNorm2d(out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if self.dropout:
            self.drop = nn.Dropout(0.5)

        if self.shrink:
            self.s_shrink = softShrink(alpha)
            #self.s_shrink = nn.Softshrink(0.0)
            #self.s_shrink = RieszQuincunx()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.batchnorm(x)
        if self.shrink:
            x = self.s_shrink(x)
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
    def __init__(self, in_channels, out_channels, idx,
                 merge_mode='concat', up_mode='bilinear'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.batchnorm = nn.BatchNorm2d(out_channels)
        #self.unflatten = UnFlatten()
        self.idx = idx

        self.upconv = upconv2x2(self.in_channels, self.out_channels, self.idx,
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
        from_up = self.batchnorm(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet_VAE_Softshrink(nn.Module):
    def __init__(self, num_classes, alpha=0.0, in_channels=3, depth=5, 
                 start_filts=64, up_mode='upsample', 
                 merge_mode='concat', enc_out_dim=1024, latent_dim=64):
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
        super(UNet_VAE_Softshrink, self).__init__()

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
        self.alpha = alpha
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False 
            dropout = True if i > (depth-3) else False
            #dropout = False
            shrink = True if i == 0 else False
            #shrink = False

            down_conv = DownConv(ins, outs, self.alpha, pooling=pooling, dropout=dropout, shrink=shrink)
            self.down_convs.append(down_conv)


        #Flatten
        self.flatten = nn.Flatten()

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks

        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, i, up_mode=up_mode,
                merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)
        #self.conv_final = conv_out(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        # the dimension before flatten is 1024 x 16 x 16 = 262144
        self.fc1 = nn.Linear(262144, latent_dim)
        self.fc2 = nn.Linear(262144, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 262144)
        self.act = nn.ReLU()

        self.reset_params()

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
        #eps = torch.randn(*mu.size())
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
        #h = self.encoder(x)
        encoder_outs = []
         
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        #print("x shape: ", x.shape)
        #print(type(x))

        x_encoded = self.flatten(x)

        if  x.detach().cpu().numpy().all() == 0:
            print("tensor equal zero")
            mu = 0
            logvar = 0
            z = x
        else:
            # calculate z_mean, z_log_var
            z, mu, logvar = self.bottleneck(x_encoded)
            z = self.act(self.fc3(z))
            #print("z shape: ", z.shape)
            z = torch.reshape(z, x.shape)

        #print(self.down_convs)
        #print(self.up_convs)

        # decoder pathway
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            if i == 0:
                x = module(before_pool, z)
            else:
                x = module(before_pool, x)

        x = self.conv_final(x)
        #x = F.relu(self.conv_final(x))

        #kl_loss = -0.5 * (1 + logvar - torch.square(mu) - torch.exp(logvar))
        #kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        #return x, mu, logvar, kl_loss
        return x