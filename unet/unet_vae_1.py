"""
Simple UNet demo
@author: ptrblck
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 stride):
        super(BaseConv, self).__init__()

        self.act = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding,
                               stride)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               padding, stride)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 stride):
        super(DownConv, self).__init__()

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_block = BaseConv(in_channels, out_channels, kernel_size,
                                   padding, stride)

    def forward(self, x):
        #x = self.pool1(x)
        x = self.conv_block(x)
        x = self.pool1(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, in_channels_skip, out_channels,
                 kernel_size, padding, stride):
        super(UpConv, self).__init__()

        self.conv_trans1 = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, padding=0, stride=2)

        self.upconv = nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            nn.Conv2d(in_channels,in_channels,kernel_size=1,groups=1,stride=1)
            )

        self.conv_block = BaseConv(
            in_channels=in_channels + in_channels_skip,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride)

    def forward(self, x, x_skip):
        #x = self.conv_trans1(x)
        x = self.upconv(x)
        x = torch.cat((x, x_skip), dim=1)
        x = self.conv_block(x)
        return x


class UNet_VAE(nn.Module):
    def __init__(self, in_channels, out_channels, num_class, kernel_size,
                 padding, stride, latent_dim = 64):
        super(UNet_VAE, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_class

        self.init_conv = BaseConv(in_channels, out_channels, kernel_size,
                                  padding, stride)

        self.down1 = DownConv(out_channels, 2 * out_channels, kernel_size,
                              padding, stride)

        self.down2 = DownConv(2 * out_channels, 4 * out_channels, kernel_size,
                              padding, stride)

        self.down3 = DownConv(4 * out_channels, 8 * out_channels, kernel_size,
                              padding, stride)

        self.down4 = DownConv(8 * out_channels, 16 * out_channels, kernel_size,
                              padding, stride)

        #Flatten
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(262144, latent_dim)
        self.fc2 = nn.Linear(262144, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 262144)
        self.act = nn.ReLU()

        self.up4 = UpConv(16 * out_channels, 8 * out_channels, 8 * out_channels,
                          kernel_size, padding, stride)

        self.up3 = UpConv(8 * out_channels, 4 * out_channels, 4 * out_channels,
                          kernel_size, padding, stride)

        self.up2 = UpConv(4 * out_channels, 2 * out_channels, 2 * out_channels,
                          kernel_size, padding, stride)

        self.up1 = UpConv(2 * out_channels, out_channels, out_channels,
                          kernel_size, padding, stride)

        self.out = nn.Conv2d(out_channels, num_class, kernel_size, padding, stride)
    
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
        # Encoder
        x = self.init_conv(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        # Decoder

        # Bottom layer process
        x_encoded = self.flatten(x4)

         # calculate z_mean, z_log_var
        z, mu, logvar = self.bottleneck(x_encoded)
        z = self.act(self.fc3(z))
        z = torch.reshape(z, x4.shape)

        x_up = self.up4(z, x3)
        x_up = self.up3(x_up, x2)
        x_up = self.up2(x_up, x1)
        x_up = self.up1(x_up, x)
        
        #x_out = F.log_softmax(self.out(x_up), 1) # using log_softmax for nn.NLLLoss()
        x_out = self.out(x_up) # using raw logits to use nn.CrossEntropyLoss()

        # get reconstruction image
        x_recon = F.relu(x_out)

        # calculate kl loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return x_out, mu, logvar, x_recon, kl_loss