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

import math as m
# pi = tf.constant(m.pi)   
pi = m.pi
eps = np.finfo(float).eps
import cmath as cm   
# j = tf.constant(cm.sqrt(-1))
j = cm.sqrt(-1)

#alpha_global = 0.5

##################################
# math functions
# Func - Isotropic polyharmonic Bspline:
def IPBspline(omega1, omega2, gamma):   
    z1 = np.exp(j*omega1)
    z2 = np.exp(j*omega2)
    
    z1_1 = z1**(-1)     # z1_1 = 1/(z1.astype(np.complex128))
    z2_1 = z2**(-1)     # z2_1 = 1/(z2.astype(np.complex128))
    
    # Isotropic Bspline:
    Lp_z = 4 - z1 - z1_1 - z2 - z2_1    
    Lm_z = 0.5 * ( 4 - z1*z2 - z1_1*z2 - z1*z2_1 - z1_1*z2_1 )
    
    V2_z = 2/3*Lp_z + 1/3*Lm_z
    beta_omega = (eps + V2_z)**(gamma/2) / (eps + omega1**2 + omega2**2)**(gamma/2)
    
    return beta_omega


# Func - Scaling autocorrelation func:
def AutocorrelationFunc(omega1, omega2, gamma):    
    Height, Width = omega1.shape
    A = np.zeros((Height, Width))    
    for m1 in range(-5, 6, 1):  # -5, 6, 1
        for m2 in range(-5, 6, 1):
            A = A + IPBspline(2*pi*m1 + omega1, 2*pi*m2 + omega2, 2*gamma)
        
    return A


# Func - Scaling autocorrelation func at scale D:
def AutocorrelationFunc_scaleD(omega1, omega2, gamma):   
    A_D = 0.5 * ( np.abs(Lowpass(omega1, omega2, gamma))**2 * AutocorrelationFunc(omega1, omega2, gamma) + np.abs(Lowpass(omega1+pi, omega2+pi, gamma))**2 * AutocorrelationFunc(omega1+pi, omega2+pi, gamma));
    return A_D


# Func - Primal lowpass filter:
def ScalingFunc_dual(omega1, omega2, gamma):   
    beta_D =IPBspline(omega1, omega2, gamma) / AutocorrelationFunc(omega1, omega2, gamma)
    return beta_D


# Func - Primal lowpass filter:
def Lowpass(omega1, omega2, gamma):   
    H = np.sqrt(2) * IPBspline(omega1 + omega2, omega1 - omega2, gamma) / IPBspline(omega1, omega2, gamma)
    return H


# Func - Primal Highpass filter:
def Highpass_primal(omega1, omega2, gamma):   
    G = - np.exp(-j*omega1) * Lowpass( -(omega1+pi), -(omega2+pi), gamma) * AutocorrelationFunc(omega1+pi, omega2+pi, gamma)
    return G


# Func - Primal lowpass filter:
def Highpass_dual(omega1, omega2, gamma):   
    G_D = - np.exp(-j*omega1) * Lowpass( -(omega1+pi), -(omega2+pi), gamma) / AutocorrelationFunc_scaleD(omega1, omega2, gamma);
    return G_D

def BsplineQuincunxScalingWaveletFuncs(Height, Width, Scales, gamma):
    # Scales = 3   # 0, 1, 2, 3  definition: 
    # Height = 256 # 128
    # Width = 256  # 128
    # gamma = 5 # 1.2
    #   
    psi_i = np.zeros((Height, Width, Scales+1), dtype=complex)    
    psi_D_i = np.zeros((Height, Width, Scales+1), dtype=complex)    
    
    #
    #for i in range(0, Scales+1):
    for i in range(0, Scales):
        # print(i)
        # i = 1
    
        # Fourier coordinate:
        omega2, omega1 = np.meshgrid(np.linspace(-pi, pi, Width), np.linspace(-pi, pi, Height))        
    
        if np.mod(i,2) == 0:
            omega1 = 2**(i/2)*omega1;
            omega2 = 2**(i/2)*omega2; 
        else:
            omega1_temp = 2**((i-1)/2)*(omega1 + omega2)
            omega2_temp = 2**((i-1)/2)*(omega1 - omega2)
            
            omega1 = omega1_temp
            omega2 = omega2_temp 
            
        # Primal/Dual wavelet funcs:
        psi_i[:,:,i] = 1/np.sqrt(2) * Highpass_primal( 0.5*(omega1+omega2), 0.5*(omega1-omega2), gamma) * IPBspline( 0.5*(omega1+omega2), 0.5*(omega1-omega2), gamma)
        psi_D_i[:,:,i] = 1/np.sqrt(2) * Highpass_dual( 0.5*(omega1+omega2), 0.5*(omega1-omega2), gamma) * ScalingFunc_dual( 0.5*(omega1+omega2), 0.5*(omega1-omega2), gamma)
    
    # Fourier coordinate:
    omega2, omega1 = np.meshgrid(np.linspace(-pi, pi, Width), np.linspace(-pi, pi, Height))        
    
    i = Scales
    
    if np.mod(i,2) == 0:
        omega1 = 2**(i/2)*omega1;
        omega2 = 2**(i/2)*omega2; 
    else:
        omega1_temp = 2**((i-1)/2)*(omega1 + omega2);
        omega2_temp = 2**((i-1)/2)*(omega1 - omega2);
            
    omega1 = omega1_temp;
    omega2 = omega2_temp;    
        
    # Primal scaling funcs:
    beta_I = IPBspline(omega1, omega2, gamma)    
    beta_D_I = ScalingFunc_dual(omega1, omega2, gamma);
    
    # Correction of the last wavelet subband:
    Matrix_one = np.ones((Height, Width), dtype=complex)    
    
    scalingFunc = np.conj(beta_D_I) * beta_I
    # check Identity:
    waveletFunc = np.zeros((Height, Width), dtype=complex)
    for i in range(1, Scales+1):
        waveletFunc = waveletFunc + np.conj(psi_D_i[:,:,i])*psi_i[:,:,i]
        
    psi_i[:,:,0] = ( Matrix_one - scalingFunc - waveletFunc ) / ( np.conj(psi_D_i[:,:,0])  ) 

    return beta_I, beta_D_I, psi_i, psi_D_i


# 1.
def QuincunxWaveletTransform_Forward(f, beta_D_I, psi_D_i):
    # alpha = 0.1   # 0.5 
    # activation_method = "SoftShrink"
    
    from numpy.fft import fft2, ifft2, fftshift, ifftshift
    
    Height, Width, Scales1 = psi_D_i.shape
    Scales = Scales1 - 1
    
    F = fftshift(fft2(f))
    
    # Scaling coefficients:
    c_I = np.real( ifft2( ifftshift( F * np.conj(beta_D_I) ) ) )
        
    # Wavelet coefficients:
    d_i = np.zeros((Height, Width, Scales+1))      
    for i in range(0, Scales+1):
        d_i[:,:,i] = np.real( ifft2( ifftshift( F * np.conj(psi_D_i[:,:,i]) ) ) )
    
    return c_I, d_i


# 2.
def QuincunxWaveletTransform_Inverse(c_I, d_i, beta_I, psi_i):
    
    from numpy.fft import fft2, ifft2, fftshift, ifftshift
    
    Height, Width, Scales1 = psi_i.shape
    Scales = Scales1 - 1
    
    
    F_re_scaling = fftshift(fft2(c_I)) * beta_I    

    F_re_wavelet = np.zeros((Height, Width), dtype=complex)
    for i in range(0, Scales+1):
        F_re_wavelet = F_re_wavelet + fftshift(fft2(d_i[:,:,i])) * psi_i[:,:,i]

    F_re = F_re_scaling + F_re_wavelet   
    
    #
    f_re = np.real( ifft2( ifftshift( F_re ) ) )
    
    return f_re


# 3.
def WaveletTruncation(d_i, alpha, activation_method):
    Scales = 3
    # alpha = 0.1   # 0.5 
    # activation_method = "SoftShrink"

    # truncation:
    for i in range(0, Scales+1):
        thres = alpha*np.max(d_i[:,:,i])
        d_i[:,:,i] = ActivationFuncs(activation_method, d_i[:,:,i], thres)
    
    return d_i

def ActivationFuncs(method, x, t):

    if method=="HardShrink":
        y = x*(np.abs(x) > t)
    elif method=="SoftShrink":
        y = x/(np.abs(x)+0.000000000000001)*np.maximum(np.abs(x)-t, 0)
    elif method=="Identity":
        y = x      
    elif method=="BinaryStep":
        y = 0*(x<0) + 1*(x>=0)
    elif method=="Sigmoid":
        y = 1/(1 + np.exp(-x))            
    elif method=="Tanh":
        y = ( np.exp(x) - np.exp(-x) )/( np.exp(x) + np.exp(-x) )            
    elif method=="ReLU":
        y = np.maximum(x,0)    
    elif method=="ReLU_min":
        y = np.minimum(x,0)          
    elif method=="LeakyReLU":
        alpha = 0.01
        y = alpha*x*(x<0) + x*(x>=0)                  
    elif method=="Softplus":
        y = np.log(1 + np.exp(x))            
    elif method=="SELU":
        lamda = 1  # 1.0507
        alpha = 1.67326
        y = lamda*( alpha*(np.exp(x) - 1)*(x<0) + x*(x>=0) )      
    elif method=="SiLU":
        y = x/(1 + np.exp(-x))              
    elif method=="Mish":
        y_1 = np.log(1 + np.exp(x))
        y_2 = ( np.exp(y_1) - np.exp(-y_1) )/( np.exp(y_1) + np.exp(-y_1) )
        y = x*y_2            
    elif method=="Gaussian":
        y = np.exp(-x**2)            
    elif method=="GCU":
        y = x*np.cos(x)    
    elif method=="DoublePareto":
        a = 0.1
        gamma = 0.1
        y = 0.5*x/np.abs(x)*( np.abs(x) - a + np.sqrt( (a - np.abs(x))**2 + 4*np.maximum(a*np.abs(x) - gamma, 0) ) )

    # else:
    #     print("Invalid choice of activation functions.")
    #     sys.exit()    
    
    return y

# RIESZ QUINCUNX: ========================
 
# Riesz Quincunx Wavelet Funcs:  
def RieszQuincunxWaveletFuncs(N, psi_i, psi_D_i):
    # N = 3
    # beta_I, beta_D_I, psi_i, psi_D_i = BsplineQuincunxScalingWaveletFuncs(Height, Width, Scales, gamma)
    
    Height, Width, Scales1 = psi_i.shape
    Scales = Scales1 - 1
    
    #
    omega2, omega1 = np.meshgrid(np.linspace(-pi, pi, Width), np.linspace(-pi, pi, Height))        
    
    # Spectrum of Riesz opt:
    Rn_omega = np.zeros((Height, Width, N+1), dtype=complex)
    for n in range(0, N+1):
        Rn_omega[:,:,n] = (-j)**N * m.sqrt( m.factorial(N) / m.factorial(n) / m.factorial(N - n) ) * omega1**n * omega2**(N-n) / ( omega1**2 + omega2**2 )**(N/2)
    
    # N-th order Riesz Quincunx wavelet:
    psi_in = np.zeros((Height, Width, Scales+1, N+1), dtype=complex)
    psi_D_in = np.zeros((Height, Width, Scales+1, N+1), dtype=complex)
    
    #
    for i in range(0, Scales+1):
        for n in range(0, N+1):
            psi_D_in[:,:,i,n] = Rn_omega[:,:,n] * psi_D_i[:,:,i]
            psi_in[:,:,i,n] = Rn_omega[:,:,n] * psi_i[:,:,i]    
    
    return psi_in, psi_D_in

# 4.
def RieszQuincunxWaveletTransform_Forward(f, beta_D_I, psi_D_in):
    # alpha = 0.1   # 0.5 
    # activation_method = "SoftShrink"
    
    from numpy.fft import fft2, ifft2, fftshift, ifftshift
    
    Height, Width, Scales1, N1 = psi_D_in.shape
    Scales = Scales1 - 1
    N = N1 - 1

    #print(psi_D_in.shape)
    #print("beta_D_I shape: ",beta_D_I.shape)

    F = fftshift(fft2(f))
    
    # Scaling coefficients:
    c_I = np.real( ifft2( ifftshift( F * np.conj(beta_D_I) ) ) )
    #print("c_I shape: ", c_I.shape)
        
    # Wavelet coefficients:
    d_in = np.zeros((Height, Width, Scales+1, N+1))
    #d_in = np.zeros((Height, Width, 64, 1))
    #print("shape d_in: ", d_in.shape)


    for i in range(0, N+1):
        for n in range(0, Scales+1):
            d_in[:,:,i,n] = np.real( ifft2( ifftshift( F * np.conj(psi_D_in[:,:,i,n]) ) ) )
    
    return c_I, d_in

# 5.
def RieszQuincunxWaveletTransform_Inverse(c_I, d_in, beta_I, psi_in):
    
    from numpy.fft import fft2, ifft2, fftshift, ifftshift
    
    Height, Width, Scales1, N1 = psi_in.shape
    Scales = Scales1 - 1
    N = N1 - 1
    
    F_re_scaling = fftshift(fft2(c_I)) * beta_I    

    F_re_wavelet = np.zeros((Height, Width), dtype=complex)

    for i in range(0, Scales+1):
        for n in range(0, N+1):
            F_re_wavelet = F_re_wavelet + fftshift(fft2(d_in[:,:,i,n])) * psi_in[:,:,i,n]
            #F_re_wavelet = fftshift(fft2(d_in[:,:,i,n])) * psi_in[:,:,i,n]

    # using both scaling coefficient and wavelet coefficient
    F_re = F_re_scaling + F_re_wavelet
    #
    f_re = np.real( ifft2( ifftshift( F_re ) ) )
    
    return f_re

# 6.
def RieszWaveletTruncation(d_in, alpha, activation_method):
    # alpha = 0.1   # 0.5 
    # activation_method = "SoftShrink"

    Height, Width, Scales1, N1 = d_in.shape
    Scales = Scales1 - 1
    N = N1 - 1    

    # truncation:
    #d_in_shrink = np.zeros((Height, Width, Scales+1, N+1))

    for i in range(0, Scales+1):
        for n in range(0, N+1):
            thres = alpha*np.max(d_in[:,:,i,n])
            #thres = alpha
            #d_in_shrink[:,:,i,n] = ActivationFuncs(activation_method, d_in[:,:,i,n], thres)
            d_in[:,:,i,n] = ActivationFuncs(activation_method, d_in[:,:,i,n], thres)

    return d_in

################################################
# Riesz-Quincunx

class RieszQuincunx(nn.Module):

    def __init__(self, alpha):
        super().__init__()

        self.scale = 3
        self.height = 256
        self.width = 256
        self.gamma = 1.2
        self.alpha = alpha

    def forward(self, x):

        # Step 1. Riesz Quincunx wavelet scaling funcs:
        # Output: beta_I, beta_D_I, psi_in, psi_D_in
            
        # Quincunx wavelet:    
        #Scales = 3      # 0, 1, 2, 3
        #Height = 256    # 128
        #Width = 256     # 128
        #gamma = 1.2       # 1.2, 5
        beta_I, beta_D_I, psi_i, psi_D_i = BsplineQuincunxScalingWaveletFuncs(self.height, self.width, self.scale, self.gamma)

        # Riesz Quincunx wavelet:
        N = 3
        psi_in, psi_D_in = RieszQuincunxWaveletFuncs(N, psi_i, psi_D_i)

        # Step 2. 
        f = x.cpu().detach().numpy()

        #f = np.reshape(f, (256,256,64,1))
        f_re = np.zeros(f.shape)

        #print("f_re shape: ", f_re.shape)

        # Case 2: Riesz Quincunx wavelet:
        # Forward wavelet:
        for j in range(f_re.shape[0]):
            for i in range(f_re.shape[1]):
                c_I, d_in = RieszQuincunxWaveletTransform_Forward(f[j,i,:,:], beta_D_I, psi_D_in)
                # Shrinkage:
                alpha = self.alpha
                activation_method = "SoftShrink"
                d_in = RieszWaveletTruncation(d_in, alpha, activation_method)

                # Inverse wavelet: 
                f_re[j,i,:,:] = RieszQuincunxWaveletTransform_Inverse(c_I, d_in, beta_I, psi_in)

                #f_re = np.reshape(f_re, (1,64,256,256))

        f_re = torch.from_numpy(f_re).float().cuda()

        return f_re

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
            #self.s_shrink = softShrink(alpha)
            #self.s_shrink = nn.Softshrink(0.0)
            self.s_shrink = RieszQuincunx(alpha)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #x = self.batchnorm(x)
        #if self.shrink:
            #x = self.s_shrink(x)
        before_pool = x
    
        if self.shrink:
            before_pool = self.s_shrink(before_pool)

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
        #rom_up = self.batchnorm(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet_VAE_RQ_old(nn.Module):
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
        super(UNet_VAE_RQ_old, self).__init__()

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

        # calculate z, mu, and logvar
        z, mu, logvar = self.bottleneck(x_encoded)
        z = self.act(self.fc3(z))
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

        x_recon = F.relu(x)

        #kl_loss = -0.5 * (1 + logvar - torch.square(mu) - torch.exp(logvar))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return x, mu, logvar, x_recon, kl_loss