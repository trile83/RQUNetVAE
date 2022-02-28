"""
Simple UNet demo
@author: ptrblck
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
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
        #self.height = 256
        #self.width = 256
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
        
        # Step 2. 
        f = x.cpu().detach().numpy()

        height = f.shape[2]
        width = f.shape[3]

        beta_I, beta_D_I, psi_i, psi_D_i = BsplineQuincunxScalingWaveletFuncs(height, width, self.scale, self.gamma)

        # Riesz Quincunx wavelet:
        N = 3
        psi_in, psi_D_in = RieszQuincunxWaveletFuncs(N, psi_i, psi_D_i)


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


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 stride, alpha = 0.0, shrink=False):
        super(BaseConv, self).__init__()

        self.act = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding,
                               stride)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               padding, stride)

        self.shrinkRQ = RieszQuincunx(alpha)
        self.shrink = shrink

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        if self.shrink:
            x = self.shrinkRQ(x)
        return x


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 stride):
        super(DownConv, self).__init__()

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_block = BaseConv(in_channels, out_channels, kernel_size,
                                   padding, stride)

    def forward(self, x):
        x = self.pool1(x)
        x = self.conv_block(x)
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


class UNet_VAE_RQ_All(nn.Module):
    def __init__(self, in_channels, out_channels, num_class, kernel_size,
                 padding, stride, alpha, latent_dim = 64):
        super(UNet_VAE_RQ_All, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_class
        self.alpha = alpha

        #self.shrinkdown = False

        self.init_conv = BaseConv(in_channels, out_channels, kernel_size,
                                  padding, stride, alpha=self.alpha)

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