import argparse
import logging
import os
from skimage import exposure
import numpy as np
import torch
from torchvision import transforms
from osgeo import gdal, gdal_array
import matplotlib.pyplot as plt
#from scipy import signal
import numpy
import cv2
from PIL import Image
from scipy.signal import fftconvolve
# import earthpy as et
# import earthpy.spatial as es

from unet import UNet_VAE
from unet import UNet_VAE_old, UNet_VAE_RQ_old, UNet_VAE_RQ_test, UNet_VAE_RQ_old_torch
from unet import UNet_VAE_RQ_new_torch, UNet_VAE_RQ_scheme3
from unet import UNet_VAE_RQ_scheme1, UNet_VAE_RQ_scheme2
from unet import UNet_test
from utils.utils import plot_img_and_mask, plot_img_and_mask_3, plot_img_and_mask_recon

#image_path = '/home/geoint/tri/github_files/test_img/number13458.TIF'
#mask_true_path = '/home/geoint/tri/github_files/test_label/number13458.TIF'
#image_path = '/home/geoint/tri/github_files/sentinel2_im/2016105_0.tif'
#mask_true_path = '/home/geoint/tri/github_files/sentinel2_im/2016105_0.tif'

#npy_path = '/home/geoint/tri/github_files/input_senegal/Tappan01_WV02_20110430_M1BS_103001000A27E100_data_568.npy'
file_path = '/home/geoint/tri/nasa_senegal/cassemance/Tappan02_WV02_20120218_M1BS_103001001077BE00_data.tif'

use_cuda = True
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#im_type = image_path[30:38]
im_type='senegal'
segment=False
alpha = 0.1
unet_option = 'unet' # options: 'unet_vae_old','unet_vae_RQ_scheme1' 'unet_vae_RQ_scheme3'
image_option = "clean" # "clean" or "noisy"

##################################
def rescale(image): ## function to rescale image for visualization
    map_img =  np.zeros(image.shape)
    for band in range(image.shape[2]):
        p2, p98 = np.percentile(image[:,:,band], (2, 98))
        map_img[:,:,band] = exposure.rescale_intensity(image[:,:,band], in_range=(p2, p98))
    return map_img

def rescale_truncate(image): ## function to rescale image for visualization
    if np.amin(image) < 0:
        image = np.where(image < 0,0,image)
    if np.amax(image) > 1:
        image = np.where(image > 1,1,image) 
    map_img =  np.zeros((256,256,3))
    for band in range(3):
        p2, p98 = np.percentile(image[:,:,band], (2, 98))
        map_img[:,:,band] = exposure.rescale_intensity(image[:,:,band], in_range=(p2, p98))
    return map_img

# def load_npy(file_path):
#     data = np.load(file_path)
#     data_1=(data - np.min(data)) / (np.max(data) - np.min(data))
#     data_1 = rescale(data_1)
#     # print(np.max(data))
#     # print(np.min(data))
#     # plt.imshow(data[:,:,:3])
#     # plt.show()

#     print("data shape: ", data_1.shape)

#     row,col,ch= data_1.shape
#     sigma = 0.01 ## choosing sigma based on the input images, 0.1-0.3 for NAIP images, 0.002 to 0.01 for sentinel2 images
#     noisy = data_1 + sigma*np.random.randn(row,col,ch)

#     transform_tensor = transforms.ToTensor()
#     if use_cuda:
#         noisy_tensor = transform_tensor(noisy).cuda()
#         tensor = transform_tensor(data_1).cuda()

#     return tensor.view([1]+list(tensor.shape)), noisy_tensor.view([1]+list(noisy_tensor.shape))

def jpg_to_tensor(filepath):

    naip_fn = filepath
    driverTiff = gdal.GetDriverByName('GTiff')
    naip_ds = gdal.Open(naip_fn, 1)
    nbands = naip_ds.RasterCount
    # create an empty array, each column of the empty array will hold one band of data from the image
    # loop through each band in the image nad add to the data array
    data = np.empty((naip_ds.RasterXSize*naip_ds.RasterYSize, nbands))
    for i in range(1, nbands+1):
        band = naip_ds.GetRasterBand(i).ReadAsArray()
        data[:, i-1] = band.flatten()

    img_data = np.zeros((naip_ds.RasterYSize, naip_ds.RasterXSize, naip_ds.RasterCount),
                    gdal_array.GDALTypeCodeToNumericTypeCode(naip_ds.GetRasterBand(1).DataType))
    for b in range(img_data.shape[2]):
        img_data[:, :, b] = naip_ds.GetRasterBand(b + 1).ReadAsArray()
        
    pil = np.array(img_data)
    pil = pil.reshape((5000,5000,8))
    #pil = pil/255

    pil = pil[256:512,512:768, :]
    print(pil.shape)

    # add noise
    row,col,ch= pil.shape
    sigma = 0.08
    noisy = pil + sigma*np.random.randn(row,col,ch)

    #pil_to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    transform_tensor = transforms.ToTensor()
    if use_cuda:
        noisy_tensor = transform_tensor(noisy).cuda()
        tensor = transform_tensor(pil).cuda()

    return tensor.view([1]+list(tensor.shape)), noisy_tensor.view([1]+list(noisy_tensor.shape))

#accept a torch tensor, convert it to a jpg at a certain path
def tensor_to_jpg(tensor):
    tensor = tensor.squeeze(0)
    if use_cuda:
        tensor = tensor.cpu()
    pil = tensor.permute(1, 2, 0).numpy()
    pil = np.array(pil)
    pil = rescale(pil)
    #pil = rescale_truncate(pil)
    return pil

#predict image
def extract_features(net,
                filepath,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    #img = img.unsqueeze(0)

    if image_option=='clean':
        img = jpg_to_tensor(filepath)[0] ## clean image
    elif image_option=='noisy':
        img = jpg_to_tensor(filepath)[1] ## noisy image

    img = img.to(device=device, dtype=torch.float32)

    print("input image shape: ", img.shape)

    ##### FEATURE EXTRACTION LOOP

    # placeholders
    PREDS = []
    FEATS = []

    # placeholder for batch features
    features = {}
    
    ##### REGISTER HOOK

    net.down_convs[1].pool.register_forward_hook(get_features(features, 'feats'))

    # forward pass [with feature extraction]
    preds = net(img)

    if unet_option == 'unet':
        preds = preds
    else:
        preds = preds[3]
    
    # add feats and preds to lists
    PREDS.append(preds.detach().cpu().numpy())
    FEATS.append(features['feats'].cpu().numpy())

    ##### INSPECT FEATURES

    PREDS = np.concatenate(PREDS)
    FEATS = np.concatenate(FEATS)

    print('- preds shape:', PREDS.shape)
    print('- feats shape:', FEATS.shape)

    return preds.detach().cpu(), FEATS, img.detach().cpu()

##### HELPER FUNCTION FOR FEATURE EXTRACTION

def get_features(features, name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

def get_false_color(image):

    false_color = np.zeros((image.shape[0], image.shape[1], 3))
    false_color[:,:,0] = image[:,:,6]
    false_color[:,:,1] = image[:,:,2]
    false_color[:,:,2] = image[:,:,1]

    return false_color

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
    window = window/numpy.sum(window)

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
        ssim_map = numpy.ones(numpy.shape(mu1))
        index = (den1 * den2) > 0
        ssim_map[index] = (num1[index] * num2[index]) / (den1[index] * den2[index])
        index = (den1 != 0) & (den2 == 0)
        ssim_map[index] = num1[index] / den1[index]

    mssim = ssim_map.mean()
    return mssim, ssim_map


def nrmse(im1, im2):
    a, b = im1.shape
    rmse = numpy.sqrt(numpy.sum((im2 - im1) ** 2) / float(a * b))
    max_val = max(numpy.max(im1), numpy.max(im2))
    min_val = min(numpy.min(im1), numpy.min(im2))
    return 1 - (rmse / (max_val - min_val))


if __name__ == '__main__':

    if unet_option == 'unet_vae_1':
        net = UNet_VAE(8)
    elif unet_option == 'unet':
        net = UNet_test(8, in_channels=8)
    elif unet_option == 'unet_vae_old':
        net = UNet_VAE_old(8, segment, in_channels=8)
    elif unet_option == 'unet_vae_RQ_old':
        net = UNet_VAE_RQ_old(8, alpha)
    # elif unet_option == 'unet_vae_RQ_allskip_trainable':
    #     net = UNet_VAE_RQ_old_trainable(8,alpha)
    elif unet_option == 'unet_vae_RQ_torch':
        net = UNet_VAE_RQ_new_torch(8, segment, alpha)
    elif unet_option == 'unet_vae_RQ_scheme3':
        net = UNet_VAE_RQ_scheme3(8, segment, alpha)
    elif unet_option == 'unet_vae_RQ_scheme1':
        net = UNet_VAE_RQ_scheme1(8, segment, alpha, in_channels=8)
    elif unet_option == 'unet_vae_RQ_scheme2':
        net = UNet_VAE_RQ_scheme2(8, segment, alpha)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    # model_saved = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_vae_old_4-18_epoch10_0.0_recon.pth'
    # model_sentinel_saved = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_vae_old_epoch20_sentinel_4-28_recon.pth'

    net.to(device=device)
    #net.load_state_dict(torch.load(model_saved, map_location=device))

    #print(net.down_convs[0])

    #print(net.down_convs[0].pool)

    logging.info('Model loaded!')
    #logging.info(f'\nPredicting image {image_path} ...')

    pred, feats, im = extract_features(net=net,
                        filepath=file_path,
                        scale_factor=1,
                        out_threshold=0.5,
                        device=device)

    im = tensor_to_jpg(im)
    #print(im.shape)
    im = im.reshape((256,256,8))

    pred = tensor_to_jpg(pred)

    im_false = get_false_color(im)
    pred_false = get_false_color(pred)

    # calculate ndvi
    ndvi_upper = im[:,:,6]-im[:,:,0]
    ndvi_upper = np.array(ndvi_upper, dtype=float)
    ndvi_lower = im[:,:,6]+im[:,:,0]
    ndvi_lower = np.array(ndvi_lower, dtype=float)
    ndvi = np.divide(ndvi_upper, ndvi_lower, out=np.zeros_like(ndvi_upper, dtype=float), where=ndvi_lower!=0)

    feats = feats.reshape((feats.shape[1],feats.shape[2],feats.shape[3]))
    print('feats shape: ', feats.shape)

    h = feats.shape[1]
    w = feats.shape[2]
    ndvi_h, ndvi_w = ndvi.shape
    bin_size = ndvi_h // h
    ndvi_res = ndvi.reshape((h, bin_size,
                                h, bin_size, 1)).max(3).max(1)

    print('ndvi_res max: ', np.max(ndvi_res))
    print('ndvi_res min: ', np.min(ndvi_res))

    b = np.zeros((feats.shape[1],feats.shape[2]))
    for i in range(feats.shape[0]):
        #plot_img_and_mask_recon(im_false, feats[:,i,:,:].reshape((feats.shape[2],feats.shape[3])))
        a = feats[i,:,:].reshape((feats.shape[1],feats.shape[2]))
        h,w = a.shape

        a = 2.*(a - np.min(a))/nan_ptp(a)-1

        # ndvi_res = np.array(ndvi_res)
        # print('ndvi type: ', type(ndvi_res))

        # print('feat max: ', np.max(a))
        # print('feat min: ', np.min(a))
        
        
        nrmse_val = nrmse(a, ndvi_res)
        if nrmse_val > -2:
            #print(nrmse_val)
            b += a
    plot_img_and_mask_recon(ndvi_res, b)

    plot_img_and_mask_recon(im_false, pred_false)
    plot_img_and_mask_recon(im[:,:,:3], pred[:,:,:3])

    