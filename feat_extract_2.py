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
import tifffile

from unet import UNet_VAE
from unet import UNet_VAE_old, UNet_VAE_RQ_old, UNet_VAE_RQ_test, UNet_VAE_RQ_old_torch
from unet import UNet_VAE_RQ_new_torch, UNet_VAE_RQ_scheme3
from unet import UNet_VAE_RQ_scheme1, UNet_VAE_RQ_scheme2
from unet import UNet_test
from unet import UNet_VAE_feat_ext
from utils.utils import plot_img_and_mask, plot_img_and_mask_3, plot_img_and_mask_recon

#image_path = '/home/geoint/tri/github_files/test_img/number13458.TIF'
#mask_true_path = '/home/geoint/tri/github_files/test_label/number13458.TIF'
#image_path = '/home/geoint/tri/github_files/sentinel2_im/2016105_0.tif'
#mask_true_path = '/home/geoint/tri/github_files/sentinel2_im/2016105_0.tif'

#npy_path = '/home/geoint/tri/github_files/input_senegal/Tappan01_WV02_20110430_M1BS_103001000A27E100_data_568.npy'


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
    map_img =  np.zeros(image.shape)
    for band in range(image.shape[2]):
        p2, p98 = np.percentile(image[:,:,band], (2, 98))
        map_img[:,:,band] = exposure.rescale_intensity(image[:,:,band], in_range=(p2, p98))
    return map_img

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
    #pil = pil.reshape((5000,5000,9))
    pil = pil.reshape((5000,5000,8))

    #pil = pil/255

    ndvi_upper = pil[:,:,6]-pil[:,:,0]
    ndvi_upper = np.array(ndvi_upper, dtype=float)
    ndvi_lower = pil[:,:,6]+pil[:,:,0]
    ndvi_lower = np.array(ndvi_lower, dtype=float)
    ndvi = np.divide(ndvi_upper, ndvi_lower, out=np.zeros_like(ndvi_upper, dtype=float), where=ndvi_lower!=0)

    #pil = pil[256:512,512:768, :]
    #pil = pil[512:768,512:768, :]
    #pil = pil[512:1024,512:1024, :]
    #pil = pil[1024:1536,2048:2560, :]

    #pil = pil[1536:2560,512:1536, :]
    pil = pil[1536:2560,1536:2560, :]
    #pil = pil[512:1536,512:1536, :]

    ndvi = ndvi[1024:1536,2048:2560]

    #print(pil.shape)

    # add noise
    row,col,ch= pil.shape
    sigma = 0.08
    noisy = pil + sigma*np.random.randn(row,col,ch)

    #pil_to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    transform_tensor = transforms.ToTensor()
    if use_cuda:
        noisy_tensor = transform_tensor(noisy).cuda()
        tensor = transform_tensor(pil).cuda()
        tensor_ndvi = transform_tensor(ndvi).cuda()

    return tensor.view([1]+list(tensor.shape)), noisy_tensor.view([1]+list(noisy_tensor.shape)), \
         tensor_ndvi.view([1]+list(tensor_ndvi.shape))


def read_image(image_option, filepath):

    if image_option=='clean':
        img = jpg_to_tensor(filepath)[0] ## clean image
    elif image_option=='noisy':
        img = jpg_to_tensor(filepath)[1] ## noisy image

    img = img.to(device=device, dtype=torch.float32)

    return img.detach().cpu()

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
        #img = jpg_to_tensor(filepath)[2] ## clean image
    elif image_option=='noisy':
        img = jpg_to_tensor(filepath)[1] ## noisy image

    img = img.to(device=device, dtype=torch.float32)

    print("input image shape: ", img.shape)

    # ##### FEATURE EXTRACTION LOOP

    # # placeholders
    # PREDS = []
    # FEATS = []

    # # placeholder for batch features
    # features = {}
    
    # ##### REGISTER HOOK

    # net.down_convs[1].pool.register_forward_hook(get_features(features, 'feats'))

    # forward pass [with feature extraction]
    preds = net(img)

    if unet_option == 'unet':
        preds = preds
    else:
        preds = preds[0].squeeze()

    preds = preds.detach().cpu()

    # full_mask = torch.argmax(preds, dim=0)
    full_mask = preds

    full_mask = torch.squeeze(full_mask).cpu().numpy()

    full_mask = full_mask.reshape((full_mask.shape[0], full_mask.shape[1], 1))

    print("mask shape: ", full_mask.shape)
    
    # # add feats and preds to lists
    # PREDS.append(preds.detach().cpu().numpy())
    # FEATS.append(features['feats'].cpu().numpy())

    # ##### INSPECT FEATURES

    # PREDS = np.concatenate(PREDS)
    # FEATS = np.concatenate(FEATS)

    # print('- preds shape:', PREDS.shape)
    # print('- feats shape:', FEATS.shape)

    return full_mask

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

    #file_path = '/home/geoint/tri/nasa_senegal/cassemance/Tappan02_WV03_20160123_M1BS_1040010018A59100_data.tif'
    #file_path = '/home/geoint/tri/nasa_senegal/cassemance/Tappan01_WV02_20110430_M1BS_103001000A27E100_data.tif'
    #file_path = '/home/geoint/tri/nasa_senegal/test/sar/Tappan02_WV03_20160123_M1BS_1040010018A59100_data_chm.tif'

    #file_path = '/home/geoint/tri/nasa_senegal/cassemance/Tappan02_WV02_20181217_M1BS_1030010089CC6D00_data.tif'
    file_path = '/home/geoint/tri/nasa_senegal/cassemance/Tappan02_WV02_20121014_M1BS_103001001B793900_data.tif'

    use_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #im_type = image_path[30:38]
    im_type='senegal'
    segment=False
    alpha = 0.0
    unet_option = 'unet_vae_feat_ext' 
    #unet_option = 'unet'
    image_option = "clean" # "clean" or "noisy"

    im = read_image(image_option, file_path)
    im = tensor_to_jpg(im)

    #im = im.reshape((256,256,8))
    #im = im.reshape((512,512,9))

    #im = im.reshape((1024,1024,9))
    im = im.reshape((1024,1024,8))

    num_band = im.shape[2]
    output_band = 1

    # plt.imshow(im[:,:,:3])
    # plt.show()

    # calculate ndvi
    ndvi_upper = im[:,:,6]-im[:,:,0]
    ndvi_upper = np.array(ndvi_upper, dtype=float)
    ndvi_lower = im[:,:,6]+im[:,:,0]
    ndvi_lower = np.array(ndvi_lower, dtype=float)
    ndvi = np.divide(ndvi_upper, ndvi_lower, out=np.zeros_like(ndvi_upper, dtype=float), where=ndvi_lower!=0)

    index_ndvi = np.ma.where(ndvi < 0.1, 1, 0)

    if unet_option == 'unet_vae_1':
        net = UNet_VAE(8)
    elif unet_option == 'unet':
        net = UNet_test(output_band, in_channels=num_band)
    elif unet_option == 'unet_vae_feat_ext':
        net = UNet_VAE_feat_ext(output_band, segment, ndvi, in_channels=num_band)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    # model_saved = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_vae_old_4-18_epoch10_0.0_recon.pth'
    # model_sentinel_saved = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_vae_old_epoch20_sentinel_4-28_recon.pth'

    net.to(device=device)
    #net.load_state_dict(torch.load(model_saved, map_location=device))


    logging.info('Model loaded!')
    #logging.info(f'\nPredicting image {image_path} ...')

    pred = extract_features(net=net,
                        filepath=file_path,
                        scale_factor=1,
                        out_threshold=0.5,
                        device=device)
    
    #pred = tensor_to_jpg(pred)
    im_false = get_false_color(im)

    # chm = im[:,:,8]
    # index_chm = np.ma.where(chm < 0.2, 1, 0)

    # index_all = index_chm + index_ndvi
    # index_all[index_all > 0] = 1

    print('pred shape: ', pred.shape)
    #pred_false = get_false_color(pred)
    # for i in range(pred.shape[2]):
    #     plot_img_and_mask_recon(im_false, pred[:,:,i])
    # plot_img_and_mask_recon(im_false, pred_false)
    plot_img_and_mask_recon(im_false, pred)
    #plot_img_and_mask_recon(im_false, ndvi)
    plot_img_and_mask_recon(im_false, index_ndvi)
    #plot_img_and_mask_recon(im_false, chm)
    #plot_img_and_mask_recon(im_false, index_chm)
    #plot_img_and_mask_recon(im_false, index_all)

    #tifffile.imsave('/home/geoint/tri/github_files/tri_exp/test_Tappan02_2016_8band_out_3.tiff', pred)
    tifffile.imsave('/home/geoint/tri/github_files/tri_exp/test_Tappan02_20181217_8band_out_1_no_func_1_im1.tiff', pred)
    #tifffile.imsave('/home/geoint/tri/github_files/tri_exp/ndvi__Tappan02_20181217.tif', ndvi)
    #tifffile.imsave('/home/geoint/tri/github_files/tri_exp/test_Tappan02_20181217_8band_false.tiff', im_false)

    