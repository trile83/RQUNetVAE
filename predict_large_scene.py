import logging
import sys
import torch
import torch.nn as nn
from torchvision import transforms
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from skimage import exposure
import glob
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import optim
from tqdm import tqdm

from unet import UNet_VAE
from unet import UNet_VAE_old, UNet_VAE_RQ_old, UNet_VAE_RQ_test, UNet_VAE_RQ_old_torch
from unet import UNet_VAE_RQ_new_torch, UNet_VAE_RQ_scheme3, RQUNet_VAE_scheme1_Pareto
from unet import UNet_VAE_RQ_scheme1, UNet_VAE_RQ_scheme2, UNet_VAE_Stacked
from utils.utils import plot_img_and_mask, plot_img_and_mask_3, plot_img_and_mask_recon


# Read from Tiff files ----------------------------------
import numpy as np
import tifffile


##################################
def rescale(image):
    map_img =  np.zeros((256,256,3))
    for band in range(3):
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

# Normalize bands into 0.0 - 1.0 scale
def normalize_image(image):
    '''
    Arg: Input is an image with dimension (channel, height, width)
    '''

    # for i in range(image.shape[0]):
    #         image[i, :, :] = (image[i, :, :] - np.min(image[i, :, :])) / (np.max(image[i, :, :]) - np.min(image[i, :, :]))

    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    #image = image/np.max(image)

    return image

# Standardize band for mean and std
def standardize_image(image):
    '''
    Arg: Input is an image with dimension (channel, height, width)
    '''
    for i in range(image.shape[0]):  # for each channel in the image
            image[i, :, :] = (image[i, :, :] - np.mean(image[i, :, :])) / \
                (np.std(image[i, :, :]) + 1e-8)

    #image = image.reshape((image.shape[1], image.shape[2], image.shape[0]))

    return image

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
def predict_img(net,
                filepath,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    #img = img.unsqueeze(0)

    # if image_option=='clean':
    #     img = jpg_to_tensor(filepath)[0] ## clean image
    # elif image_option=='noisy':
    #     img = jpg_to_tensor(filepath)[1] ## noisy image

    
    ## test
    # get all the image and mask path and number of images

    arr = tifffile.imread('2019059.tif')
    # normalization
    img = np.asarray(arr[:,:,:])

    print(img.shape)
    # image will have dimension (h,w,c) and don't need to reshape
    # if the image is (C,H,W)
    # img = img.reshape((img.shape[2],img.shape[1],img.shape[0]))

    img = normalize_image(img)

    # ---------------------------------------------------------------

    # image_input_path = 'C:/Users/hthai7/Desktop/Python/VA_Construction_10Band_Plus_QA/Image_Construction/'

    # Pre-processing data:
    # if np.amin(img) < 0:
    #     img = np.where(img < 0, 0, img)
    # if np.amax(img) > 1:
    #     img = np.where(img > 1, 1, img)
    # img = np.float32(img)
    # print(np.amin(img),np.max(img))
    # img = np.transpose(img, (1, 2, 0))

    h, w, c = img.shape

    train_size = 100 
    test_size = 10  
    I = np.random.randint(0, h-256, size=train_size+test_size)
    J = np.random.randint(0, w-256, size=train_size+test_size)
    
    X = np.array([img[i:(i+256), j:(j+256),:] for i, j in zip(I, J)])

    #######
    # get first image 
    img = X[0]
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if unet_option == 'unet':
            output = output
            return output.cpu()
        elif unet_option=='unet_vae_stacked':
            output = output[1]
            return output.cpu()
        elif unet_option == 'unet_vae_RQ_scheme3':
            err = output[5]
            output = output[3]
            print("relative error: ", err)
            plt.plot(err.cpu())
            plt.show()

            return output.cpu()
        elif unet_option == 'rqunet_vae_scheme1_pareto':
            s = output[6]
            Wy = output[5]
            output = output[3]

            return output.cpu(), s.cpu().numpy(), Wy.cpu().numpy()


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_saved = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_vae_old_4-18_epoch10_0.0_recon.pth'
    model_sentinel_saved = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_vae_old_epoch19_sentinel_5-7_recon.pth'


    dir_checkpoint = Path('/home/geoint/tri/github_files/github_checkpoints/')
    #use cuda, or not? be prepared for a long wait if you don't have cuda capabilities.
    use_cuda = False
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    im_type = 'sentinel'
    print('image type: ', im_type)
    segment=False
    alpha = 0.0
    unet_option = 'unet_vae_stacked' # options: 'unet_vae_old','unet_vae_RQ_scheme1' 'unet_vae_RQ_scheme3'
    image_option = "clean" # "clean" or "noisy"

    
    image_path = '2019059.tif'


    if unet_option == 'unet_vae_1':
        net = UNet_VAE(3)
    elif unet_option == 'unet_vae_old':
        net = UNet_VAE_old(3, segment)
    elif unet_option == 'unet_vae_RQ_old':
        net = UNet_VAE_RQ_old(3, alpha)
    # elif unet_option == 'unet_vae_RQ_allskip_trainable':
    #     net = UNet_VAE_RQ_old_trainable(3,alpha)
    elif unet_option == 'unet_vae_RQ_torch':
        net = UNet_VAE_RQ_old_torch(3, segment, alpha)
    elif unet_option == 'unet_vae_RQ_scheme3':
        net = UNet_VAE_RQ_scheme3(3, segment, alpha)
    elif unet_option == 'unet_vae_RQ_scheme1':
        net = UNet_VAE_RQ_scheme1(3, segment, alpha)
    elif unet_option == 'unet_vae_RQ_scheme2':
        net = UNet_VAE_RQ_scheme2(3, segment, alpha)
    elif unet_option == 'unet_vae_stacked':
        net = UNet_VAE_Stacked(3, segment, alpha, device, model_sentinel_saved)

    elif unet_option == 'rqunet_vae_scheme1_pareto':
        net = RQUNet_VAE_scheme1_Pareto(3, segment, alpha)

    
    #logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)

    if unet_option != 'unet_vae_stacked':
        if im_type == 'sentinel':
            net.load_state_dict(torch.load(model_sentinel_saved, map_location=device))
        else:
            net.load_state_dict(torch.load(model_saved, map_location=device))

    else:
        net = net

    logging.info('Model loaded!')
    logging.info(f'\nPredicting image {image_path} ...')


    # if unet_option == 'rqunet_vae_scheme1_pareto':
    #     mask, s, Wy = predict_img(net=net,
    #                     filepath=image_path,
    #                     scale_factor=1,
    #                     out_threshold=0.5,
    #                     device=device)

    #     x_range = np.arange(65536)
    #     plt.plot(s, x_range, color='blue', label = 's')
    #     plt.plot(Wy, x_range, color='red', label = 'Wy')
    #     plt.legend()
    #     plt.show()

        
    # else:
    mask = predict_img(net=net,
                        filepath='[image_path]',
                        scale_factor=1,
                        out_threshold=0.5,
                        device=device)

    

    # out_files = 'out/predict_va_vae_recon_epoch1'
    # im_out_files = 'out/img'
    
    # if not args.no_save:
    #     out_filename = out_files
    #     logging.info(f'Mask saved to {out_filename}')

    mask = tensor_to_jpg(mask)

    plt.imshow(mask[:,:,:3])


    # if image_option=='clean':
    #     img = jpg_to_tensor(image_path)[0]
    # else:
    #     img = jpg_to_tensor(image_path)[1]
    # img = tensor_to_jpg(img)

    # plot_img_and_mask_recon(img, mask)