import argparse
import logging
import os
import rasterio as rio
import re
from skimage import exposure
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from osgeo import gdal, gdal_array
import matplotlib.pyplot as plt
import tifffile

from unet import UNet_VAE
from unet import UNet_VAE_old, UNet_VAE_RQ_old, UNet_VAE_RQ_test, UNet_VAE_RQ_old_torch
from unet import UNet_VAE_RQ_new_torch, UNet_VAE_RQ_scheme3, RQUNet_VAE_scheme1_Pareto
from unet import UNet_VAE_RQ_scheme1, UNet_VAE_RQ_scheme2, UNet_VAE_Stacked
from utils.utils import plot_img_and_mask, plot_img_and_mask_3, plot_img_and_mask_recon

##################################
def rescale(image): ## function to rescale image for visualization
    map_img =  np.zeros((256,256,3))
    for band in range(3):
        p2, p98 = np.percentile(image[:,:,band], (2, 98))
        map_img[:,:,band] = exposure.rescale_intensity(image[:,:,band], in_range=(p2, p98))
    return map_img

def rescale_image(
            image: np.ndarray,
            rescale_type: str = 'per-image',
            highest_value: int = 1
        ):
    """
    Rescale image [0, 1] per-image or per-channel.
    Args:
        image (np.ndarray): array to rescale
        rescale_type (str): rescaling strategy
    Returns:
        rescaled np.ndarray
    """
    image = image.astype(np.float32)
    mask = np.where(image[0, :, :] >= 0, True, False)

    if rescale_type == 'per-image':
        image = (image - np.min(image, initial=highest_value, where=mask)) \
            / (np.max(image, initial=highest_value, where=mask)
                - np.min(image, initial=highest_value, where=mask))
    elif rescale_type == 'per-ts':
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

    elif rescale_type == 'per-channel':
        for i in range(image.shape[-1]):
            image[:, :, i] = (
                image[:, :, i]
                - np.min(image[:, :, i], initial=highest_value, where=mask)) \
                / (np.max(image[:, :, i], initial=highest_value, where=mask)
                    - np.min(
                        image[:, :, i], initial=highest_value, where=mask))
    else:
        logging.info(f'Skipping based on invalid option: {rescale_type}')
    return image

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

#accept a file path to a jpg, return a torch tensor
def jpg_to_tensor(filepath, im_type):

    img_data = tifffile.imread(filepath)

    pil = np.array(img_data)
    pil = np.array(pil[:,:,1:4])

    if im_type == "naip":
        pil=pil/255
    elif im_type == 'hls':
        pil = pil*0.0001
        # pil=(pil - np.min(pil)) / (np.max(pil) - np.min(pil))
        # pil = rescale_image(pil)
        # pil=pil
    elif im_type == "sentinel":
        pil=(pil - np.min(pil)) / (np.max(pil) - np.min(pil))

    # pil = np.array(pil[:,:,1:4])

    h, w, c = pil.shape
    input_size = 256
    
    # I = np.random.randint(0, h-input_size, size=1)
    # J = np.random.randint(0, w-input_size, size=1)
    
    # pil = np.array([pil[i:(i+input_size), j:(j+input_size),:] for i, j in zip(I, J)])
    pil = pil[100:(100+input_size), 100:(100+input_size),:]
    pil = np.squeeze(pil)
    # pil = rescale_image(pil)
    pil=(pil - np.min(pil)) / (np.max(pil) - np.min(pil))

    print(pil.shape)

    # print(np.max(pil))
    # print(np.min(pil))

    row,col,ch= pil.shape
    sigma = 0.002 ## choosing sigma based on the input images, 0.1-0.3 for NAIP images, 0.002 to 0.01 for sentinel2 images
    noisy = pil + sigma*np.random.randn(row,col,ch)

    pil = np.transpose(pil, (2,0,1))

    transform_tensor = transforms.ToTensor()
    if use_cuda:
        noisy_tensor = torch.tensor(noisy).cuda()
        tensor = torch.tensor(pil).cuda()

    return tensor.view([1]+list(tensor.shape)), noisy_tensor.view([1]+list(noisy_tensor.shape))

#accept a torch tensor, convert it to a jpg at a certain path
def tensor_to_jpg(tensor):
    tensor = tensor.squeeze(0)
    if use_cuda:
        tensor = tensor.cpu()
    pil = tensor.permute(1, 2, 0).numpy()
    pil = np.array(pil)
    pil = rescale(pil)

    # pil = rescale_truncate(pil)
    return pil

#predict image
def predict_img(net,
                filepath,
                im_type,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    #img = img.unsqueeze(0)

    if image_option=='clean':
        img = jpg_to_tensor(filepath, im_type)[0] ## clean image
    elif image_option=='noisy':
        img = jpg_to_tensor(filepath, im_type)[1] ## noisy image

    img = img.to(device=device, dtype=torch.float32)

    print(img.shape)

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
        
        elif unet_option == 'unet_vae_RQ_scheme1':
            output = output[3]
            return output.cpu()
        
        elif unet_option == 'unet_vae_old':
            output = output[3]
            return output.cpu()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='github_checkpoints/checkpoint_unet_vae_old_epoch20_0.0_recon.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_saved = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_vae_old_4-18_epoch10_0.0_recon.pth'
    model_sentinel_saved = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_vae_old_epoch19_sentinel_5-7_recon.pth'
    model_hls_saved = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_vae_old_epoch97_senegal_hls_rgb_06-20-2023_recon_new.pth'

    #image_path = '/home/geoint/tri/github_files/test_img/number13458.TIF'
    #mask_true_path = '/home/geoint/tri/github_files/test_label/number13458.TIF'
    # image_path = '/home/geoint/tri/github_files/sentinel2_im/2016105_0.tif'
    # mask_true_path = '/home/geoint/tri/github_files/sentinel2_im/2016105_0.tif'

    # global image_path 
    image_path = '/home/geoint/PycharmProjects/tensorflow/out_hls/HLS.S30.T28PEV.2021189T112119.v2.0.tif'
    # mask_true_path = '/home/geoint/PycharmProjects/tensorflow/out_hls/HLS.S30.T28PEV.2021004T112451.v2.0.tif'

    name = re.search(r'/out_hls/(.*?).tif', image_path).group(1)
    print(name)

    use_cuda = True
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # im_type = image_path[30:38]
    im_type = 'hls'
    print('image type: ', im_type)
    segment=False
    alpha = 0.007
    unet_option = 'unet_vae_stacked' # options: 'unet_vae_old','unet_vae_RQ_scheme1' 'unet_vae_RQ_scheme3'
    image_option = "clean" # "clean" or "noisy"
    num_classes=3
    channels=3

    if unet_option == 'unet_vae_1':
        net = UNet_VAE(num_classes)
    elif unet_option == 'unet_vae_old':
        net = UNet_VAE_old(num_classes, segment,in_channels=channels)
    elif unet_option == 'unet_vae_RQ_old':
        net = UNet_VAE_RQ_old(num_classes, alpha)
    # elif unet_option == 'unet_vae_RQ_allskip_trainable':
    #     net = UNet_VAE_RQ_old_trainable(3,alpha)
    elif unet_option == 'unet_vae_RQ_torch':
        net = UNet_VAE_RQ_old_torch(num_classes, segment, alpha)
    elif unet_option == 'unet_vae_RQ_scheme3':
        net = UNet_VAE_RQ_scheme3(num_classes, segment, alpha)
    elif unet_option == 'unet_vae_RQ_scheme1':
        net = UNet_VAE_RQ_scheme1(num_classes, segment, alpha, in_channels=channels)
    elif unet_option == 'unet_vae_RQ_scheme2':
        net = UNet_VAE_RQ_scheme2(num_classes, segment, alpha)
    elif unet_option == 'unet_vae_stacked':
        net = UNet_VAE_Stacked(num_classes, segment, alpha, device, model_hls_saved, in_channels=channels, unet_num_block=10)

    elif unet_option == 'rqunet_vae_scheme1_pareto':
        net = RQUNet_VAE_scheme1_Pareto(num_classes, segment, alpha)

    #logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)

    if unet_option != 'unet_vae_stacked':
        if im_type == 'sentinel':
            net.load_state_dict(torch.load(model_sentinel_saved, map_location=device))
        elif im_type == 'hls':
            net.load_state_dict(torch.load(model_hls_saved, map_location=device))
        else:
            net.load_state_dict(torch.load(model_saved, map_location=device))

    else:
        net = net

    logging.info('Model loaded!')
    logging.info(f'\nPredicting image {image_path} ...')


    if unet_option == 'rqunet_vae_scheme1_pareto':
        mask, s, Wy = predict_img(net=net,
                        filepath=image_path,
                        im_type=im_type,
                        scale_factor=1,
                        out_threshold=0.5,
                        device=device)

        x_range = np.arange(65536)
        plt.plot(s, x_range, color='blue', label = 's')
        plt.plot(Wy, x_range, color='red', label = 'Wy')
        plt.legend()
        plt.show()

    else:
        mask = predict_img(net=net,
                        filepath=image_path,
                        im_type=im_type,
                        scale_factor=1,
                        out_threshold=0.5,
                        device=device)
        
    print('mask shape: ', mask.shape)

    print(np.max(mask.numpy()))


    # out_files = 'out/predict_va_vae_recon_epoch1'
    # im_out_files = 'out/img'
    
    # if not args.no_save:
    #     out_filename = out_files
    #     logging.info(f'Mask saved to {out_filename}')

    # mask = tensor_to_jpg(mask)
    if image_option=='clean':
        img = jpg_to_tensor(image_path, im_type)[0]
    else:
        img = jpg_to_tensor(image_path, im_type)[1]

    mask = mask.numpy()
    mask = np.squeeze(mask)
    print('mask',mask.shape)
    mask = np.transpose(mask[:,:,:], (1,2,0))
    np.save(f"/home/geoint/tri/stacked-unetvae-hls-video/{name}.npy", mask)
    mask = rescale_truncate(mask[:,:,::-1])

    img = img.cpu().numpy()
    img = np.squeeze(img)
    print('img', img.shape)
    img = np.transpose(img[:,:,:], (1,2,0))
    img = rescale_truncate(img[:,:,::-1])

    print('mask shape: ',mask.shape)
    print('image shape: ', img.shape)

    plot_img_and_mask_recon(img, mask, name)