import argparse
import logging
import os
import rasterio
from skimage import exposure
import numpy as np
import torch
import matplotlib.pyplot as plt

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

# Standardize images
def standardize_image(image):
    '''
    Arg: Input is an image with dimension (channel, height, width)
    '''
    # for i in range(image.shape[0]):  # for each channel in the image
    #         image[i, :, :] = (image[i, :, :] - np.mean(image[i, :, :])) / \
    #             (np.std(image[i, :, :]) + 1e-8)

    image = image.reshape((image.shape[1], image.shape[2], image.shape[0]))

    return image

# Normalize bands into 0.0 - 1.0 scale
def normalize_image(image):
    '''
    Arg: Input is an image with dimension (channel, height, width)
    '''

    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    #image = image/np.max(image)

    return image

#accept a file path to a jpg, return a torch tensor
def jpg_to_tensor(filepath):

    image = rasterio.open(filepath).read()

    if im_type != "sentinel":
        pil = image/255
    else:
        pil = normalize_image(image)

    ch,row,col= pil.shape
    sigma = 0.01 ## choosing sigma based on the input images, 0.1-0.3 for NAIP images, 0.002 to 0.01 for sentinel2 images
    noisy = pil + sigma*np.random.randn(ch,row,col)

    if use_cuda:
        noisy_tensor = torch.tensor(noisy)
        tensor = torch.tensor(pil)

    return tensor.view([1]+list(tensor.shape)), noisy_tensor.view([1]+list(noisy_tensor.shape))

#accept a torch tensor, convert it to a jpg at a certain path
def tensor_to_jpg(tensor):
    tensor = tensor.squeeze(0)
    if use_cuda:
        tensor = tensor.cpu()
    pil = tensor.permute(2, 1, 0).numpy()
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

    if image_option=='clean':
        img = jpg_to_tensor(filepath)[0] ## clean image
    elif image_option=='noisy':
        img = jpg_to_tensor(filepath)[1] ## noisy image

    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if unet_option == 'unet':
            output = output
            return output.permute(2, 1, 0).numpy()
        elif unet_option=='unet_vae_stacked':
            output = output[1]
            return output.permute(2, 1, 0).numpy()
        elif unet_option == 'unet_vae_RQ_scheme3':
            err = output[5]
            output = output[3]
            print("relative error: ", err)
            plt.plot(err.cpu())
            plt.show()

            return output.permute(2, 1, 0).numpy()
        elif unet_option == 'rqunet_vae_scheme1_pareto':
            s = output[6]
            Wy = output[5]
            output = output[3]

            return output.permute(2, 1, 0).numpy(), s.cpu().numpy(), Wy.cpu().numpy()
        else:

            output = output[3]
            output = output.cpu().squeeze()

            print('output shape: ', output.shape)
 
            return output.permute(2, 1, 0).numpy()
            #return output.numpy()


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
    #args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_saved = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_vae_old_4-18_epoch10_0.0_recon.pth'
    model_sentinel_saved = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_vae_old_epoch14_sentinel_6-28_recon.pth'

    #image_path = '/home/geoint/tri/github_files/test_img/number13458.TIF'
    #mask_true_path = '/home/geoint/tri/github_files/test_label/number13458.TIF'
    image_path = '/home/geoint/tri/github_files/sentinel2_im/2016105_0.tif'
    mask_true_path = '/home/geoint/tri/github_files/sentinel2_im/2016105_0.tif'

    use_cuda = True
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    im_type = image_path[30:38]
    print('image type: ', im_type)
    segment=False
    alpha = 0.4
    unet_option = 'unet_vae_RQ_scheme2' # options: 'unet_vae_old','unet_vae_RQ_scheme1' 'unet_vae_RQ_scheme3' 'rqunet_vae_scheme1_pareto'
    image_option = "noisy" # "clean" or "noisy"


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


    if unet_option == 'rqunet_vae_scheme1_pareto':
        mask, s, Wy = predict_img(net=net,
                        filepath=image_path,
                        scale_factor=1,
                        out_threshold=0.5,
                        device=device)
    else:
        mask = predict_img(net=net,
                        filepath=image_path,
                        scale_factor=1,
                        out_threshold=0.5,
                        device=device)

    x_range = np.arange(65536)
    # plt.plot(s, x_range, color='blue', label = 's')
    # plt.plot(Wy, x_range, color='red', label = 'Wy')
    # plt.legend()
    # plt.show()

    # out_files = 'out/predict_va_vae_recon_epoch1'
    # im_out_files = 'out/img'
    
    # if not args.no_save:
    #     out_filename = out_files
    #     logging.info(f'Mask saved to {out_filename}')

    print('mask shape: ', mask.shape)
    #mask = mask.reshape((mask.shape[2], mask.shape[3], mask.shape[1]))

    #print(mask)

    mask = rescale(mask)


    if image_option=='clean':
        img = jpg_to_tensor(image_path)[0]
    else:
        img = jpg_to_tensor(image_path)[1]

    img = tensor_to_jpg(img)

    print('image shape: ', img.shape)
    # img = img.cpu().numpy()
    #img = img.reshape((img.shape[2], img.shape[3], img.shape[1]))

    plot_img_and_mask_recon(img, mask)