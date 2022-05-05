import argparse
import logging
import os
import rasterio as rio
from skimage import exposure
import numpy as np
import torch
from torchvision import transforms
from osgeo import gdal, gdal_array
import matplotlib.pyplot as plt

from unet import UNet_VAE
from unet import UNet_VAE_old, UNet_VAE_RQ_old, UNet_VAE_RQ_test, UNet_VAE_RQ_old_trainable, UNet_VAE_RQ_old_torch
from unet import UNet_VAE_RQ_new_torch, UNet_VAE_RQ_scheme3
from unet import UNet_VAE_RQ_scheme1, UNet_VAE_RQ_scheme2
from utils.utils import plot_img_and_mask, plot_img_and_mask_3, plot_img_and_mask_recon

#image_path = '/home/geoint/tri/github_files/test_img/number13458.TIF'
#mask_true_path = '/home/geoint/tri/github_files/test_label/number13458.TIF'
#image_path = '/home/geoint/tri/github_files/sentinel2_im/2016105_0.tif'
#mask_true_path = '/home/geoint/tri/github_files/sentinel2_im/2016105_0.tif'

npy_path = '/home/geoint/tri/github_files/input_senegal/Tappan01_WV02_20110430_M1BS_103001000A27E100_data_568.npy'

use_cuda = True
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#im_type = image_path[30:38]
im_type='senegal'
segment=False
alpha = 0.1
unet_option = 'unet_vae_RQ_scheme1' # options: 'unet_vae_old','unet_vae_RQ_scheme1' 'unet_vae_RQ_scheme3'
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

def load_npy(file_path):
    data = np.load(file_path)
    data_1=(data - np.min(data)) / (np.max(data) - np.min(data))
    data_1 = rescale(data_1)
    # print(np.max(data))
    # print(np.min(data))
    # plt.imshow(data[:,:,:3])
    # plt.show()

    print("data shape: ", data_1.shape)

    row,col,ch= data_1.shape
    sigma = 0.01 ## choosing sigma based on the input images, 0.1-0.3 for NAIP images, 0.002 to 0.01 for sentinel2 images
    noisy = data_1 + sigma*np.random.randn(row,col,ch)

    transform_tensor = transforms.ToTensor()
    if use_cuda:
        noisy_tensor = transform_tensor(noisy).cuda()
        tensor = transform_tensor(data_1).cuda()

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
        img = load_npy(filepath)[0] ## clean image
    elif image_option=='noisy':
        img = load_npy(filepath)[1] ## noisy image

    img = img.to(device=device, dtype=torch.float32)

    print("input image shape: ", img.shape)

    ##### FEATURE EXTRACTION LOOP

    # placeholders
    PREDS = []
    FEATS = []

    # placeholder for batch features
    features = {}
    
    ##### REGISTER HOOK

    net.down_convs[0].pool.register_forward_hook(get_features(features, 'feats'))

    # forward pass [with feature extraction]
    preds = net(img)

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

    if unet_option == 'unet_vae_1':
        net = UNet_VAE(3)
    elif unet_option == 'unet_vae_old':
        net = UNet_VAE_old(8)
    elif unet_option == 'unet_vae_RQ_old':
        net = UNet_VAE_RQ_old(3, alpha)
    elif unet_option == 'unet_vae_RQ_allskip_trainable':
        net = UNet_VAE_RQ_old_trainable(8,alpha)
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

    model_saved = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_vae_old_4-18_epoch10_0.0_recon.pth'
    model_sentinel_saved = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_vae_old_epoch20_sentinel_4-28_recon.pth'

    net.to(device=device)
    #net.load_state_dict(torch.load(model_saved, map_location=device))

    #print(net.down_convs[0])

    #print(net.down_convs[0].pool)

    logging.info('Model loaded!')
    #logging.info(f'\nPredicting image {image_path} ...')

    pred, feats, im = extract_features(net=net,
                        filepath=npy_path,
                        scale_factor=1,
                        out_threshold=0.5,
                        device=device)

    im = tensor_to_jpg(im)
    im = im.reshape((256,256,8))

    pred = tensor_to_jpg(pred)

    im_false = get_false_color(im)
    pred_false = get_false_color(pred)

    # for i in range(feats.shape[1]):
    #     plot_img_and_mask_recon(false_col_im, feats[:,i,:,:].reshape((feats.shape[2],feats.shape[3])))

    plot_img_and_mask_recon(im_false, pred_false)

    