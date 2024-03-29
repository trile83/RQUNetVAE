import argparse
import logging
import os
import math
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
from cnn_denoise.rdn import RDN
from utils.utils import plot_img_and_mask, plot_img_and_mask_3, plot_img_and_mask_recon

# image_path = '/home/geoint/tri/github_files/test_img/number13458.TIF'
# mask_true_path = '/home/geoint/tri/github_files/test_label/number13458.TIF'
# image_path = '/home/geoint/tri/github_files/sentinel2_im/2016105_0.tif'
# mask_true_path = '/home/geoint/tri/github_files/sentinel2_im/2016105_0.tif'

image_path = '/home/geoint/tri/RQUNet-sup-exp/data/2019059/train/sat/2019059.hdf'
mask_true_path = '/home/geoint/tri/RQUNet-sup-exp/data/2019059/train/sat/2019059.hdf'

np.random.seed(123)
use_cuda = True
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
im_type = image_path[30:38]
print('image type: ', im_type)
segment=False
alpha = 0.0
unet_option = 'unet_vae_old' # options: 'unet_vae_old','unet_vae_RQ_scheme1' 'unet_vae_RQ_scheme3'
image_option = "noisy" # "clean" or "noisy"

##################################
def rescale(image): ## function to rescale image for visualization
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
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
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
def jpg_to_tensor(filepath=image_path):
    '''
    get 1 image at a time
    '''
    if filepath == '/home/geoint/tri/RQUNet-sup-exp/data/2019059/train/sat/2019059.hdf':
        train_size = 1 
        test_size = 0
        input_size = 256
        X = read_large_scene(filepath, train_size, test_size, input_size)
        pil = X

    else:
        img_data= tifffile.imread(filepath)
        pil = np.array(img_data)

    if im_type != "sentinel":
        pil=pil/255
    # else:
    #     pil=(pil - np.min(pil)) / (np.max(pil) - np.min(pil))

    # print(pil.shape)
    pil = np.squeeze(pil, axis=0)

    # img_flat = pil.flatten()
    # plt.hist(img_flat, 100)
    # plt.show()

    pil = rescale(pil)
    pil = normalize_image(pil)

    row,col,ch= pil.shape
    sigma = 0.05
    ## choosing sigma based on the input images, 0.1-0.3 for NAIP images, 0.002 to 0.01 for sentinel2 images
    noisy = pil + sigma*np.random.randn(row,col,ch)

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
    # pil = rescale(pil)

    pil = rescale_truncate(pil)
    return pil


def read_large_scene(filepath, train_size, test_size, input_size):
    '''
    Args:
    '''
    file = gdal.Open(filepath, gdal.GA_ReadOnly)
    subDatasets = file.GetSubDatasets()
    bands = []
    # append rgb bands
    for iband in range(1,4):   # 9
        bands.append(gdal.Open(subDatasets[iband][0]).ReadAsArray())
    img = np.asarray(bands) * 0.0001
    if np.amin(img) < 0:
        img = np.where(img < 0, 0, img)
    if np.amax(img) > 1:
        img = np.where(img > 1, 1, img)
    img = np.float32(img)
    # print(np.amin(img),np.max(img))
    # print(img.shape)
    
    img = np.transpose(img, (1,2,0))

    # print(img.shape)

    # image will have dimension (h,w,c) and don't need to reshape
    # ---------------------------------------------------------------

    h, w, c = img.shape

    I = np.random.randint(0, h-input_size, size=train_size+test_size)
    J = np.random.randint(0, w-input_size, size=train_size+test_size)
    
    X = np.array([img[i:(i+input_size), j:(j+input_size),:] for i, j in zip(I, J)])

    return X

def test(epoch, testing_data_loader, model, criterion):
    print('===> Testing # %d epoch'%(epoch))
    avg_psnr = 0
    # model.cuda()
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch['image'].to(device=device, dtype=torch.float32), \
                 batch['image'].to(device=device, dtype=torch.float32)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * math.log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.6f} dB".format(avg_psnr / len(testing_data_loader)))

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
            return output.cpu()
        elif unet_option=='unet_vae_stacked':
            output = output[1]
            return output.cpu()
        elif unet_option=='unet_vae_old':
            output = output[0]
            return img.cpu(), output.cpu()
            
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

        elif unet_option == 'cnn-denoise':
            output = output

            return img.cpu(), output.cpu()



def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument(
        '--model', '-m', default='github_checkpoints/checkpoint_unet_vae_old_epoch20_0.0_recon.pth',
         metavar='FILE',
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
    if "unet_vae" in unet_option:
        model_saved = \
            '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_vae_old_epoch50_va059_11-29_denoise.pth'
        model_sentinel_saved = \
            '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_vae_old_epoch19_sentinel_5-7_recon.pth'
    else:
        model_saved = \
            '/home/geoint/tri/github_files/github_checkpoints/checkpoint_cnn-denoise_epoch50_va059_11-29_denoise.pth'


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

    elif unet_option == 'cnn-denoise':
        net = RDN(channel = 3,growth_rate = 64,rdb_number = 3,upscale_factor=1, learning_rate=1e-4)

    
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

        x_range = np.arange(65536)
        plt.plot(s, x_range, color='blue', label = 's')
        plt.plot(Wy, x_range, color='red', label = 'Wy')
        plt.legend()
        plt.show()

        
    else:
        img, mask = predict_img(net=net,
                        filepath=image_path,
                        scale_factor=1,
                        out_threshold=0.5,
                        device=device)

    

    # out_files = 'out/predict_va_vae_recon_epoch1'
    # im_out_files = 'out/img'
    
    # if not args.no_save:
    #     out_filename = out_files
    #     logging.info(f'Mask saved to {out_filename}')

    mask = tensor_to_jpg(mask)
    img = tensor_to_jpg(img)

    # mask = rescale_truncate(mask)

    plot_img_and_mask_recon(img, mask)