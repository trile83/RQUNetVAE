import argparse
import logging
import os
import rasterio as rio
from skimage import exposure

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision
from osgeo import gdal, gdal_array
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset

from unet import UNet_VAE
from unet import UNet_VAE_old, UNet_VAE_RQ_old, UNet_VAE_RQ_test, UNet_VAE_RQ_old_trainable, UNet_VAE_RQ_old_torch
from unet import UNet_VAE_RQ_new_torch, UNet_VAE_RQ_scheme3
from unet import UNet_VAE_RQ_scheme1
from utils.utils import plot_img_and_mask, plot_img_and_mask_3, plot_img_and_mask_recon

image_path = '/home/geoint/tri/github_files/sentinel2_im/2016002_0.tif'
mask_true_path = '/home/geoint/tri/github_files/sentinel2_im/2016002_0.tif'

use_cuda = True
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

im_type = image_path[30:38]
#print(im_type)
segment=False
alpha = 0.2
unet_option = 'unet_vae_RQ_scheme1' # options: 'unet_vae_old', 'unet_vae_RQ_old', 'unet_vae_RQ_allskip_trainable', 'unet_vae_RQ_torch', 'unet_vae_RQ_scheme3'
image_option = "noisy" # "clean" or "noisy"

##################################
def rescale(image):
    map_img =  np.zeros((256,256,3))
    for band in range(3):
        p2, p98 = np.percentile(image[:,:,band], (2, 98))
        map_img[:,:,band] = exposure.rescale_intensity(image[:,:,band], in_range=(p2, p98))
    return map_img

def rescale_truncate(image):
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
    if im_type != "sentinel":
        pil=pil/255

    row,col,ch= pil.shape
    sigma = 0.1
    noisy = pil + sigma*np.random.randn(row,col,ch)


    #pil_to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    transform_tensor = transforms.ToTensor()
    if use_cuda:
        noisy_tensor = transform_tensor(noisy).cuda()
        tensor = transform_tensor(pil).cuda()

    return tensor.view([1]+list(tensor.shape)), noisy_tensor.view([1]+list(noisy_tensor.shape))

#accept a torch tensor, convert it to a jpg at a certain path
def tensor_to_jpg(tensor):
    #tensor = tensor.view(tensor.shape[1:])
    tensor = tensor.squeeze(0)
    if use_cuda:
        tensor = tensor.cpu()
    #tensor_to_pil = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
    #pil = tensor_to_pil(tensor)
    pil = tensor.permute(1, 2, 0).numpy()
    pil = np.array(pil)
    pil = rescale_truncate(pil)
    return pil

#########################
# get data
## Load data
import os
from collections import defaultdict
import pickle
from osgeo import gdal, gdal_array

# load image folder path and image dictionary
class_name = "sentinel2_xiqi" ## or va059
data_dir = "/home/geoint/tri/"
data_dir = os.path.join(data_dir, class_name)

# Create training data 
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def load_image_paths(path, name, mode, images):
    images[name] = {mode: defaultdict(dict)}
    # test, train, valid
    ttv = os.listdir(path)
    
    for ttv_typ in ttv: 
        typ_path = os.path.join(path, ttv_typ) # typ_path = ../train/
        ms = os.listdir(typ_path)
        
        for ms_typ in ms: # ms_typ is either 'sat' or 'map'
            ms_path = os.path.join(typ_path, ms_typ)
            ms_img_fls = os.listdir(ms_path) # list all file path
            ms_img_fls = [fl for fl in ms_img_fls if fl.endswith(".tiff") or fl.endswith(".tif")]
            scene_ids = [fl.replace(".tiff", "").replace(".tif", "") for fl in ms_img_fls]
            ms_img_fls = [os.path.join(ms_path, fl) for fl in ms_img_fls]           
            # Record each scene
            
            for fl, scene_id in zip(ms_img_fls, scene_ids):
                if ms_typ == 'map':
                    images[name][ttv_typ][ms_typ][scene_id] = fl

                elif ms_typ == "sat":
                    images[name][ttv_typ][ms_typ][scene_id] = fl
                 
def data_generator(files, size=256, mode="train", batch_size=6):
    while True:
        all_scenes = list(files[mode]['sat'].keys())
        
        # Randomly choose scenes to use for data
        scene_ids = np.random.choice(all_scenes, size=batch_size, replace=True)
        
        X_fls = [files[mode]['sat'][scene_id] for scene_id in scene_ids]
        Y_fls = [files[mode]['map'][scene_id] for scene_id in scene_ids]        
        
        #print(Y_fls)
        # read in image to classify with gdal
        X_lst=[]
        for j in range(len(X_fls)):
            naip_fn = X_fls[j]
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
                
            if img_data.shape == (256,256,3):
                X_lst.append(img_data)


        #X = np.array(X_lst).astype(np.float32)
        X = np.array(X_lst)
        if im_type != "sentinel":
            X = X/255
            for i in range(len(X)):
                X[i] = (X[i] - np.min(X[i])) / (np.max(X[i]) - np.min(X[i]))

        X_noise = []

        if image_option == 'noisy':
            row,col,ch= X[0].shape
            sigma = 0.01
            for img in X:
                noisy = img + sigma*np.random.randn(row,col,ch)
                X_noise.append(noisy)

            X_noise = np.array(X_noise)

            yield X_noise, X

        else:

            yield X, X

class satDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, Y):
        'Initialization'
        self.data = X
        self.targets = Y
        self.transforms = transforms.ToTensor()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.data[index]
        Y = self.targets[index]
        
        #X = Image.fromarray(self.data[index].astype(np.uint8))
        X = self.transforms(X)
        Y = self.transforms(Y)
        #Y = label
        return {
            #'image': torch.as_tensor(X.copy()).float(),
            'image': X,
            'mask': X
            #'mask': torch.as_tensor(X.copy()).float()
        }

#####################

#predict image
def predict_img(net,
                filepath,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    #img = img.unsqueeze(0)

    ### get data
    images = {}
    load_image_paths(data_dir, class_name, 'train', images)
    train_data_gen = data_generator(images[class_name], size=256, mode="train", batch_size=10)

    images, labels = next(train_data_gen)

    images = images[:5]
    labels = labels[:5]

    # 2. Split into train / validation partitions
    n_pred = len(images)

    print("total input: ", n_pred)

    # 3. Create data loaders
    loader_args = dict(batch_size=n_pred, num_workers=4, pin_memory=True)

    sat_dataset = satDataset(X=images, Y=labels)
    im_loader = DataLoader(sat_dataset, shuffle=True, **loader_args)

    #if image_option=='clean':
        #img = jpg_to_tensor(filepath)[0] ## clean image
    #elif image_option=='noisy':
        #img = jpg_to_tensor(filepath)[1] ## noisy image

    #img = img.to(device=device, dtype=torch.float32)

    for batch in im_loader:

        images = batch['image']
        true_masks = batch['mask']

        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.float32)

        #print("true mask shape: ", true_masks.shape)

        images = torch.reshape(images, (5,3,256,256))
        true_masks = torch.reshape(true_masks, (5,3,256,256))

        print("image shape: ", images.shape)
        #print("true mask shape: ", true_masks.shape)

        with torch.no_grad():
            output = net(images)

            if unet_option == 'unet':
                output = output
            else:
                output = output[3]

            print(output.shape)

        # if  output.detach().cpu().numpy().all() == 0:
        #     print("output is zero")
        #     print(output.cpu())

        #full_mask = output.cpu()
        #full_mask = full_mask.reshape(256,256,3)
        #print(full_mask.shape)

        # tf = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((256, 256)),
        #     transforms.ToTensor()
        # ])

        #full_mask = tf(output.cpu()).squeeze()
        #print(full_mask.shape)

    return output.cpu(), images.cpu()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_vae_old_epoch10_0.0_recon.pth', metavar='FILE',
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
    #in_files = args.input
    #out_files = get_output_filenames(args)

    if unet_option == 'unet_vae_1':
        net = UNet_VAE(3)
    elif unet_option == 'unet_vae_old':
        net = UNet_VAE_old(3)
    
    elif unet_option == 'unet_vae_RQ_old':
        net = UNet_VAE_RQ_old(3, alpha)
    
    elif unet_option == 'unet_vae_RQ_allskip_trainable':
        net = UNet_VAE_RQ_old_trainable(3,alpha)

    elif unet_option == 'unet_vae_RQ_torch':
        #net = UNet_VAE_RQ_old_torch(3, alpha = alpha)
        net = UNet_VAE_RQ_new_torch(3, segment, alpha)

    elif unet_option == 'unet_vae_RQ_scheme3':
        net = UNet_VAE_RQ_scheme3(3, segment, alpha)
    elif unet_option == 'unet_vae_RQ_scheme1':
        net = UNet_VAE_RQ_scheme1(3, segment, alpha)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    #for i, filename in enumerate(in_files):
    logging.info(f'\nPredicting image {image_path} ...')
    #img = Image.open(filename)

    preds, imgs = predict_img(net=net,
                        filepath=image_path,
                        scale_factor=1,
                        out_threshold=0.5,
                        device=device)


    #out_files = 'out/predict_va_softshrink_all_0.02.tif'
    out_files = 'out/predict_va_vae_recon_epoch1'
    im_out_files = 'out/img'
    
    if not args.no_save:
        out_filename = out_files
        #result = mask_to_image(mask)
        #arr_to_tif(raster_f=image_path, segments=mask, out_tif=out_files)
        #result.save(out_filename)
        logging.info(f'Mask saved to {out_filename}')

    for i in range(preds.size(0)):
        pred = tensor_to_jpg(preds[i])
        #print(mask)
        # if image_option=='clean':
        #     img = jpg_to_tensor(image_path)[0]
        # else:
        #     img = jpg_to_tensor(image_path)[1]
        img = tensor_to_jpg(imgs[i])

        plot_img_and_mask_recon(img, pred)

    if args.viz:
        logging.info(f'Visualizing results for image {image_path}, close to continue...')
        #plot_img_and_mask(img, mask)