import wandb
import argparse
import logging
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.flatten import Unflatten
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from torch.nn import init
from pathlib import Path
from skimage import exposure
import glob
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from torch import optim
from tqdm import tqdm

from unet import UNet_VAE, UNet_VAE_old
from unet import UNet_VAE_RQ_old, UNet_VAE_RQ_test, UNet_VAE_RQ_new_torch, UNet_VAE_RQ_old_trainable
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate

dir_checkpoint = Path('/home/geoint/tri/github_files/github_checkpoints/')
#use cuda, or not? be prepared for a long wait if you don't have cuda capabilities.
use_cuda = True

##################################
def rescale(image):
    map_img =  np.zeros((256,256,3))
    for band in range(3):
        p2, p98 = np.percentile(image[:,:,band], (2, 98))
        map_img[:,:,band] = exposure.rescale_intensity(image[:,:,band], in_range=(p2, p98))
    return map_img

###########
# get data
## Load data
import os
from collections import defaultdict
from osgeo import gdal, gdal_array

# load image folder path and image dictionary
class_name = "sentinel2_xiqi"
data_dir = "/home/geoint/tri/"
data_dir = os.path.join(data_dir, class_name)

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

        X_noise = []

        # for image in X:
        #     ## add gaussian noise
        #     row,col,ch= image.shape
        #     mean = 0
        #     var = 0.1
        #     sigma = var**0.5
        #     gauss = np.random.normal(mean,sigma,(row,col,ch))
        #     gauss = gauss.reshape(row,col,ch)
        #     noisy = image + gauss
        #     X_noise.append(noisy)

        # X_noise = np.array(X_noise)

        for i in range(len(X)):
            X[i] = (X[i] - np.min(X[i])) / (np.max(X[i]) - np.min(X[i]))

        print('max X[0]: ', np.max(X[0]))
        print('min X[0]: ', np.min(X[0]))

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

def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):

    ### get data
    images = {}
    load_image_paths(data_dir, class_name, 'train', images)
    #im_dict = load_obj("images_dict")

    #print(images[class_name]['train'])

    train_data_gen = data_generator(images[class_name], size=256, mode="train", batch_size=70)
    images, labels = next(train_data_gen)

    train_images = images[:60]
    train_labels = labels[:60]

    val_images = images[60:70]
    val_labels = labels[60:70]

    # 2. Split into train / validation partitions
    n_val = len(val_images)
    n_train = len(train_images)
    #train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)

    transformed_dataset = satDataset(X=train_images, Y=train_labels)
    train_loader = DataLoader(transformed_dataset, shuffle=True, **loader_args)

    transformed_dataset_val = satDataset(X=val_images, Y=val_labels)
    val_loader = DataLoader(transformed_dataset_val, shuffle=True, **loader_args)

    print("train loader size",len(train_loader))
    print("val loader size",len(val_loader))

    # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    # experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #                               val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
    #                               amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    #network optimizer set up
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.MSELoss()
    global_step = 0

    #dummy index to provide names to output files
    save_img_ind = 0
    loss_items = {}
    loss_items['recon_loss'] = []
    loss_items['kl_loss'] = []
    loss_items['total_loss'] = []
    
    min_valid_loss = np.inf
    for epoch in range(epochs):
        #get the network output
        net.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:

                images = batch['image']
                true_masks = batch['mask']

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                #print("true mask shape: ", true_masks.shape)

                images = torch.reshape(images, (batch_size,3,256,256))
                true_masks = torch.reshape(true_masks, (batch_size,3,256,256))

                print("image shape: ", images.shape)
                #print("true mask shape: ", true_masks.shape)

                with torch.cuda.amp.autocast(enabled=False):

                    output = net(images)

                    if unet_option == 'unet' or unet_option == 'unet_1':
                        masked_output = output

                    elif unet_option == 'simple_unet':
                        masked_output = output
                    else:
                        masked_output = output[3]
                    #masked_output = output

                    print("masked_output shape: ", masked_output.shape)
                    print("true mask shape: ", true_masks.shape)

                    mu = output[1]
                    logvar = output[2]
                    
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    #kl_loss = torch.sum(output[4])
                    print("kl loss: ", kl_loss)
                    loss_items['kl_loss'].append(kl_loss.detach().cpu())

                    #recon_loss = torch.sum((masked_output - true_masks)**2)
                    recon_loss = criterion(masked_output, true_masks)
                    loss_items['recon_loss'].append(recon_loss.detach().cpu())
                    print("reconstruction loss: ", recon_loss)


                    #loss = recon_loss
                    loss = recon_loss + kl_loss
                    loss_items['total_loss'].append(loss.detach().cpu())
                    print("total loss: ", loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #print('At step {}, loss is {}'.format(step, loss.data.cpu()))

                #for p in net.parameters():
                    #print(p)

                #grad_scaler.scale(loss).backward()
                #grad_scaler.step(optimizer)
                #grad_scaler.update()

                # pbar.update(images.shape[0])
                # global_step += 1
                # epoch_loss += loss.item()
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                # pbar.set_postfix(**{'loss (batch)': loss.item()})

                
            # Validation
            valid_loss = 0.0
            net.eval()     # Optional when not using Model Specific layer
            for batch_val in val_loader:
                # Transfer Data to GPU if available
                images_val = batch_val['image']
                true_masks_val = batch_val['mask']

                images_val = images_val.to(device=device, dtype=torch.float32)
                true_masks_val = true_masks_val.to(device=device, dtype=torch.float32)

                #print("true mask shape: ", true_masks.shape)

                images_val = torch.reshape(images_val, (batch_size,3,256,256))
                true_masks_val = torch.reshape(true_masks_val, (batch_size,3,256,256))

                # Forward Pass
                output_val = net(images)

                if unet_option == 'unet' or unet_option == 'unet_1':
                    output_val_recon = output_val
                    loss = criterion(output_val_recon, true_masks_val)
                elif unet_option == 'simple_unet':
                    output_val_recon = output_val
                    loss = criterion(output_val_recon, true_masks_val)
                else:
                    output_val_recon = output_val[3]

                mu = output_val[1]
                logvar = output_val[2]
                kl_loss_val = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
                # Find the Loss
                recon_loss_val = criterion(output_val_recon, true_masks_val)
                loss_val = recon_loss_val + kl_loss_val
                # Calculate Loss
                valid_loss += loss_val.item()

            print(f'Epoch {epoch+1} \t\t Training Loss: {epoch_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(val_loader)}')
            
            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss
                
                # print("valid_loss: ", valid_loss)
                # Saving State Dict
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_{model}_epoch{number}_sentinel_5-7_recon.pth'.format(model=unet_option, number=epoch + 1, alpha=alpha)))
                    

    #plt.plot(loss_items['total_loss'])
    plt.plot(loss_items['recon_loss'], 'r--', loss_items['kl_loss'], 'b--', loss_items['total_loss'], 'g')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(labels = ['reconstruction loss','kl loss','total loss'],loc='upper right')
    plt.show()

if __name__ == '__main__':
    #args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    
    alpha = 0.0
    unet_option = "unet_vae_old"
    segment = False

    if unet_option == 'unet_vae_1':
        net = UNet_VAE(3)
    elif unet_option == 'unet_vae_old':
        net = UNet_VAE_old(3, segment)
    elif unet_option == 'unet_vae_RQ_old':
        net = UNet_VAE_RQ_old(3, alpha)
    elif unet_option == 'unet_vae_RQ_allskip_trainable':
        net = UNet_VAE_RQ_old_trainable(3,alpha)

    ### check parameters
    #for name, param in net.named_parameters():
        #print(name)

    #bind the network to the gpu if cuda is enabled
    if use_cuda:
        net.cuda()

    logging.info(f'Network:\n'
                 f'\t{net.in_channels} input channels\n'
                 f'\t{net.num_classes} output channels (classes)')

    '''
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')
    '''

    try:
        train_net(net=net,
                  epochs=20,
                  batch_size=1,
                  learning_rate=1e-4,
                  device=device,
                  img_scale=1,
                  val_percent=10/100,
                  amp=True)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)

    
    #clean up any mess we're leaving on the gpu
    if use_cuda:
        torch.cuda.empty_cache()