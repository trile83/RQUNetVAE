import wandb
import argparse
import logging
import sys
import math

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
import tifffile

from unet import UNet_VAE, UNet_VAE_old, UNet_VAE_update_param, UNet_VAE_RQ_old_torch
from unet import UNet_VAE_RQ_old, UNet_VAE_RQ_test, UNet_VAE_RQ_new_torch, UNet_VAE_RQ_trainable
from unet.unet_vae_RQ_scheme1 import UNet_VAE_RQ_scheme1
from cnn_denoise.rdn import RDN
from utils.dice_score import dice_loss
from evaluate import evaluate
import os
from collections import defaultdict
from osgeo import gdal, gdal_array

dir_checkpoint = Path('/home/geoint/tri/github_files/github_checkpoints/')
#use cuda, or not? be prepared for a long wait if you don't have cuda capabilities.
use_cuda = True

###########
# get data

# load image folder path and image dictionary
class_name = "va059"
data_dir = "/home/geoint/tri/"
data_dir = os.path.join(data_dir, class_name)

# Create training data 
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
            ms_img_fls = [fl for fl in ms_img_fls if fl.endswith(".tiff") or fl.endswith(".TIF")]
            scene_ids = [fl.replace(".tiff", "").replace(".TIF", "") for fl in ms_img_fls]
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
        
        X_lst=[]
        for j in range(len(X_fls)):
            img_data = tifffile.imread(X_fls[j])
                
            if img_data.shape == (256,256,3):
                X_lst.append(img_data)

        X = np.array(X_lst)
        X = X/255

        X_noise = []

        row,col,ch= X[0].shape
        sigma = 0.08
        for img in X:
            noisy = img + sigma*np.random.randn(row,col,ch)
            X_noise.append(noisy)
        X_noise = np.array(X_noise)
        yield X_noise, X

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
            'image': X,
            'mask': Y
        }

# Normalize bands into 0.0 - 1.0 scale
def normalize_image(image):
    '''
    Arg: Input is an image with dimension (channel, height, width)
    '''

    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    return image


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
    print(np.amin(img),np.max(img))
    print(img.shape)
    
    img = np.transpose(img, (1,2,0))

    print(img.shape)
    # image will have dimension (h,w,c) and don't need to reshape
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
    input_size = 256

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
    # images = {}
    # load_image_paths(data_dir, class_name, 'train', images)

    # train_data_gen = data_generator(images[class_name], size=256, mode="train", batch_size=130)
    # images, labels = next(train_data_gen)

    # train_images = images[:100]
    # train_labels = labels[:100]

    # val_images = images[100:105]
    # val_labels = labels[100:105]

    train_size = 100 
    test_size = 10 
    input_size = 256

    filepath = '/home/geoint/tri/RQUNet-sup-exp/data/2019059/train/sat/2019059.hdf'
    X = read_large_scene(filepath, train_size, test_size, input_size)

    x_train = X[:train_size]
    x_test = X[train_size:]

    train_images = x_train
    train_labels = x_train

    val_images = x_test
    val_labels = x_test

    # 2. Split into train / validation partitions
    n_val = len(val_images)
    n_train = len(train_images)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)

    transformed_dataset = satDataset(X=train_images, Y=train_labels)
    train_loader = DataLoader(transformed_dataset, shuffle=False, **loader_args)

    transformed_dataset_val = satDataset(X=val_images, Y=val_labels)
    val_loader = DataLoader(transformed_dataset_val, shuffle=False, **loader_args)

    print("train loader size",len(train_loader))
    print("val loader size",len(val_loader))

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

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
    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.MSELoss()
    global_step = 0


    criterion = net.criterion
    optimizer = net.optimizer
    scheduler = net.scheduler

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

                images = torch.reshape(images, (batch_size,3,256,256))
                true_masks = torch.reshape(true_masks, (batch_size,3,256,256))

                print("image shape: ", images.shape)
                #print("true mask shape: ", true_masks.shape)

                #with torch.cuda.amp.autocast(enabled=True):

                output = net(images)

                if unet_option == 'unet' or unet_option == 'unet_jaxony':
                    masked_output = output
                    kl_loss = torch.zeros((1)).cuda()

                elif unet_option == 'simple_unet':
                    masked_output = output
                    kl_loss = torch.zeros((1)).cuda()


                elif unet_option == 'cnn-denoise':
                    output = output

                    loss = criterion(output, true_masks)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_items['total_loss'].append(loss.detach().cpu())

                else:
                    masked_output = output[0]
                    #masked_output = output

                    print("masked_output shape: ", masked_output.shape)
                    print("true mask shape: ", true_masks.shape)

                    mu = output[1]
                    logvar = output[2]
                
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                    #kl_loss = torch.sum(output[4])
                    print("kl loss: ", kl_loss)
                    loss_items['kl_loss'].append(kl_loss.detach().cpu())

                    #recon_loss = criterion(masked_output, true_masks)
                    recon_loss = torch.sum((masked_output-true_masks)**2)/(256*256)
                    loss_items['recon_loss'].append(recon_loss.detach().cpu())
                    print("reconstruction loss: ", recon_loss)

                    scaled_kl_loss = kl_loss * 1
                    #loss = recon_loss
                    net.tau += 1
                    print('net tau: ', net.tau)
                    loss = recon_loss + scaled_kl_loss
                    loss_items['total_loss'].append(loss.detach().cpu())
                    print("total loss: ", loss)

                    optimizer.zero_grad()
                    #loss.backward(retain_graph=True)
                    loss.backward()

                    #nn.utils.clip_grad_norm_(net.parameters(), 1)
                    optimizer.step()

                # grad_scaler.scale(loss).backward()
                # grad_scaler.step(optimizer)
                # grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                
            # Validation
            valid_loss = 0.0
            test(epoch, val_loader, net, criterion)
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
                output_val = net(images_val)

                if unet_option == 'unet' or unet_option == 'unet_jaxony':
                    output_val_segment = output_val
                    val_loss = torch.sum((output_val-true_masks_val)**2)/(256*256)
                    kl_loss_val = torch.zeros((1)).cuda()
                elif unet_option == 'simple_unet':
                    output_val = output_val
                    val_loss = torch.sum((output_val-true_masks_val)**2)/(256*256)
                    kl_loss_val = torch.zeros((1)).cuda()

                elif unet_option == 'cnn-denoise':
                    output_val = output_val

                    val_loss = criterion(output_val, true_masks_val)
                    optimizer.zero_grad()

                    # Calculate Loss
                    valid_loss += val_loss.item()

                else:
                    output_val_segment = output_val[0]

                    mu_val = output_val[1]
                    logvar_val = output_val[2]
                    kl_loss_val = -0.5 * torch.sum(1 + logvar_val - mu_val.pow(2) - logvar_val.exp())

                    scaled_kl_val = kl_loss_val*1
                
                    # Find the Loss
                    recon_loss_val = criterion(output_val_segment, true_masks_val)
                    val_loss = recon_loss_val + scaled_kl_val
                    #loss = recon_loss_val
                    # Calculate Loss
                    valid_loss += val_loss.item()

            print(f'Epoch {epoch+1} \t\t Training Loss: {epoch_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(val_loader)}')
                
            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss
                
                # print("valid_loss: ", valid_loss)
                # Saving State Dict
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_{model}_epoch{number}_va059_11-27_denoise.pth'.format(model=unet_option, number=epoch + 1, alpha=alpha)))

    plt.plot(loss_items['total_loss'])
    # plt.plot(loss_items['recon_loss'], 'r--', loss_items['kl_loss'], 'b--', loss_items['total_loss'], 'g')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.legend(labels = ['reconstruction loss','kl loss','total loss'],loc='upper right')
    plt.legend(labels = ['total loss'],loc='upper right')
    plt.show()
        

if __name__ == '__main__':
    #args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    
    alpha = 0.3
    segment = False
    class_num = 3
    unet_option = "cnn-denoise"

    if unet_option == 'unet_vae_1':
        net = UNet_VAE(class_num)
    elif unet_option == 'unet_vae_old':
        net = UNet_VAE_old(class_num, segment)
    elif unet_option == 'unet_vae_RQ_old':
        net = UNet_VAE_RQ_old(class_num, alpha)
    elif unet_option == 'unet_vae_RQ_trainable':
        net = UNet_VAE_RQ_trainable(class_num, segment, alpha)
    elif unet_option == 'unet_vae_RQ_update_param':
        net = UNet_VAE_update_param(class_num, segment, alpha)
    elif unet_option == 'unet_vae_RQ_scheme1':
        net = UNet_VAE_RQ_scheme1(class_num, segment, alpha)
    elif unet_option == 'unet_vae_RQ_torch':
        net = UNet_VAE_RQ_old_torch(class_num, segment, alpha)

    elif unet_option == 'cnn-denoise':
        net = RDN(channel = 3,growth_rate = 64,rdb_number = 3,upscale_factor=1)

    ### check parameters
    # for name, param in net.named_parameters():
    #     if name == 'tau':
    #         print(name)
    #         print(param)

        
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
                  epochs=50,
                  batch_size=5,
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