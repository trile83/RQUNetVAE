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
import unet
from pathlib import Path
from skimage import exposure
import glob
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset, TensorDataset
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
import pickle
from osgeo import gdal, gdal_array

# load image folder path and image dictionary
class_name = "sentinel"
#data_dir = "F:\\NAIP\\"
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
            
            if ms_typ == 'map':
                scene_ids_map = [fl.replace(".tiff", "").replace(".tif", "") for fl in ms_img_fls]
                scene_ids = [fl[5:] for fl in scene_ids_map]
                #print(scene_ids)
            else:
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

        # label set
        Y_lst=[]
        for j in range(len(Y_fls)):
            print(Y_fls[j])
            naip_fn = Y_fls[j]
            #print(naip_fn)
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

            #img_data = img_data-1

            img_data[img_data == 1] = 3
            
            if np.max(img_data)==255:
                img_data[img_data == 255] = 3

            # if np.max(img_data)>2:
            #     img_data[img_data > 2] = 2

            img_data = img_data-2

            print(np.unique(img_data))
            print(img_data.shape)

            #im = img_data.reshape(256,256)

            #plt.plot(img_data)
            #plt.imshow()
            #plt.show()

            break

            Y_lst.append(img_data)
         
        X = np.array(X_lst)
        Y = np.array(Y_lst)

        print("Max value of Y", np.max(Y))
        print("Min value of Y", np.min(Y))

        # normalized input images
        for i in range(len(X)):
            X[i] = (X[i] - np.min(X[i])) / (np.max(X[i]) - np.min(X[i]))

        print("Max value of X", np.max(X))
        print("Min value of X", np.min(X))

        yield X, Y

class satDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, Y):
        'Initialization'
        self.data = X
        #self.targets = torch.LongTensor(Y)
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

        X = self.transforms(X)
        Y = torch.LongTensor(Y)
        return {
            'image': X,
            'mask': Y
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

    train_data_gen = data_generator(images[class_name], size=256, mode="train", batch_size=53)
    images, labels = next(train_data_gen)

    train_images = images[:45]
    train_labels = labels[:45]

    val_images = images[45:50]
    val_labels = labels[45:50]

    # 2. Split into train / validation partitions
    n_val = len(val_images)
    n_train = len(train_images)
    #train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)

    transformed_dataset = satDataset(X=train_images, Y=train_labels)
    train_loader = DataLoader(transformed_dataset, shuffle=True, **loader_args)

    #transformed_dataset_val = satDataset(X=val_images, Y=val_labels)
    #val_loader = DataLoader(transformed_dataset_val, shuffle=True, **loader_args)

    print("train loader size",len(train_loader))
    #print("val loader size",len(val_loader))

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
    decayRate = 0.96
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    #criterion  = nn.NLLLoss()
    global_step = 0

    #dummy index to provide names to output files
    save_img_ind = 0
    loss_items = {}
    loss_items['crossentropy_loss'] = []
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
                true_masks = true_masks.to(device=device, dtype=torch.long)

                print("true mask shape: ", true_masks.shape)

                images = torch.reshape(images, (batch_size,3,256,256))
                true_masks = torch.reshape(true_masks, (batch_size,256,256))

                #print("image shape: ", images.shape)
                #print("true mask shape: ", true_masks.shape)

                with torch.cuda.amp.autocast(enabled=False):

                    output = net(images)

                    if unet_option == 'unet' or unet_option == 'unet_jaxony':
                        masked_output = output
                        #print("masked output shape: ", masked_output.shape)

                        print("Max values of predicted image: ",np.max(masked_output.detach().cpu().numpy()))
                        print("Min values of predicted image: ",np.min(masked_output.detach().cpu().numpy()))

                        loss = criterion(masked_output, true_masks) 
                            #+ dice_loss(F.softmax(masked_output, dim=1).float(),F.one_hot(true_masks, net.num_classes).permute(0, 3, 1, 2).float(),multiclass=True)
                        loss_items['crossentropy_loss'].append(loss.detach().cpu())
                        print("crossentropy loss: ", loss)

                    else:
                        masked_output = output[0]

                        print("Max values of predicted image: ",np.max(masked_output.detach().cpu().numpy()))
                        print("Min values of predicted image: ",np.min(masked_output.detach().cpu().numpy()))

                        print("masked_output shape: ", masked_output.shape)
                        print("true mask shape: ", true_masks.shape)
                        
                        kl_loss = torch.sum(output[4])
                        #kl_loss = -0.5 * torch.sum(1 + output[2] - output[1].pow(2) - output[2].exp())

                        print("kl loss: ", kl_loss)
                        scaled_kl = kl_loss*0.001
                        loss_items['kl_loss'].append(scaled_kl.detach().cpu())

                        loss = criterion(masked_output, true_masks) 
                                #+ dice_loss(F.softmax(masked_output, dim=1).float(),F.one_hot(true_masks, net.num_classes).permute(0, 3, 1, 2).float(),multiclass=True) 

                        loss_items['crossentropy_loss'].append(loss.detach().cpu())
                        print("crossentropy loss: ", loss)

                        loss = scaled_kl + loss
                        loss_items['total_loss'].append(loss.detach().cpu())
                        print("total loss: ", loss)

                #optimizer.zero_grad(set_to_none=True)
                optimizer.zero_grad()
                #print('At step {}, loss is {}'.format(step, loss.data.cpu()))
                loss.backward()
                optimizer.step()

                #grad_scaler.scale(loss).backward()
                #grad_scaler.step(optimizer)
                #grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                #_, predicted = torch.max(masked_output.data, 1)
                #total_train += true_masks.nelement()
                #correct_train += predicted.eq(true_masks.data).sum().item()
                #train_accuracy = 100 * correct_train/ total_train
                #logging.info('Training accuracy: {}'.format(train_accuracy))

                #print(net.named_parameters())
                

                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            #histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            #histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        #val_score = evaluate(net, val_loader, device, unet_option)
                        #scheduler.step(val_score)

                        #logging.info('Validation loss score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            #'validation loss': val_score,
                            #'images': wandb.Image(images[0,:,:,:2].cpu()),
                            'masks': {
                                #'true': wandb.Image(true_masks[0].float().cpu()),
                                #'pred': wandb.Image(torch.softmax(masked_output, dim=1).argmax(dim=1)[0].float().cpu())
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

                
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_{model}_epoch{number}_{alpha}_batchnorm_segment.pth'.format(model=unet_option, number=epoch + 1, alpha=alpha)))
            #torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_unet_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')

    if unet_option == 'unet' or unet_option == 'unet_1':
        plt.plot(loss_items['crossentropy_loss'], 'r--')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(labels= 'crossentropy loss',loc='upper right')
        plt.show()
    else:
        plt.plot(loss_items['crossentropy_loss'], 'r--', loss_items['kl_loss'], 'b--', loss_items['total_loss'], 'g--')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(labels = ['crossentropy loss','kl loss','total loss'],loc='upper right')
        plt.show()

        #if use_cuda:
            #noise.data += sigma * torch.randn(noise.shape).cuda()
        #else:
            #noise.data += sigma * torch.randn(noise.shape)

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
    segment = False ## which means adding batchnorm layers, better for segmentation

    if unet_option == 'unet_vae_1':
        net = UNet_VAE(3)
    elif unet_option == 'unet_vae_old':
        net = UNet_VAE_old(3, segment)
    elif unet_option == 'unet_vae_RQ_old':
        net = UNet_VAE_RQ_old(3, alpha)
    elif unet_option == 'unet_vae_RQ_allskip_trainable':
        net = UNet_VAE_RQ_old_trainable(3,alpha)

    #bind the network to the gpu if cuda is enabled
    if use_cuda:
        net.cuda()

    logging.info(f'Network:\n'
                 f'\t{net.in_channels} input channels\n'
                 f'\t{net.num_classes} output channels (classes)')


    #if args.load:
    #model_checkpoint = 'checkpoints/checkpoint_unet_vae_1_epoch10_0.5_segment_dice_kl_0.001.pth'
    #net.load_state_dict(torch.load(model_checkpoint, map_location=device))
    #logging.info(f'Model loaded from {model_checkpoint}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=30,
                  batch_size=5,
                  learning_rate=1e-5,
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


