import wandb
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
import torch.nn.functional as F
from tqdm import tqdm
import tifffile
from sklearn.utils.class_weight import compute_class_weight

from unet import UNet_VAE, UNet_VAE_old, UNet_test
from unet import UNet_VAE_RQ_old, UNet_VAE_RQ_test, UNet_VAE_RQ_new_torch
from utils.dice_score import dice_loss
from evaluate import evaluate

dir_checkpoint = Path('/home/geoint/tri/github_files/github_checkpoints/')
#use cuda, or not? be prepared for a long wait if you don't have cuda capabilities.
use_cuda = True

##################################
def rescale(image):
    map_img =  np.zeros(image.shape)
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

    #image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = image/np.max(image)

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

class satDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, image_paths, target_paths, train=True):   # initial logic happens like transform

        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transforms = transforms.ToTensor()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image = tifffile.imread(self.image_paths[index])
        mask = tifffile.imread(self.target_paths[index])
        image = np.asarray(image)
        mask = np.asarray(mask)
        # image = rasterio.open(self.image_paths[index]).read()
        # image = rescale(image)
        # image = standardize_image(image)
        # mask = rasterio.open(self.target_paths[index]).read(1)

        mask[mask==2]=1
        mask[mask==3]=1
        mask[mask==4]=2

        mask = mask-1

        t_image = self.transforms(image)
        #t_mask = self.transforms(mask)
        t_mask = torch.LongTensor(mask)
        # print(torch.max(t_image))
        # print(torch.unique(t_mask))
        return {
            'image': t_image,
            'mask': t_mask
            # 'weight': class_weights
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

    ## test
     # get all the image and mask path and number of images
    folder_data = glob.glob("/home/geoint/tri/sentinel/train/sat/*.tif")
    folder_mask = glob.glob("/home/geoint/tri/sentinel/train/map/*.tif")

    # folder_data = folder_data[:50]
    # folder_data = folder_mask[:50]

    # split these path using a certain percentage
    len_data = len(folder_data)
    print('number of images: ', len_data)
    print('number of masks: ', len(folder_mask))
    train_size = (1-val_percent)

    train_image_paths = folder_data[:int(len_data*train_size)]
    test_image_paths = folder_data[int(len_data*train_size):]

    train_mask_paths = folder_mask[:int(len_data*train_size)]
    test_mask_paths = folder_mask[int(len_data*train_size):]

    n_train = len(train_image_paths)
    n_val = len(test_image_paths)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)

    transformed_dataset = satDataset(image_paths=train_image_paths, target_paths=train_mask_paths, train=True)
    train_loader = DataLoader(transformed_dataset, shuffle=True, **loader_args)

    transformed_dataset_val = satDataset(image_paths=test_image_paths, target_paths=test_mask_paths, train=False)
    val_loader = DataLoader(transformed_dataset_val, shuffle=True, **loader_args)

    print("train loader size",len(train_loader))
    print("val loader size",len(val_loader))

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

    weights = [0.2,1.0]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
    # criterion  = nn.NLLLoss()
    global_step = 0

    #dummy index to provide names to output files
    save_img_ind = 0
    loss_items = {}
    loss_items['crossentropy_loss'] = []
    loss_items['kl_loss'] = []
    loss_items['total_loss'] = []
    loss_items['val_loss'] = []
    
    min_valid_loss = np.inf
    for epoch in range(epochs):
        #get the network output
        net.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:

                images = batch['image']
                true_masks = batch['mask']
                # class_weights = batch['weight']

                # criterion = nn.CrossEntropyLoss(weight=class_weights,reduction='mean')

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                #print("true mask shape: ", true_masks.shape)

                images = torch.reshape(images, (batch_size,3,256,256))
                true_masks = torch.reshape(true_masks, (batch_size,256,256))

                print("image shape: ", images.shape)
                #print("true mask shape: ", true_masks.shape)

                with torch.cuda.amp.autocast(enabled=False):

                    output = net(images)

                    if unet_option == 'unet' or unet_option == 'unet_jaxony':
                        masked_output = output
                        # masked_output = F.log_softmax(output, dim=1)
                        kl_loss = torch.zeros((1)).cuda()

                    elif unet_option == 'simple_unet':
                        masked_output = output
                        kl_loss = torch.zeros((1)).cuda()
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

                    crossentropy_loss = criterion(masked_output, true_masks)
                    loss_items['crossentropy_loss'].append(crossentropy_loss.detach().cpu())
                    print("crossentropy loss: ", crossentropy_loss)

                    scaled_kl_loss = kl_loss * 0.01
                    #loss = recon_loss
                    loss = crossentropy_loss + scaled_kl_loss
                    loss_items['total_loss'].append(loss.detach().cpu())
                    print("total loss: ", loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
                true_masks_val = true_masks_val.to(device=device, dtype=torch.long)

                #print("true mask shape: ", true_masks.shape)

                images_val = torch.reshape(images_val, (batch_size,3,256,256))
                true_masks_val = torch.reshape(true_masks_val, (batch_size,256,256))

                # Forward Pass
                output_val = net(images_val)

                if unet_option == 'unet' or unet_option == 'unet_jaxony':
                    output_val_segment = output_val
                    # output_val_segment = F.log_softmax(output_val, dim=1)
                    loss = criterion(output_val_segment, true_masks_val)
                    kl_loss_val = torch.zeros((1)).cuda()
                elif unet_option == 'simple_unet':
                    output_val_segment = output_val
                    loss = criterion(output_val_segment, true_masks_val)
                    kl_loss_val = torch.zeros((1)).cuda()
                else:
                    output_val_segment = output_val[0]

                    mu_val = output_val[1]
                    logvar_val = output_val[2]
                    kl_loss_val = -0.5 * torch.sum(1 + logvar_val - mu_val.pow(2) - logvar_val.exp())

                scaled_kl_val = kl_loss_val*0.01
            
                # Find the Loss
                crossentropy_loss_val = criterion(output_val_segment, true_masks_val)
                val_loss = crossentropy_loss_val + scaled_kl_val

                loss_items['val_loss'].append(val_loss.detach().cpu())
                # Calculate Loss
                valid_loss += val_loss.item()

            print(f'Epoch {epoch+1} \t\t Training Loss: {epoch_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(val_loader)}')
                
            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss
                
                # print("valid_loss: ", valid_loss)
                # Saving State Dict
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_{model}_epoch{number}_7-6_segment_sentinel.pth'.format(model=unet_option, number=epoch + 1, alpha=alpha)))
                           

    #plt.plot(loss_items['total_loss'])
    plt.plot(loss_items['crossentropy_loss'],'r--',loss_items['kl_loss'],'b--',loss_items['total_loss'],'g',loss_items['val_loss'], 'c')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(labels = ['crossentropy loss','kl loss','total loss','val loss'],loc='upper right')
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
    unet_option = "unet_jaxony"
    segment = True
    class_num = 2

    if unet_option == 'unet_vae_1':
        net = UNet_VAE(class_num)
    elif unet_option == 'unet_jaxony':
        net = UNet_test(class_num)
    elif unet_option == 'unet_vae_old':
        net = UNet_VAE_old(class_num, segment)
    elif unet_option == 'unet_vae_RQ_old':
        net = UNet_VAE_RQ_old(class_num, alpha)

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
                  epochs=40,
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