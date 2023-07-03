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
from utils.utils import AverageMeter

from unet import UNet_VAE, UNet_VAE_old
from unet import UNet_VAE_RQ_old, UNet_VAE_RQ_test, UNet_VAE_RQ_new_torch
from utils.dice_score import dice_loss
from evaluate import evaluate

dir_checkpoint = Path('/home/geoint/tri/github_files/github_checkpoints/')
#use cuda, or not? be prepared for a long wait if you don't have cuda capabilities.
use_cuda = True


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


# Normalize bands into 0.0 - 1.0 scale
def normalize_image(image):
    '''
    Arg: Input is an image with dimension (channel, height, width)
    '''

    # for i in range(image.shape[0]):
    #         image[i, :, :] = (image[i, :, :] - np.min(image[i, :, :])) / (np.max(image[i, :, :]) - np.min(image[i, :, :]))

    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    #image = image/255

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

    ## test
    # get all the image and mask path and number of images

    arr = tifffile.imread('/home/geoint/PycharmProjects/tensorflow/out_hls/HLS.S30.T28PEV.2021004T112451.v2.0.tif')
    # normalization
    #img = np.asarray(arr[1:4,:,:]) * 0.0001   
    img = np.asarray(arr[:,:,:])

    print(arr.shape)

    # if the image is (C,H,W)
    # img = img.reshape((img.shape[2],img.shape[1],img.shape[0]))
    # img = np.transpose(img, (1,2,0))

    # print(img.shape)
    # image will have dimension (h,w,c) and don't need to reshape

    # img = standardize_image(img/10000)
    # img = normalize_image(img)

    # img = rescale_image(img)

    ## use RGB image
    img = img[:,:,1:4]

    ## normalize based on scale
    img = img * 0.0001

    ## standardize min max
    # img=(img - np.min(img)) / (np.max(img) - np.min(img))
    # img = rescale_image(img)

    print('max pixel ', np.max(img))
    print('min pixel ', np.min(img))

    # ---------------------------------------------------------------

    h, w, c = img.shape

    train_size = 200 
    test_size = 10 
    input_size = 256

    I = np.random.randint(0, h-input_size, size=train_size+test_size)
    J = np.random.randint(0, w-input_size, size=train_size+test_size)
    
    X = np.array([img[i:(i+input_size), j:(j+input_size),:] for i, j in zip(I, J)])

    # X = rescale_image(X)
    for i in range(len(X)):
        # X[i] = rescale_image(X[i])
        X[i]=(X[i] - np.min(X[i])) / (np.max(X[i]) - np.min(X[i]))

    print("X shape: ",X.shape)
    print("max X rescale: ", np.max(X))
    print("min X rescale: ", np.min(X))

    x_train = X[:train_size]
    x_test = X[train_size:]


    train_images = x_train
    train_labels = x_train

    val_images = x_test
    val_labels = x_test

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
    loss_items['train_loss'] = []

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

                # print(images.shape)

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                images = torch.reshape(images, (batch_size,images.shape[1],images.shape[2],images.shape[3]))
                true_masks = torch.reshape(true_masks, (batch_size,images.shape[1],images.shape[2],images.shape[3]))

                with torch.cuda.amp.autocast(enabled=False):

                    output = net(images)

                    if unet_option == 'unet' or unet_option == 'unet_1':
                        masked_output = output
                        recon_loss = criterion(masked_output, true_masks)
                        loss = recon_loss
                        loss_items['train_loss'].append(loss.detach().cpu())
                        epoch_loss+=loss.item()
                        print("total loss: ", loss)

                    elif unet_option == 'simple_unet':
                        masked_output = output
                        recon_loss = criterion(masked_output, true_masks)
                        loss = recon_loss
                        loss_items['train_loss'].append(loss.detach().cpu())
                        epoch_loss+=loss.item()
                        print("train loss: ", loss)
                        
                    else:
                        masked_output = output[3]
                        #masked_output = output

                        # print("masked_output shape: ", masked_output.shape)
                        # print("true mask shape: ", true_masks.shape)

                        mu = output[1]
                        logvar = output[2]
                        
                        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        #kl_loss = torch.sum(output[4])
                        # print("kl loss: ", kl_loss)
                        loss_items['kl_loss'].append(kl_loss.detach().cpu())

                        #recon_loss = torch.sum((masked_output - true_masks)**2)
                        recon_loss = criterion(masked_output, true_masks)
                        loss_items['recon_loss'].append(recon_loss.detach().cpu())
                        # print("reconstruction loss: ", recon_loss)

                        #loss = recon_loss
                        loss = recon_loss + kl_loss
                        loss_items['train_loss'].append(loss.detach().cpu())
                        epoch_loss+=loss.item()
                        # print("total loss: ", loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                
            # Validation
            # valid_loss = 0.0
            # net.eval()     # Optional when not using Model Specific layer
            # for batch_val in val_loader:
            #     # Transfer Data to GPU if available
            #     images_val = batch_val['image']
            #     true_masks_val = batch_val['mask']

            #     images_val = images_val.to(device=device, dtype=torch.float32)
            #     true_masks_val = true_masks_val.to(device=device, dtype=torch.float32)

            #     #print("true mask shape: ", true_masks.shape)

            #     images_val = torch.reshape(images_val, (batch_size,images_val.shape[1],images_val.shape[2],images_val.shape[3]))
            #     true_masks_val = torch.reshape(true_masks_val, (batch_size,images_val.shape[1],images_val.shape[2],images_val.shape[3]))

            #     # Forward Pass
            #     output_val = net(images)

            #     if unet_option == 'unet' or unet_option == 'unet_1':
            #         output_val_recon = output_val
            #         recon_loss_val = criterion(output_val_recon, true_masks_val)
            #         loss_val = recon_loss_val
            #         loss_items['val_loss'].append(loss_val.detach().cpu())

            #     elif unet_option == 'simple_unet':
            #         output_val_recon = output_val
            #         recon_loss_val = criterion(output_val_recon, true_masks_val)
            #         loss_val = recon_loss_val
            #         loss_items['val_loss'].append(loss_val.detach().cpu())

            #     else:
            #         output_val_recon = output_val[3]

            #         print('max pixel output val recon', torch.max(output_val_recon))

            #         mu_val = output_val[1]
            #         logvar_val = output_val[2]
            #         kl_loss_val = -0.5 * torch.sum(1 + logvar_val - mu_val.pow(2) - logvar_val.exp())
                
            #         # Find the Loss
            #         recon_loss_val = criterion(output_val_recon, true_masks_val)
            #         loss_val = recon_loss_val + kl_loss_val
            #         loss_items['val_loss'].append(loss_val.detach().cpu())
            #         # Calculate Loss
            #         valid_loss += loss_val.item()

            # print(f'Epoch {epoch+1} \t\t Training Loss: {epoch_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(val_loader)}')
            
            if min_valid_loss > epoch_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{epoch_loss:.6f}) \t Saving The Model')
                min_valid_loss = epoch_loss

                print('max pixel output val recon', torch.max(masked_output))
                
                # print("valid_loss: ", valid_loss)
                # Saving State Dict
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_{model}_epoch{number}_senegal_hls_rgb_06-20-2023_recon_new.pth'.format(model=unet_option, number=epoch + 1, alpha=alpha)))
                    

    #plt.plot(loss_items['total_loss'])
    plt.plot(loss_items['train_loss'], 'g', loss_items['val_loss'], 'r--')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(labels = ['train loss', 'validation loss'],loc='upper right')
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
    num_classes = 3
    channels = 3

    if unet_option == 'unet_vae_1':
        net = UNet_VAE(num_classes)
    elif unet_option == 'unet_vae_old':
        net = UNet_VAE_old(num_classes, segment, in_channels=channels)
    elif unet_option == 'unet_vae_RQ_old':
        net = UNet_VAE_RQ_old(num_classes, alpha)

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
                  epochs=100,
                  batch_size=10,
                  learning_rate=1e-3,
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