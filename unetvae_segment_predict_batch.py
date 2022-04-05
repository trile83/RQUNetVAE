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
from collections import defaultdict
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools

from unet import UNet_VAE
from unet import UNet_VAE_old, UNet_VAE_RQ_old, UNet_VAE_RQ_test, UNet_VAE_RQ_old_trainable, UNet_VAE_RQ_old_torch
from unet import UNet_VAE_RQ_new_torch, UNet_VAE_RQ_scheme3
from unet import UNet_VAE_RQ_scheme1, UNet_test
from utils.utils import plot_img_and_mask, plot_img_and_mask_3, plot_img_and_mask_recon
from sklearn.metrics import confusion_matrix  
import numpy as np


def compute_iou(y_pred, y_true):
    # ytrue, ypred is a flatten vector
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU)


def confusion_matrix_func(y_true=[], y_pred=[], nclasses=3, norm=True):
    """
    Args:
        y_true:   2D numpy array with ground truth
        y_pred:   2D numpy array with predictions (already processed)
        nclasses: number of classes
    Returns:
        numpy array with confusion matrix
    """

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    y_true = y_true-1
    y_true[y_true == 3] == 2
    if np.max(y_true)==255:
        y_true[y_true == 255] = 2
    y_true[y_true > 2] = 2

    #print("label unique values",np.unique(y_true))
    #print("prediction unique values",np.unique(y_pred))

    # get overall weighted accuracy
    accuracy = accuracy_score(y_true, y_pred, sample_weight=None)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred, sample_weight=None)

    #print(classification_report(y_true, y_pred))
    ## get confusion matrix
    con_mat = confusion_matrix(
        y_true, y_pred
    )

    if norm:
        con_mat = np.around(
            con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis],
            decimals=2
        )

    where_are_NaNs = np.isnan(con_mat)
    con_mat[where_are_NaNs] = 0
    return con_mat, accuracy, balanced_accuracy


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

#accept a torch tensor, convert it to a jpg at a certain path
def tensor_to_jpg(tensor):
    #tensor = tensor.view(tensor.shape[1:])
    tensor = tensor.squeeze(0)
    if use_cuda:
        tensor = tensor.cpu()
    pil = tensor.permute(1, 2, 0).numpy()
    pil = np.array(pil)
    pil = rescale_truncate(pil)
    return pil

#########################
# get data
# load image folder path and image dictionary
class_name = "pa101" ## va059 or sentinel2_xiqi
data_dir = "/home/geoint/tri/"
data_dir = os.path.join(data_dir, class_name)

# Create data 
    
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
                 
def data_generator(files, sigma=0.08, mode="test", batch_size=6):
    while True:
        all_scenes = list(files[mode]['sat'].keys())
        #print(files[mode]['map'].keys())
        
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

            if img_data.shape == (256,256,1):
                Y_lst.append(img_data)

        X = np.array(X_lst)
        Y = np.array(Y_lst)
        print("X shape: ", X.shape)
        print("Y shape: ", Y.shape)
        if im_type == "sentinel":
            for i in range(len(X)):
                X[i] = (X[i] - np.min(X[i])) / (np.max(X[i]) - np.min(X[i]))
        else:
            X = X/255

        X_noise = []

        if image_option == 'noisy':
            row,col,ch= X[0].shape
            sigma = 0.08
            for img in X:
                noisy = img + sigma*np.random.randn(row,col,ch)
                X_noise.append(noisy)
            X_noise = np.array(X_noise)
            yield X_noise, Y
        else:
            yield X, Y

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
        
        X = self.transforms(X)
        Y = torch.LongTensor(Y)
        return {
            'image': X,
            'mask': Y
        }

#####################

#predict image
def predict_img(net,
                device,
                sigma,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    #img = img.unsqueeze(0)

    ### get data
    images = {}
    load_image_paths(data_dir, class_name, 'test', images)
    train_data_gen = data_generator(images[class_name], sigma, mode="test", batch_size=20)

    images, labels = next(train_data_gen)

    images = images
    labels = labels

    # 2. Split into train / validation partitions
    n_pred = len(images)

    print("total input: ", n_pred)

    # 3. Create data loaders
    loader_args = dict(batch_size=n_pred, num_workers=4, pin_memory=True)

    sat_dataset = satDataset(X=images, Y=labels)
    im_loader = DataLoader(sat_dataset, shuffle=False, **loader_args)

    for batch in im_loader:

        images = batch['image']
        true_masks = batch['mask']

        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.float32)

        #print("true mask shape: ", true_masks.shape)

        images = torch.reshape(images, (20,3,256,256))
        true_masks = torch.reshape(true_masks, (20,256,256))

        #print("image shape: ", images.shape)
        #print("true mask shape: ", true_masks.shape)

        with torch.no_grad():
            output = net(images)

            if unet_option == 'unet' or unet_option == 'simple_unet' or unet_option == 'unet_jaxony':
                output = output.squeeze()
            else:
                output = output[0].squeeze()

            #print("output squeeze shape: ", output.shape)

            if net.num_classes > 1:
                #probs = F.softmax(output, dim=1)
                probs = output
                #probs = F.log_softmax(output, dim=1)
            else:
                probs = torch.sigmoid(output[0])[0]

            tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])

            #print("probs shape: ", probs.shape)

            probs = probs.detach().cpu()
            full_mask = torch.argmax(probs, dim=1)

            #print(torch.unique(full_mask))
            full_mask = torch.squeeze(full_mask).cpu().numpy()

            #print("full masks shape: ",full_mask.shape)

        if net.num_classes == 1:
            return (full_mask > out_threshold).numpy(), images.cpu(), true_masks.cpu()
        else:
            return full_mask, images.cpu(), true_masks.cpu().numpy()



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

    use_cuda = True
    im_type = "va059" ## sentinel or naip
    segment=True
    alpha = 0.46
    sigma = 0.08
    unet_option = 'unet_jaxony' # options: 'unet_jaxony', 'unet_vae_old', 'unet_vae_RQ_torch', 'unet_vae_RQ_scheme3'
    image_option = "clean" # "clean" or "noisy"

    if unet_option == 'unet_vae_1':
        net = UNet_VAE(3)
    elif unet_option == 'unet_jaxony':
        net = UNet_test(3)
    elif unet_option == 'unet_vae_old':
        net = UNet_VAE_old(3, segment)
    
    elif unet_option == 'unet_vae_RQ_allskip_trainable':
        net = UNet_VAE_RQ_old_trainable(3,alpha)

    elif unet_option == 'unet_vae_RQ_torch':
        net = UNet_VAE_RQ_old_torch(3, segment, alpha)
        #net = UNet_VAE_RQ_new_torch(3, segment, alpha)

    elif unet_option == 'unet_vae_RQ_scheme3':
        net = UNet_VAE_RQ_scheme3(3, segment, alpha)
    elif unet_option == 'unet_vae_RQ_scheme1':
        net = UNet_VAE_RQ_scheme1(3, segment, alpha)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    model_unet_jaxony = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_jaxony_2_epoch20_0.5_batchnorm_segment.pth'
    model_unet_vae = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_vae_old_3-28_epoch30_0.0_va059_segment.pth'

    net.to(device=device)
    if unet_option == 'unet_jaxony':
        net.load_state_dict(torch.load(model_unet_jaxony, map_location=device))
    else:
        net.load_state_dict(torch.load(model_unet_vae, map_location=device))
    

    logging.info('Model loaded!')

    preds, imgs, labels = predict_img(net=net,
                        sigma=sigma,
                        scale_factor=1,
                        out_threshold=0.5,
                        device=device)

    # print("prediction shape: ", preds.shape)
    # print("images shape: ", imgs.shape)
    # print("labels shape: ", labels.shape)

    acc_lst = []
    bal_acc_lst = []
    mean_accuracy = 0
    for i in range(imgs.size(0)):
        pred = preds[i]
        img = tensor_to_jpg(imgs[i])
        label = labels[i]

        cnf_matrix, accuracy, balanced_accuracy = confusion_matrix_func(
            y_true=label, y_pred=pred, nclasses=3, norm=True
        )

        bal_acc_lst.append(balanced_accuracy)
        acc_lst.append(accuracy)

        #plot_img_and_mask_3(img, label, pred, balanced_accuracy)

    acc_np = np.array(acc_lst)
    bal_acc_np = np.array(bal_acc_lst)
    mean_accuracy = np.mean(acc_np)
    mean_bal_accuracy = np.mean(bal_acc_np)
    std_acc = np.std(acc_np)
    std_bal_acc = np.std(bal_acc_np)
    print("average accuracy: ", mean_accuracy)
    print("standard deviation accuracy: ", std_acc)
    print("average balanced accuracy: ", mean_bal_accuracy)
    print("standard deviation balanced accuracy: ", std_bal_acc)