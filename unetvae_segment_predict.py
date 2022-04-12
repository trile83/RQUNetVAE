import argparse
import logging
import os
import rasterio as rio
#import opencv as cv

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torchvision
from osgeo import gdal, gdal_array
import matplotlib.pyplot as plt
from skimage import exposure
import cv2
import matplotlib.colors as pltc
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools

from unet import UNet_VAE, UNet_RQ
from unet import UNet_VAE_old, UNet_VAE_RQ_old, UNet_VAE_RQ_test, UNet_VAE_RQ_old_trainable, UNet_VAE_RQ_old_torch
from unet import UNet_VAE_RQ_new_torch, UNet_VAE_RQ_scheme3, UNet_test, UNet_VAE_RQ
from unet import UNet_VAE_RQ_scheme1
from utils.utils import plot_img_and_mask, plot_img_and_mask_3, plot_img_and_mask_2, plot_img_and_mask_4


########
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
    if np.max(y_true)>2:
        y_true[y_true > 2] = 2

    #print("label unique values",np.unique(y_true))
    #print("prediction unique values",np.unique(y_pred))

    # get overall weighted accuracy
    accuracy = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
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


def plot_confusion_matrix(cm, class_names=['a', 'b', 'c']):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names: list with classes for confusion matrix
    Return: confusion matrix figure.
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    # Use white text if squares are dark; otherwise black.
    threshold = 0.55  # cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.savefig('/home/geoint/tri/nasa_senegal/confusion_matrix/{}_chm_int_cfn_matrix.png'.format(label_name[:-4]))
    plt.show()

def rescale(image):
    map_img =  np.zeros((256,256,3))
    for band in range(3):
        p2, p98 = np.percentile(image[:,:,band], (2, 98))
        map_img[:,:,band] = exposure.rescale_intensity(image[:,:,band], in_range=(p2, p98))
    return map_img

#accept a file path to a jpg, return a torch tensor
def jpg_to_tensor(filepath):

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
    #if im_type != "sentinel":
    pil=pil/255
    #pil = (pil - np.min(pil)) / (np.max(pil) - np.min(pil))


    row,col,ch= pil.shape
    sigma = 0.08
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
    #pil = tensor_to_pil(tensor)
    pil = tensor.permute(1, 2, 0).numpy()
    pil = np.array(pil)
    #pil = rescale(pil)
    
    return pil

#predict image
def predict_img(net,
                filepath,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    if image_option=='clean':
        img = jpg_to_tensor(filepath)[0] ## clean image
    elif image_option=='noisy':
        img = jpg_to_tensor(filepath)[1] ## noisy image
    img = img.to(device=device, dtype=torch.float32)

    #print("img shape: ", img.shape)

    with torch.no_grad():
        output = net(img)

        test_output = output
        #print("output shape: ", output.shape)

        if unet_option == 'unet' or unet_option == 'unet_jaxony' or unet_option == 'unet_rq':
            #output = output[0]
            output = output.squeeze()
            #output = output
        else:
            #output = output[0][0]
            output = output[0].squeeze()

        #print("output squeeze shape: ", output.shape)
        #print(torch.unique(output))

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

        #print(probs)

        #full_mask = tf(probs.cpu()).squeeze()
        #full_mask = probs.cpu()

        probs = probs.detach().cpu()
        full_mask = torch.argmax(probs, dim=0)

        #print(torch.unique(full_mask))
        full_mask = torch.squeeze(full_mask).cpu().numpy()

        #print("full mask shape: ",full_mask.shape)

    if net.num_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        #img = F.one_hot(full_mask.argmax(dim=0), net.num_classes).permute(2, 0, 1).numpy()
        #img_2 = full_mask.argmax(dim=0).numpy()
        return full_mask

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='/home/geoint/tri/github_files/github_checkpoints/', metavar='FILE',
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

def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        #return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))
        return (np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8)


if __name__ == '__main__':
    args = get_args()

    #image_path = '/home/geoint/tri/sentinel/train/sat/2016105_10.tif'
    #mask_true_path = '/home/geoint/tri/sentinel/train/map/nlcd_2016105_10.tif'

    image_path = '/home/geoint/tri/va059/train/sat/number34823.TIF'
    mask_true_path = '/home/geoint/tri/va059/train/map/number34823.TIF'

    #image_path = '/home/geoint/tri/pa101/test/sat/number10698.TIF'
    #mask_true_path = '/home/geoint/tri/pa101/test/map/number10698.TIF'

    use_cuda = True
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    im_type = image_path[17:25]
    segment=True
    alpha = 1
    unet_option = 'unet_vae_RQ' # options: 'unet_vae_old', 'unet_jaxony', 'unet_vae_RQ_torch', 'unet_vae_RQ_scheme3', 'unet_vae_RQ_scheme1'
    image_option = "clean" # "clean" or "noisy"

    if unet_option == 'unet_vae_1':
        net = UNet_VAE(3)
    elif unet_option == 'unet_jaxony':
        net = UNet_test(3)
    elif unet_option == 'unet_rq':
        net = UNet_RQ(3, alpha)

    elif unet_option == 'unet_vae_old':
        net = UNet_VAE_old(3, segment)
    
    elif unet_option == 'unet_vae_RQ_old':
        net = UNet_VAE_RQ_old(3, alpha)
    
    elif unet_option == 'unet_vae_RQ_allskip_trainable':
        net = UNet_VAE_RQ_old_trainable(3,alpha)

    elif unet_option == 'unet_vae_RQ_torch':
        net = UNet_VAE_RQ_old_torch(3, segment, alpha = alpha)
        #net = UNet_VAE_RQ_new_torch(3, segment, alpha)

    elif unet_option == 'unet_vae_RQ':
        net = UNet_VAE_RQ(3, segment, alpha = alpha)

    elif unet_option == 'unet_vae_RQ_scheme3':
        net = UNet_VAE_RQ_scheme3(3, segment, alpha)
    elif unet_option == 'unet_vae_RQ_scheme1':
        net = UNet_VAE_RQ_scheme1(3, segment, alpha)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    model_unet_jaxony = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_jaxony_4-07_epoch30_0.0_va059_segment.pth'
    model_unet_vae = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_vae_old_4-08_epoch30_0.0_va059_segment.pth'

    net.to(device=device)
    if unet_option == 'unet_jaxony' or unet_option == 'unet_rq':
        net.load_state_dict(torch.load(model_unet_jaxony, map_location=device))
        print('Model loaded! ', model_unet_jaxony)
    else:
        net.load_state_dict(torch.load(model_unet_vae, map_location=device))
        print('Model loaded! ', model_unet_vae)

    #for i, filename in enumerate(in_files):
    logging.info(f'\nPredicting image {image_path} ...')

    mask = predict_img(net=net,
                        filepath=image_path,
                        scale_factor=1,
                        out_threshold=0.5,
                        device=device)


    #out_files = 'out/predict_va_softshrink_all_0.02.tif'
    out_files = 'out/predict_va_unet_epoch40_new.tif'
    
    out_filename = out_files
    logging.info(f'Mask saved to {out_files}')
    
    ## get image
    if image_option=='clean':
        img = jpg_to_tensor(image_path)[0]
    else:
        img = jpg_to_tensor(image_path)[1]
    img = tensor_to_jpg(img)

    #print(naip_ds)

    ## get ground truth label
    naip_fn = mask_true_path
    #print(naip_fn)
    driverTiff = gdal.GetDriverByName('GTiff')
    naip_ds = gdal.Open(naip_fn, 1)
    nbands = naip_ds.RasterCount
    # create an empty array, each column of the empty array will hold one band of data from the image
    # loop through each band in the image and add to the data array
    data = np.empty((naip_ds.RasterXSize*naip_ds.RasterYSize, nbands))
    for i in range(1, nbands+1):
        band = naip_ds.GetRasterBand(i).ReadAsArray()
        data[:, i-1] = band.flatten()

    img_data = np.zeros((naip_ds.RasterYSize, naip_ds.RasterXSize, naip_ds.RasterCount),
                    gdal_array.GDALTypeCodeToNumericTypeCode(naip_ds.GetRasterBand(1).DataType))
    for b in range(img_data.shape[2]):
        img_data[:, :, b] = naip_ds.GetRasterBand(b + 1).ReadAsArray()

    label = np.array(img_data)
    #print(label.shape)
    label = label.reshape((256,256))

    classes = ['Tree', 'Grass', 'Concrete']  # 6-Cloud not present
    colors = ['forestgreen', 'lawngreen', 'orange']
    colormap = pltc.ListedColormap(colors)

    cnf_matrix, accuracy, balanced_accuracy = confusion_matrix_func(
            y_true=label, y_pred=mask, nclasses=len(classes), norm=True
        )

    print("Overall Accuracy: ", accuracy)
    print("Balanced Accuracy: ", balanced_accuracy)
    plot_confusion_matrix(cnf_matrix, class_names=classes)
    if im_type == 'sentinel':
        plot_img_and_mask_4(img, label, mask)
    else:
        plot_img_and_mask_3(img, label, mask, balanced_accuracy)
    #plot_img_and_mask_2(img, mask)