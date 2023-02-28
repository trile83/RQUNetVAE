import argparse
import logging
import os
from matplotlib import image
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
import tifffile
import cv2
import matplotlib.colors as pltc
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import jaccard_score
import itertools

from unet import UNet_VAE, UNet_RQ
from unet import UNet_VAE_old, UNet_VAE_RQ_old, UNet_VAE_RQ_test, UNet_VAE_RQ_old_torch
from unet import UNet_VAE_RQ_new_torch, UNet_VAE_RQ_scheme3, UNet_test, UNet_VAE_RQ
from unet import UNet_VAE_RQ_scheme1, UNet_VAE_RQ_scheme2, UNet_VAE_Stacked
from utils.utils import plot_img_and_mask, plot_img_and_mask_3, plot_img_and_mask_2, plot_img_and_mask_4
from utils.utils import plot_img_and_mask_recon, plot_pred_only


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

    # y_true = y_true-1
    # y_true[y_true == 3] == 2
    # if np.max(y_true)>2:
    #     y_true[y_true > 2] = 2

    # print('y true label: ', np.unique(y_true))
    # print('y pred label: ', np.unique(y_pred))

    #print("label unique values",np.unique(y_true))
    #print("prediction unique values",np.unique(y_pred))

    # get overall weighted accuracy
    accuracy = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred, sample_weight=None)

    f1 = f1_score(y_true, y_pred, average=None)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)

    iou = jaccard_score(y_pred, y_true, average="micro")

    # print(classification_report(y_true, y_pred))

    # f1 = np.around(f1, 3)
    # precision = np.around(precision, 3)
    # recall = np.around(recall, 3)

    # iou = np.around(iou, 3)

    #print(f1)

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
    return con_mat, accuracy, balanced_accuracy, f1, precision, recall, iou


def plot_confusion_matrix(cm, class_names=['a', 'b', 'c'], name=''):
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
    conf_mat_name = '/home/geoint/tri/github_files/results_paper1/image_1/{}_conf_mat.png'.format(name)
    
    plt.savefig(conf_mat_name, bbox_inches='tight')
    
    plt.close()

def rescale(image):
    map_img =  np.zeros((256,256,3))
    for band in range(3):
        p2, p98 = np.percentile(image[:,:,band], (2, 98))
        map_img[:,:,band] = exposure.rescale_intensity(image[:,:,band], in_range=(p2, p98))
    return map_img

#accept a file path to a jpg, return a torch tensor
def jpg_to_tensor(filepath):

    # naip_fn = filepath
    # driverTiff = gdal.GetDriverByName('GTiff')
    # naip_ds = gdal.Open(naip_fn, 1)
    # nbands = naip_ds.RasterCount
    # # create an empty array, each column of the empty array will hold one band of data from the image
    # # loop through each band in the image nad add to the data array
    # data = np.empty((naip_ds.RasterXSize*naip_ds.RasterYSize, nbands))
    # for i in range(1, nbands+1):
    #     band = naip_ds.GetRasterBand(i).ReadAsArray()
    #     data[:, i-1] = band.flatten()

    # img_data = np.zeros((naip_ds.RasterYSize, naip_ds.RasterXSize, naip_ds.RasterCount),
    #                 gdal_array.GDALTypeCodeToNumericTypeCode(naip_ds.GetRasterBand(1).DataType))
    # for b in range(img_data.shape[2]):
    #     img_data[:, :, b] = naip_ds.GetRasterBand(b + 1).ReadAsArray()
    
    img_data = tifffile.imread(filepath)
    pil = np.array(img_data)
    pil = pil.reshape((256,256,3))
    pil = pil/255

    # add noise
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
    pil = rescale(pil)
    
    return pil

#predict image
def predict_img(net,
                filepath,
                img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    # if image_option=='clean':
    #     img = jpg_to_tensor(filepath)[0] ## clean image
    # elif image_option=='noisy':
    #     img = jpg_to_tensor(filepath)[1] ## noisy image

    img = img.to(device=device, dtype=torch.float32)

    #print("img shape: ", img.shape)

    with torch.no_grad():
        output = net(img)

        test_output = output
        #print("output shape: ", output.shape)

        if unet_option == 'unet' or unet_option == 'unet_jaxony' or unet_option == 'unet_rq':
            output = output.squeeze()
        else:
            #output = output[0][0]
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

        probs = probs.detach().cpu()
        full_mask = torch.argmax(probs, dim=0)
        full_mask = torch.squeeze(full_mask).cpu().numpy()

    if net.num_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
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

    # image_path = '/home/geoint/tri/va059/train/sat/number34823.TIF'
    # mask_true_path = '/home/geoint/tri/va059/train/map/number34823.TIF'

    # image_path = '/home/geoint/tri/va059/train/sat/number13458.TIF'
    # mask_true_path = '/home/geoint/tri/va059/train/map/number13458.TIF'

    # image_path = '/home/geoint/tri/pa101/test/sat/number10698.TIF'
    # mask_true_path = '/home/geoint/tri/pa101/test/map/number10698.TIF'

    # image_path = '/home/geoint/tri/pa101/test/sat/number13376.TIF'
    # mask_true_path = '/home/geoint/tri/pa101/test/map/number13376.TIF'

    image_path = '/home/geoint/tri/md013/val/sat/number30719.TIF'
    mask_true_path = '/home/geoint/tri/md013/val/map/number30719.TIF'

    im_name = image_path[-15:-4]

    use_cuda = True
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    im_type = image_path[17:25]
    segment=True
    alpha = 0.4
    unet_option = 'unet_vae_RQ_torch' # options: 'unet_vae_old', 'unet_jaxony', 'unet_vae_RQ_torch', 'unet_vae_RQ_scheme3', 'unet_vae_RQ_scheme1'
    image_option = "noisy" # "clean" or "noisy"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    model_unet_jaxony = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_jaxony_epoch11_va059_5-16_segment2class.pth'
    model_unet_vae = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_vae_old_epoch11_va059_5-16_segment2class.pth'

    if unet_option == 'unet_vae_1':
        net = UNet_VAE(2)
    elif unet_option == 'unet_jaxony':
        net = UNet_test(2)
    elif unet_option == 'unet_vae_old':
        net = UNet_VAE_old(2, segment)
    elif unet_option == 'unet_vae_RQ_old':
        net = UNet_VAE_RQ_old(2, alpha)
    # elif unet_option == 'unet_vae_RQ_allskip_trainable':
    #     net = UNet_VAE_RQ_old_trainable(2,alpha)
    elif unet_option == 'unet_vae_RQ_torch':
        net = UNet_VAE_RQ_old_torch(2, segment, alpha)
        #net = UNet_VAE_RQ_new_torch(3, segment, alpha)
    elif unet_option == 'unet_vae_RQ':
        net = UNet_VAE_RQ(2, segment, alpha = alpha)
    elif unet_option == 'unet_vae_RQ_scheme3':
        net = UNet_VAE_RQ_scheme3(2, segment, alpha)
    elif unet_option == 'unet_vae_RQ_scheme1':
        net = UNet_VAE_RQ_scheme1(2, segment, alpha)
    elif unet_option == 'unet_vae_RQ_scheme2':
        net = UNet_VAE_RQ_scheme2(2, segment, alpha)
    elif unet_option == 'unet_vae_stacked':
        net = UNet_VAE_Stacked(2, segment, device, model_unet_vae)

    
    net.to(device=device)
    if unet_option == 'unet_jaxony' or unet_option == 'unet_rq':
        net.load_state_dict(torch.load(model_unet_jaxony, map_location=device))
        print('Model loaded! ', model_unet_jaxony)
    elif unet_option != 'unet_vae_stacked':
        net.load_state_dict(torch.load(model_unet_vae, map_location=device))
        print('Model loaded! ', model_unet_vae)

    # baseline unet
    net_1 = UNet_test(2)
    net_1.to(device=device)
    net_1.load_state_dict(torch.load(model_unet_jaxony, map_location=device))
    
    ###
    classes = ['Tree', 'Concrete']  # 6-Cloud not present
    colors = ['forestgreen','orange']
    colormap = pltc.ListedColormap(colors)

    ## get image
    if image_option=='clean':
        img = jpg_to_tensor(image_path)[0]
    else:
        img = jpg_to_tensor(image_path)[1]


    img_1 = tensor_to_jpg(img)

    ## get ground truth label
    naip_fn = mask_true_path
    # data= rio.open(naip_fn)
    # img_data = data.read([1])
    img_data = tifffile.imread(naip_fn)

    label = np.array(img_data)
    label = label.reshape((256,256))
    label = label - 1
    label[label == 1] = 0
    label[label == 2] = 1
    label[label == 3] = 1

    #for i, filename in enumerate(in_files):
    logging.info(f'\nPredicting image {image_path} ...')

    iteration = 20

    # arrays for typical UNet results
    base_balanced_acc_arr = np.zeros((iteration))
    base_f1_arr = np.zeros((iteration,2))
    base_precision_arr = np.zeros((iteration,2))
    base_recall_arr = np.zeros((iteration,2))
    base_iou_arr = np.zeros((iteration))

    ## arrays for RQUNet-VAE results
    balanced_acc_arr = np.zeros((iteration))
    f1_arr = np.zeros((iteration, 2))
    precision_arr = np.zeros((iteration,2))
    recall_arr = np.zeros((iteration,2))
    iou_arr = np.zeros((iteration))

    
    for i in range(iteration):

        baseline_mask = predict_img(net=net_1,
                        filepath=image_path,
                        img = img,
                        scale_factor=1,
                        out_threshold=0.5,
                        device=device)

        mask = predict_img(net=net,
                            filepath=image_path,
                            img = img,
                            scale_factor=1,
                            out_threshold=0.5,
                            device=device)
        
        # baseline plot

        base_cnf_matrix, base_accuracy, base_balanced_accuracy, \
            base_f1, base_precision, base_recall, base_iou = confusion_matrix_func(
                y_true=label, y_pred=baseline_mask, nclasses=len(classes), norm=True
            )

        base_balanced_acc_arr[i] = base_balanced_accuracy
        base_f1_arr[i,:]= base_f1
        base_precision_arr[i,:] = base_precision
        base_recall_arr[i,:] = base_recall
        base_iou_arr[i] = base_iou

        ## rqunet denoise
        cnf_matrix, accuracy, balanced_accuracy, \
             f1, precision, recall, iou = confusion_matrix_func(
                y_true=label, y_pred=mask, nclasses=len(classes), norm=True
            )

        balanced_acc_arr[i] = balanced_accuracy
        f1_arr[i,:]= f1
        precision_arr[i,:] = precision
        recall_arr[i,:] = recall
        iou_arr[i] = iou

    #print(f1_arr)

    plot_confusion_matrix(base_cnf_matrix, class_names=classes, name="base")
    base_pred_name = '/home/geoint/tri/github_files/results_paper1/image_1/base_unet_pred.png'
    plot_pred_only(baseline_mask,base_pred_name, base_balanced_accuracy)

    plot_confusion_matrix(cnf_matrix, class_names=classes, name='rqunet_vae')
    rqunetvae_pred_name = '/home/geoint/tri/github_files/results_paper1/image_1/rqunet_vae_pred.png'
    plot_pred_only(mask,rqunetvae_pred_name, balanced_accuracy)

    base_bal_acc_mean = np.mean(base_balanced_acc_arr)
    base_bal_acc_std = np.std(base_balanced_acc_arr)
    base_f1_mean = np.mean(base_f1_arr, axis=0)
    base_f1_std = np.std(base_f1_arr, axis=0)
    base_precision_mean = np.mean(base_precision_arr, axis=0)
    base_precision_std = np.std(base_precision_arr, axis=0)
    base_recall_mean = np.mean(base_recall_arr, axis=0)
    base_recall_std = np.std(base_recall_arr, axis=0)

    base_iou_mean = np.mean(base_iou_arr)
    base_iou_std = np.std(base_iou_arr)


    #print("Baseline Overall Accuracy Mean: ", base_accuracy)
    print("Baseline Balanced Accuracy Mean: ", base_bal_acc_mean)
    print("Baseline Balanced Accuracy Std: ", base_bal_acc_std)
    print("Baseline F1 Mean: ", base_f1_mean)
    print("Baseline F1 Std: ", base_f1_std)
    print("Baseline Precision Mean: ", base_precision_mean)
    print("Baseline Precision Std: ", base_precision_std)
    print("Baseline Recall Mean: ", base_recall_mean)
    print("Baseline Recall Std: ", base_recall_std)
    print("Baseline mIoU Mean: ", base_iou_mean)
    print("Baseline mIoU Std: ", base_iou_std)

    bal_acc_mean = np.mean(balanced_acc_arr)
    bal_acc_std = np.std(balanced_acc_arr)
    f1_mean = np.mean(f1_arr, axis=0)
    f1_std = np.std(f1_arr, axis=0)
    precision_mean = np.mean(precision_arr, axis=0)
    precision_std = np.std(precision_arr, axis=0)
    recall_mean = np.mean(recall_arr, axis=0)
    recall_std = np.std(recall_arr, axis=0)

    iou_mean = np.mean(iou_arr)
    iou_std = np.std(iou_arr)

    #print("Overall Accuracy: ", accuracy)
    print("Balanced Accuracy Mean: ", bal_acc_mean)
    print("Balanced Accuracy Std: ", bal_acc_std)
    print("F1 Score Mean: ", f1_mean)
    print("F1 Score Std: ", f1_std)
    print("Precision Mean: ", precision_mean)
    print("Precision Std: ", precision_std)
    print("Recall Mean: ", recall_mean)
    print("Recall Std: ", recall_std)
    print("mIoU Mean: ", iou_mean)
    print("IoU Std: ", iou_std)

    file = open('/home/geoint/tri/github_files/results_paper1/image_1/stats_results.txt', 'w')

    file.write(f'Baseline Typical UNet for {im_name}\n')
    file.write(f'Class: [impervious vegetation]\n')
    file.writelines(f'Baseline accuracy: {np.round(base_bal_acc_mean,3)} +- {np.round(base_bal_acc_std,3)}\n')
    file.writelines(f'Baseline F1 score: {np.round(base_f1_mean,3)} +- {np.round(base_f1_std,3)}\n')
    file.writelines(f'Baseline recall: {np.round(base_recall_mean,3)} +- {np.round(base_recall_std,3)}\n')
    file.writelines(f'Baseline precision: {np.round(base_precision_mean,3)} +- {np.round(base_precision_std,3)}\n')
    file.writelines(f'Baseline IoU: {np.round(base_iou_mean,3)} +- {np.round(base_iou_std,3)}\n')

    file.writelines(f'RieszQuincunx-UNet-VAE for {im_name}\n')
    file.writelines(f'accuracy: {np.round(bal_acc_mean,3)} +- {np.round(bal_acc_std,3)}\n')
    file.writelines(f'F1 score: {np.round(f1_mean,3)} +- {np.round(f1_std,3)}\n')
    file.writelines(f'recall: {np.round(recall_mean,3)} +- {np.round(recall_std,3)}\n')
    file.writelines(f'precision: {np.round(precision_mean,3)} +- {np.round(precision_std,3)}\n')
    file.writelines(f'IoU: {np.round(iou_mean,3)} +- {np.round(iou_std,3)}\n')

    file.close()