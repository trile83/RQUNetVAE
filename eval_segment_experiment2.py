import argparse
import logging
import os
#import opencv as cv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from osgeo import gdal, gdal_array
import matplotlib.pyplot as plt
from skimage import exposure
import cv2
import matplotlib.colors as pltc
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import pickle

from unet import UNet_VAE
from unet import UNet_VAE_old, UNet_VAE_RQ_old, UNet_VAE_RQ_test, UNet_VAE_RQ_old_trainable, UNet_VAE_RQ_old_torch
from unet import UNet_VAE_RQ_new_torch, UNet_VAE_RQ_scheme3, UNet_test
from unet import UNet_VAE_RQ_scheme1, UNet_RQ
from utils.utils import plot_img_and_mask, plot_img_and_mask_3, plot_img_and_mask_5, plot_img_and_mask_4
from utils.utils import plot_img_and_mask_recon, plot_3D

use_cuda = True

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
    if im_type != "sentinel":
        pil=pil/255
    else:
        pil = (pil - np.min(pil)) / (np.max(pil) - np.min(pil))


    row,col,ch= pil.shape
    sigma = 0.08
    noisy = pil + sigma*np.random.randn(row,col,ch)

    #pil_to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    transform_tensor = transforms.ToTensor()
    if use_cuda:
        noisy_tensor = transform_tensor(noisy).cuda()
        tensor = transform_tensor(pil).cuda()

    return tensor.view([1]+list(tensor.shape)), noisy_tensor.view([1]+list(noisy_tensor.shape))

# accept a torch tensor, convert it to a jpg at a certain path
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
                img,
                unet_option,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    print("img shape: ", img.shape)

    with torch.no_grad():
        output = net(img)

        test_output = output
        #print("output shape: ", output.shape)

        if unet_option == 'unet_rq' or unet_option == 'unet_jaxony':
            #output = output[0]
            output = output.squeeze()
            #output = output
        else:
            #output = output[0][0]
            output = output[0].squeeze()

        print("output squeeze shape: ", output.shape)

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

        #print(probs)

        probs = probs.detach().cpu()
        print("probs shape: ", probs.shape)
        full_mask = torch.argmax(probs, dim=0)
        full_mask = torch.squeeze(full_mask).cpu().numpy()

        #print("full mask shape: ",full_mask.shape)

    if net.num_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return probs  #full_mask

def get_f_seg(image,label):

    f_seg = np.zeros((256,256,3))
    y = np.zeros((256,256,3))
    m = np.zeros((3,3)) # number of channels x number of class
    for value in np.unique(label):
        itemindex = np.ma.where(label == value, 1, 0)
        for i in range(3): # number of channel
            y[:,:,i] = image[:,:,i]*itemindex
            m[i,value] = np.sum(y[:,:,i])/np.sum(itemindex)
            f_seg[:,:,i] = f_seg[:,:,i] + m[i,value]*itemindex

    return f_seg

def get_f_seg_mode(preds): # preds have 3x256x256 dim
    f_seg_mode = np.zeros((256,256))
    for i in range(preds.shape[1]):
        for j in range(preds.shape[2]):
            mode_pix = np.bincount(preds[:,i,j]).argmax()
            f_seg_mode[i,j]=mode_pix
    return f_seg_mode

def get_pix_acc(pred, label):
    accuracy = []
    for index in range(pred.shape[1]):
        label_pix = label[index]
        count = 0
        for j in range(pred.shape[0]):
            pred_pix = pred[j, index]
            if pred_pix == label_pix:
                count += 1

        accuracy.append(count/loop_num)

    accuracy = np.array(accuracy)

    return accuracy

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


if __name__ == '__main__':
    args = get_args()

    #image_path = '/home/geoint/tri/sentinel/train/sat/2016105_10.tif'
    #mask_true_path = '/home/geoint/tri/sentinel/train/map/nlcd_2016105_10.tif'

    image_path = '/home/geoint/tri/va059/train/sat/number34823.TIF'
    mask_true_path = '/home/geoint/tri/va059/train/map/number34823.TIF'

    #image_path = '/home/geoint/tri/pa101/test/sat/number10698.TIF'
    #mask_true_path = '/home/geoint/tri/pa101/test/map/number10698.TIF'

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    im_type = image_path[17:25]
    segment=True
    alpha = 0.1
    unet_option = 'unet_vae_RQ_torch' # options: 'unet_vae_old', 'unet_jaxony', 'unet_vae_RQ_torch', 'unet_vae_RQ_scheme3', 'unet_vae_RQ_scheme1'
    image_option = 'clean' # "clean" or "noisy"

    if image_option=='clean':
        image = jpg_to_tensor(image_path)[0] ## clean image
    elif image_option=='noisy':
        image = jpg_to_tensor(image_path)[1] ## noisy image
    image = image.to(device=device, dtype=torch.float32)

    if unet_option == 'unet_vae_1':
        net = UNet_VAE(3)
    elif unet_option == 'unet_jaxony':
        net = UNet_test(3)
    elif unet_option == 'unet_vae_old':
        net = UNet_VAE_old(3, segment)
    elif unet_option == 'unet_rq':
        net = UNet_RQ(3, segment, alpha)
    elif unet_option == 'unet_vae_RQ_old':
        net = UNet_VAE_RQ_old(3, alpha)
    
    elif unet_option == 'unet_vae_RQ_allskip_trainable':
        net = UNet_VAE_RQ_old_trainable(3, alpha)

    elif unet_option == 'unet_vae_RQ_torch':
        net = UNet_VAE_RQ_old_torch(3, segment, alpha)
        #net = UNet_VAE_RQ_new_torch(3, segment, alpha)

    elif unet_option == 'unet_vae_RQ_scheme3':
        net = UNet_VAE_RQ_scheme3(3, segment, alpha)
    elif unet_option == 'unet_vae_RQ_scheme1':
        net = UNet_VAE_RQ_scheme1(3, segment, alpha)

    
    #logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    model_unet_jaxony = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_jaxony_4-07_epoch30_0.0_va059_segment.pth'
    model_unet_vae = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_vae_old_4-05_epoch30_0.0_va059_segment.pth'

    net.to(device=device)

    if unet_option == 'unet_jaxony' or unet_option == 'unet_rq':
        net.load_state_dict(torch.load(model_unet_jaxony, map_location=device))
    else:
        net.load_state_dict(torch.load(model_unet_vae, map_location=device))

    logging.info('Model loaded!')

    #for i, filename in enumerate(in_files):
    logging.info(f'\nPredicting image {image_path} ...')

    ## get image for visualization
    # if image_option=='clean':
    #     sat_image = jpg_to_tensor(image_path)[0]
    # else:
    #     sat_image = jpg_to_tensor(image_path)[1]
    # sat_image = tensor_to_jpg(sat_image)

    ## get ground truth label
    naip_fn = mask_true_path
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
    print(label.shape)
    label = label.reshape((256,256))
    label = label - 1
    if np.max(label)>2:
        label[label > 2] = 2

    #print(np.unique(label))

    rgb_im = tensor_to_jpg(image)
    print("rgb shape: ", rgb_im.shape)

    f_seg_gt = get_f_seg(rgb_im, label)
    print("f_seg shape: ", f_seg_gt.shape)

    #f_seg_gt = rescale(f_seg_gt)

    print(np.unique(f_seg_gt))

    # visualize f_seg_gt
    plot_img_and_mask_5(rgb_im, label, f_seg_gt)



    # looping 50 times
    loop_num = 50
    # pred_masks = []
    # pred_probs = torch.zeros((loop_num,3,256,256))
    # for i in range(loop_num):

    #     probs = predict_img(net=net,
    #                         img=image,
    #                         unet_option=unet_option,
    #                         scale_factor=1,
    #                         out_threshold=0.5,
    #                         device=device)

    #     pred_probs[i,:,:,:] = probs
        
    # #pred_probs = np.array(pred_probs)

    # print("pred probs shape: ", pred_probs.shape)

    # mean_probs = torch.zeros(probs.shape)
    # for i in range(3):
    #     mean_probs[i,:,:] = torch.mean(pred_probs[:,i,:,:])

    # mask = torch.argmax(mean_probs, dim=0)
    # mask = torch.squeeze(mask).cpu().numpy()
    # pred_masks.append(mask)

    # pred_masks = np.array(pred_masks)

    # #file_pickle_name = '/home/geoint/tri/github_files/unet_RQ_exp2.pickle'
    # file_pickle_name = '/home/geoint/tri/github_files/unet_vae_RQ_mean_exp2.pickle'

    # # save pickle file
    # with open(file_pickle_name, 'wb') as handle:
    #     pickle.dump(pred_masks, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # file_pickle_name = '/home/geoint/tri/github_files/unet_RQ_exp2.pickle'
    # # load pickle file
    # with open(file_pickle_name, 'rb') as input_file:
    #     pred_masks_unetrq = pickle.load(input_file)

    #pred_masks_unetrq = pred_masks

    # print("unet rq shape: ", pred_masks_unetrq.shape)

    #print("predicted masks shape: ", pred_masks_unetrq.shape)

    file_pickle_name = '/home/geoint/tri/github_files/unet_vae_RQ_exp2.pickle'
    # load pickle file
    with open(file_pickle_name, 'rb') as input_file:
        pred_masks_unetvaerq = pickle.load(input_file)

    print("unet vae rq shape: ", pred_masks_unetvaerq.shape)

    # get f_seg for predictions results:
    # f_seg_preds = []
    # for i in range(loop_num):
    #     #plot_img_and_mask_3(img, label, pred_masks[i], 0)
    #     f_seg_pr = get_f_seg(rgb_im, pred_masks_unetvaerq[i,:,:])
    #     f_seg_preds.append(f_seg_pr)
    # f_seg_preds = np.array(f_seg_preds) # 50,256,256,3

    # print("f_seg_preds: ", f_seg_preds.shape)

    # get f_segmode
    f_segmode = get_f_seg_mode(pred_masks_unetvaerq)
    plot_img_and_mask_3(rgb_im, label, f_segmode)


    #####
    # get 1 line of pixel in the ground truth
    index_line = 127

    label_line_arr = label[index_line,:,]

    pred_line_unetvaerq_arr = pred_masks_unetvaerq[:,index_line,:]

    # get index line for segmented image (new definition) 256x256x3
    # pred_lines = f_seg_preds[:,index_line,:]


    # get accuracy
    accuracy = get_pix_acc(pred_line_unetvaerq_arr, label_line_arr)


    # get f line at index line
    rgb_line = rgb_im[index_line,:,:]
    pred_line_gt = f_seg_gt[index_line,:,:]
    # for i in range(f_seg_preds.shape[2]):
    #     f_seg_preds[:,:,i]
    # plot_3D(rgb_line, pred_lines)


    # change class number to class name for categorical display
    label_line_arr = label_line_arr.astype(str)
    label_line_arr[label_line_arr == '0'] = "Tree"
    label_line_arr[label_line_arr == '1'] = "Grass"
    label_line_arr[label_line_arr == '2'] = "Concrete"

    #print("pred mean line: ", pred_line_unetvaerq_arr.shape)

    plt.rcParams["figure.figsize"] = [20, 10]
    #plt.rcParams["figure.autolayout"] = True

    ax1 = plt.subplot()
    ax1.set_ylabel('class number')
    l1, = ax1.plot(label_line_arr, label='train label', color='red')
    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy')
    l2, = ax2.plot(accuracy, 'bo')

    plt.legend([l1, l2], ['train label', 'accuracy'])
    
    # plt.plot(range(256), pred_line_unetvaerq_arr, label=('U_mean'))
    # for index in range(pred_line_unetvaerq_arr.shape[0]):
    #     plt.plot(range(256), pred_line_unetvaerq_arr[index], '--')
    # plt.plot(label_line_arr, label='train label', color='red', linewidth=2)
    # plt.plot(range(256), mean_pred_vaerq_line, label = 'RQUnet-VAE')
    # plt.fill_between(range(256), mean_pred_vaerq_line-std_pred_vaerq_line, mean_pred_vaerq_line+std_pred_vaerq_line, alpha=0.5)
    # plt.plot(range(256), mean_pred_rq_line, label = 'RQUnet')
    # plt.fill_between(range(256), mean_pred_rq_line-std_pred_rq_line, mean_pred_rq_line+std_pred_rq_line, alpha=0.5)
    # plt.legend()
    plt.show()