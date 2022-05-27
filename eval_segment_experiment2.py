import argparse
import logging
import os
from xmlrpc.client import DateTime
#import opencv as cv
import numpy as np
import scipy.stats as stats
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
from numpy import linalg as LA
from datetime import date

from unet import UNet_VAE
from unet import UNet_VAE_old, UNet_VAE_RQ_old, UNet_VAE_RQ_test, UNet_VAE_RQ_old_trainable, UNet_VAE_RQ_old_torch
from unet import UNet_VAE_RQ_new_torch, UNet_VAE_RQ_scheme3, UNet_test
from unet import UNet_VAE_RQ_scheme1, UNet_RQ
from utils.utils import plot_img_and_mask, plot_img_and_mask_3, plot_img_and_mask_5, plot_img_and_mask_4
from utils.utils import plot_img_and_mask_recon, plot_accu_map

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

    # print("img shape: ", img.shape)
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
        return full_mask, probs

def get_f_seg(image, label, num_class):

    image = image.reshape((256,256,3))
    f_seg = np.zeros((256,256,3))
    y = np.zeros((256,256,3)) # image after class process (normal color for class and black for non-class)
    m = np.zeros((3,num_class)) # number of channels x number of class
    #print('get_f_seg func: ', np.unique(label))
    for value in range(num_class):
        itemindex = np.ma.where(label == value, 1, 0)
        for i in range(3): # number of channel
            y[:,:,i] = image[:,:,i]*itemindex
            m[i,value] = np.sum(y[:,:,i])/np.sum(itemindex)
            f_seg[:,:,i] += m[i,value]*itemindex
    return f_seg

# get label_mode
def get_label_mode(preds): # preds is the Torch tensor have 50x256x256 dim
    preds = preds.astype(np.int64)
    label_mode = np.zeros((256,256))
    for i in range(preds.shape[1]):
        for j in range(preds.shape[2]):
            mode_pix = np.argmax(np.bincount(preds[:,i,j])) # get the label with the highest count
            label_mode[i,j]=mode_pix
    return label_mode.astype(np.uint8)

# get covariance matrix
def get_covar_mat(f_seg_preds, f_seg_mode):
    # f_seg_preds has dimension of (50 x 256 x 256 x 3)
    # f_seg_mode has dimension of (256 x 256 x 3)
    mean = np.zeros((256,256,3))
    for i in range(f_seg_preds.shape[0]):
        mean += f_seg_preds[i,:,:,:]
    mean = mean/50
    # mean = np.mean(f_seg_preds, axis=0)

    V = {}
    for i in range(f_seg_preds.shape[1]):
        V[i] = {}
        for j in range(f_seg_preds.shape[2]):
            V[i][j] = np.zeros((3,3))
            b = f_seg_mode[i,j,:].reshape((3,1))
            # b = mean[i,j,:].reshape((3,1))
            for n in range(f_seg_preds.shape[0]):
                a = f_seg_preds[n,i,j,:].reshape((3,1))
                
                #A = np.matmul((a-b),np.transpose(a-b))
                A = (a-b) @ np.transpose(a-b)
                V[i][j] += A

            V[i][j] = V[i][j]/f_seg_preds.shape[0]
    return V

# get heat map
def get_accuracy_map(preds, label, loop_num, img):

    img_name = '/home/geoint/tri/github_files/results_paper1/input_img.png'
    label_name = '/home/geoint/tri/github_files/results_paper1/groundtruth.png'
    accu_map_name = '/home/geoint/tri/github_files/results_paper1/accuracy_map_5_12.png'

    img = img.reshape((256,256,3))
    accu_map = np.zeros((256,256))
    for i in range(label.shape[0]):
        accu= get_pix_acc(preds[:,i,:], label[i,:], loop_num)
        accu_map[i] = accu

    std = np.sqrt( accu_map/loop_num * ( np.ones((256,256)) - accu_map ) )

    #plot_accu_map(img, label, accu_map)

    colors = ['forestgreen','orange']
    colormap = pltc.ListedColormap(colors)

    plt.imshow(img)
    plt.axis('off')
    plt.savefig(img_name, bbox_inches='tight')
    plt.show()


    plt.imshow(label, cmap = colormap)
    plt.axis('off')
    plt.savefig(label_name, bbox_inches='tight')
    plt.show()


    plt.imshow(accu_map, cmap = 'coolwarm')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(accu_map_name, bbox_inches='tight')
    plt.show()

    return accu_map, std

def get_pix_acc(pred, label, loop_num): # get pixel accuracy for each row of image
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
    
    return accuracy # dim (256,)

def compute_determinant_covar(label_mode, var_mat):
 
    det = np.zeros((label_mode.shape))
    for i in range(label_mode.shape[0]):
        for j in range(label_mode.shape[1]):
            det[i,j] = LA.det(var_mat[i][j])

    classes = np.unique(label_mode)
    class_maxdet_index = {}
    for i in classes:
        index_mat = np.ma.where(label_mode == i, 1, 0)
        class_det = det * index_mat
        max_det = np.max(class_det)
        itemindex = np.where(det == max_det)
        class_maxdet_index[i] = itemindex

    return det, class_maxdet_index


def draw_accu_norm_dist(P, std, label, class_num_list = []):

    name = '/home/geoint/tri/github_files/results_paper1/allclass_accuracy_normal_density_plot.png'

    cluster_means = {}
    for class_num in class_num_list:
        itemindex = np.where(label == class_num)
        h_ind_lst = itemindex[0]
        w_ind_lst = itemindex[1]

        mean_cluster = 0
        total = 0

        for i in range(len(h_ind_lst)):
            a = P[h_ind_lst[i],w_ind_lst[i]]
            total += a

        mean_cluster = total/len(h_ind_lst)
        #std = np.sqrt(mean_cluster * (1-mean_cluster))

        cluster_means[class_num]=mean_cluster

        # loop through all pixels
        # Num = 1000
        # x = np.linspace(mean_cluster-1, mean_cluster+1, num=Num)
        # x = x.reshape((1000,1))

    # for i in range(len(h_ind_lst)):
    #     a = P[h_ind_lst[i],w_ind_lst[i]]
    #     total += a
    #     b = std[h_ind_lst[i],w_ind_lst[i]]
    #     u = stats.norm.pdf(x, a, b)
    #     if b != 0:
    #         plt.plot(x, u)

    Num = 1000
    x = np.linspace(-0.5, 1.7, num=Num)
    x = x.reshape((1000,1))
    for class_num in cluster_means.keys():
        std = np.sqrt(cluster_means[class_num] * (1-cluster_means[class_num]))
        u = stats.norm.pdf(x, cluster_means[class_num], std)
        if class_num == 0:
            plot_name = 'tree+grass'
            color = 'blue'
        elif class_num == 1:
            plot_name = 'concrete'
            color = 'orange'
        else:
            plot_name = 'concrete'
            color = 'green'
        plt.plot(x, u, label=plot_name)
        # Plot the average line
        plt.axvline(cluster_means[class_num], color=color, linestyle='dashed', linewidth=1)
    plt.legend()
    plt.xlabel('pixel accuracy')
    plt.ylabel('probability density')
    plt.savefig(name, bbox_inches='tight')
    plt.show()



    # Num = 1000
    # x = np.linspace(a-0.5, a+0.5, num=Num)
    # #y = 1/( np.sqrt( 2*np.pi*b**2 ) ) * np.exp( -1/(2*b**2)*( x - a*np.ones([Num,1]) )**2 )
    # u = stats.norm.pdf(x, a, b)
    
    # x = x.reshape((1000,1))
    # plt.plot(x, u)
    # plt.savefig(name, bbox_inches='tight')
    # plt.show()


def draw_accu_norm_pixel(P, std, label, class_num):

    name = '/home/geoint/tri/github_files/results_paper1/class_{}_accuracy_normal_density.png'.format(class_num)

    itemindex = np.where(label == class_num)
    h_ind_lst = itemindex[0]
    w_ind_lst = itemindex[1]
    # print("a: ", a)  # a:  0.26
    # print("b: ", b)  # b:  0.06203224967708329

    Num = 1000
    x = np.linspace(-0.5, 1.7, num=Num)
    x = x.reshape((1000,1))

    # y = 1/( np.sqrt( 2*np.pi*b**2 ) ) * np.exp( -1/(2*b**2)*( x - a*np.ones([Num,1]) )**2 )
    # u = stats.norm.pdf(x, a, b)

    for i in range(len(h_ind_lst)):
        a = P[h_ind_lst[i],w_ind_lst[i]]
        b = std[h_ind_lst[i],w_ind_lst[i]]
        u = stats.norm.pdf(x, a, b)
        #if b != 0:
        plt.plot(x, u)
    plt.savefig(name, bbox_inches='tight')
    plt.show()

# plot accuracy and confidence interval
def plot_accuracy(label, accuracy, std, index):

    name = '/home/geoint/tri/github_files/results_paper1/accuracy_line_plot.png'
    
    # get 1 line of pixel in the ground truth
    index_line = index
    label_line= label[index_line,:,]
    accuracy_line = accuracy[index_line,:] # (50,256,256)
    std_line = std[index_line,:]

    # change class number to class name for categorical display
    label_line = label_line.astype(str)
    label_line[label_line == '0'] = "Tree+Grass"
    label_line[label_line == '1'] = "Concrete"
    #label_line[label_line == '2'] = "Concrete"

    plt.rcParams["figure.figsize"] = [20, 10]
    # plt.rcParams["figure.autolayout"] = True
    ax1 = plt.subplot()
    ax1.set_ylabel('class name')
    l1, = ax1.plot(label_line, label='train label', color='red')
    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy')
    l2 = ax2.errorbar(range(256), accuracy_line, yerr = std_line, barsabove=True, fmt="bo", label='accuracy')
    plt.legend([l1, l2], ['train label', 'accuracy'])
    plt.savefig(name, bbox_inches='tight')
    plt.show()


# get red line for f_seg_preds
def get_red_line_plot(image, f_seg_preds):
    line_index = 127
    img_line_red = image[line_index, :, 0]
    f_seg_preds_line = f_seg_preds[:, line_index, :, 0]

    plt.plot(img_line_red, label='clean red', color='red')
    for i in range(f_seg_preds_line.shape[0]):
        plt.plot(f_seg_preds_line[i])

    plt.legend()
    plt.show()


def read_image(image_path):
    ## get ground truth label
    naip_fn = image_path
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
    label = label.reshape((256,256))

    # label = label - 1
    # if np.max(label)>2:
    #     label[label > 2] = 2

    label = label - 1
    label[label == 1] = 0
    label[label == 2] = 1
    label[label == 3] = 1

    return label

def loop_predict(image, net, loop_num, unet_option):
    # loop through number of runs
    today = date.today()
    # Month abbreviation, day and year	
    d4 = today.strftime("%m-%d-%Y")
    pred_masks = np.zeros((loop_num,image.shape[2],image.shape[3]))
    print("pred mask shape: ", pred_masks.shape)
    for i in range(loop_num):

        mask, probs = predict_img(net=net,
                            img=image,
                            unet_option=unet_option,
                            scale_factor=1,
                            out_threshold=0.5,
                            device=device)

        pred_masks[i,:,:] = mask

    # pred_masks = np.array(pred_masks)
    file_pickle_name = '/home/geoint/tri/github_files/unet_vae_RQ_exp2_{}.pickle'.format(d4)
    # file_pickle_name = '/home/geoint/tri/github_files/unet_vae_RQ_mean_exp2.pickle'

    # save pickle file for 50 predictions
    with open(file_pickle_name, 'wb') as handle:
        pickle.dump(pred_masks, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save noisy image
    file_pickle_name = '/home/geoint/tri/github_files/exp2_noisy_im_{}.pickle'.format(d4)
    with open(file_pickle_name, 'wb') as handle:
        pickle.dump(image, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return pred_masks

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

    #image_path = '/home/geoint/tri/va059/train/sat/number34823.TIF'
    #mask_true_path = '/home/geoint/tri/va059/train/map/number34823.TIF'

    image_path = '/home/geoint/tri/pa101/test/sat/number10698.TIF'
    mask_true_path = '/home/geoint/tri/pa101/test/map/number10698.TIF'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    im_type = image_path[17:25]
    segment=True
    alpha = 0.5
    class_num = 2
    unet_option = 'unet_vae_RQ_torch' # options: 'unet_vae_old', 'unet_jaxony', 'unet_vae_RQ_torch', 'unet_vae_RQ_scheme3', 'unet_vae_RQ_scheme1'
    image_option = 'noisy' # "clean" or "noisy"

    if image_option=='clean':
        image = jpg_to_tensor(image_path)[0] ## clean image
    elif image_option=='noisy':
        image = jpg_to_tensor(image_path)[1] ## noisy image
    image = image.to(device=device, dtype=torch.float32)

    if unet_option == 'unet_vae_1':
        net = UNet_VAE(class_num)
    elif unet_option == 'unet_jaxony':
        net = UNet_test(class_num)
    elif unet_option == 'unet_vae_old':
        net = UNet_VAE_old(class_num, segment)
    elif unet_option == 'unet_rq':
        net = UNet_RQ(class_num, segment, alpha)
    elif unet_option == 'unet_vae_RQ_old':
        net = UNet_VAE_RQ_old(class_num, alpha)
    elif unet_option == 'unet_vae_RQ_allskip_trainable':
        net = UNet_VAE_RQ_old_trainable(class_num, alpha)
    elif unet_option == 'unet_vae_RQ_torch':
        net = UNet_VAE_RQ_old_torch(class_num, segment, alpha)
        #net = UNet_VAE_RQ_new_torch(3, segment, alpha)
    elif unet_option == 'unet_vae_RQ_scheme3':
        net = UNet_VAE_RQ_scheme3(class_num, segment, alpha)
    elif unet_option == 'unet_vae_RQ_scheme1':
        net = UNet_VAE_RQ_scheme1(class_num, segment, alpha)

    
    #logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    # model_unet_jaxony = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_jaxony_4-07_epoch30_0.0_va059_segment.pth'
    # model_unet_vae = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_vae_old_4-05_epoch30_0.0_va059_segment.pth'

    model_unet_jaxony = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_jaxony_epoch11_va059_5-16_segment2class.pth'
    model_unet_vae = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_vae_old_epoch11_va059_5-16_segment2class.pth'

    net.to(device=device)

    if unet_option == 'unet_jaxony' or unet_option == 'unet_rq':
        net.load_state_dict(torch.load(model_unet_jaxony, map_location=device))
    else:
        net.load_state_dict(torch.load(model_unet_vae, map_location=device))

    logging.info('Model loaded!')

    #for i, filename in enumerate(in_files):
    logging.info(f'\nPredicting image {image_path} ...')

    label = read_image(mask_true_path)

    # print("unique label class: ", np.unique(label))
    # print("label data type: ", label.dtype)

    rgb_im = tensor_to_jpg(image)

    # looping 50 times for predictions
    loop_num = 50
    # pred_masks = loop_predict(image, net, loop_num, unet_option)

    # load pickle file
    file_pickle_name = '/home/geoint/tri/github_files/unet_vae_RQ_exp2_05-20-2022.pickle'
    with open(file_pickle_name, 'rb') as input_file:
        pred_masks_unetvaerq = pickle.load(input_file)

    print("unet vae rq shape: ", pred_masks_unetvaerq.shape)

    # load noisy image
    file_pickle_name = '/home/geoint/tri/github_files/exp2_noisy_im_05-20-2022.pickle'
    with open(file_pickle_name, 'rb') as input_file:
        noisy_im = pickle.load(input_file)

    # get f_seg for predictions results:
    f_seg_preds = []
    for i in range(loop_num):
        f_seg_pr = get_f_seg(noisy_im.cpu().numpy(), pred_masks_unetvaerq[i,:,:], class_num)
        f_seg_preds.append(f_seg_pr)
    f_seg_preds = np.array(f_seg_preds) # 50,256,256,3 # TODO: 

    print("f_seg_preds: ", f_seg_preds.shape)
    # plot_img_and_mask_5(noisy_im, label, f_seg_preds[0])

    # get f_segmode
    label_mode = get_label_mode(pred_masks_unetvaerq) # 50,256,256
    # plot_img_and_mask_3(rgb_im, label, f_segmode)

    # save label_mode image
    # file_pickle_name = '/home/geoint/tri/github_files/exp2_label_mode_4-25.pickle'
    # with open(file_pickle_name, 'wb') as handle:
    #     pickle.dump(label_mode, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    f_seg_mode = get_f_seg(noisy_im.cpu().numpy(), label_mode, class_num)
    print("f_segmode_seg: ", f_seg_mode.shape)
    # plot_img_and_mask_5(noisy_im, f_segmode, f_segmode_seg)

    # get covariance matrix
    var_mat = get_covar_mat(f_seg_preds, f_seg_mode)

    #plot_img_and_mask_5(rgb_im, label, mean)

    varmat_pickle_name = '/home/geoint/tri/github_files/unet_vae_RQ_varmat_5-20.pickle'
    # save pickle file
    with open(varmat_pickle_name, 'wb') as input_file:
        pickle.dump(var_mat, input_file, protocol=pickle.HIGHEST_PROTOCOL)

    print(var_mat[1][2])

    w, v = LA.eig(var_mat[1][2])
    print("eigenvalues: ", w)

    #get_red_line_plot(noisy_im, f_seg_preds)

    accuracy_map, accu_std = get_accuracy_map(pred_masks_unetvaerq, label, loop_num, tensor_to_jpg(noisy_im))
    plot_accuracy(label, accuracy_map, accu_std, index=127)

    draw_accu_norm_dist(accuracy_map, accu_std, label, class_num_list = [0,1])
    # draw_accu_norm_dist(accuracy_map, accu_std, label, class_num = 1)
    # draw_accu_norm_dist(accuracy_map, accu_std, label, class_num = 2)

    draw_accu_norm_pixel(accuracy_map, accu_std, label, class_num = 0)
    draw_accu_norm_pixel(accuracy_map, accu_std, label, class_num = 1)
    # draw_accu_norm_pixel(accuracy_map, accu_std, label, class_num = 2)

    det_mat, class_maxdet_index = compute_determinant_covar(label_mode, var_mat)
    # print(det_mat)
    # print(class_maxdet_index)
    # for i in class_maxdet_index.keys():
    #     h_ind = class_maxdet_index[i][0][0]
    #     w_ind = class_maxdet_index[i][1][0]
    #     print("determinant for class "+str(i)+": "+ str(det_mat[h_ind,w_ind]))
    #     print(var_mat[h_ind][w_ind])

    