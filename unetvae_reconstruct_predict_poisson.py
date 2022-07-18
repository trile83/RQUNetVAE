import argparse
import logging
import os
# import rasterio as rio
from skimage import exposure
from skimage.transform import resize
import cv2 
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision
from osgeo import gdal, gdal_array
import matplotlib.pyplot as plt

from unet import UNet_VAE
from unet import UNet_VAE_old, UNet_VAE_RQ_old, UNet_VAE_RQ_test, UNet_VAE_RQ_old_torch
from unet import UNet_VAE_RQ_new_torch, UNet_VAE_RQ_scheme3
from unet import UNet_VAE_RQ_scheme1
from utils.utils import plot_img_and_mask, plot_img_and_mask_3, plot_img_and_mask_recon

import graphlearning as gl
from tifffile import imread
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter


image_path = 'CH_R001_2018-05-07_03.tif'
mask_true_path = 'CH_R001_2018-05-07_03.tif'


use_cuda = True
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

im_type = image_path[30:38]
#print(im_type)
segment=False
alpha = 0.1
unet_option = 'unet_vae_RQ_scheme3' # options: 'unet_vae_old', 'unet_vae_RQ_old', 'unet_vae_RQ_allskip_trainable', 'unet_vae_RQ_torch', 'unet_vae_RQ_scheme3'
image_option = "noisy" # "clean" or "noisy"

##################################
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

#accept a file path to a jpg, return a torch tensor
def jpg_to_tensor(filepath=image_path):

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
    # if im_type != "sentinel":
    #     pil=pil/255

    ## add gaussian noise
    # row,col,ch= pil.shape
    # mean = 0
    # var = 0.01
    # sigma = var**0.5
    # gauss = np.random.normal(mean,sigma,(row,col,ch))
    # gauss = gauss.reshape(row,col,ch)
    # noisy = pil + 0*gauss

    row,col,ch= pil.shape
    sigma = 0.1
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
    #tensor_to_pil = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
    #pil = tensor_to_pil(tensor)
    pil = tensor.permute(1, 2, 0).numpy()
    pil = np.array(pil)
    pil = rescale_truncate(pil)
    return pil

#predict image
def predict_img(net,
                filepath,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    #img = img.unsqueeze(0)

    if image_option=='clean':
        img = jpg_to_tensor(filepath)[0] ## clean image
    elif image_option=='noisy':
        img = jpg_to_tensor(filepath)[1] ## noisy image

    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if unet_option == 'unet':
            output = output
        else:
            output = output[3]

        # print(output.shape)

        # if  output.detach().cpu().numpy().all() == 0:
        #     print("output is zero")
        #     print(output.cpu())

        #full_mask = output.cpu()
        #full_mask = full_mask.reshape(256,256,3)
        #print(full_mask.shape)

        # tf = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((256, 256)),
        #     transforms.ToTensor()
        # ])

        #full_mask = tf(output.cpu()).squeeze()
        #print(full_mask.shape)

    return output[3].cpu()


def random_fourier_feature(x, D): 
    #x - 1,100
    # x-100 dimension (with 50 numbers)
    # D -3
    num = x.shape[0]
    p = x.shape[1]
    
    # num = 4
    # p = 100
    # D = 3
    # x = np.ones((num,p))
    z_x = np.zeros((D, 1))
    

    
    b = np.random.uniform(0, 2*np.pi, D)
    mean = np.zeros(p)
    cov = np.identity(p)
    w = np.random.multivariate_normal(mean, cov, D).T
    eta = np.random.multivariate_normal(mean, cov, D).T

    # case 1:
    # aa = np.sqrt(2/D) * np.cos(x@w+b)

    b = b[:,None]
    # case 1:
    # z_x = np.sqrt(2/D) * np.cos((x@w).T+b)
    # case 2:
    z_x = np.sqrt(2/D) * np.exp((x@eta).T) * np.cos((x@w).T+b)

    z_x = z_x.T
    
    return z_x

       
def random_fourier_feature_c3(x, D): 
    #x - 1,100
    # x-100 dimension (with 50 numbers)
    # D -3
    num = x.shape[0]
    p = x.shape[1]
    
    # num = 4
    # p = 100
    # D = 3
    # x = np.ones((num,p))
    z_x = np.zeros((D, 1))
    

    
    b = np.random.uniform(0, 2*np.pi, D)
    mean = np.zeros(p)
    cov = np.identity(p)
    w = np.random.multivariate_normal(mean, cov, D).T


    # case 1:
    # aa = np.sqrt(2/D) * np.cos(x@w+b)

    b = b[:,None]
    # case 1:
    # z_x = np.sqrt(2/D) * np.cos((x@w).T+b)

    # case 2:
    # eta = np.random.multivariate_normal(mean, cov, D).T
    # z_x = np.sqrt(2/D) * np.exp((x@eta).T) * np.cos((x@w).T+b)

    # case 3:
    eta = np.random.normal(0, 1, D)
    eta = eta[:,None]
    z_x = np.sqrt(2/D) * np.exp(eta) * np.cos((x@w).T+b)

#    # case 4:
#    eta = np.random.gamma(1, 1, D)
#    eta = eta[:,None]
#    z_x = np.sqrt(2/D) * np.exp(eta) * np.cos((x@w).T+b)

    z_x = z_x.T
    
    return z_x


# def random_fourier_feature_c3_new(x, D): 
#     #x - 1,100
#     # x-100 dimension (with 50 numbers)
#     # D -3
#     n = x.shape[0]
#     P = x.shape[1]
#     p = n
    
#     # n = 4   # n - sample size
#     # P = 100 # P - high dimension
#     # p = n   # p - for kernel dimension 
#     # D = 2   # D - low dimension
    
#     # x = np.random.rand(n,P)
#     # x = np.ones((n,P))
#     z_x = np.zeros((p, 1))
    
    
#     b = np.random.uniform(0, 2*np.pi, p)
#     mean = np.zeros(P)
#     cov = np.identity(P)
#     w = np.random.multivariate_normal(mean, cov, p).T


#     # case 1:
#     # aa = np.sqrt(2/D) * np.cos(x@w+b)

#     b = b[:,None]
#     # case 1:
#     # z_x = np.sqrt(2/D) * np.cos((x@w).T+b)

#     # case 2:
#     # eta = np.random.multivariate_normal(mean, cov, D).T
#     # z_x = np.sqrt(2/D) * np.exp((x@eta).T) * np.cos((x@w).T+b)

#     # case 3:
#     eta = np.random.normal(0, 1, p)
#     eta = eta[:,None]
#     PHI = np.sqrt(2/p) * np.exp(eta) * np.cos((x@w).T+b)
#     ones_n = np.ones((n,1))
#     K_tilde = PHI.T @ np.conj(PHI) - 1/n * np.kron(ones_n.T, PHI.T @ np.conj(PHI) @ ones_n) - 1/n * np.kron(ones_n, ones_n.T @ PHI.T @ np.conj(PHI)) + 1/n**2 * (ones_n.T @ PHI.T @ np.conj(PHI) @ ones_n) * np.ones((n,n)) 

#     # Lambda - e-value; C - e-vectors 
#     Lambda, C = np.linalg.eig(1/n * K_tilde)
    
    
#     M_phi = 1/n * np.sum(PHI,1)
#     M_phi = M_phi[:,None]
#     V = ( PHI - np.kron(ones_n.T, M_phi))@C
    

#     F = V[:, :D]

    
# #    # case 4:
# #    eta = np.random.gamma(1, 1, D)
# #    eta = eta[:,None]
# #    z_x = np.sqrt(2/D) * np.exp(eta) * np.cos((x@w).T+b)

#     # z_x = z_x.T
    
#     return F


def random_fourier_feature_c3_new(x, p, D): 
    #x - 1,100
    # x-100 dimension (with 50 numbers)
    # D -3
    n = x.shape[0]
    P = x.shape[1]
    
    # p = np.floor(n/100) 
  
    # # ---------
    # n = 4   # n - sample size
    # P = 100 # P - high dimension
    # p = 3   # p < n - for kernel dimension
    # D = 2   # D - low dimension
   
    # x = np.random.rand(n,P)
    # # x = np.ones((n,P))
    # # ---------
    
    z_x = np.zeros((p, 1))
   
    b = np.random.uniform(0, 2*np.pi, p)
    mean = np.zeros(P)
    cov = np.identity(P)
    w = np.random.multivariate_normal(mean, cov, p).T

    # case 1:
    # aa = np.sqrt(2/D) * np.cos(x@w+b)

    b = b[:,None]
    # case 1:
    # z_x = np.sqrt(2/D) * np.cos((x@w).T+b)

    # case 2:
    # eta = np.random.multivariate_normal(mean, cov, D).T
    # z_x = np.sqrt(2/D) * np.exp((x@eta).T) * np.cos((x@w).T+b)

    # case 3:
    eta = np.random.normal(0, 1, p)
    eta = eta[:,None]
    PHI = np.sqrt(2/p) * np.exp(eta) * np.cos((x@w).T+b)
    ones_n = np.ones((n,1))
    K_tilde = PHI.T @ np.conj(PHI) - 1/n * np.kron(ones_n.T, PHI.T @ np.conj(PHI) @ ones_n) - 1/n * np.kron(ones_n, ones_n.T @ PHI.T @ np.conj(PHI)) + 1/n**2 * (ones_n.T @ PHI.T @ np.conj(PHI) @ ones_n) * np.ones((n,n))

    # Lambda - e-value; C - e-vectors
    Lambda, C = np.linalg.eig(1/n * K_tilde)
   
   
    M_phi = 1/n * np.sum(PHI,1)
    M_phi = M_phi[:,None]
    PHI_tilde = PHI - np.kron(ones_n.T, M_phi) 
    V = PHI_tilde @ C
   
    # D-truncated eigenvectors:
    V_D = V[:, :D]
    
    # Feature set by projecting the dataset x to the space of D-truncated eigenvectors:
    F = PHI_tilde.T @ V_D    
     
    #    # case 4:
    #    eta = np.random.gamma(1, 1, D)
    #    eta = eta[:,None]
    #    z_x = np.sqrt(2/D) * np.exp(eta) * np.cos((x@w).T+b)

    # z_x = z_x.T
   
    return F



def random_fourier_feature_c4(x, D): 
    #x - 1,100
    # x-100 dimension (with 50 numbers)
    # D -3
    num = x.shape[0]
    p = x.shape[1]
    
    # num = 4
    # p = 100
    # D = 3
    # x = np.ones((num,p))
    z_x = np.zeros((D, 1))
    

    
    b = np.random.uniform(0, 2*np.pi, D)
    mean = np.zeros(p)
    cov = np.identity(p)
    w = np.random.multivariate_normal(mean, cov, D).T


    # case 1:
    # aa = np.sqrt(2/D) * np.cos(x@w+b)

    b = b[:,None]
    # case 1:
    # z_x = np.sqrt(2/D) * np.cos((x@w).T+b)

    # case 2:
    # eta = np.random.multivariate_normal(mean, cov, D).T
    # z_x = np.sqrt(2/D) * np.exp((x@eta).T) * np.cos((x@w).T+b)

    # # case 3:
    # eta = np.random.normal(0, 1, D)
    # eta = eta[:,None]
    # z_x = np.sqrt(2/D) * np.exp(eta) * np.cos((x@w).T+b)

    # case 4:
    eta = np.random.gamma(1, 1, D)
    eta = eta[:,None]
    z_x = np.sqrt(2/D) * np.exp(eta) * np.cos((x@w).T+b)


    z_x = z_x.T
    
    return z_x
    
def get_f_seg(image,label):
    f_seg = np.zeros((256,256,3))
    y = np.zeros((256,256,3))
    m = np.zeros((3,3)) # number of channels x number of class
    for value in np.unique(label):
        itemindex = np.ma.where(label == value, 1, 0)
        for i in range(3): # number of channel
            y[:,:,i] = image[:,:,i]*itemindex
            m[i,value] = np.sum(y[:,:,i])/np.sum(itemindex)
            f_seg[:,:,i] += m[i,value]*itemindex

    return f_seg


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='/home/geoint/tri/CPC_CGRU/models/checkpoints/checkpoint_unet_vae_old_epoch20_0.0_recon.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    #parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', default='F:\\NAIP\\256\\pa101\\test\\sat\\number13985.TIF', help='Filenames of input images', required=True)
    #parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', default='out/predict1.tif', help='Filenames of output images')
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
    #in_files = args.input
    #out_files = get_output_filenames(args)

    if unet_option == 'unet_vae_1':
        net = UNet_VAE(3)
    elif unet_option == 'unet_vae_old':
        net = UNet_VAE_old(3)
    
    elif unet_option == 'unet_vae_RQ_old':
        net = UNet_VAE_RQ_old(3, alpha)

    elif unet_option == 'unet_vae_RQ_torch':
        #net = UNet_VAE_RQ_old_torch(3, alpha = alpha)
        net = UNet_VAE_RQ_new_torch(3, segment, alpha)

    elif unet_option == 'unet_vae_RQ_scheme3':
        net = UNet_VAE_RQ_scheme3(3, segment, alpha)
    elif unet_option == 'unet_vae_RQ_scheme1':
        net = UNet_VAE_RQ_scheme1(3, segment, alpha)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    #for i, filename in enumerate(in_files):
    logging.info(f'\nPredicting image {image_path} ...')
    #img = Image.open(filename)

    net.eval()

    if image_option=='clean':
        img = jpg_to_tensor(image_path)[0] ## clean image
    elif image_option=='noisy':
        img = jpg_to_tensor(image_path)[1] ## noisy image

    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)


    mask = output[3].cpu()
    
    s_i_shrink_temp =  output[6]
    
    z_ori_temp =  output[7]


    s_i_shrink_new = []
    for i in s_i_shrink_temp:
        s_i_shrink_new.append(s_i_shrink_temp[i].cpu()[0,:,:,:].permute(1, 2, 0).numpy())
    z_ori_new = z_ori_temp.cpu()[0,:,:,:].permute(1, 2, 0).numpy()
    
    
    feature_stack_list = []
    for idx, s_i in enumerate(s_i_shrink_new):
        if idx <= 1:
            print(s_i.shape)
            resized = resize(s_i, (256,256))
            print(resized.shape)
            feature_stack_list.append(resized)
            
    # feature_stack_list.append(resize(z_ori_new, (256,256)))
    # feature_stack = np.dstack(feature_stack_list)
    
    depth = 2
    s_i_list = []
    for i in range(depth+1):
        padding_size = 2**(depth-i-1)
        s_i = s_i_shrink_new[i]
        if padding_size >= 1:
            s_i = cv2.copyMakeBorder(s_i, padding_size, padding_size, padding_size, padding_size,cv2.BORDER_REFLECT)
        s_i_list.append(s_i)
        
    temp_list = []
    for i in range(256):
        for j in range(256):
            
            for k in range(depth+1):
                padding_size = 2**(depth-k-1)
                if padding_size >= 1:
                    window_size = 2*padding_size+1
                else:
                    window_size = 1
                x = i//2**k
                y = j//2**k
                s_i_sub = s_i_list[k][x:x+window_size,y:y+window_size,:]
                # feature_temp = s_i_sub.reshape((-1,s_i_sub.shape[2]))
                feature_temp = s_i_sub.flatten()

                if k == 0:
                    feature_combine = feature_temp
                else:
                    feature_combine = np.concatenate((feature_combine, feature_temp))
            temp_list.append(feature_combine)

    pixel_vals = np.asarray(temp_list)

    # poisson segmentation for noisy/origin image
    # temp = feature_stack.reshape((-1,feature_stack.shape[2]))
    # # Convert to float32 type
    # pixel_vals = np.float32(temp)
    ind = np.arange(65536).reshape((256,256))
    # 0 - barren, 1 - tree, 2 - building
    train_ind =np.array([ind[120][25],ind[252][210],ind[120][70],ind[20][230],ind[40][235],ind[229][250]])
    train_labels = np.array([0,0,1,1,2,2])

    ###############

    X = pixel_vals
    print("Negative Values: ", X[X<0])
    #build dataset
    gl.datasets.save(X,X,'s2',overwrite=True)
    #Build a knn graph
    k = 1000
    W = gl.weightmatrix.knn(data=X, k=k)
    num_train_per_class = 3
    #Run Poisson learning
    # labels_poisson = gl.graph_ssl(W,train_ind,train_labels,algorithm='poisson')
    model = gl.ssl.poisson(W, solver='gradient_descent')
    labels_poisson = model.fit_predict(train_ind, train_labels)
    
    segmented_image = labels_poisson.reshape((256,256))
    
    
    plt.figure(figsize=(5,5))
    plt.imshow(segmented_image,  cmap='tab20')
    plt.title("stacked_feature(s_i_shrink_01)_poisson")
    plt.show()
    
    
    # plot_img_and_mask_recon(img, mask)
    img1 = tensor_to_jpg(img).copy()
    # img1 = img.copy()
    img1[120,25,:] = np.array([1, 0, 0])
    img1[252,210,:] = np.array([1, 0, 0])
    img1[120,70,:] = np.array([0, 1, 0])
    img1[20,230,:] = np.array([0, 1, 0])
    img1[40,235,:] = np.array([0, 0, 1])
    img1[229,250,:] = np.array([0, 0, 1])
    
    
    plt.figure(figsize=(20,20))
    plt.subplot(1,3,1)
    plt.title("Sat")
    plt.imshow(img1)
    plt.subplot(1,3,2)
    plt.title("Reconstruction")
    #values = np.unique(y.ravel())
    plt.imshow(tensor_to_jpg(mask))
    # plt.imshow(mask)
    plt.subplot(1,3,3)
    plt.title("stacked_feature(s_i_shrink_01)_poisson")
    #values = np.unique(y.ravel())
    plt.imshow(segmented_image,  cmap='tab20')
    plt.show()
    
    clear_img = np.transpose(np.float32(jpg_to_tensor(image_path)[0].cpu())[0,:,:,:], (1, 2, 0))
    f_seg_result = get_f_seg(clear_img, segmented_image)
    
    f_seg_result_transpose = np.transpose(f_seg_result, (2, 0, 1))
    plt.figure()
    plt.imshow(f_seg_result)
    plt.show()
    
    rescale_f_seg_result = rescale(f_seg_result)
    plt.figure()
    plt.imshow(rescale_f_seg_result)
    plt.show()

    ##############

    # # pixel_vals_rff = random_fourier_feature(pixel_vals, 1000)
    # # pixel_vals_rff = random_fourier_feature_c3(pixel_vals, 1000)
    # # pixel_vals_rff = random_fourier_feature_c4(pixel_vals, 100)
    
    # p = int(np.floor(65536/1000)) 
    # f16_pixel_vals = np.float16(pixel_vals)
    # pixel_vals_rff = random_fourier_feature_c3_new(f16_pixel_vals, p, 10)
    # # poisson segmentation for noisy/origin image
    # # temp = feature_stack.reshape((-1,feature_stack.shape[2]))
    # # # Convert to float32 type
    # # pixel_vals = np.float32(temp)
    # ind = np.arange(65536).reshape((256,256))
    # # 0 - barren, 1 - tree, 2 - building
    # train_ind =np.array([ind[120][25],ind[252][210],ind[120][70],ind[20][230],ind[40][235],ind[229][250]])
    # train_labels = np.array([0,0,1,1,2,2])

    # X = pixel_vals_rff        
    # #build dataset
    # gl.datasets.save(X,X,'s2',overwrite=True)
    # #Build a knn graph
    # k = 1000
    # W = gl.weightmatrix.knn(data=X, k=k)
    # num_train_per_class = 3
    # #Run Poisson learning
    # # labels_poisson = gl.graph_ssl(W,train_ind,train_labels,algorithm='poisson')

    # model = gl.ssl.poisson(W, solver='gradient_descent')
    # labels_poisson = model.fit_predict(train_ind, train_labels)
    
    # segmented_image = labels_poisson.reshape((256,256))
    
    # plt.figure(figsize=(5,5))
    # plt.imshow(segmented_image,  cmap='tab20')
    # plt.title("stacked_feature(s_i_shrink_012)_generalized_rff_poisson_case3_PCA_new_D10_p655")
    # plt.show()

