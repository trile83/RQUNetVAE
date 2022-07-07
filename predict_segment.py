import argparse
import logging
import os
from skimage import exposure
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import tifffile

from unet import UNet_VAE, UNet_test
from unet import UNet_VAE_old, UNet_VAE_RQ_old, UNet_VAE_RQ_test, UNet_VAE_RQ_old_torch
from unet import UNet_VAE_RQ_new_torch, UNet_VAE_RQ_scheme3, RQUNet_VAE_scheme1_Pareto
from unet import UNet_VAE_RQ_scheme1, UNet_VAE_RQ_scheme2, UNet_VAE_Stacked
from utils.utils import plot_img_and_mask, plot_img_and_mask_3, plot_img_and_mask_recon, plot_img_and_mask_4

##################################
def rescale(image): ## function to rescale image for visualization
    map_img =  np.zeros((256,256,3))
    for band in range(3):
        p2, p98 = np.percentile(image[:,:,band], (2, 98))
        map_img[:,:,band] = exposure.rescale_intensity(image[:,:,band], in_range=(p2, p98))
    return map_img

def rescale_2(image):
    map_img =  np.zeros(image.shape)
    for band in range(3):
        p2, p98 = np.percentile(image[band,:,:], (2, 98))
        map_img[band,:,:] = exposure.rescale_intensity(image[band,:,:], in_range=(p2, p98))
    return map_img

def rescale_truncate(image): ## function to rescale image for visualization
    if np.amin(image) < 0:
        image = np.where(image < 0,0,image)
    if np.amax(image) > 1:
        image = np.where(image > 1,1,image) 
    map_img =  np.zeros((256,256,3))
    for band in range(3):
        p2, p98 = np.percentile(image[:,:,band], (2, 98))
        map_img[:,:,band] = exposure.rescale_intensity(image[:,:,band], in_range=(p2, p98))
    return map_img

# Standardize images
def standardize_image(image):
    '''
    Arg: Input is an image with dimension (channel, height, width)
    '''
    for i in range(image.shape[0]):  # for each channel in the image
            image[i, :, :] = (image[i, :, :] - np.mean(image[i, :, :])) / \
                (np.std(image[i, :, :]) + 1e-8)

    # image = image.reshape((image.shape[1], image.shape[2], image.shape[0]))

    return image

# Normalize bands into 0.0 - 1.0 scale
def normalize_image(image):
    '''
    Arg: Input is an image with dimension (channel, height, width)
    '''

    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    #image = image/np.max(image)

    return image

#accept a file path to a jpg, return a torch tensor
def jpg_to_tensor(filepath):

    #image = rasterio.open(filepath).read()
    image= tifffile.imread(filepath)

    if im_type != "sentinel":
        pil = image/255
    else:
        # pil = normalize_image(image)
        # pil = standardize_image(pil)
        pil = np.asarray(image)
        #pil = rescale(pil)

    # print(np.max(pil))
    # print(np.min(pil))

    row,col,ch= pil.shape
    sigma = 0.01 ## choosing sigma based on the input images, 0.1-0.3 for NAIP images, 0.002 to 0.01 for sentinel2 images
    noisy = pil + sigma*np.random.randn(row,col,ch)

    transform_tensor = transforms.ToTensor()

    # noisy_tensor = torch.tensor(noisy)
    noisy_tensor = transform_tensor(noisy).cuda()
    # tensor = torch.tensor(pil)
    tensor = transform_tensor(pil).cuda()

    #print(tensor.shape)

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
            output = output.squeeze()
        else:
            #output = output[0][0]
            output = output[0].squeeze()

        #print("output squeeze shape: ", output.shape)

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        if net.num_classes > 1:
            probs = F.softmax(output, dim=0)
            #probs = output
            #probs = F.log_softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output[0])[0]


        probs = probs.detach().cpu()
        full_mask = torch.argmax(probs, dim=0)
        full_mask = torch.squeeze(full_mask).cpu().numpy()


        # full_mask = tf(probs.cpu()).squeeze()
        # print(full_mask.shape)

    if net.num_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        #return full_mask.argmax(dim=0).numpy()
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


if __name__ == '__main__':
    args = get_args()

    # image_path = '/home/geoint/tri/sentinel/train/sat/2016105_6.tif'
    # mask_true_path = '/home/geoint/tri/sentinel/train/map/nlcd_2016105_6.tif'

    image_path = '/home/geoint/tri/sentinel/train/sat/2016105_9.tif'
    mask_true_path = '/home/geoint/tri/sentinel/train/map/nlcd_2016105_9.tif'

    # image_path = '/home/geoint/tri/va059/train/sat/number34823.TIF'
    # mask_true_path = '/home/geoint/tri/va059/train/map/number34823.TIF'

    # image_path = '/home/geoint/tri/va059/train/sat/number13458.TIF'
    # mask_true_path = '/home/geoint/tri/va059/train/map/number13458.TIF'

    # image_path = '/home/geoint/tri/pa101/test/sat/number10698.TIF'
    # mask_true_path = '/home/geoint/tri/pa101/test/map/number10698.TIF'

    # image_path = '/home/geoint/tri/pa101/test/sat/number13376.TIF'
    # mask_true_path = '/home/geoint/tri/pa101/test/map/number13376.TIF'

    use_cuda = True
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    im_type = 'sentinel'
    class_num = 2
    segment=True
    alpha = 0.4
    unet_option = 'unet_jaxony' # options: 'unet_vae_old', 'unet_jaxony', 'unet_vae_RQ_torch', 'unet_vae_RQ_scheme3', 'unet_vae_RQ_scheme1'
    image_option = "clean" # "clean" or "noisy"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    model_unet_jaxony = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_jaxony_epoch34_7-6_segment_sentinel.pth'
    model_unet_vae = '/home/geoint/tri/github_files/github_checkpoints/checkpoint_unet_vae_old_epoch11_va059_5-16_segment2class.pth'

    if unet_option == 'unet_vae_1':
        net = UNet_VAE(class_num)
    elif unet_option == 'unet_jaxony':
        net = UNet_test(class_num)
    elif unet_option == 'unet_vae_old':
        net = UNet_VAE_old(class_num, segment)
    elif unet_option == 'unet_vae_RQ_old':
        net = UNet_VAE_RQ_old(class_num, alpha)
    # elif unet_option == 'unet_vae_RQ_allskip_trainable':
    #     net = UNet_VAE_RQ_old_trainable(2,alpha)
    elif unet_option == 'unet_vae_RQ_torch':
        net = UNet_VAE_RQ_old_torch(class_num, segment, alpha)
        #net = UNet_VAE_RQ_new_torch(3, segment, alpha)
    # elif unet_option == 'unet_vae_RQ':
    #     net = UNet_VAE_RQ(2, segment, alpha = alpha)
    elif unet_option == 'unet_vae_RQ_scheme3':
        net = UNet_VAE_RQ_scheme3(class_num, segment, alpha)
    elif unet_option == 'unet_vae_RQ_scheme1':
        net = UNet_VAE_RQ_scheme1(class_num, segment, alpha)
    elif unet_option == 'unet_vae_RQ_scheme2':
        net = UNet_VAE_RQ_scheme2(class_num, segment, alpha)
    elif unet_option == 'unet_vae_stacked':
        net = UNet_VAE_Stacked(class_num, segment, device, model_unet_vae)

    
    net.to(device=device)
    if unet_option == 'unet_jaxony' or unet_option == 'unet_rq':
        net.load_state_dict(torch.load(model_unet_jaxony, map_location=device))
        print('Model loaded! ', model_unet_jaxony)
    elif unet_option != 'unet_vae_stacked':
        net.load_state_dict(torch.load(model_unet_vae, map_location=device))
        print('Model loaded! ', model_unet_vae)

    mask = predict_img(net=net,
                        filepath=image_path,
                        scale_factor=1,
                        out_threshold=0.5,
                        device=device)

    print(mask.shape)


    # get image
    if image_option=='clean':
        img = jpg_to_tensor(image_path)[0]
    else:
        img = jpg_to_tensor(image_path)[1]
    img = tensor_to_jpg(img)

    # img = rasterio.open(image_path).read()
    # img = np.array(img).astype(np.uint8)
    # img = img.reshape((256,256,3))

    #img = read_sentinel2(image_path)

    #print(naip_ds)

    ## get ground truth label
    label_file = mask_true_path
    label = tifffile.imread(label_file)

    label = np.array(label)
    #print(label.shape)
    #label = label.reshape((256,256))
    label[label==2]=1
    label[label==3]=1
    label[label==4]=2
    label = label - 1 

    print(np.unique(label))

    error = mask - label
    #print("errors: ", error)
    print(np.unique(mask))
    if im_type == 'sentinel':
        plot_img_and_mask_4(img, label, mask)
    else:
        plot_img_and_mask_3(img, label, mask)

    