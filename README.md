# Riesz-Quincunx-UNet Variational Auto-Encoder using Pytorch
![Model Architecture](/figs/rqunetvae_architecture.png) <br>

This is the primary code for the [paper](https://arxiv.org/pdf/2208.12810.pdf)
## Getting Started
The code is constructed using PyTorch and users can the requirement/environment.yaml to clone the conda environment. The code require at least Python 3.8.<br>
The code is currently using a DataLoader from PyTorch to load data inside the model for training, therefore, need to modify the DataLoader code to correct data input path.<br>

## Dataset
In this study, we used satellite images from National Agriculture Imagery Program [NAIP](https://www.usgs.gov/search?keywords=Products%20and%20Datasets) dataset with 3 bands (RGB) for segmentation experiment and [Sentinel-2](https://scihub.copernicus.eu/) data with 3 bands (RGB) for reconstruction and denoising experiments. 
The data is preprocessed to the size 256x256 for training and prediction stages.
The data generation process for PyTorch DataLoader in the training script is used for the data with a specific path type. For example, Sentinel-2 input images have the path "sentinel/train/sat/<image_name>.tif" and the input masks have the path "sentinel/train/map/<image_name>.tif". Input image and mask must have the same name.<br>

## Results
![Denoised Video](/figs/video_rgb_1932_new.gif) <br>

## On the main branch:
### When users already have sets of small cut of satellite images (e.g. 256x256) for efficient computation.
To run the training file for segmentation, users can run the following command: <br>
```python unet_vae_2class_segment_train.py```
-Users can specify the path to store the model since it would save every epoch.<br>

To run the predict file for segmentation, users can run the the following command: <br>
```python unet_vae_2class_segment_predict.py```
-User will need to specify the path to load the saved model in "model_saved" variable in the script (usually inside the checkpoints directory).<br>

To run the training file for reconstruction, users can run the the following command: <br>
```python unet_vae_recon_train.py```
-Users can specify the path to store the model since it would save every epoch.<br>

To run the predict file for reconstruction for one image, users can run the following command: <br>
```python unetvae_reconstruct_predict.py```
-User will need to specify the path to load the saved model in "model_saved" variable in the script (usually inside the checkpoints directory).<br>

To run the predict file for batch of images, users can run the following command: <br>
```python unetvae_recon_predict_batch.py```
-User will need to specify the path to load the saved model in "model_saved" variable in the script.<br>

In the ```unetvae_reconstruct_predict.py``` file<br>
1/ Users can change the UNet option to perform prediction of reconstruction: 'unet_vae_old', 'unet_vae_RQ_scheme1', or 'unet_vae_RQ_scheme3'.<br>
2/ Users can change the image option to perform prediction of reconstruction: 'clean' or 'noisy' ('noisy' option to add Gaussian noise into original image).<br>
3/ Users can change the segmentation option to perform prediction of reconstruction: "segment=False" is for reconstruction.<br>
4/ Users can change the alpha level to perform prediction of reconstruction: between 0 and 1 for 'unet_vae_RQ_scheme1'. The larger the alpha, the smoother the image after reconstruction, to perform image denoising.<br>
5/ Users can change the image type to perform prediction of reconstruction: the current setup is to condition between Sentinel2 and NAIP data, using image path, to determine the normalization process, should change it accordingly. Recommendations: 'im_type=sentinel'.<br>

### When users only have large satellite imagery that requires cutting smaller tiles on the fly
Run the script ```train_large_scene.py``` and ```predict_large_scene.py```.

In the ```predict_large_scene.py``` file<br>
1/ Users can change the UNet option to perform prediction of reconstruction: 'unet_vae_old', 'unet_vae_RQ_scheme1', or 'unet_vae_RQ_scheme3'.<br>
2/ Users can change the image option to perform prediction of reconstruction: 'clean' or 'noisy' ('noisy' option to add Gaussian noise into original image).<br>
3/ Users can change the segmentation option to perform prediction of reconstruction: "segment=False" is for reconstruction.<br>
4/ Users can change the alpha level to perform prediction of reconstruction: between 0 and 1 for 'unet_vae_RQ_scheme1'. The larger the alpha, the smoother the image after reconstruction, to perform image denoising.<br>
5/ Users can change the image type to perform prediction of reconstruction: the current setup is to condition between Sentinel2 and NAIP data, using image path, to determine the normalization process, should change it accordingly. Recommendations: 'im_type=sentinel'.<br>

### Running Autolabeler code
File name: ```autolabeler_dmv_dtree_lc_combined_agg.py``` <br>
Extract all the pixels in the HLS imagery, and sample them according to the sample size. The sampled pixel has attributes: 'nlcd_class','B02','B03','B04','B05','B06','B07','Fmask','PixID','NDVI','SAVI','NDBSI','NBRI','NDSI','NDBBBI',and 'Prod'(product). 

Then train a Decision Tree with sampled pixels and apply it to classify land cover label 

To run the file, users can run the following command:<br>
```python autolabeler_dmv_dtree_lc_combined_agg.py```

