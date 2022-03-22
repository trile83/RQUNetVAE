# UNet VariationalEncoder RieszQuincunxShrinkage using Pytorch
 
The Unet code is constructed using PyTorch and users can the requirement/environment.yaml to clone the conda environment in the "test" branch.<br>

The Unet code is using a DataLoader from PyTorch to load data inside the model for training, therefore, need to modify the DataLoader code to correct data input path.<br>
The data generation process in the training script is used for the data with a specific path type. For example, input images have the path "sentinel/train/sat/<image_name>.tif" and the input masks have the path "sentinel/train/map/<image_name>.tif". Input image and mask must have the same name.<br>
## On the test branch:<br>
To run the training file, users can run the the following command: <br>
```python unet_vae_recon_train.py```
-Users can specify the path to store the model since it would save every epoch.<br>

To run the predict file for one image, users can run the following command: <br>
```python unetvae_reconstruct_predict.py```
-User will need to specify the path to load the saved model in "model_saved" variable in the script.<br>

To run the predict file for batch of images, users can run the following command: <br>
```python unetvae_recon_predict_batch.py```
-User will need to specify the path to load the saved model in "model_saved" variable in the script.<br>

In the "unetvae_reconstruct_predict.py" file<br>
1/ Users can change the UNet option to perform prediction of reconstruction: 'unet_vae_old', 'unet_vae_RQ_scheme1', or 'unet_vae_RQ_scheme3'.<br>
2/ Users can change the image option to perform prediction of reconstruction: 'clean' or 'noisy' ('noisy' option to add Gaussian noise into original image).<br>
3/ Users can change the segmentation option to perform prediction of reconstruction: "segment=False" is for reconstruction.<br>
4/ Users can change the alpha level to perform prediction of reconstruction: between 0 and 1 for 'unet_vae_RQ_scheme1'. The larger the alpha, the smoother the image after reconstruction.<br>
5/ Users can change the image type to perform prediction of reconstruction: the current setup is to condition between Sentinel2 and NAIP data, using image path, to determine the normalization process, should change it accordingly. Recommendations: 'im_type=sentinel'.<br>



