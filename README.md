# UNet VariationalEncoder RieszQuincunxShrinkage using Pytorch
 
The Unet code is constructed using PyTorch and users can the requirement/environment.yaml to clone the conda environment in the "test" branch.

The Unet code is using a DataLoader from PyTorch to load data inside the model for training, therefore, need to modify the DataLoader code to correct data input path.

To run the training file, users can run the the following command: <br>
```python unet_vae_recon_train.py```

To run the predict file, users can run the following command: <br>
```python unetvae_reconstruct_predict.py```
