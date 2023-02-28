# -*- coding: utf-8 -*-
"""
Created on Wed Apr  14 15:12:16 2021

Extract all the pixels in the HLS imagery, and Train a Decision Tree with sampled pixels and apply it to classify land cover label 

@author: Xiqi Fei
"""

from osgeo import gdal
import pyproj
import matplotlib.pyplot as plt
import seaborn as sb
import math
import numpy as np
import random
import pandas as pd
import glob, os
from datetime import datetime
from collections import Counter
from sklearn.metrics import accuracy_score
import sklearn as sk
import sklearn.tree
import sklearn.model_selection
import graphviz as gv




def FeatureExtraction(**kwargs):
    """Extract features of all pxiels, and sample them according to the sample size
    
    Parameters
    ----------

    hls_path : path where hls data are stored

    nlcd_path : path where nlcd are stored
    
    sample_size : number of pixel per class for sampling


    Returns
    -------
    df1 :datafrane which contains sampled pixels with all the attributes
    

    """
    
    hls_path = kwargs.get('hls_path', None)
    nlcd_path = kwargs.get('nlcd_path', None)
    sample_size = kwargs.get('sample_size', None)
    
    # HLS S30 (Sentinel2) selected bands
    s_list = ['B02','B03','B04','B8A','B11','B12']
    # HLS L30 (Landsat 8 OLI) selected bands
    l_list = ['B02','B03','B04','B05','B06','B07']
    
    
    # NLCD selected classes - reselected
    nlcd_class_list = [11,12,22,23,24,31,41,42,43,52,71,81,82]
    
    # changes the current working directory to the folder where downloaded HLS files are stored
    os.chdir(hls_path)
    all_files = [f for f in glob.glob("*.tif")]
    file_list = list(set([x.split('.')[0]+'.'+x.split('.')[1]+'.'+x.split('.')[2]+'.'+x.split('.')[3]+'.'+ x.split('.')[4]+'.'+ x.split('.')[5] for x in all_files]))
    tiles = list(set([x.split('.')[2] for x in all_files]))
    
    # # iteratet through tiles
    # for tile in tiles:
    tile = tiles[1]    
    #get all the files related with the tile
    files = [f for f in file_list if tile in f]
    df = pd.DataFrame(columns = ['nlcd_class','B02','B03','B04','B05','B06','B07','Fmask','PixID','NDVI','SAVI','NDBSI','NBRI','NDSI','NDBBBI','Prod'] )
       
    nlcd_pix_loc_list = []
    for count, fname in enumerate(files):
        print(fname)
        # read Fmask file first - filter to get pixels with no clouds...
        fmask = gdal.Open(fname + '.Fmask.subset.tif')
        fm = fmask.ReadAsArray()
        
        # get HLS projection information
        proj_hls = fmask.GetProjection()
        ulx, xres, xskew, uly, yskew, yres  = fmask.GetGeoTransform()
        lrx = ulx + (fmask.RasterXSize * xres)
        lry = uly + (fmask.RasterYSize * yres)
        sizex = fmask.RasterXSize
        sizey = fmask.RasterYSize
    
        # for each tile, only open NLCD file once    
        if count==0:        
            # NLCD - clip NLCD to match with the HLS file
            # change the path to where the NLCD files are stored
            nlcd = gdal.Open(nlcd_path + tile[1:4] + '_nlcd.img')
            geotransform = nlcd.GetGeoTransform()
            proj_nlcd = nlcd.GetProjection()
            dsClip = gdal.Warp(nlcd_path +'clip.tif', nlcd, format = "GTiff",  outputBounds = (ulx,lry,lrx,uly))
            nlcd_arr= dsClip.ReadAsArray()
            dsClip = None
            df_nlcd = pd.DataFrame(nlcd_arr)
            for nlcd_class in nlcd_class_list:
                if nlcd_class == 22 or  nlcd_class == 23 or  nlcd_class == 71 or  nlcd_class == 81:
                    divide_num = 2
                else:
                    divide_num = 1
                    
                # search for pixels of the defined nlcd class, and save them into pix_list
                i, c = np.where(df_nlcd == nlcd_class)
                pix_list = [[],[]]
                if len(i) > 0:
                    print("number of avaliable NLCD class"+ str(nlcd_class)+" pixels: " + str(len(i)))
                    for x in range(len(i)):
                        pix_list[0].append(i[x])
                        pix_list[1].append(c[x])
                else:
                    print("number of avaliable NLCD class"+ str(nlcd_class)+" pixels: " + str(len(i)) + ", no avaliable NLCD pixel")
                nlcd_pix_loc_list.append(pix_list)
            
    
        # read all the data layers of the S30 product
        if fname[4:7]=='S30':
            print('S30')
            bands = []
            for xn in s_list:
                fn = fname + '.' + xn + '.subset.tif'
                ds = gdal.Open(fn)
                ds_arr = ds.ReadAsArray()
                bands.append(ds_arr)
            # adding the FMASK layer     
            bands.append(fm)    
            hls_arr = np.dstack(bands)
            
            for nlcd_class, pix_list in zip(nlcd_class_list,nlcd_pix_loc_list):
                # search for pixels of the defined nlcd class, and save them into pix_list
                # if this nlcd classes has pixels (not empty)
                if pix_list[0]:
                    pixels = []
                    #the following two classes needs to be aggregated
                    if nlcd_class == 22:
                        nlcd_class = 23
                    if nlcd_class == 81:
                        nlcd_class =71
                    
                    pixid_list = []
                    for x, y  in zip(pix_list[0],pix_list[1]):
                        # add pixel ID (tileID + x + y) to a list
                        pixid_list.append(tile + str(x).zfill(4) + str(y).zfill(4))        
                        # add land cover class to the first column
                        pixels.append(np.insert(hls_arr[x,y],0,nlcd_class))
    
                    pix_arr = np.stack(pixels,axis =0)
                    df_pix_arr = pd.DataFrame(pix_arr)   
                    df_pix_arr.columns =['nlcd_class','B02','B03','B04','B8A','B11','B12','Fmask'] 
                    df_pix_arr['PixID'] = pixid_list
    
                    # Calculate spectral features 
                    # ndvi = (8A-4)/(8A+4)
                    # savi = 1.5 * (8A-4)/ (8A+4+0.5) 
                    # ndbsi = (11-8A)/(11+8A)
                    # nbri = (8A-12)/(8A+12)
                    # ndsi = (3-11)/(3+11)
                    # ndbbbi =(2-11)/(2+11) 
                    df_pix_arr['NDVI'] = (df_pix_arr['B8A']-df_pix_arr['B04']) / (df_pix_arr['B8A']+df_pix_arr['B04'])
                    df_pix_arr['SAVI'] = 1.5 * (df_pix_arr['B8A']-df_pix_arr['B04']) / (df_pix_arr['B8A']+df_pix_arr['B04']+0.5)
                    df_pix_arr['NDBSI'] =  (df_pix_arr['B11']-df_pix_arr['B8A']) / (df_pix_arr['B11']+df_pix_arr['B8A'])
                    df_pix_arr['NBRI'] = (df_pix_arr['B8A']-df_pix_arr['B12']) / (df_pix_arr['B8A']+df_pix_arr['B12'])
                    df_pix_arr['NDSI'] = (df_pix_arr['B03']-df_pix_arr['B11']) / (df_pix_arr['B03']+df_pix_arr['B11'])
                    df_pix_arr['NDBBBI'] = (df_pix_arr['B02']-df_pix_arr['B11']) / (df_pix_arr['B02']+df_pix_arr['B11'])
                    df_pix_arr.columns =['nlcd_class','B02','B03','B04','B05','B06','B07','Fmask','PixID','NDVI','SAVI','NDBSI','NBRI','NDSI','NDBBBI'] 
                    df_pix_arr['Prod'] = fname[4:7]
                    # Collection month of the HLS tile
                    month = datetime.strptime(fname[15:19] + "-" +fname[19:22], "%Y-%j").month
                    df_pix_arr['month'] = month
                    df_pix_arr['day'] = fname[19:22]
                    df = df.append(df_pix_arr, ignore_index=True)                   
        # read all the data layers of the L30 product
        elif fname[4:7] =='L30':
            print('L30')
            bands = []
            for xn in l_list:
                fn = fname + '.' + xn + '.subset.tif'
                ds = gdal.Open(fn)
                ds_arr = ds.ReadAsArray()
                bands.append(ds_arr)
            # adding the FMASK layer     
            bands.append(fm)    
            hls_arr = np.dstack(bands)
            
            for nlcd_class, pix_list in zip(nlcd_class_list,nlcd_pix_loc_list):
                # search for pixels of the defined nlcd class, and save them into pix_list
                # if this nlcd classes has pixels (not empty)
                if pix_list[0]:
                    pixels = []
                    #the following two classes needs to be aggregated
                    if nlcd_class == 22:
                        nlcd_class = 23
                    if nlcd_class == 81:
                        nlcd_class =71
                    
                    pixid_list = []
                    for x, y  in zip(pix_list[0],pix_list[1]):
                        # add pixel ID (tileID + x + y) to a list
                        pixid_list.append(tile + str(x).zfill(4) + str(y).zfill(4))        
                        # add land cover class to the first column
                        pixels.append(np.insert(hls_arr[x,y],0,nlcd_class))
    
                    pix_arr = np.stack(pixels,axis =0)
                    df_pix_arr = pd.DataFrame(pix_arr)   
                    df_pix_arr.columns =['nlcd_class','B02','B03','B04','B05','B06','B07','Fmask'] 
                    df_pix_arr['PixID'] = pixid_list
    
        
                    # Calculate spectral features- L30
                    # ndvi = (5-4)/(5+4)
                    # savi = 1.5 * (5-4)/ (5+4+0.5)
                    # ndbsi = (6-5)/(6+5)
                    # nbri = (5-7)/(5+7)
                    # ndsi = (3-6)/(3+6)
                    # ndbbbi =(2-6)/(2+6)
                    df_pix_arr['NDVI'] = (df_pix_arr['B05']-df_pix_arr['B04']) / (df_pix_arr['B05']+df_pix_arr['B04'])
                    df_pix_arr['SAVI'] = 1.5 * (df_pix_arr['B05']-df_pix_arr['B04']) / (df_pix_arr['B05']+df_pix_arr['B04']+0.5)
                    df_pix_arr['NDBSI'] =  (df_pix_arr['B06']-df_pix_arr['B05']) / (df_pix_arr['B06']+df_pix_arr['B05'])
                    df_pix_arr['NBRI'] = (df_pix_arr['B05']-df_pix_arr['B07']) / (df_pix_arr['B05']+df_pix_arr['B07'])
                    df_pix_arr['NDSI'] = (df_pix_arr['B03']-df_pix_arr['B06']) / (df_pix_arr['B03']+df_pix_arr['B06'])
                    df_pix_arr['NDBBBI'] = (df_pix_arr['B02']-df_pix_arr['B06']) / (df_pix_arr['B02']+df_pix_arr['B06'])
                    df_pix_arr['Prod'] = fname[4:7]
                    # Collection month of the HLS tile
                    month = datetime.strptime(fname[15:19] + "-" +fname[19:22], "%Y-%j").month
                    df_pix_arr['month'] = month
                    df_pix_arr['day'] = fname[19:22]
                    df = df.append(df_pix_arr, ignore_index=True)
    os.remove(nlcd_path +'clip.tif')
    
    # filter samples 
    df_temp = df[(df["Fmask"]==0) & (df["B02"]>0.01)]
    df_temp_agg = df_temp.groupby(['nlcd_class','PixID']).agg({'B02':'mean','B03':'mean','B04':'mean','B05':'mean','B06':'mean','B07':'mean','NDVI':'mean','SAVI':'mean','NDBSI':'mean','NBRI':'mean','NDSI':'mean','NDBBBI':'mean'})
    df_agg = df_temp_agg.reset_index()
    # print(Counter(df_agg["nlcd_class"]).keys() )
    # print(Counter(df_agg["nlcd_class"]).values() )
    
    nlcd_class_list = df_agg.nlcd_class.unique()
    df_new_list = []
    #select num_pix samples per class for dtree
    for x in nlcd_class_list:
        df_new_list.append(df_agg[(df_agg["nlcd_class"]==x)].sample(n = sample_size))
    df1 = pd.concat(df_new_list)
        
    return df1

def decisionTree(df,predictor_att,class_label,**kwargs):
    """Quick decision tree builder for comparison analysis. Will either save 
        or pop-up a decision tree visulaizion, and will output classier,
        confusion matrix, and accuracy of model
    
    Parameters
    ----------
    df : dataframe containing data that is to be used to compute decision tree
        column names should be attributes with one column being class labels

    predictor_att : 1darray containing strings of atrritbutes to be used for
        training
        
    class_label : variable containing string name for column of known classes
    
    fpath : path where decision tree will be saved
    
    tree_depth : integer representing the decision tree depth
    
    test_size : number representing the train/test split size
        example: train/test = 80/20 -> test_size = 0.2

    Returns
    -------
    acc : value containing the accuracy of the classifier based on the 
        train/test split
        
    classifier : classifier object. values accessible by ".classifiername"
        convention. ex: df_class.
        
    conf : confusion matrix generated from train/test split
    
    Outputs
    -------
    DecisionTree.png: A visual representation of the built decision tree.
        This will be rendered as a popup at the completion of the processor
        or be saved to a specified folder location and will not pop up at the
        completion of the processor.

    """

    
    # Assigning the values of kwargs to their corresponding values in
    #   in processor
    fpath = kwargs.get('fpath', None)
    fname = kwargs.get('fname', None)
    tree_depth = kwargs.get('tree_depth', None)
    test_size = kwargs.get('test_size', None)
    
    # Breaking out the data into train test split variables.
    # df[predictor_att] grabs the attributes to be passed from the dataframe
    # df[class_label] grabs the class labels from the dataframe
    pred_train, pred_test, lbl_train, lbl_test = sklearn.model_selection.train_test_split(
        df[predictor_att], df[class_label], 
        test_size=test_size, random_state=1)
    
    
    classifier = sklearn.tree.DecisionTreeClassifier(max_depth=tree_depth)
    classifier = classifier.fit(pred_train, lbl_train)
    
    
    # Grabbing the labels in the predicted attributes in order of how they are
    #   structured to maintain fidelity with structure of future confusion 
    #   matrix. Then forcing object type as string.
    c_label = lbl_test.unique().astype('str').tolist()
    

    # Creating graphical decision tree
    g = sklearn.tree.export_graphviz(classifier, out_file = None,
                                feature_names=predictor_att, 
                                class_names=c_label,
                                filled = True, rounded = True)
    graph = gv.Source(g)
    graph.format = 'png'
    if fpath is None:
        # Since fpath is not defined, this will just render as a popup
        graph.render(filename = 'DecisionTree', view=True)
    else:
        # Saving the decision tree to the passed in filepath. As such, the
        #   render will not popup.
        graph.render(filename = 'DecisionTree', directory = fpath, 
                     cleanup = True, view=False)
    
    # Grabbing predicted labels
    lbl_pred = classifier.predict(pred_test)
    
    # Building a confusion matrix
    conf = sk.metrics.confusion_matrix(lbl_test,lbl_pred)
    
    # Putting the confusion matrix into dataframe form for ease of access to
    #   user and to have proper class labels associated
    df_conf = pd.DataFrame(conf,columns = c_label,index = c_label)
    
    # Pulling out sklearn's auto-calculated accuracy
    acc = sk.metrics.accuracy_score(lbl_test,lbl_pred)
    
    # Plot confusion matrix
    print('accuracy: ', str(acc))
    f, ax = plt.subplots(figsize=(11, 10))
    sb.set(font_scale=1.5)
    sb.heatmap(conf.T, annot=True, fmt="d",cmap='cividis', linewidths=.3,cbar=False)
    ax.set_title('Decision Tree - Depth: {0} \nAccuracy:{1:.3%}\n'.format(str(num_td),acc))
    ax.set_xlabel('Truth')
    ax.set_ylabel('Predicted')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    
    return classifier, df_conf, acc
    





# number of pixel per class for sampling
sample_size = 2000
# filepath where hls are stored
hls_path = 'F:/Auto-labeler/data/hls_dmv_both/'
# filepath where nlcd are stored
nlcd_path ='F:/Auto-labeler/data/NLCD/'
# Passing a filepath to the processor where decision tree will be saved
fpath = r"F:\Auto-labeler\result\dtree"
# Choosing the attributes the classifier is to be trained on.
predictor_att = ['B02','B03','B04','B05','B06','B07','NDVI','SAVI','NDBSI','NBRI','NDSI','NDBBBI']
# Naming the column which contains the class labels
class_label = "nlcd_class"
# Decision Tree depth
num_td = 6


# feature extraction
df = FeatureExtraction(hls_path=hls_path, nlcd_path=nlcd_path,sample_size=sample_size)

#  decsion tree 
df_classifier, conf, acc = decisionTree(df,predictor_att,class_label, fpath = fpath, tree_depth = num_td, test_size = 0.2)





    
