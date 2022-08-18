# -*- coding: utf-8 -*-
"""
Preprocessing part analyzing component of OAT.

@author: Alex-932
@version: 0.7
"""

import os
import time
import pandas as pd
import re
import numpy as np

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

from modules.utils.filemanager import filemanager
from modules.utils.clustering import clustering
from modules.utils.image import image
from modules.utils.tools import tools
from modules.utils.figures import figures
from modules.import_data import import_data

class preprocessing():
    
    def trackmate_instructions(img_dirpath, out_dirpath, fiji_dir):
        """
        Create "OAT_instructions.txt" in the fiji directory for 
        trackmate_script.py to retrieve the input image file (tif) and the path
        to the .csv output. 

        """
        #Creating the instruction file in the fiij directory.
        instructions = open(fiji_dir+"\\OAT_instructions.txt", "w")
        
        #The formatting of the instructions are as bellow:
        #   Each row correspond to an image file.
        #   The row is seperated in half by a comma.
        #   Before the comma is the input image file path.
        #   After the comma is the output .csv file path.
        
        img_paths = filemanager.search_file(img_dirpath, "tif")
        
        img_names = filemanager.search_file(img_dirpath, "tif", 
                                            fullpath = False)
        out_paths = [out_dirpath+"\\"+re.split("\.", name)[0]+"_spots.csv\n" 
                     for name in img_names]
        
        for img in range(len(img_paths)):
            instructions.write(img_paths[img]+","+out_paths[img])
    
            
    def get_ROI(df, std = 15, eps = 2, min_samples = 3, roi_offset = 5):
        """
        Processing the limits of the ROI (region of interest).
        The ROI is the smallest cubic volume where the organoid is, 
        whatever the time point is.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing spots coordinates and time points.
        std : int, optional
            Standard deviation threshold. If below that threshold, 
            the dispersion isn't enough to correctly use DBSCAN.
            The default is 15.
        eps : int, optional
            Radius of search for the DBSCAN algorithm. The default is 2.
        min_samples : int , optional
            Min. number of neighbors for DBSCAN. The default is 3.
        roi_offset : float, optional
            ROI offset. The default is 5.
            
        Returns
        -------
        ROI : pd.DataFrame
            Dataframe containing ROI limits on each axis, for each time point.
            Contains as well the "global" ROI which contains the organoïd 
            whatever the time point is.
        
        """
        ## Creating the dataframe that contains all the ROI limits.
        labels = ["X_min", "X_max", "Y_min", "Y_max", "Z_min", "Z_max"]
        ROI = pd.DataFrame(columns = labels, dtype = "float")
        
        # Getting the filenames of the processed images.
        TP = df["TP"].unique().tolist()
        
        # Computing the ROI frame by frame, and adding it to localROI. 
        for tp in TP :
            subdf = df[df["TP"] == tp]
            subdf = subdf[subdf["CLUSTER_SELECT"]]
            
            arr = np.array(subdf["COORD"].tolist())
            
            try :
                ROI.loc[tp] = [arr[:, 0].min(), arr[:, 0].max(),
                               arr[:, 1].min(), arr[:, 1].max(),
                               arr[:, 2].min(), arr[:, 2].max()]
            except :
                print("Warning : 1 or 0 selected cluster in time point : "+str(tp))
                
        globalROI = []
        
        # Running selectROI() method on each column of ROI and adding the
        # resulting value to ROI.
        for col in labels :
            subdf = ROI[col]

            globalROI.append(clustering.select_ROI(subdf, std, eps, 
                                                   min_samples, roi_offset))
            
        ROI.loc["Global"] = globalROI
               
        return ROI
       
    def denoise_timelapse(csv_dirpath, img_dirpath, savepath = None,
                  std = 15, roi_eps = 2, roi_min_samples = 3, 
                  roi_offset = 5, clust_eps = 40, clust_min_samples = 3, 
                  clustering_on_distance = True, centroid_method = "gradient",
                  rescaling = [1, 1, 1]):
        """
        Create a "denoised" timelapse from tifs images in img_dirpath, using 
        spots segmentation of csv files in csv_dirpath. 
        Each image and csv file correspond to a specific time point.
        
        The timelapse is saved in savepath which also contains the name 
        of the file.

        Parameters
        ----------
        csv_dirpath : str
            Path to the csv file folder.
        img_dirpath : str
            Path to the images folder.
        savepath : str
            Full path of the timelapse file.
        std : int, optional
            Standard deviation threshold. If below that threshold, 
            the dispersion isn't enough to correctly use DBSCAN.
            The default is 15.
        roi_eps : float, optional
            Radius of search for the DBSCAN algorithm when selecting ROI 
            limits. 
            The default is 2.
        roi_min_samples : int, optional
            Min. number of neighbors for DBSCAN when selecting ROI limits. 
            The default is 3.
        roi_offset : float, optional
            ROI offset. The default is 5.
        clust_eps : float, optional
            Radius of search for the DBSCAN algorithm when main clustering. 
            The default is 40.
        clust_min_samples : int, optional
            Min. number of neighbors for DBSCAN when main clustering. 
            The default is 3.
        clustering_on_distance : bool, optional
            If True, refine the clustering using the distance from the centroid.
            The default is True.
        centroid_method : str, optional
            Method used to get the centroid:
                - mean : simple mean on each axis.
                - gradient : using gradient slope to converge on the point that
                             have the least sum of distance with all spots.
                - sampled : subsampling multiple times and computing the 
                            subsample centroid.
            The default is "gradient" which seems to be the best.
        rescaling : list, optional
            Rescaling matrix where coordinates are multiplied by the given 
            factor : [X, Y, Z]. The default is [1, 1, 1].

        Returns
        -------
        df : pd.DataFrame
            Full dataframe containing spots, time points, cluster info.

        """
        start_time = time.time()
        
        ## Opening and merging every csv in the 
        step_time = time.time()
        print("Opening files ...", end = " ")
        df = import_data.read_spots(csv_dirpath)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        
        ## Clustering the spots.
        step_time = time.time()
        print("Clustering spots ...", end = " ")
        df = clustering.compute_clusters(df, "COORD", clust_eps, clust_min_samples, 
                                         clustering_on_distance, centroid_method)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        
        ## Getting the ROI using clustering results.
        step_time = time.time()
        print("Getting the ROI using clustering results ...", end = " ")
        ROI = preprocessing.get_ROI(df, std, roi_eps, roi_min_samples, roi_offset)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        
        cleanArray = []
        
        ## Retrieving all paths for tif files in the given directory.
        imgpath = filemanager.search_file(img_dirpath, "tif")
        
        ## Denoising images
        step_time = time.time()
        print("Keeping the ROI and creating timelapse ...", end = " ")
        for tp in df["TP"].unique().tolist() :
            imarray = image.load_tif(imgpath[tp])
            cleanArray.append(image.denoise(ROI.loc["Global"], imarray))
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        
        if savepath is not None:
            image.save_tif(np.array(cleanArray), savepath)
            print("")
            print("Image saved as :", savepath)
        print("Total time :", str(round(time.time()-start_time, 2))+"s")
        
        return df
        
    
            
            
            
            
            