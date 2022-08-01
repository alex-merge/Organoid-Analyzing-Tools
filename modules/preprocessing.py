# -*- coding: utf-8 -*-
"""
Preprocessing part analyzing component of OAT.

@author: Alex-932
@version: 0.7
"""

import os
import time
import pandas as pd
from modules.utils.filemanager import *
from modules.utils.clustering import *
from modules.utils.image import *

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
    
    def read_spots(dirpath):
        """
        Load segmentation result of Trackmate detection as a dataframe.

        Parameters
        ----------
        dirpath : str
            Path to the directory containing files.

        Returns
        -------
        df : pd.DataFrame
            Dataframe where index are spots and columns are 
            ("QUALITY", "X", "Y", "Z", "TP").

        """
        ## Importing the files.
        filespath = filemanager.search_file(dirpath, "csv")

        df = pd.DataFrame(columns=["QUALITY", "X", "Y", "Z"],
                          dtype = "float")
        
        ## For the moment, files are sorted and imported as if the time point 
        ## starts at 0.
        for tp in range(len(filespath)):
            
            ## Reading the CSV.
            stream = pd.read_csv(filespath[tp], index_col = "LABEL")
            
            # Renaming columns to have smaller names.
            stream.rename(columns = {"POSITION_"+axis: axis for axis in 
                                     ["X", "Y", "Z"]},
                          inplace = True)
            
            # Keeping useful columns.
            stream = stream.loc[:,["QUALITY", "X", "Y", "Z"]]
            
            # Setting every values to float type.
            stream = stream.astype("float")
            
            ## Adding the time point.
            stream["TP"] = stream.shape[0]*[tp]
            
            ## Merging the dataframe to the main one.
            df = pd.concat([df, stream])
        
        df["TP"] = df["TP"].astype("int")    
        
        return df
            
    def get_ROI(df, std = 15, eps = 2, min_samples = 3, offset = 5):
        """
        Processing the limits of the ROI (region of interest).
        The ROI is the smallest cubic volume where the organoid is, 
        whatever the time point is.
        These limits are saved in the self.ROI.
        Limits for each frames are stored in the self.localROI.
        
        Parameters
        ----------
        std : int
            Standard deviation threshold.
        eps : int, optional
            Radius of search for the DBSCAN algorithm. The default is 2.
        min_samples : int , optional
            Min. number of neighbors for DBSCAN. The default is 3.
        offset : float, optional
            Value to add or retrieve to the max or the min value to take into 
            account the size of the real object. 
            The default is 5.
        
        """
        # Creating the dataframe that contains all the ROI limits.
        labels = ["X_min", "X_max", "Y_min", "Y_max", "Z_min", "Z_max"]
        ROI = pd.DataFrame(columns = labels, dtype = "float")
        
        # Getting the filenames of the processed images.
        TP = df["TP"].unique().tolist()
        
        # Computing the ROI frame by frame, and adding it to localROI. 
        for tp in TP :
            subdf = df[df["TP"] == tp]
            subdf = subdf[subdf["F_SELECT"]]
            ROI.loc[tp] = [subdf["X"].min(), subdf["X"].max(),
                           subdf["Y"].min(), subdf["Y"].max(),
                           subdf["Z"].min(), subdf["Z"].max()]
        
        globalROI = []
        
        # Running selectROI() method on each column of ROI and adding the
        # resulting value to ROI.
        for col in labels :
            subdf = ROI[col]

            globalROI.append(clustering.select_ROI(subdf, std, eps, 
                                                   min_samples, offset))
            
        ROI.loc["Global"] = globalROI
               
        return ROI
       
    def denoise_timelapse(csv_dirpath, img_dirpath, savepath,
                  std = 15, roi_eps = 2, roi_min_samples = 3, 
                  offset = 5, clust_eps = 40, clust_min_samples = 3, 
                  cIter = 1000, cSample = 10, threshold = 10, 
                  rescaling = [1, 1, 1]):
        
        start_time = time.time()
        
        ## Opening and merging every csv in the 
        step_time = time.time()
        print("Opening files ...", end = " ")
        df = preprocessing.read_spots(csv_dirpath)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        
        
        center = image.get_center(img_dirpath)
        
        ## Clustering the spots.
        step_time = time.time()
        print("Clustering spots ...", end = " ")
        df = clustering.compute_clusters(df, center, clust_eps, 
                                        clust_min_samples, cIter, cSample, 
                                        threshold, rescaling)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        
        ## Getting the ROI using clustering results.
        step_time = time.time()
        print("Getting the ROI using clustering results ...", end = " ")
        ROI = preprocessing.get_ROI(df, std, roi_eps, roi_min_samples, offset)
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
            
        image.save_tif(np.array(cleanArray), savepath)
        print("")
        print("Image saved as :", savepath)
        print("Total time :", str(round(time.time()-start_time, 2))+"s")