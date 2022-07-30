# -*- coding: utf-8 -*-
"""
Preprocessing section analyzing component of OAT.

@author: Alex-932
@version: 0.7
"""

import os
import pandas as pd
from modules.utils.fileimport import *
from modules.utils.clustering import *
from modules.utils.image import *

class preprocessing():
    
    def readSpots(filepath):
        """
        Load segmentation result of Trackmate detection as a dataframe, 
        remove the unwanted columns and return the dataframe.

        Parameters
        ----------
        filepath : str
            Path to the file.

        Returns
        -------
        spots_df : pd.DataFrame
            Dataframe where index are spots and columns are 
            ("QUALITY", "X", "Y", "Z").

        """
        # Importing the csv as a dataframe adn dropping the 3 first columns.
        spots_df = pd.read_csv(filepath)
        spots_df.drop(index = [0, 1, 2], inplace = True)
        
        # Setting the index to be the IDs of the spots.
        spots_df.index = spots_df["LABEL"]
        
        # Renaming columns to have smaller names.
        for axis in ["X", "Y", "Z"]:
            # We rename the column as it is shorter.
            spots_df.rename(columns = {"POSITION_"+axis:axis},
                                  inplace = True)
        # Keeping useful columns.
        spots_df = spots_df.loc[:,["QUALITY", "X", "Y", "Z"]]
        
        # Setting every values to float type.
        spots_df = spots_df.astype("float")
        
        return spots_df
    
    def getSpots(directory):
        """
        Importing and merging all spots.csv files and saving them as 
        self.spots.

        """
        files = os.listdir(directory)
        files.sort()
        filepath = [directory+"\\"+file for file in files]

        data = pd.DataFrame(columns=["QUALITY", "X", "Y", "Z", "TP"],
                            dtype = "float")
        
        ## For the moment, files are sorted and imported as if the time point 
        ## starts at 0.
        for tp in range(len(filepath)):
            ## Importing the csv.
            subdf = preprocessing.readSpots(filepath[tp])
            
            ## Adding the time point.
            subdf["TP"] = subdf.shape[0]*[tp]
            
            ## Merging the dataframe to the main one.
            data = pd.concat([data, subdf])
        
        data["TP"] = data["TP"].astype("int")    
        
        return data
    
    def getROI(df, std = 15, eps = 2, min_samples = 3, offset = 5):
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

            globalROI.append(clustering.selectROI(subdf, std, eps, 
                                                  min_samples, offset))
            
        ROI.loc["Global"] = globalROI
               
        return ROI
       
    def createImg(fileDirpath, imgDirpath, savepath,
                  std = 15, roi_eps = 2, roi_min_samples = 3, 
                  offset = 5, clust_eps = 40, clust_min_samples = 3, 
                  cIter = 1000, cSample = 10, threshold = 10, 
                  rescaling = [1, 1, 1]):
        
        df = preprocessing.getSpots(fileDirpath)
        
        center = image.getVolCenter(imgDirpath)
        
        df = clustering.computeClusters(df, center, clust_eps, 
                                        clust_min_samples, cIter, cSample, 
                                        threshold, rescaling)
        
        ROI = preprocessing.getROI(df, std, roi_eps, roi_min_samples, offset)
        
        cleanArray = []
        
        imgs = os.listdir(imgDirpath)
        imgs.sort()
        imgpath = [imgDirpath+"\\"+img for img in imgs]
        
        for tp in df["TP"].unique().tolist() :
            imarray = image.loadTif(imgpath[tp])
            cleanArray.append(image.denoising(ROI.loc["Global"], imarray))
            
        image.saveTif(np.array(cleanArray), savepath)