# -*- coding: utf-8 -*-
"""
Preprocessing part analyzing component of OAT.

@author: Alex-932
@version: 0.7
"""

import os
import time
import pandas as pd

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

from modules.utils.filemanager import *
from modules.utils.clustering import *
from modules.utils.image import *
from modules.utils.tools import *

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
                                                   min_samples, roi_offset))
            
        ROI.loc["Global"] = globalROI
               
        return ROI
       
    def denoise_timelapse(csv_dirpath, img_dirpath, savepath,
                  std = 15, roi_eps = 2, roi_min_samples = 3, 
                  roi_offset = 5, clust_eps = 40, clust_min_samples = 3, 
                  centroid_iter = 1000, centroid_sample = 10, min_cells = 10, 
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
        centroid_iter : int, optional
            Number of iteration for searching the best centroid. 
            The default is 1000.
        centroid_sample : int, optional
            Number of random spots to compute the centroid from. 
            The default is 10.
        min_cells : int, optional
            Minimal cells for a cluster to be considered. Overpassed if none
            of the clusters contains more than this threshold.
            The default is 10.
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
        df = preprocessing.read_spots(csv_dirpath)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        
        
        center = image.get_center(img_dirpath)
        
        ## Clustering the spots.
        step_time = time.time()
        print("Clustering spots ...", end = " ")
        df = clustering.compute_clusters(df, center, clust_eps, 
                                         clust_min_samples, centroid_iter, centroid_sample, 
                                         min_cells, rescaling)
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
            
        image.save_tif(np.array(cleanArray), savepath)
        print("")
        print("Image saved as :", savepath)
        print("Total time :", str(round(time.time()-start_time, 2))+"s")
        
        return df
        
    def show_spots(df, TP, show = True, save = False, savepath = None,
                   show_centroids = True, color_clusters = True):
        """
        Create a figure showing spots for a given time point.
        The figure can be saved. See parameters below.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing coordinates and time points (X, Y, Z, TP).
        TP : int
            Time point to show the spots.
        show : bool, optional
            Show the figure. The default is True.
        save : bool, optional
            Save the figure, need savepath to work. The default is False.
        savepath : str, optional
            Full path of the output figure. The default is None.
        show_centroids : bool, optional
            Add the centroid to the figure. The default is True.
        color_clusters : bool, optional
            Add cluster information to the figure. The default is True.

        """
        
        ## Setting which time points to display.
        if not isinstance(TP, int) :
            raise TypeError("TP must be int")
            
        ## Creating the figure.
        fig = plt.figure(figsize=(10, 7), dpi = 400)
        plt.style.use("seaborn-paper")
        
        ## Set global font
        plt.rcParams.update({'font.family':'Montserrat'})
        
        legend = []
            
        subdf = df[df["TP"] == TP].copy()
        
        ax = fig.add_subplot(111, projection = "3d")
        
        if color_clusters and "F_SELECT" in subdf.columns:
            t_df = subdf[subdf["F_SELECT"]]
            f_df = subdf[subdf["F_SELECT"] == False]
            
            ax.scatter(t_df["X"], t_df["Y"], t_df["Z"], c = "green", s = 50)
            ax.scatter(f_df["X"], f_df["Y"], f_df["Z"], c = "orange", s = 20)
            
            legend.append(Line2D([0], [0], marker = 'o', 
                                 color = "green", 
                                 label = 'Selected spots',
                                 markerfacecolor = "green", 
                                 markersize=7, ls = ''))
            
            
        else :
            ax.scatter(subdf["X"], subdf["Y"], subdf["Z"], c = "skyblue")
        
        if show_centroids :
            cx, cy, cz = tools.get_centroid(subdf)
            ax.scatter(cx, cy, cz, c="red", marker = "^", s = 50)
            
            legend.append(Line2D([0], [0], marker = '^', 
                                 color = "red", 
                                 label = 'Raw centroid',
                                 markerfacecolor = "red", 
                                 markersize=7, ls = ''))
        
        ax.set_title("Detected spots for time point "+str(TP), fontsize = 15)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        ax.legend(handles = legend, loc = 'best')
        
        if show :
            plt.show()
            
        if save and savepath is not None :
            plt.savefig(savepath, dpi = 400)
            
        plt.close()
            
            
            
            
            