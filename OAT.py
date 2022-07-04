# -*- coding: utf-8 -*-
"""
organoid_tracking_tools (OAT) is a set of methods that integrates FIJI's 
Trackmate csv files output to process cell displacement within an organoid.

@author: Alex-932
@version: 0.5.2
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
#import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
from skimage import io
import tifffile
import cv2
from scipy.spatial import ConvexHull #, convex_hull_plot_2d
from sklearn.decomposition import PCA
from PyVTK import PyVTK

class OAT():
    
    def __init__(self, fiji_dir, wrk_dir = None):
        """
        Initialize the sample analysis by creating the directories needed.

        Parameters
        ----------
        fiji_dir : str
            Directory path for Fiji.app folder.
        wrk_dir : str, optional
            Working directory i.e. the dir. where the tree will be built. 
            The default is the script directory.

        """
        # Creating the directory table
        self.dir = pd.Series(dtype="str")
        
        # Adding the fiji directory
        if not os.path.exists(fiji_dir):
            raise FileNotFoundError("No such fiji directory")
        self.dir["fiji"] = fiji_dir
        
        # Adding the working directory
        if wrk_dir == None :
            self.dir["root"] = os.getcwd()
        else :
            if not os.path.exists(wrk_dir):
                os.makedirs(wrk_dir)
            self.dir["root"] = wrk_dir
            
        # Creating the file table    
        self.files = pd.DataFrame(dtype="str")
        
        # Building the directories tree
        self.buildTree()
        self.version = "0.5.2"      
         
    def buildTree(self):
        """
        Create the directories tree in the working directory.
        """
        # Setting the paths for the diverse components of the pipeline.
        root = self.dir.loc["root"]
        self.dir["tifs"] = root+'\\data\\organoid_images'
        self.dir["spots"] = root+'\\data\\spots'
        self.dir["tracks"] = root+'\\data\\tracks'
        self.dir["out"] = root+'\\output'
        self.dir["figs"] = root+'\\output\\figs'
        self.dir["spotsFigs"] = root+'\\output\\figs\\spots'
        self.dir["vectorsFigs"] = root+'\\output\\figs\\vectors'
        self.dir["anim"] = root+'\\output\\animation'
        self.dir["vtk"] = root+'\\output\\vtk_export'
        
        # Creating the directories if they don't already exist.
        for path in self.dir:
            if not os.path.exists(path):
                os.makedirs(path)
        
    def unstackTif(filepath, suffix, savedir):
        """
        Unstack an image in the given save directory.

        Parameters
        ----------
        filepath : str
            Path to the image file.
        suffix : str
            Suffix name for the output images.
        savedir : str
            Save directory.

        """
        image = io.imread(filepath)
        imarray = np.array(image, dtype="unint8")
        # Browsing the image through the time axis
        for tp in range(imarray.shape[0]):
            tifffile.imwrite(savedir+"\\"+suffix+"_"+str(tp)+".tif", 
                             np.array(imarray[tp]), 
                             imagej=True, metadata={'axes': 'ZYX'})
                
    def importFiles(self, mode):
        """
        Retrieve and check if the given type of input file are good to work 
        with. 

        Parameters
        ----------
        mode : str
            Mode (i.e. target) for the search : 
                - "tifs" for images files
                - "spots" for csv with spots data
                - "tracks" for csvs with edges and tracks data

        """
        # Checking tifs files.
        if mode == "tifs" :
            # Looking for tif images.
            tifs = [file for file in os.listdir(self.dir["tifs"]) 
                         if re.split(r'\.', file)[-1] == "tif"]
            # Checking if there are images.
            if len(tifs) == 0 :
                # If there are no files.
                raise ImportError("No tif image in the directory")
            else :
                self.files["tifs"] = tifs
                self.files.index = [re.split("\.tif", file)[0] 
                                    for file in tifs]
                self.files["time"] = [k for k in range(len(self.files.index))]
                self.sample = os.path.commonprefix(list(self.files["tifs"]))
                
        # Checking csv files related to spots.
        if mode == "spots" :
            # Looking for csv files.
            spots = [file for file in os.listdir(self.dir["spots"])
                                if re.split('\.', file)[-1] == 'csv']
            if len(spots) == 0 :
                self.setInstructions()
                raise ImportError("No .csv found in the spots directory")
            else :
                # Adding and linking them to the file table.
                spots = pd.Series(spots, index = [re.split("\_rs.csv", file)[0]
                                                  for file in spots],
                                  name = "spots", dtype = "str")
                self.files = pd.concat([self.files, spots], axis = 1)
                
        # Checking csv files related to tracks and edges.
        if mode == "tracks" :
            # Looking for csv files containing tracks.
            tracks_csv = self.dir["tracks"]+"\\"+"tracks.csv"
            edges_csv = self.dir["tracks"]+"\\"+"edges.csv"
            if os.path.exists(tracks_csv) :
                self.tracks_csv = tracks_csv
            else : 
                self.tracks_csv = None
            if os.path.exists(edges_csv) :
                self.edges_csv = edges_csv
            else : 
                self.edges_csv = None
            if self.tracks_csv == None or self.edges_csv == None :
                raise ImportError("One or all .csv are missing in the directory")
                                  
    def getVolShape(self):
        """
        Save the shape of the volume as a tuple (Z, Y, X) called self.VolShape.
        Load the first image and get its volume.
        
        """
        # Opening the first tif file.
        file = io.imread(self.dir["tifs"]+"\\"+self.files["tifs"][0])
        
        # Converting it to a numpy array.
        array = np.array(file)
        
        # Saving the shapes
        self.VolShape = array.shape
        
    def loadTif(self):
        """
        Loading the Tifs files from the "\\data\\3D Organoids" directory.

        """
        self.importFiles("tifs")
        self.getVolShape()
                
    def setInstructions(self):
        """
        Create "OAT_instructions.txt" in the fiji directory for 
        trackmate_script.py to retrieve the input image file (tif) and the path
        to the .csv output. 

        """
        #Creating the instruction file in the fiij directory.
        instructions = open(self.fiji_dir+"\\OAT_instructions.txt", "w")
        
        #The formatting of the instructions are as bellow:
        #   Each row correspond to an image file.
        #   The row is seperated in half by a comma.
        #   Before the comma is the input image file path.
        #   After the comma is the output .csv file path.
        
        for file in self.files["tifs"]:
            instructions.write(
                self.dir["tifs"]+"\\"+file+","+self.dir["spots"]+"\\"+\
                                   re.split("\.", file)[0]+"_rs.csv\n")
        
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
    
    def getSpots(self):
        """
        Importing and merging all spots.csv files and saving them as 
        self.spots.

        """
        # Checking the .csv files and getting the list of names.
        self.importFiles("spots")
        
        # Creating the spots dataframe which contains info regarding each spots
        # for each image file (or timepoint).
        self.spots = pd.DataFrame(columns=["QUALITY", "X", "Y", "Z",
                                           "FILE"], dtype=("float"))
        
        # Setting the correct type for the "FILE" and 'CLUSTER' columns.
        self.spots["FILE"] = self.spots["FILE"].astype("str")
        
        # Reading each file and adding them to the spots dataframe.
        for file in self.files["spots"] :
            read = OAT.readSpots(self.dir["spots"]+"\\"+file)
            read["FILE"] = len(read.index)*[re.split("_rs.csv", file)[0]]
            self.spots = pd.concat([self.spots, read])
    
    def euclidDist(PointA, PointB):
        """
        Return the euclid distance between PointA and PointB. 
        Works in 3D as well as in 2D. 
        Just make sure that both points have the same dimension.

        Parameters
        ----------
        PointA : list
            List of the coordinates (length = dimension).
        PointB : list
            List of the coordinates (length = dimension).

        Returns
        -------
        float
            Euclid distance between the 2 given points.

        """
        # Computing (xa-xb)², (ya-yb)², ...
        sqDist = [(PointA[k]-PointB[k])**2 for k in range(len(PointA))]
        
        return np.sqrt(sum(sqDist)) 
    
    def getCentroid(spots_df):
        """
        Process the "euclid" centroid from a set of points. Get the centroid by
        getting the mean value on each axis.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with the points and their coordinates.

        Returns
        -------
        centroid : list
            List that contains the coordinates of the computed centroid.

        """
        centroid = []
        for axis in ["X", "Y", "Z"]:
            centroid.append(spots_df[axis].mean())
            
        return centroid
    
    def reScaling(df, ratio = [1, 1, 1]):
        """
        Rescale the coordinates of the given axis by the given ratio. 

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe which contains axis as columns and spots as index.
        ratio : list
            Coordinates multiplier for each axis : [Xratio, Yratio, Zratio].

        Returns
        -------
        df : pandas.DataFrame
            Same as the input one but with rescaled coordinates.

        """
        out_df = pd.DataFrame(index = df.index, columns = df.columns)        
        out_df["X"] = df.loc[:, "X"]*ratio[0]
        out_df["Y"] = df.loc[:, "Y"]*ratio[1]
        out_df["Z"] = df.loc[:, "Z"]*ratio[2]
        
        return out_df

    def selectCluster(subdf, center, column, threshold = 10):
        """
        Search for the cluster that is the most likely of being the
        organoid cluster.
        
        The algorithm compute the centroid of each cluster.
        
        Then it computes the distance between each centroid and the center of 
        the volume.
        
        Finally, it returns the cluster that is the closest to the center and
        that contains more than {threshold} spots.

        Parameters
        ----------
        subdf : pd.DataFrame
            Dataframe that contains spots coordinates as well as clustering
            results.
        center : list 
            Center coordinates of the whole volume as follow [X, Y, Z].
        column : str
            Name of the column that contains cluster IDs.
        threshold : int, optional
            Number of spots a cluster must have to be selected. The default is
            10.

        Returns
        -------
        selected : pd.Series
            Index is the ID of the spots, values are booleans.
            

        """
        # Retrieving the ID of the clusters as well as the number of spots they
        # contains.
        clustersInfo = subdf[column].value_counts()
        clustersID = clustersInfo.index
        
        # Creating a Series to store the distances between the centroid and the
        # center.
        dist = pd.Series(dtype = "float")
        
        # Computing the distance for each clusters.
        for ID in clustersID :
            centroid = OAT.getCentroid(subdf[subdf[column] == ID])
            distance = OAT.euclidDist(centroid, center)
            dist.loc[ID] = distance
        
        # Sorting from the lowest to the greatest distance.
        dist.sort_values(ascending = True, inplace = True)
        
        # Going through the closest to farthest cluster until it contains more 
        # than {threshold} spots. If there are no cluster that meets both 
        # conditions, we take the first one.
        selectedClusterID = 0
        for idx in range(len(dist.index)):
            if clustersInfo[dist.index[idx]] >= threshold:
                selectedClusterID = dist.index[idx]
                break
        
        # Returning the selection result as a pd.Series
        return subdf[column] == selectedClusterID      
    
    def clusteringEngine(subdf, center, cIter = 100, cSample = 10, 
                         eps = 40, min_samples = 3, threshold = 10):
        """
        Cluster and select the spots that are more likely to be part of the
        organoid.
        
        First, the algorithm compute the centroid of the organoid by taking 
        {cSample} random spots and computing their centroid. 
        It's repeated {cIter} times then it takes the average centroid.
        
        A DBSCAN is then runned on the distance between the spots and the 
        centroid as we expect a spike at a certain distance given all spots 
        that are part of the organoid should be at the same distance.
        DBSCAN give the spots for each spikes and we select the right spike by 
        taking the one that is closer to the centroid.
        
        A second DBSCAN is runned on the spots of the selected spikes to 
        separate the ones that are close but not part of the organoid. The 
        cluster is also selected by the selectCluster method.  

        Parameters
        ----------
        subdf : pd.DataFrame
            Spots to cluster. Same formatting as self.spots expected.
        center : list 
            Center coordinates of the whole volume as follow [X, Y, Z].
        cIter :  int, optional
            cIter number for the centroid location. The default is 100.
        cSample : int, optional
            Number of spots to compute the centroid. The default is 10.
        eps : int, optional
            Radius of search for the 2nd DBSCAN algorithm. The default is 40.
        min_samples : int , optional
            Min. number of neighbors for the 2nd DBSCAN. The default is 3.
        threshold : int, optional
            Number of spots a cluster must have to be selected. The default is
            10.

        Returns
        -------
        Results : pd.DataFrame
            Dataframe where index are the spots ID and columns are :
                A_CLUSTER : Clusters ID (int) for the first clustering step.
                A_SELECT : Selected spots for the first clustering (bool).
                F_CLUSTER : Clusters ID (int) for the 2nd clustering step.
                F_SELECT : Selected spots for the second clustering (bool).

        """
        # Getting the centroid.
        centroids = [OAT.getCentroid(subdf.sample(cSample, axis=0)) \
                     for k in range(cIter)]
        centroids = pd.DataFrame(centroids, columns=["X", "Y", "Z"])
        centroid = [centroids["X"].median(), 
                    centroids["Y"].median(),
                    centroids["Z"].median()]
        
        # Computing the distance between each point and the centroid.
        distance = pd.Series(dtype = "float", name = "DISTANCE")
        for points in subdf.index:
            distance[points] = OAT.euclidDist(
                list(subdf.loc[points, ["X", "Y", "Z"]]), 
                centroid)
        
        # Clustering the distances and saving it as a pd.Series.
        cluster = DBSCAN(eps=5, min_samples=6).fit_predict(distance.to_frame())
        cluster = pd.Series(cluster, index = subdf.index, name = "A_CLUSTER")
        
        # Creating the final dataframe with the first clustering results.
        Results = cluster.to_frame()
        
        # Selecting the cluster based on the OAT.selectCluster method.    
        selected = OAT.selectCluster(pd.concat([subdf, cluster], axis = 1),
                                     center, column = "A_CLUSTER",
                                     threshold = threshold)
        selected.name = "A_SELECT"
        
        # Adding the selection results to Results dataframe.
        Results = pd.concat([Results, selected], axis = 1)
        
        # Keeping the selected spots for the next clustering step.
        subdf = subdf[selected].loc[:,["X", "Y", "Z"]]
        
        # Clustering the spots.
        cluster = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(subdf)
        cluster = pd.Series(cluster, index = subdf.index, name = "F_CLUSTER")
        
        # Merging the clusters ID to the Results dataframe. 
        Results = pd.concat([Results, cluster], axis = 1)
        
        # Selecting the right cluster once again using the same method.
        selected = OAT.selectCluster(pd.concat([subdf, cluster], axis = 1),
                                     center, column = "F_CLUSTER", 
                                     threshold = threshold)
        selected.name = "F_SELECT"
        
        # Merging the selection to the Results dataframe.
        Results = pd.concat([Results, selected], axis = 1)
        
        # Filling the NaN values in the 2nd clustering results as some spots
        # were not computed.
        Results["F_CLUSTER"].fillna(100, inplace = True)
        Results["F_SELECT"].fillna(False, inplace = True)
        
        return Results
        
    def getClusters(self, df = "spots", eps = 40, min_samples = 3, 
                    cIter = 1000, cSample = 10, threshold = 10, 
                    rescaling = [1, 1, 1]):
        """
        Clustering the spots for each frame using the .clusteringEngine() 
        method.

        Parameters
        ----------
        df : str, optional
            Name of the dataframe. The default is "spots".
            It can be either "spots" or "tracks".
        eps : int, optional
            See .clusteringEngine() method. The default is 40.
        min_samples : int , optional
            See .clusteringEngine() method. The default is 3.
        cIter :  int, optional
            See .clusteringEngine() method. The default is 1000.
        cSample : int, optional
            See .clusteringEngine() method. The default is 10.
        threshold : int, optional
            See .clusteringEngine() method. The default is 10.
        rescaling : list, otpional
            Rescale the spots coordinates on each axis by the given value :
                [Xratio, Yratio, Zratio].
            The default is [1, 1, 1].

        """
        # Selecting the dataframe on which the clustering will be done.
        if df == "tracks" and hasattr(self, "tracks"): 
            data = self.tracks.loc[:,["X", "Y", "Z", "FILE"]]
        elif  df == "spots" and hasattr(self, "spots"):
            data = self.spots.loc[:,["X", "Y", "Z", "FILE"]]
        else :
            raise ValueError("The df is not supported or it doesn't exist")
        
        # Computing the coordinates of the volume's center.
        center = [self.VolShape[-axis]/2 for axis in range(len(
            self.VolShape))]
        
        # Clustering every spots, frame by frame and adding the results to the
        # res temporary datafame.
        Results = pd.DataFrame(dtype = "float")
        for file in self.files.index :
            subdf = data[data["FILE"] == file]
            subdf = OAT.reScaling(subdf, ratio = rescaling)
            clusterResults = OAT.clusteringEngine(subdf, center, cIter,
                                                  cSample, eps, min_samples,
                                                  threshold)
            Results = pd.concat([Results, clusterResults])
            
        # Adding the cluster infos to the right dataframe. 
        if df == "tracks":
            self.tracks = pd.concat([self.tracks, Results], axis = 1)
        elif df == "spots":
            self.spots = pd.concat([self.spots, Results], axis = 1)
    
    def showSpots(self, filename, ROI = False, save = False, df = "spots", 
                  figsize = (20, 8), dpi = 400, color = "b", cmap = 'tab10'):
        """
        Create and shows a set of 3 scatterplots for each plane.
        Each one represent the spots of the given file, colored by cluster.

        Parameters
        ----------
        filename : str or list
            Name of the image file or list of the name of the image files. 
            Do not include the '.tif'.
            Use "all" to show the spots for all frames. 
        ROI : bool, optional
            If True, only show the spots within the ROI. The default is False.
        save : bool, optional
            If True, save the figure(s) in the \\output\\clustering directory.
            The default is False.
        df : str, optional
            Name of the dataframe. The default is "spots".
            It can be either "spots" or "tracks".
        figsize : couple, optional
            Size of the figure as matplotlib accept it. The default is (20, 8).
        dpi : int, optional
            DPI of the figure. The default is 400.
        color : str, optional
            Default matplotlib color if no clustering info. The default is "b".
        cmap : str, optional
            matplotlib cmap used when showing clusters. The default is "tab10".

        """
        # Setting the cmap.
        cmap = plt.cm.get_cmap(cmap)
        
        # If the user wants to see all frames.
        if filename == "all":
            for file in self.files.index:
                self.showSpots(file, ROI, save, df)
            return None
        
        # If the user wants several frames. 
        elif type(filename) == list :
            for file in filename:
                self.showSpots(file, ROI, save, df)
            return None
        
        # The actual figure generation.
        elif type(filename) == str :
            fig, axs = plt.subplots(1, 3, figsize = figsize, dpi = dpi)
            
            # Saving the different columns to look up depending on the view.
            planes = [["X","Y"],["X","Z"],["Y","Z"]]
            ROIcol = [[0, 2], [0, 4], [2, 4]]
            
            # Selecting the dataframe.
            if df == "spots":
                subdf = self.spots[self.spots["FILE"] == filename]
            elif df == "tracks":
                subdf = self.tracks[self.tracks["FILE"] == filename]
                
            # 3 plane, 3 axes.
            for idx in range(3):
                # Coloring the clusters if info are available.
                if "F_SELECT" in subdf.columns :
                    color = subdf["F_CLUSTER"].map(cmap)
                
                # Plotting and labeling axis and title.
                axs[idx].scatter(subdf[planes[idx][0]],
                                 subdf[planes[idx][1]],
                                 c = color)
                axs[idx].set_xlabel(planes[idx][0])
                axs[idx].set_ylabel(planes[idx][1])
                axs[idx].set_title("File : "+filename+", View : "+\
                                   planes[idx][0]+"*"+planes[idx][1])
                    
                # If cluster info are available, adding a legend to show the
                # selected one's color.
                if "F_SELECT" in subdf.columns :
                    cluster_id = subdf[subdf["F_SELECT"]]["F_CLUSTER"][0]
                    legend = [Line2D([0], [0], marker = 'o', 
                                     color = cmap(cluster_id), 
                                     label = 'Selected spots', 
                                     markerfacecolor = cmap(cluster_id), 
                                     markersize=10)]
                    axs[idx].legend(handles = legend, loc = 'best')
                
                # If ROI has been computed and the user want to crop the 
                # volume.
                if hasattr(self, "ROI") and ROI:
                    axs[idx].set_xlim([self.ROI[ROIcol[idx][0]], 
                                       self.ROI[ROIcol[idx][0]+1]])
                    axs[idx].set_ylim([self.ROI[ROIcol[idx][1]], 
                                       self.ROI[ROIcol[idx][1]+1]])
        
        # Adding the version of OAT.
        fig.text(0.08, 0.05, "OAT Version : "+str(self.version))
        
        # Saving if wanted. 
        if save :
            plt.savefig(self.dir["spotsFigs"]+"\\"+filename+".png", dpi = dpi)
        
        # Showing the plot and closing it.
        plt.show()
        plt.close(fig)
        
    def exportSpotsVTK(self, filename, organoid = False, df = "spots"):
        """
        Export the given files to a points type .vtk for visualization in 
        paraview.

        Parameters
        ----------
        filename : str or list
            Name of the image file or list of the name of the image files. 
            Do not include the '.tif'

        organoid : bool, optional (default is False)
            True : Export only the spots that are supposed to be in the 
            organoid.        

        """
        # Exporting all spots, frame by frame, by calling back the method.
        if filename == "all":
            for file in self.files.index:
                self.exportSpotsVTK(file, organoid, df)
        
        #Exporting the spots from the given frames, by calling back the method.
        elif type(filename) == list:
            for file in filename:
                self.exportSpotsVTK(file, organoid, df) 
        
        # Actual export of the spots for the given file.
        elif type(filename) == str:
            
            # Selecting the wanted dataframe.
            if df == "spots":
                subdf = self.spots[self.spots["FILE"] == filename]
            elif df == "tracks":
                subdf = self.tracks[self.tracks["FILE"] == filename]
                
            # Selecting spots if available and if the user wants it.
            if "F_SELECT" in subdf.columns and organoid:
                subdf = subdf[subdf["IS_ORGANOID"]]
                PyVTK(filename+"_"+df+"_organoid", subdf['X'], subdf['Y'], 
                  subdf['Z'], self.dir["vtk"], "points")
            
            # Else, showing everything.
            else :
                PyVTK(filename+"_"+df, subdf['X'], subdf['Y'], 
                      subdf['Z'], self.dir["vtk"], "points")
                
                
    def selectROI(subdf, std = 15, eps = 2, min_samples = 3, offset = 5):
        """
        Method to get the most representative value from a pd.Series in the 
        context of the search of the ROI.

        Parameters
        ----------
        subdf : pd.Series
            Series or df column that contain values for a certain category.
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

        Returns
        -------
        float
            Best fitting value for the limit of the ROI.

        """
        # If the standard deviation is small, we don't need to cluster, all
        # spots are given the same clusterID (0).
        if subdf.std() <= std :
            results = pd.Series(len(subdf.index)*[0], 
                                index = subdf.index, 
                                name = subdf.name)
            
        # Else, clustering with DBSCAN to get the most representative values.
        else :
            results = DBSCAN(eps = eps, min_samples = min_samples).fit_predict(
                subdf.to_frame())
            results = pd.Series(results, 
                                index = subdf.index, 
                                name = subdf.name)
            
        # Getting the biggest cluster ID.
        biggestClusterID = results.value_counts(ascending = False).index[0]
        
        # Getting the side of the limit (min or max)
        extreme = re.split("\_", subdf.name)[-1]
        
        # Returning the limit value +/- an offset depending 
        # on the side (min/max).
        if extreme == "min":
            value = subdf[results == biggestClusterID].min()
            return value-offset
        elif extreme == "max":
            value = subdf[results == biggestClusterID].max()
            return value+offset
                   
    def getROI(self, std = 15, eps = 2, min_samples = 3, offset = 5):
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
        # Importing the spots..
        self.getSpots()
        
        # Clustering the spots.
        self.getClusters()
        
        # Creating the dataframe that contains all the ROI limits.
        labels = ["X_min", "X_max", "Y_min", "Y_max", "Z_min", "Z_max"]
        localROI = pd.DataFrame(columns = labels, dtype = "float")
        
        # Getting the filenames of the processed images.
        filenames = self.spots["FILE"].value_counts().index
        
        # Computing the ROI frame by frame, and adding it to localROI. 
        for name in filenames :
            subdf = self.spots[self.spots["FILE"] == name]
            subdf = subdf[subdf["F_SELECT"]]
            localROI.loc[name] = [subdf["X"].min(), subdf["X"].max(),
                                  subdf["Y"].min(), subdf["Y"].max(),
                                  subdf["Z"].min(), subdf["Z"].max()]
        
        # Creating the Series that will contain the global ROI.
        self.ROI = pd.Series(index = labels, name = "ROI", dtype = "float")
        
        # Running selectROI() method on each column of localROI and adding the
        # resulting value to self.ROI.
        for col in labels :
            subdf = localROI[col]
            self.ROI[col] = OAT.selectROI(subdf, std, eps, min_samples, offset)
            
        # Adding the global ROI in the local ROI dataframe.
        self.localROI = localROI
        self.localROI.loc["GLOBAL"] = list(self.ROI)
        
    def getArea(spot, radius, volShape):
        return [[x for x in range(int(spot["X"])-radius, 
                                  int(spot["X"])+radius+1)],
                [y for y in range(int(spot["Y"])-radius, 
                                  int(spot["Y"])+radius+1)],
                [z for z in range(int(spot["Z"])-radius, 
                                  int(spot["Z"])+radius+1)]]
        
    def denoising(ROI, file):
        """
        Load a tif image and set all pixels that are not in the ROI to 0.
        Used in the cleanImage method.

        Parameters
        ----------
        ROI : pd.Series
            Formatting is the same as self.ROI : Index are 
            ["X_min", "X_max", "Y_min", "Y_max", "Z_min", "Z_max"].
        file : str
            Path to the image.

        Returns
        -------
        imarray : np.array
            Denoised array of the image.

        """
        # Opening the image.
        image = io.imread(file)
        
        # Converting into a Numpy array. The shape is in this order : Z, Y, X.
        imarray = np.array(image)
        
        
        # For each axis, we get the coordinate (1D) of the pixels that needs to 
        # be set to 0. 
        X_values = [X for X in range(imarray.shape[2]) if
                     X < ROI["X_min"] or X > ROI["X_max"]]
        Y_values = [Y for Y in range(imarray.shape[1]) if
                     Y < ROI["Y_min"] or Y > ROI["Y_max"]]
        Z_values = [Z for Z in range(imarray.shape[0]) if
                     Z < ROI["Z_min"] or Z > ROI["Z_max"]]
        
        # Setting the given pixel to 0 on each axis.
        imarray[Z_values,:,:] = 0
        imarray[:,Y_values,:] = 0
        imarray[:,:,X_values] = 0
        
        return imarray
    
        ### IDEA : Removing spots from unwanted clusters.
        # # Removing last spots that could still be in the frames.
        # filename = re.split("\.tif", re.split(r"\\", file)[-1])[0]
        # subdf = self.spots[self.spots["FILE"] == filename]
        # for spot_id in subdf[subdf["F_SELECT"] == False].index :
        #     spot = subdf.loc[spot_id]
        #     if spot["X"] not in X_values and spot["Y"] not in Y_values and \
        #         spot["Z"] not in Z_values:
        #             area = OAT.getArea(spot, 10, self.VolShape)
        #             print(area)
        #             imarray[area[2], area[1], area[0]] = 0
        
        
    def cleanImage(self):
        """
        Load all input images in the \\data\\organoid images directory. 
        Then, runs the denoising() function on it. 
        Merge and save the cleaned image in \\data directory.

        """
        #Creating the 4D array.
        imarray = []
        
        #Adding each 3D array in the list in chronological order.
        for file in self.files["tifs"]:
            imarray.append(OAT.denoising(self.ROI,
                                           self.dir["tifs"]+'\\'+file))
            
        #Saving the tif with th correct metadata.
        tifffile.imwrite(self.dir["root"]+'\\data\\'+self.sample+"_tp.tif", 
                         np.array(imarray), 
                         imagej=True, metadata={'axes': 'TZYX'})
        
    def computeSpotsLinks(self):
        """
        Generate a Series containing , for each track (row), the list of 
        spots ID that they are composed of, in chronological order.
        The Series is called self.spotsLinks.

        """
        # Getting the target for each spots.
        links = self.tracks["TARGET"]
        
        # Every Nan means that the spots has no target, so it is the last one.
        # We will build the lists by going backward.
        enders = links[links.isnull()].index
        
        # Creating the final list and creating the sublists with the enders ID.
        spotsLinks = [[ID] for ID in enders]
        
        # Looping the spots until no backward link is established.
        unfinished = True
        
        while unfinished:
            unfinished = False
            
            for track in spotsLinks:
                # Trying to add the ID of the previous spots from the last spot
                # in the sublist.
                try : 
                    track.append(links[links == track[-1]].index[0])
                    # If it works, then there is a connection.
                    unfinished = True
                except :
                    pass
        
        # Reversing each sublist.
        for track in spotsLinks :
            track.reverse()
        
        # Saving the info.
        self.spotsLinks = pd.Series(spotsLinks, 
                                    index = [
                                        self.tracks.loc[idx[0]]["TRACK_ID"] 
                                        for idx in spotsLinks])
        
    def readTracks(self):
        """
        Load tracks file (tracks.csv and edges.csv) in the 
        \\data\\tracks directory as a DataFrame called self.tracks. 
        
        tracks.csv correspond to the .csv you can get by saving the dataset 
        found in tracks>spots.
        
        tracks.csv correspond to the .csv you can get by saving the dataset 
        found in tracks>edges.

        """
        # Importing the files.
        self.importFiles("tracks")
        tracks_df = pd.read_csv(self.tracks_csv)
        edges_df = pd.read_csv(self.edges_csv)
        
        # Removing the 3 first rows as they are redundant with the labels
        tracks_df.drop(index = [0, 1, 2], inplace = True)
        edges_df.drop(index = [0, 1, 2], inplace = True)
        
        # Setting the spots ID as index in both dataframes.
        tracks_df.index = tracks_df["LABEL"]
        edges_df.index = [re.split(" ", _lbl)[0] for _lbl in 
                               edges_df["LABEL"]]
        
        # Renaming some labels in order to be cleaner
        for axis in ["X", "Y", "Z"]:
            tracks_df.rename(columns = {"POSITION_"+axis:axis},
                                  inplace = True)
        edges_df.rename(columns = {"SPOT_TARGET_ID":"TARGET"}, 
                        inplace = True)
        tracks_df.rename(columns = {"POSITION_T":"TP"}, inplace = True)
        
        # Keeping the interesting columns 
        tracks_df = tracks_df.loc[:,["TRACK_ID", "QUALITY", "X", "Y", "Z", "T", 
                                     "FRAME"]]
        edges_df = edges_df.loc[:,"TARGET"]
        
        # Setting the dataframes' values type
        tracks_df = tracks_df.astype("float")
        tracks_df["TRACK_ID"] = tracks_df["TRACK_ID"].astype("int")
        edges_df = edges_df.astype("float")
        
        # Setting the correct names for the files
        tracks_df.rename(columns = {"FRAME": "FILE"}, inplace = True)
        tracks_df["FILE"] = [self.files.iloc[int(k)].name \
                             for k in tracks_df["FILE"]]
            
        # Modifying the "Target" values to be spots ID.
        edges_df["TARGET"] = ["ID"+str(int(_id)) for _id 
                                   in edges_df["TARGET"]]
        
        # Merging the 2 dataframes.
        self.tracks = tracks_df.join(edges_df)
        
        # Computing the links between spots.
        self.computeSpotsLinks()
        
    def computeVectors(PtA, PtB, toList = False):
        """
        Return the 2D or 3D displacement vector between 2 points.
        The vector is oriented from PtA to PtB. 
        Both input must have the same dimension.
        
        Parameters
        ----------
        PtA : list or pd.Series
            List or Series containing the coordinates values. For example :
            [X, Y, Z].
        PtB : list or pd.Series
            List or Series containing the coordinates values. For example :
            [X, Y, Z].
        toList : bool, optional
            Force to save the vectors coordinates as a list.
        
        Returns
        -------
        list or pd.Series
            Return the coordinates of the vector in the same format as PtA.
    
        """
        vect = [PtB[axis] - PtA[axis] for axis in range(len(PtA))]
        
        if type(PtA) == list or toList:
            return vect
        else :
            return pd.Series(vect, index = PtA.index, dtype="float") 
    
    def getVectors(self, filtering = False, reimport = False):
        """
        Compute displacement vectors for every spots in the tracks dataframe 
        and add them to it.
        
        Vectors are computed based on the sequence of the track they're in.
        That means that vectors are computed between 2 following spots of a 
        given track.
        
        They are saved in the same line as the origin spot of the vector.
        
        Parameters
        ----------
        filtering : bool, optional
            If True, use getClusters() on tracks dataframe and keep the 
            selected ones (F_SELECT = True).
        reimport : bool, optional
            If True, reimport self.tracks.
    
        """
        # Importing the tracks if not done or if user wants to reimport it.
        if not hasattr(self, "tracks") or reimport:
            self.readTracks()
        
        # Removing the previous computation of the vectors in case it already 
        # has been done.
        if "uX" in self.tracks.columns:
            self.tracks.drop(columns = ["uX", "vY", "wZ"], inplace = True)
                      
        # Clustering and removing bad spots in self.tracks dataframe if it 
        # hasn't already been done.     
        if filtering and "F_SELECT" not in self.tracks.columns:
            self.getClusters(df="tracks")
            self.tracks = self.tracks[self.tracks["F_SELECT"]]
            
        # Creating a dataframe to store vectors as they are computed.
        # Each row is a vector with its index being the ID of its origin spot.
        vectors = pd.DataFrame(columns = ["uX", "vY", "wZ"], dtype = "float")
        
        # Computing vectors, track by track.
        for trackID in self.spotsLinks.index :
            for spot in range(len(self.spotsLinks[trackID])-1) :
                
                # Retrieving spot and target spot IDs.
                spotID = self.spotsLinks[trackID][spot]
                nextSpotID = self.spotsLinks[trackID][spot+1]
                
                # Retrieving the spot and target spot coordinates
                spotInfo = self.track.loc[spotID, ["X", "Y", "Z"]]
                nextSpotInfo = self.track.loc[nextSpotID, ["X", "Y", "Z"]]
                
                # Adding the computed vectors to the vectos dataframe.
                vectors[spotID] = OAT.computeVectors(spotInfo, 
                                                     nextSpotInfo,
                                                     toList = True)
                
            # Adding a null vector as the last spot don't have any vector.
            vectors[self.spotsLinks[trackID][-1]] = 3*[np.nan]
            
        # Merging the vectors dataframe to self.tracks. 
        self.tracks = pd.concat([self.tracks, vectors], axis = 1) 
        
    def showVectors(self, TP, angles = None, label = "3D",
                    translation = False, show = True, 
                    save = False):
        """
        Create a figure with a representation of the vector field. The figure 
        is then saved.
    
        Parameters
        ----------
        TP : float
            Time point.
        angles : tuple, optional
            Viewing angle as follow (lat,long). The default is np.nan as for 
            default.
        label : str, optional
            Name of the representation. The default is "3D".
        lim : list, optional
            Limits for the axis. Format is as follow : 
                [[xmin, xmax], [ymin, ymax], [zmin, zmax]] 
            The default is np.nan.
        translation : bool, optional
            Use the translated points (self.translationCoord must
            have been used to generate them)
    
        Returns
        -------
        Save the vector field as a picture in the "Vector field" folder.
    
        """
        #Subsampling the dataframe with the spots of the given timepoint.
        subdf = self.tracks[self.tracks["EDGE_TIME"] == TP]
        #Initializing the figure and its unique axes.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #Plotting the vector field. Syntax : ax.quiver(x, y, z, u, v, w)
        if not translation:
            ax.quiver(subdf["X"], subdf["Y"], subdf["Z"], 
                      subdf["uX"], subdf["vY"], subdf["wZ"])
            if "RA_uX" in self.data.columns:
                RA = self.data.loc[TP-0.5]
                ax.quiver(RA["cX"], RA["cY"], RA["cZ"], 
                          RA["RA_uX"], RA["RA_vY"], RA["RA_wZ"],
                          color = "red", length = 5, pivot = "middle")
        else :
            ax.quiver(subdf["tX"], subdf["tY"], subdf["tZ"], 
                      subdf["uX"], subdf["vY"], subdf["wZ"])
        #Labeling axis
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        #Giving a title to the figure (see further if an angle is provided)
        ax.set_title("Time point : "+str(TP))
        #Setting limits to axis if limits are given
        if label == "X":
            ymax, ymin = subdf["Y"].max(), subdf["Y"].min()
            ax.set_ylim3d([ymax-(ymax-ymin)/2, ymax+10])
            ax.set_yticks([])
            #print([ymax-(ymax-ymin)/2, ymax+10])
        elif label == "Y":
            xmax, xmin = subdf["X"].max(), subdf["X"].min()
            ax.set_xlim3d([xmax-(xmax-xmin)/2, xmax+10])
            ax.set_xticks([])
            #print([xmax-(xmax-xmin)/2, xmax+10])
        elif label == "Z":
            zmax, zmin = subdf["Z"].max(), subdf["Z"].min()
            ax.set_zlim3d([zmax-(zmax-zmin)/2, zmax+10])
            ax.set_zticks([])
            #print([zmax-(zmax-zmin)/2, zmax+10])
        #Setting the viewing angle if provided and renaming the figure to
        #include the angle information
        if type(angles) == tuple:
            ax.view_init(angles[0],angles[1])
            ax.set_title("Timepoint : "+str(TP)+', Angle : ('+str(angles[0])+\
                         ","+str(angles[1])+")")
        #Saving the figure as an image. Name template is as follow :
        #{name of the file (argument given in _init_)}_vf_({TP})_{label}.png
        if show :
            plt.show()
        if save :
            plt.savefig(self.dir["root"]+"\\output\\figs\\"+self.sample+\
                        "_vf_("+str(TP)+")_"+label+".png", dpi=400)
        plt.close(fig)
        
    def animVectors(self, translation = False):
        """
        Method to generate a film showing the displacement vector field at 
        every timepoint available.  
    
        Returns
        -------
        Save the film in the "Vector field" folder as a .avi file.
    
        """
        #Retrieving available timepoints and keeping those who have more than 
        #10 spots. That way we remove incorrect vectors. 
        TP_list = [k for k in self.tracks["EDGE_TIME"].value_counts().index 
                    if self.tracks["EDGE_TIME"].value_counts()[k] >= 10]
        #Sorting the list.
        TP_list.sort()
        #Setting the different views we want in our representation along with 
        #limits for each one of them.
        angles = [None, (0, 90), (0, 0), (90, 0)]
        labels = ["3D","X","Y","Z"]
        #img_array will save the various images opened with opencv.
        img_array = []
        #Browsing the angle of each representation and making 1 figure for each
        #timepoint.
        for idx in range(len(angles)) :
            for TP in TP_list:
                #Creating a figure and saving it as an image with the 
                #showVectors method.
                self.showVectors(TP, angles[idx], labels[idx],
                                 translation = translation,
                                 show = False, save = True)
                #opening the image that have just been created.
                img = cv2.imread(
                    self.dir["root"]+"\\output\\figs\\"+self.sample+\
                    "_vf_("+str(TP)+")_"+labels[idx]+".png")
                #Retrieving the image size to set the vido size.
                height, width, layers = img.shape
                size = (width,height)
                #Adding the opencv object containing the image in the
                #img_array.
                img_array.append(img)
        #Creating and opening the videofile container using opencv. 
        out = cv2.VideoWriter(self.dir["anim"]+"\\"+self.sample+"_vf.avi", 0,  
                               cv2.VideoWriter_fourcc(*'DIVX'), fps = 1, 
                               frameSize = size)
        #Loading every images present in img_array into the video container.
        for i in range(len(img_array)):
            out.write(img_array[i])
        #Releasing and effectively saving the videofile
        out.release()
        
    def exportVectorsVTK(self, tp = "all", tracks = "all"):
        #Retrieving available timepoints and keeping those who have more than 
        #10 spots. That way we remove incorrect vectors.
        
        # Creating the list of time points to show by the user choice.
        if tp == "all":
            TP_list = [k for k in self.tracks["EDGE_TIME"].value_counts().index 
                        if self.tracks["EDGE_TIME"].value_counts()[k] >= 10]
        elif type(tp) == list:
            TP_list = tp
        elif type(tp) == int:
            TP_list = [tp]
        
        # Creating the list of tracks that will be exported.
        if tracks == "all":
            tracks_list = [k for k in 
                           self.tracks["TRACK_ID"].value_counts().index]
        elif type(tracks) == list:
            tracks_list = tracks
        elif type(tracks) == int:
            tracks_list = [tracks]
        
        # Sorting the list of time point because we want to increment following
        # the chronological order. 
        TP_list.sort()
        tracks_list.sort()
        
        #Saving points coordinates in 3 dataframes, one for each axis. Rows are
        #timepoints and columns are tracks.
        list_X, list_Y, list_Z = [], [], []
        #Filling previous lists with sublists. Each sublist has the 
        #corresponding axis coordinates of all points for 1 timepoint. It's 
        #like an image of the cells on the X axis at a certain timepoint.
        for tp in TP_list :
            #Subsampling tracking_df with the values for a given timepoints
            subdf = self.tracks[self.tracks["EDGE_TIME"] == tp]
            list_X.append([])
            list_Y.append([])
            list_Z.append([])
            for track in tracks_list :
                #Retrieving the point data  
                _point_data = subdf.loc[subdf["TRACK_ID"] == track]
                #Checking if the point exist. If it doesn't then we enter no 
                #data (represented by a NaN value.) We will deal with these 
                #holes later
                if _point_data.empty :
                    list_X[-1].append(np.nan)
                    list_Y[-1].append(np.nan)
                    list_Z[-1].append(np.nan)
                else :
                    list_X[-1].append(_point_data["X"][0])
                    list_Y[-1].append(_point_data["Y"][0])
                    list_Z[-1].append(_point_data["Z"][0])
        Xpoints = pd.DataFrame(list_X, index = TP_list, 
                                    columns = tracks_list)
        Ypoints = pd.DataFrame(list_Y, index = TP_list, 
                                    columns = tracks_list)
        Zpoints = pd.DataFrame(list_Z, index = TP_list, 
                                    columns = tracks_list)
        #Filling the empty values : if the track start later, we fill previous
        #positions with the starting one (bfill). We do the same thing the 
        #other way when it stop sooner (ffill).
        Xpoints.fillna(method = 'ffill', inplace = True)
        Ypoints.fillna(method = 'ffill', inplace = True)
        Zpoints.fillna(method = 'ffill', inplace = True)
        Xpoints.fillna(method = 'bfill', inplace = True)
        Ypoints.fillna(method = 'bfill', inplace = True)
        Zpoints.fillna(method = 'bfill', inplace = True)
        PyVTK(self.sample+"_tracks", Xpoints, Ypoints, Zpoints, 
                    self.dir["vtk"], "polydata")
        
    def convexHull(self):
        volume, spots = [], []
        for file in self.files.index:
            subdf = self.tracks[self.tracks["FILE"] == file]
            hull = ConvexHull(subdf.loc[:,["X", "Y", "Z"]])
            volume.append(hull.volume) 
            spots += list(subdf.iloc[hull.vertices.tolist()].index)
        isHull = pd.Series(name = "isHull", dtype="bool")
        for idx in self.tracks.index :
            if idx in spots:
                isHull[idx] = True
            else :
                isHull[idx] = False
        self.tracks = pd.concat([self.tracks, isHull], axis = 1)
        volume = pd.Series(volume, index = self.files["time"], 
                           name = "volume", dtype = "float")
        if not hasattr(self, "data"):
            self.data = pd.DataFrame(index=self.files["time"])
        self.data = pd.concat([self.data, volume], axis = 1)
        self.data["radius"] = [(3*V/(4*np.pi))**(1/3) 
                               for V in self.data["volume"]]      
        
    def translationCoord(self):
        center = [self.VolShape[-axis]/2 
                  for axis in range(len(self.VolShape))]
        coords = pd.DataFrame(columns = ["tX", "tY", "tZ"], dtype = "float")
        for file in self.files.index:
            subdf = self.tracks[self.tracks["FILE"] == file]
            centroid = OAT.getCentroid(subdf.loc[:,["X", "Y", "Z"]])
            temp_df = pd.DataFrame([center, centroid], 
                                   columns = ["X", "Y", "Z"])
            translation = OAT.computeVectors(temp_df)
            new_coords = []
            for spot in subdf.index:
                new_coords.append([subdf.loc[spot, "X"]+translation["X"],
                                  subdf.loc[spot, "Y"]+translation["Y"],
                                  subdf.loc[spot, "Z"]+translation["Z"]])
            coords = pd.concat([coords, 
                                pd.DataFrame(new_coords, index = subdf.index, 
                                             columns = ["tX", "tY", "tZ"], 
                                             dtype = "float")])
        self.tracks = pd.concat([self.tracks, coords], axis = 1)        
        
    def getDrift(self):
        centroids = pd.DataFrame(dtype = "float", columns = ["X", "Y", "Z"])
        drift = pd.DataFrame(dtype = "float", columns = ["distance"])
        vectors = pd.DataFrame(dtype = "float", 
                               columns = ["c_uX", "c_vY", "c_wZ"])
        for ID in range(len(self.files.index)):
            subdf = self.tracks[self.tracks["FILE"] == self.files.index[ID]]
            index = self.files.iloc[ID]["time"]
            centroids.loc[index] = OAT.getCentroid(
                subdf.loc[:,["X", "Y", "Z"]])
            if ID >= 1 :
                vectors.loc[index-1] = list(
                    OAT.computeVectors(centroids.iloc[ID-1:ID+1]))
                drift.loc[index] = OAT.euclidDist(centroids.iloc[ID], 
                                                       centroids.iloc[ID-1])
        if not hasattr(self, "data"):
            self.data = pd.DataFrame(index=self.files["time"])
        centroids.columns = ["cX", "cY", "cZ"]
        self.data = pd.concat([self.data, centroids], axis = 1)
        self.data = pd.concat([self.data, vectors], axis = 1)
        self.data = pd.concat([self.data, drift], axis = 1)
        
    def crossProduct(df):
        # Return the cross product of the 2 vectors.
        A, B = df.iloc[0], df.iloc[1]
        return pd.Series([(A["vY"]*B["wZ"]-A["wZ"]-B["vY"]),
                          -(A["uX"]*B["wZ"]-A["wZ"]*B["uX"]),
                          (A["uX"]*B["vY"]-A["vY"]*B["uX"])],
                         index = ["uX", "vY", "wZ"], dtype = "float")
        
    def computeRotationAxis(self, drift = True):
        #Compute the rotation axis for each frame or time point
        rotationAxisVectors = pd.DataFrame(columns = ["uX", "vY", "wZ"], 
                                           dtype = "float")
        CP = pd.DataFrame(columns = ["uX", "vY", "wZ"], 
                                    dtype = "float")
        time = []
        for file in self.files.index:
            subdf = self.tracks[self.tracks["FILE"] == file]
            res = pd.DataFrame(columns = ["uX", "vY", "wZ"], 
                                        dtype = "float")
            time += 1000*[self.files.loc[file, 'time']]
            for iteration in range(1000):
                sample = subdf.sample(2)
                sample = sample.loc[:,["uX", "vY", "wZ"]]
                res.loc[iteration] = OAT.crossProduct(sample)
            CP = pd.concat([CP, res], axis = 0, ignore_index = True)
            res.dropna(inplace=True)
            rotationAxisVectors.loc[
                self.files.loc[file, "time"]] = res[["uX", "vY", "wZ"]].mean()
        CP["time"] = time
        CP.dropna(inplace=True)
        dX, dY, dZ = [], [], []
        for tp in self.files["time"] :
            subdf = CP[CP["time"] == tp]
            dX += list(self.data.loc[tp ,"cX"]+subdf["uX"])
            dY += list(self.data.loc[tp, "cY"]+subdf["vY"])
            dZ += list(self.data.loc[tp, "cZ"]+subdf["wZ"])
        CP["DX"] = dX
        CP["DY"] = dY
        CP["DZ"] = dZ    
        self.CrossProducts = CP
        if drift and "c_uX" in self.data.columns: 
            for tp in self.data.index:
                rotationAxisVectors["uX"] = (
                    rotationAxisVectors["uX"]-self.data["c_uX"])
                rotationAxisVectors["vY"] = (
                    rotationAxisVectors["vY"]-self.data["c_vY"])
                rotationAxisVectors["wZ"] = (
                    rotationAxisVectors["wZ"]-self.data["c_wZ"])
        if not hasattr(self, "data"):
            self.data = pd.DataFrame(index=self.files["time"])
        rotationAxisVectors.columns = ["RA_uX", "RA_vY", "RA_wZ"]
        self.data = pd.concat([self.data, rotationAxisVectors], axis = 1)               
        
    def countCells(self):
        NumberOfCells = []
        for file in self.files.index:
            NumberOfCells.append(
                len(self.tracks[self.tracks["FILE"] == file].index))
        if not hasattr(self, "data"):
            self.data = pd.DataFrame(index=self.files["time"])
        self.data["NumberOfCells"] = NumberOfCells
        
    def computeStats(self):
        self.convexHull()
        self.getDrift()
        self.summary = pd.Series(name = "Summary", dtype = "float")
        self.summary["Total_Distance"] = self.data["distance"].sum()
        print("The Organoid travelled ", self.summary["Total_Distance"], 
              " pixels.")
        self.summary["distance/radius"] = (self.summary["Total_Distance"]/
                                            self.data.loc[self.data.index[-1], 
                                                          "radius"])
        print("This roughly correspond to ", self.summary["distance/radius"], 
              " of the radius of the organoid.")
        self.computeRotationAxis()
        
    def plot_figs(fig_num):
        fig = plt.figure(fig_num, figsize=(4, 3))
        plt.clf()
        ax = fig.add_subplot(111, projection="3d")
        X = T.CrossProducts[T.CrossProducts["time"] == 0][["DX","DY","DZ"]]
        #ax.scatter(X["DX"], X["DY"], X["DZ"], alpha=0.1)
    
        # Using SciPy's SVD, this would be:
        # _, pca_score, Vt = scipy.linalg.svd(Y, full_matrices=False)
    
        pca = PCA(n_components=3)
        pca.fit(X)
        V = pca.components_.T
    
        x_pca_axis, y_pca_axis, z_pca_axis = 3 * V
        x_pca_plane = np.r_[x_pca_axis[:2], -x_pca_axis[1::-1]]
        y_pca_plane = np.r_[y_pca_axis[:2], -y_pca_axis[1::-1]]
        z_pca_plane = np.r_[z_pca_axis[:2], -z_pca_axis[1::-1]]
        x_pca_plane.shape = (2, 2)
        y_pca_plane.shape = (2, 2)
        z_pca_plane.shape = (2, 2)
        ax.plot_surface(x_pca_plane, y_pca_plane, z_pca_plane)
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        plt.show
            
    def showData(self):
        features = self.data.columns
        # fig, axs = plt.subplots(1, len(features), figsize=(20, 6), dpi = 400)
        if "volume" in features:
            plt.plot(self.data.index, self.data["volume"])
            plt.xlabel("Time point")
            plt.ylabel("Volume")
            plt.title("Volume of the organoid through time")
            #plt.text(1, 1, "OAT Version : "+str(self.version))
            plt.show()
            plt.close()
        if "NumberOfCells" in features:
            plt.plot(self.data.index, self.data["NumberOfCells"])
            plt.xlabel("Time point")
            plt.ylabel("Number of cells")
            plt.title("Number of cells through time")
            plt.show()
            plt.close()
        if "distance" in features:
            plt.plot(self.data.index, self.data["distance"])
            plt.xlabel("Time point")
            plt.ylabel("Travelled distance between 2 time points")
            plt.title("Travelled distance through time")
            plt.show()
            plt.close()
                  
if __name__ == "__main__":
    T = OAT(fiji_dir = r"C:\Apps\Fiji.app", wrk_dir = r"D:\Wrk\Datasets\1")
    T.loadTif()
    #T.getROI()
    T.getVectors(filtering = False)
    # T.computeStats()
    # T.showData()
    # T.animVectors()

            
### NO USE AT THE MOMENT ------------------------------------------------------

# def minimizing_func(spots_df, center_coords):
#     x, y, z, r = center_coords
#     _sum = 0
#     for _index in spots_df.index:
#         point = [spots_df.loc[_index][axis] 
#                  for axis in ["X", "Y", "Z"]]
#         _sum += (OAT.euclidDist(point, [x, y, z])-r)
#     return _sum   

# def get_spheroid_model(spots_df):
#     # "X", "Y", "Z", "Radius", "Squared root sum"
#     #Getting the centroid
#     centroid = OAT.getCentroid(spots_df)
#     #The aim is to model spheroids that best suits the clusters
#     bounds = [[centroid[0]-10, centroid[0]+10],
#               [centroid[1]-10, centroid[1]+10],
#               [centroid[2]-10, centroid[2]+10],
#               [0, 200]]
#     results = dual_annealing(OAT.minimizing_func, bounds)
#     return [results['x'][0], results['x'][1], results['x'][2], 
#             results['x'][2], OAT.minimizing_func(results['x'])]

# def selectCluster(self, spots_df):
#     """
#     Method to return a filtered version of the spots_df where only the 
#     spots that appear to be part of the organoid remains.

#     Parameters
#     ----------
#     spots_df : pandas.DataFrame
#         Dataframe with clustering data.

#     Returns
#     -------
#     pandas.DataFrame
#         Dataframe with the spots that are in the correct cluster.

#     """
#     # Getting ids of all clusters as well as the number of spots inside of
#     # them. The obtained ids are sorted from most represented to least.
#     # spots_per_cluster is a pandas.Series that contains ids as index and
#     # the number of spots as values.
#     spots_per_cluster = spots_df["CLUSTER"].value_counts().sort_values(
#         ascending = False)
#     cluster_ids = spots_per_cluster.index
#     # Leaderboard is a list where each sublist contain the position of 
#     # clusters for the given criteria.
#     leaderboard = [[], []]
#     distance_to_center = []
#     center = list(self.volume_shape)
#     center.reverse()
#     for idx in cluster_ids :
#         centroid = OAT.getCentroid(spots_df[spots_df["CLUSTER"] == idx])
#         distance = OAT.euclidDist(centroid, center)
#         distance_to_center.append(distance)
#     distance_to_center = pd.Series(distance_to_center, index = cluster_ids)
#     distance_to_center.sort_values(ascending = False, inplace = True)
#     for idx in range(len(cluster_ids)) :
#         # Scoring by the number of spots.
#         leaderboard[0].append(idx)
#         # Scoring by the distance to the center of the cluster's centroid
#         for position in range(len(distance_to_center.index)):
#             if cluster_ids[idx] == distance_to_center.index[position]:
#                 # If True, position directly returns the place.
#                 leaderboard[1].append(position)
#     leaderboard = pd.DataFrame(leaderboard, index = 
#                                ["Number of spots", "Distance to center"],
#                                columns = cluster_ids)
#     # Retrieveing the index (cluster_id) for which the sum of the score is
#     # the lowest in all clusters.
#     cluster_id = leaderboard.sum().sort_values(ascending = True).index[0]
#     return spots_df[spots_df["CLUSTER"] == cluster_id]

# def clustering(subdf, eps = 40, min_samples = 3):
#     """
#     Clustering the spots to assess what spots are in the organoid.

#     """
#     #Clustering using DBSCAN.
#     cluster = DBSCAN(
#         eps=eps, min_samples=min_samples).fit_predict(
#             subdf.loc[:,["X", "Y", "Z"]])
#     #Adding the cluster results to the spots dataframe.
#     cluster = pd.Series(cluster, index = subdf.index, 
#                         name = "CLUSTER")
#     return cluster   

# def reverseDistance(self):
#     r_tracks = self.tracks.loc[:,["X", "Y", "Z", "FILE"]]
    
#     # Computing the distance between each point and the centroid.
#     distance = pd.Series(dtype = "float", name = "DISTANCE")
#     res, index = [], []
#     c_df = []
#     for file in self.files.index :
#         centroid = OAT.getCentroid(r_tracks[r_tracks["FILE"] == file])
#         for spot in r_tracks[r_tracks["FILE"] == file].index:
#             index.append(spot)
#             c_df.append(centroid)
#             distance[spot] = OAT.euclidDist(
#                 list(r_tracks.loc[spot, ["X", "Y", "Z"]]), 
#                 centroid
#                 )
#             tempdf = pd.DataFrame(
#                 [list(r_tracks.loc[spot, ["X", "Y", "Z"]]),
#                  centroid], columns = ["X", "Y", "Z"])
#             res.append(list(OAT.computeVectors(tempdf)))
#     res = pd.DataFrame(res, columns = ["dX","dY","dZ"], index = index, 
#                        dtype = "float")
#     c_df = pd.DataFrame(c_df, index = index, columns = ["cX", "cY", "cZ"],
#                         dtype = "float")
#     r_tracks = pd.concat([r_tracks, distance], axis = 1)
#     r_tracks = r_tracks.join(c_df)
#     r_tracks = r_tracks.join(res)
#     r_tracks["1/D"] = [r_tracks["DISTANCE"].max()/d \
#                        for d in r_tracks.loc[:,"DISTANCE"]]
#     new_coord = []
#     for spot in r_tracks.index:
#         series = r_tracks.loc[spot]
#         new_coord.append([
#             series["cX"]+series["1/D"]*series["dX"],
#             series["cY"]+series["1/D"]*series["dY"],
#             series["cZ"]+series["1/D"]*series["dZ"]
#             ])
#     new_coord = pd.DataFrame(new_coord, index = r_tracks.index, 
#                              columns = ["nX","nY","nZ"])
#     r_tracks = r_tracks.join(new_coord)
#     fig, axs = plt.subplots(1, 2, figsize=(20, 6), dpi=400)
#     axs[0].scatter(r_tracks["X"], r_tracks["Y"])
#     axs[1].scatter(r_tracks["nX"], r_tracks["nY"])
#     plt.show()
#     r_tracks.drop(columns = ["X", "Y", "Z", "FILE"], inplace = True)
#     self.tracks = self.tracks.join(r_tracks)
        