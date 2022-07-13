# -*- coding: utf-8 -*-
"""
organoid_tracking_tools (OAT) is a set of methods that integrates FIJI's 
Trackmate csv files output to process cell displacement within an organoid.

@author: Alex-932
@version: 0.6.1
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
from scipy.io import savemat
from scipy.spatial import ConvexHull #, convex_hull_plot_2d
from sklearn.decomposition import PCA
from PyVTK import PyVTK

class OAT():
    
    def __init__(self, fiji_dir = r"C:\Apps\Fiji.app", wrk_dir = None):
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
        self.version = "0.6.1"      
         
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
        self.dir["distFigs"] = root+'\\output\\figs\\distance'
        self.dir["anim"] = root+'\\output\\animation'
        self.dir["vtk"] = root+'\\output\\vtk_export'
        self.dir["mat"] = root+'\\output\\matlab_export'
        
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
                self.files["TP"] = [k for k in range(len(self.files.index))]
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
        instructions = open(self.dir["fiji"]+"\\OAT_instructions.txt", "w")
        
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
        
        return Results, distance, centroids
        
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
        
        # Creating a dataframe to store informations about the distance (see
        # clusteringEngine). 
        clustDist = pd.DataFrame(columns = ["Distance", "TP"], 
                                 dtype = "object")
        
        # Creating a dataframe to save the centroids coordinates that have been
        # computed (for debug reasons).
        clustCent = pd.DataFrame(columns = ["X", "Y", "Z", "TP"], 
                                 dtype = "object")
        
        for file in self.files.index :
            subdf = data[data["FILE"] == file]
            subdf = OAT.reScaling(subdf, ratio = rescaling)
            clusterResults, dist, cent = OAT.clusteringEngine(subdf, center, 
                                                              cIter, cSample, 
                                                              eps, min_samples,
                                                              threshold)
            
            dist = dist.to_frame()
            dist.columns = ["Distance"]
            
            # Adding time points info to the distance and centroid dataframes.
            dist["TP"] = [self.files.loc[file, "TP"]]*dist.shape[0]
            cent["TP"] = [self.files.loc[file, "TP"]]*cent.shape[0]
            
            Results = pd.concat([Results, clusterResults])
            clustDist = pd.concat([clustDist, dist])
            
            
        # Adding the cluster infos to the right dataframe. 
        if df == "tracks":
            self.tracks = pd.concat([self.tracks, Results], axis = 1)
        elif df == "spots":
            self.spots = pd.concat([self.spots, Results], axis = 1)
        
        # Saving the distance dataframe.
        clustDist = pd.concat([clustDist, Results], axis = 1)
        self.clustDist = clustDist
        # self.clustDist["TP"] = self.clustDist["TP"].astype("int")
        
    def showCentroids(self, TP = "all", figsize = (20, 8), dpi = 400, 
                      show = True, save = False):
        pass
        
    def showDistances(self, TP = "all", figsize = (20, 8), dpi = 400, 
                      bins = 30, show = True, save = False, cmap = 'tab10'):
        """
        Create a figure for selected Time Points to show the distance from the 
        centroid.

        Parameters
        ----------
        TP : list or int, optional
            TP to plot. The default is "all".
        figsize : couple, optional
            Size of the figure as matplotlib accept it. The default is (20, 8).    
        dpi : int, optional
            DPI of the figure. The default is 400.
        save : bool, optional
            If True, save the figure(s) in the \\output\\clustering directory.
            The default is False.

        """
        # Setting the cmap.
        cmap = plt.cm.get_cmap(cmap)
        
        # Setting TP variable according to the user choice.
        if type(TP) in [int, float] :
            TP = [int(TP)]
        elif TP == "all" :
            TP = self.clustDist["TP"].value_counts(ascending = True).index
            TP = list(TP)
        
        for tp in TP :
            # Retrieving the data we will need.
            data = self.clustDist[self.clustDist["TP"] == tp].copy()
            fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
            
            # Showing cluster data if available.
            if "A_CLUSTER" in data.columns :
                # Setting the colors.
                data["Color"] = data.loc[:,"A_CLUSTER"].map(cmap)
                
                # Iterating over clusters ID.
                for cluster in data["A_CLUSTER"
                                    ].value_counts(ascending = True).index :
                    
                    # Getting the rows for 1 cluster.
                    subdata = data[data["A_CLUSTER"] == cluster]
                    
                    # Plotting the histogram with colors.
                    ax.hist(subdata["Distance"], color = subdata["Color"][0], 
                            bins = bins, edgecolor = "white")
            else :
                # Plotting the histogram without color presets.
                ax.hist(data["Distance"], bins = bins, edgecolor = "white")
            
            # Labelling axis and the figure.
            ax.set_xlabel("Distance (in pixels)")
            ax.set_ylabel("Number of spots")
            ax.set_title("Spots by the distance from the centroid")
            
            if show :
                plt.show()
            if save :
                plt.savefig(self.dir["distFigs"]+"\\"+tp+".png", dpi = dpi)
            
            plt.close()
            
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
        
    def readTracks(self, rescaling = [1, 1, 1]):
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
        tracks_df.index = "ID"+tracks_df["ID"].astype("str")
        edges_df.index = "ID"+edges_df["SPOT_SOURCE_ID"].astype("str")
        
        # Renaming some labels in order to be cleaner
        for axis in ["X", "Y", "Z"]:
            tracks_df.rename(columns = {"POSITION_"+axis:axis},
                                  inplace = True)
        edges_df.rename(columns = {"SPOT_TARGET_ID":"TARGET"}, 
                        inplace = True)
        tracks_df.rename(columns = {"POSITION_T":"TP"}, inplace = True)
        
        # Keeping the interesting columns 
        tracks_df = tracks_df.loc[:,["TRACK_ID", "QUALITY", "X", "Y", "Z", 
                                     "TP", "FRAME"]]
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
        edges_df = "ID" + edges_df.astype(int).astype("str")
        
        # Merging the 2 dataframes.
        self.tracks = pd.concat([tracks_df, edges_df], axis = 1)
        
        # Computing the links between spots.
        self.computeSpotsLinks()
        
        # Rescaling the coordinates in case we need to.
        self.tracks["X"] = self.tracks["X"]*rescaling[0]
        self.tracks["Y"] = self.tracks["Y"]*rescaling[1]
        self.tracks["Z"] = self.tracks["Z"]*rescaling[2]
        
        # Creating self.data to store informations at time point level.
        self.data = pd.DataFrame(index = self.files["TP"])
        
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
    
    def getVectors(self, filtering = False, reimport = False, aligned = False,
                   rescaling = [1, 1, 1]):
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
        aligned : bool, optional
            If True, compute displacement vectors for all aligned coordinates.
            See self.alignRotAxis().
        rescaling : list, optional
            Scaling factor for the coordinates. List must be as follow :
            [Xfactor, Yfactor, Zfactor].
    
        """
        # Importing the tracks if not done or if user wants to reimport it.
        if not hasattr(self, "tracks") or reimport:
            self.readTracks(rescaling)
        
        # Removing any previous computation of the vectors in case it already 
        # has been done.
        if "uX" in self.tracks.columns and not aligned:
            self.tracks.drop(columns = ["uX", "vY", "wZ"], inplace = True)
                      
        # Clustering and removing bad spots in self.tracks dataframe if it 
        # hasn't already been done.     
        if filtering and "F_SELECT" not in self.tracks.columns:
            self.getClusters(df="tracks")
            self.tracks = self.tracks[self.tracks["F_SELECT"]]
            
        # Creating a dataframe to store vectors as they are computed.
        # Each row is a vector with its index being the ID of its origin spot.
        if aligned :
            vectors = pd.DataFrame(columns = ["Aligned_uX", "Aligned_vY", 
                                              "Aligned_wZ"], 
                                   dtype = "float")
        else :
            vectors = pd.DataFrame(columns = ["uX", "vY", "wZ"], 
                                   dtype = "float")
        
        # Computing vectors, track by track.
        for trackID in self.spotsLinks.index :
            for spot in range(len(self.spotsLinks[trackID])-1) :
                
                # Retrieving spot and target spot IDs.
                spotID = self.spotsLinks[trackID][spot]
                nextSpotID = self.spotsLinks[trackID][spot+1]
                
                # Retrieving the spot and target spot coordinates
                if aligned :
                    spotInfo = self.tracks.loc[spotID, ["Aligned_X", 
                                                        "Aligned_Y", 
                                                        "Aligned_Z"]]
                    nextSpotInfo = self.tracks.loc[nextSpotID, ["Aligned_X", 
                                                                "Aligned_Y", 
                                                                "Aligned_Z"]]
                else :
                    spotInfo = self.tracks.loc[spotID, ["X", "Y", "Z"]]
                    nextSpotInfo = self.tracks.loc[nextSpotID, ["X", "Y", "Z"]]
                
                # Adding the computed vectors to the vectos dataframe.
                vectors.loc[spotID] = OAT.computeVectors(spotInfo, 
                                                     nextSpotInfo,
                                                     toList = True)
                
            # Adding a null vector as the last spot don't have any vector.
            vectors.loc[self.spotsLinks[trackID][-1]] = 3*[np.nan]
            
        # Merging the vectors dataframe to self.tracks. 
        self.tracks = pd.concat([self.tracks, vectors], axis = 1) 
        
    def showVectors(self, TP, df = "default", angles = None, lim = None,
                    rotAxis = True, show = True, label = "3D",
                    save = False, cellVoxels = False, vectorColor = "black"):
        """
        Create a figure with a representation of the vector field. The figure 
        is then saved.
    
        Parameters
        ----------
        TP : float
            Time point.
        df : str, optional
            Select the data to show:
            - default : raw vectors.
            - translated : translated vectors if computed.
            - aligned : translated and the rotation axis is the Z axis.
        angles : tuple, optional
            Viewing angle as follow (azimuth, elevation). The default is None.
        lim : list, optional
            Limits for the axis. Format is as follow : 
                [[xmin, xmax], [ymin, ymax], [zmin, zmax]] 
            The default is None.    
        rotAxis : bool, optional
            If True, show the rotation axis if available. The default is True.
        show : bool, optional
            If True, show the figure. Default is True.
        label : str, optional
            Name of the representation. The default is "3D".
        save : bool, optional
            If True, save the figures in \\output\\figs\\vectors.
        cellVoxels : bool, optional
            Computationally heavy, use with caution !
            If True, show the cells as voxels. Voxels are obtained using the
            getCellVoxels().
        vectorColor : str, optional
            Set the color of the vectors. The default is black.
    
        """
        # Initializing the figure and its unique axes.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Normal vectors, directly computed from trackmate.
        subdf = self.tracks[self.tracks["TP"] == TP].copy()
        
        if df == "default" :
            if np.isin(["RA_uX", "Cent_X"], self.data.columns).all() and \
                rotAxis : 
                RA = self.data.loc[TP].copy()
                RA.loc[["Cent_X", "Cent_X", "Cent_X"]] = [0, 0, 0]
        
        # Using translated coordinates if desired and available.
        if df == "translated" and "Trans_X" in subdf.columns :
            subdf = subdf.drop(columns = ["X", "Y", "Z"])
            subdf.rename(columns = {"Trans_X": "X",
                                    "Trans_Y": "Y",
                                    "Trans_Z": "Z"},
                         inplace = True)
            
            # Preparing rotation axis data if available and wanted.
            if "RA_uX" in self.data.columns and rotAxis : 
                RA = self.data.loc[TP].copy()
                RA.loc[["Cent_X", "Cent_X", "Cent_X"]] = [0, 0, 0]
                
        # Using aligned coordinates if desired and available.    
        if df == "aligned" and "Aligned_X" in subdf.columns :
            subdf = subdf.drop(columns = ["X", "Y", "Z", "uX", "vY", "wZ"])
            subdf.rename(columns = {"Aligned_X": "X", "Aligned_Y": "Y",
                                    "Aligned_Z": "Z", "Aligned_uX": "uX",
                                    "Aligned_vY": "vY", "Aligned_wZ": "wZ"},
                         inplace = True)
            
            if "Aligned_RA_uX" in self.data.columns and rotAxis :
                RA = self.data.loc[TP].copy()
                RA.loc[["Cent_X", "Cent_X", "Cent_X"]] = [0, 0, 0]
                RA = RA.drop(columns = ["RA_uX", "RA_vY", "RA_wZ"])
                RA.rename(index = {"Aligned_RA_uX": "RA_uX",
                                     "Aligned_RA_vY": "RA_vY",
                                     "Aligned_RA_wZ": "RA_wZ"},
                          inplace = True)
           
        # Plotting the vector field according to the user choice.
        ax.quiver(subdf["X"], subdf["Y"], subdf["Z"], 
                  subdf["uX"], subdf["vY"], subdf["wZ"],
                  color = vectorColor)
        
        # Plotting the axis of rotation if desired and available.
        if rotAxis :
            ax.quiver(RA["Cent_X"], RA["Cent_X"], RA["Cent_X"], 
                      RA["RA_uX"], RA["RA_vY"], RA["RA_wZ"],
                      color = "red", length = 5, pivot = "middle")
            
        # Showing cell voxels if so desired.
        if hasattr(self, "cellArray") and cellVoxels and label == "3D" and \
            TP in self.cellArray.index :
                
            ax.voxels(self.cellArray[TP], shade = True)
            
        # Labeling axis
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        # Giving a title to the figure (renamed further if an angle is 
        # provided)
        ax.set_title("Time point : "+str(TP))
        
        # Setting limits to axis.
        # If nothing has been provided : limit are set only for special views.
        # -> The first front half is shown when "2D" angling.
        if lim == None :
            if angles == (0, 90):
                ymax, ymin = subdf["Y"].max(), subdf["Y"].min()
                ax.set_ylim3d([ymax-(ymax-ymin)/2, ymax+10])
                ax.set_yticks([])
    
            elif angles == (0, 0):
                xmax, xmin = subdf["X"].max(), subdf["X"].min()
                ax.set_xlim3d([xmax-(xmax-xmin)/2, xmax+10])
                ax.set_xticks([])
    
            elif angles == (90, 0):
                zmax, zmin = subdf["Z"].max(), subdf["Z"].min()
                ax.set_zlim3d([zmax-(zmax-zmin)/2, zmax+10])
                ax.set_zticks([])
                
            else :
                ymax, ymin = subdf["Y"].max(), subdf["Y"].min()
                xmax, xmin = subdf["X"].max(), subdf["X"].min()
                zmax, zmin = subdf["Z"].max(), subdf["Z"].min()
                ax.set_ylim3d([ymin-10, ymax+10])
                ax.set_xlim3d([xmin-10, xmax+10])
                ax.set_zlim3d([zmin-10, zmax+10])
                
        # If limits are provided.        
        else :
            ax.set_xlim3d(lim[0])
            ax.set_ylim3d(lim[1])
            ax.set_zlim3d(lim[2])
            
        # Setting the viewing angle if provided and renaming the figure to
        # include the angle information
        if type(angles) == tuple:
            ax.view_init(angles[0],angles[1])
            ax.set_title("Timepoint : "+str(TP)+', Angle : ('+str(angles[0])+\
                         ","+str(angles[1])+")")
                
        if show :
            plt.show()
            
        # Saving the figure as an image. Name template is as follow :
        # {name of the file (argument given in _init_)}_vf_({TP})_{label}.png
        if save :
            plt.savefig(self.dir["vectorsFigs"]+"\\"+self.sample+\
                        "_vf_("+str(TP)+")_"+label+".png", dpi=400)
                
        plt.close(fig)
        
    def animVectors(self, TP = "all", fps = 1, lim = None, df = "default", 
                    rotAxis = True, cellVoxels = False, 
                    vectorColor = "black"):
        """
        Generate a film showing the displacement vector field at 
        every time point available. 
        The video is saved in \\output\\animation.

        Parameters
        ----------
        TP : list, optional
            First element is the first time point, the 2nd is the last 
            time point (included). The default is "all".
        fps : int, optional
            Frames per second for the video. The default is 1.
        lim : list, optional
            Limits for the axis. Format is as follow : 
                [[xmin, xmax], [ymin, ymax], [zmin, zmax]] 
            The default is None.
        mode : str, optional
            Select the mode :
            - default : video will contains vectors, for each time points
            - translated : vectors are translated to [0, 0, 0].
            - aligned : use the spots coordinates where the axis of rotation
                        is aligned to the Z axis.
        rotAxis : bool, optional
            If True, show the rotation axis if available. The default is True.
        cellVoxels : bool, optional
            Computationally heavy, use with caution !
            If True, show the cells as voxels. Voxels are obtained using the
            getCellVoxels(). 
        vetorColor : str, optional
            Set the color of the vectors. The default is black.

        """        
        # Setting time points according to user inputs.
        if TP == "all":
            TP = self.files["TP"][:-1]
            
        elif type(TP) == list:
            # Checking if there are None values and replacing them.
            if TP[0] == None:
                TP[0] = self.files["TP"].min()
                
            elif TP[1] == None:
                TP[1] = self.files["TP"].max()
                
            # Creating TP to include all time points in the desired range.
            TP = [tp for tp in range(TP[0], TP[1]) if tp in self.files["TP"]]
        
        # arrays will save the various images opened with opencv.
        arrays = []
        
        # Setting angles
        angles = [None, (0, 90), (0, 0), (90, 0)]
        labels = ["3D", "X", "Y", "Z"]
        
        # Iterating over all angles.
        for idx in range(len(angles)) :
            
            # Iterating over all time points.
            for tp in TP:
                
                # Creating a figure and saving it as an image.
                self.showVectors(TP = tp, df = df, angles = angles[idx],
                                 lim = lim, label = labels[idx],
                                 rotAxis = rotAxis,
                                 show = False, save = True, 
                                 cellVoxels = cellVoxels, 
                                 vectorColor = vectorColor)
                
                # Opening the image that have just been created.
                img = cv2.imread(self.dir["vectorsFigs"]+"\\"+self.sample+\
                                 "_vf_("+str(tp)+")_"+labels[idx]+".png")
                    
                # Retrieving the image size to set the video shapes.
                height, width, layers = img.shape
                size = (width,height)
                
                # Adding the opencv object containing the image in the
                # img_array.
                arrays.append(img)       
        
        # Creating and opening the videofile container using opencv. 
        out = cv2.VideoWriter(self.dir["anim"]+"\\"+self.sample+"_"+df+".avi", 
                              0, cv2.VideoWriter_fourcc(*'DIVX'), fps = fps, 
                              frameSize = size)
        
        # Loading every images present in arrays into the video container.
        for img in arrays:
            out.write(img)
            
        # Closing the video.
        out.release()
        
    def exportVectorsVTK(self, TP = "all", tracks = "all"):
        """
        Export the displacement vectors coordinates in a polydata .vtk format.

        Parameters
        ----------
        TP : int or list, optional
            Singe (int) or multiple (list) time points. The default is "all".
        tracks : int, optional
            Track ID. The default is "all".

        Returns
        -------
        Save the .vtk in \\output\\vtk_export

        """
        
        # Creating the list of time points to show by the user choice.
        if TP == "all":
            TP = self.files["TP"]
        elif type(TP) == int:
            TP = [TP]
        
        # Creating the list of tracks that will be exported.
        if tracks == "all":
            tracks = [k for k in self.tracks["TRACK_ID"].value_counts().index]
        elif type(tracks) == int:
            tracks = [tracks]
        
        # Sorting the list of time point because we want to increment following
        # the chronological order. 
        tracks.sort()
        
        # Saving points coordinates in 3 dataframes, one for each axis. 
        # Rows are timepoints and columns are tracks.
        list_X, list_Y, list_Z = [], [], []
        
        # Filling previous lists with sublists. Each sublist has the 
        # corresponding axis coordinates of all points for 1 time point. It's 
        # like an image of the cells on the X axis at a certain time point.
        for tp in TP :
            
            # Working with the subDataframe corresponding to the time point.
            subdf = self.tracks[self.tracks["TP"] == tp]
            list_X.append([])
            list_Y.append([])
            list_Z.append([])
            
            # Adding coordinates values for each track
            for track in tracks :
                # Retrieving the point data  
                pointData = subdf[subdf["TRACK_ID"] == track]
                # Checking if the point exist. If it doesn't then we enter no 
                # data (represented by a NaN value.) We will deal with these 
                # later on.
                if pointData.empty :
                    list_X[-1].append(np.nan)
                    list_Y[-1].append(np.nan)
                    list_Z[-1].append(np.nan)
                    
                else :
                    list_X[-1].append(pointData["X"][0])
                    list_Y[-1].append(pointData["Y"][0])
                    list_Z[-1].append(pointData["Z"][0])
                    
        # Creating the Dataframes that will be used by PyVTK. 
        Xpoints = pd.DataFrame(list_X, index = TP, columns = tracks)
        Ypoints = pd.DataFrame(list_Y, index = TP, columns = tracks)
        Zpoints = pd.DataFrame(list_Z, index = TP, columns = tracks)
        
        # Filling the empty values : if the track start later, we fill previous
        # positions with the starting one (bfill). We do the same thing the 
        # other way when it stop sooner (ffill).
        Xpoints.fillna(method = 'ffill', inplace = True)
        Ypoints.fillna(method = 'ffill', inplace = True)
        Zpoints.fillna(method = 'ffill', inplace = True)
        Xpoints.fillna(method = 'bfill', inplace = True)
        Ypoints.fillna(method = 'bfill', inplace = True)
        Zpoints.fillna(method = 'bfill', inplace = True)
        
        # Running PyVTK
        PyVTK(self.sample+"_tracks", Xpoints, Ypoints, Zpoints, 
              self.dir["vtk"], "polydata")
        
    def computeCellVoxels(self, TP = "all", offset = 10, outerThres = 0.9):
        """
        Compute voxels of cells. For each time point, the corresponding image
        is loaded as an array. We get a threshold by getting the minimal pixel
        value for the pixels that are at spots coordinates, for a given 
        time point. 

        Parameters
        ----------
        TP : int or list, optional
            Time points to compute. The default is "all".
        offset : int, optional
            Added value to the min and max to make sure everything is inside. 
            The default is 10.
        outerThres : float, optional
            Threshold determining that a pixels is inside the voxel and need to
            be set to 0. Used to improve plotting speed. The default is 0.9.

        """
        # Selecting the timepoint based on the user choice
        if TP == "all":
            TP = self.files["TP"]
        elif type(TP) == int:
            TP = [TP]
        
        # Creating a pd.Series in which we will save the arrays if not already.
        if not hasattr(self, "cellArray"):
            self.cellArray = pd.Series(dtype = "object")
        
        # Iterating over all desired time points.
        for tp in TP:
            filename = self.files[self.files["TP"] == tp]["tifs"].values[0]
            
            # Opening the image.
            image = io.imread(self.dir["tifs"]+"\\"+filename)
            
            # Converting into a Numpy array. The shape is in this order : Z, Y, X.
            imarray = np.array(image)
            
            # Getting the minimal value of pixels at spots coordinates.
            subdf = self.tracks[self.tracks["TP"] == tp].copy()
            subdf = subdf.astype("int")
            values = imarray[subdf["Z"], subdf["Y"], subdf["X"]].tolist()
            minimal = min(values)
            
            # Setting the outside as 0 and the inside as 1.
            imarray[imarray < (minimal-offset)] = 0
            imarray[imarray >= (minimal-offset)] = 1
            
            # Transposing the array that way it is oriented the same as the 
            # other objects (spots, vectors).
            imarray = np.transpose(imarray, (2, 1, 0))
            
            # Setting the inner pixels to 0 as well to only get the outer 
            # shell.
            toChange = []
            for x in range(1, imarray.shape[0]-1):
                for y in range(1, imarray.shape[1]-1):
                    for z in range(1, imarray.shape[2]-1):
                        
                        # Getting the 3*3 square array centered on (x,y,z). 
                        neighbors = imarray[x-1:x+2, y-1:y+2, z-1:z+2]
                        
                        # Summing the values and if the number of ones is 
                        # greater than the threshold, saving the pixel coord.
                        if neighbors.sum()/27 >= outerThres :
                            toChange.append([x, y, z])
            
            # Setting the values of the selected pixels to 0.
            for coord in toChange:
                imarray[coord[0], coord[1], coord[2]] = 0
            
            # Saving the array with its time point index.
            self.cellArray.loc[tp] = imarray           
            
    def computeConvexHull(self):
        """
        Use the Convex Hull algorithm of scipy to get the cells that are 
        forming the outershell as well as the volume of the organoid. 

        """
        # Creating 2 buffer lists.
        volume, spots = [], []
        
        # Iterating over files.
        for tp in self.files["TP"]:
            
            # Getting the sub dataframe.
            subdf = self.tracks[self.tracks["TP"] == tp]
            
            # Using the Convex Hull algorithm.
            hull = ConvexHull(subdf.loc[:,["X", "Y", "Z"]])
            
            # Saving the volume and the spots that are the outershell.
            volume.append(hull.volume) 
            spots += list(subdf.iloc[hull.vertices.tolist()].index)
            
        # Setting the bool value for the question : is this spot part of the 
        # outershell ?
        isHull = pd.Series(name = "isHull", dtype="bool")
        for idx in self.tracks.index :
            if idx in spots:
                isHull[idx] = True
            else :
                isHull[idx] = False
                
        # Merging the bool isHull to self.tracks.
        self.tracks = pd.concat([self.tracks, isHull], axis = 1)
        
        # Converting the volume list to a Series to add time point informations
        volume = pd.Series(volume, index = self.files["TP"], 
                           name = "volume", dtype = "float")
        
        # Adding volume and mean radius to self.data
        self.data = pd.concat([self.data, volume], axis = 1)
        self.data["radius"] = [(3*V/(4*np.pi))**(1/3) 
                               for V in self.data["volume"]]
        
        # Creating a summary series if not already.
        if not hasattr(self, "summary"):
            self.summary = pd.Series(name = "Summary", dtype = "float")
        self.summary["Mean_radius"] = self.data["radius"].mean()
        
    def computeDrift(self):
        """
        Compute the drift of the organoid between time points.

        """
        # Creating DataFrames to hold computation results. 
        centroids = pd.DataFrame(dtype = "float", 
                                 columns = ["Cent_X", "Cent_X", "Cent_X"])
        drift = pd.DataFrame(dtype = "float", columns = ["drift_distance"])
        vectors = pd.DataFrame(dtype = "float", 
                               columns = ["drift_uX", "drift_vY", "drift_wZ"])
        
        # Iterating over files index.
        for tp in range(len(self.files["TP"])):
            
            # Extracting the subdataframe containing the information for a 
            # given file.
            subdf = self.tracks[self.tracks["TP"] == tp]
            
            # Getting the centroid for this tp.
            centroids.loc[tp] = OAT.getCentroid(subdf.loc[:,["X", "Y", "Z"]])
            
            # If we're not at the first file, we can compute the drift vector 
            # between the centroids from the n-1 and n time point.
            # The drift vector is saved with the n-1 time point index. 
            if tp >= 1 :
                vectors.loc[tp-1] = OAT.computeVectors(centroids.iloc[tp-1],
                                                       centroids.iloc[tp],
                                                       toList = True)
                drift.loc[tp] = OAT.euclidDist(centroids.iloc[tp], 
                                                       centroids.iloc[tp-1])
            
        # Merging the several dataframes to self.data.
        self.data = pd.concat([self.data, centroids], axis = 1)
        self.data = pd.concat([self.data, vectors], axis = 1)
        self.data = pd.concat([self.data, drift], axis = 1)
        
        # Creating a summary series if not already.
        if not hasattr(self, "summary"):
            self.summary = pd.Series(name = "Summary", dtype = "float")
        self.summary["Total_Distance"] = self.data["drift_distance"].sum()
        if "Mean_radius" in self.data.columns:
            self.summary["D/R"] = (self.summary["Total_Distance"]/
                                   self.data["Mean_radius"].mean())
        
    def translateCoord(self):
        """
        Translate the coordinates of all points within self.tracks to get the
        centroid of the organoid at [0, 0, 0].
        The translated coordinates are added to self.tracks.

        """
        # Setting the center coordinates.
        center = pd.Series([0, 0, 0], index = ["X", "Y", "Z"])
        
        # Creating a DataFrame to store the translated coordinates.
        coords = pd.DataFrame(columns = ["Trans_X", "Trans_Y", "Trans_Z"], 
                              dtype = "float")
        
        # Iterating over files.
        for file in self.files.index:
            
            # Getting the wanted rows.
            subdf = self.tracks[self.tracks["FILE"] == file]
            
            # Computing the centroid as well as the translation between the 
            # centroid and the center coordinates.
            centroid = OAT.getCentroid(subdf.loc[:,["X", "Y", "Z"]])
            translation = OAT.computeVectors(center, centroid)
            
            # Creating a buffer list to store new coordinates.
            new_coords = []
            
            # Iterating over spot IDs and computing the translated coordinates.
            for spot in subdf.index:
                new_coords.append([subdf.loc[spot, "X"]-translation["X"],
                                  subdf.loc[spot, "Y"]-translation["Y"],
                                  subdf.loc[spot, "Z"]-translation["Z"]])
            
            # Adding the translated coordinates to the DataFrame.
            coords = pd.concat([coords, 
                                pd.DataFrame(new_coords, index = subdf.index, 
                                             columns = ["Trans_X", "Trans_Y", 
                                                        "Trans_Z"], 
                                             dtype = "float")])
            
        # Merging self.tracks and the Dataframe containing the translated 
        # coords.
        self.tracks = pd.concat([self.tracks, coords], axis = 1)        
           
    def crossProduct(df):
        """
        Compute the cross product of 2 vectors.

        Parameters
        ----------
        df : pd.DataFrame
            Each row is a vector and columns are ["uX", "vY", "wZ"].

        Returns
        -------
        pd.Series
            Series where index are ["uX", "vY", "wZ"].

        """
        # Retrieve the vectors as Series from the DataFrame.
        A, B = df.iloc[0], df.iloc[1]
        
        return pd.Series([(A["vY"]*B["wZ"]-A["wZ"]*B["vY"]),
                          -(A["uX"]*B["wZ"]-A["wZ"]*B["uX"]),
                          (A["uX"]*B["vY"]-A["vY"]*B["uX"])],
                         index = ["uX", "vY", "wZ"], dtype = "float")
    
    def computeRotationAxis(self):
        """
        Compute the rotation axis of the dataset, at each time point.
        Update self.data with the colinear vectors of the rotation axis.

        """
        # Creating a dataframe to store both vectors forming the PCA plane as
        # well as the crossproduct of those 2.
        componentVectors = pd.DataFrame(columns = ["V1_uX", "V1_vY", "V1_wZ",
                                                   "V2_uX", "V2_vY", "V2_wZ",
                                                   "RA_uX", "RA_vY", "RA_wZ"], 
                                        dtype = "float")
        
        # Iterating over timepoints.
        for tp in self.files["TP"]:
            subdf = self.tracks[self.tracks["TP"] == tp
                                ].loc[:, ["uX", "vY", "wZ"]]
            subdf = subdf.dropna()
            
            # Checking if the dataframe is empty meaning we can't compute the 
            # axis.
            if not subdf.empty:   
            
                pca = PCA(n_components = 2)
                pca.fit(subdf.loc[:, ["uX", "vY", "wZ"]])
                
                V1 = pca.components_[0]
                V2 = pca.components_[1]
                
                # Creating a temporary df for crossProduct().
                tempDF = pd.DataFrame(pca.components_, 
                                      columns = ["uX", "vY", "wZ"])
                
                # Computingthe crossproduct.
                RA = list(OAT.crossProduct(tempDF))
                
                # Saving coordinates to th dataframe.
                componentVectors.loc[tp] = [V1[0], V1[1], V1[2], 
                                            V2[0], V2[1], V2[2],
                                            RA[0], RA[1], RA[2]]
        
        # Merging componentVectors with self.data.
        self.data = pd.concat([self.data, componentVectors], axis = 1)
    
    def alignRotAxis(self):
        """
        Rotate the points of self.tracks to get the axis of rotation aligned 
        with the Z axis. New coordinates are saved in self.tracks in 
        "Aligned_..." columns.

        """
        # Running required functions
        if not hasattr(self, "data") :
            self.computeDrift()
        if not "Cent_X" in self.data.columns :
            self.computeDrift()
        if not "RA_uX" in self.data.columns :
            self.computeRotationAxis()
        if not "Trans_X" in self.tracks.columns :
            center = pd.Series([0, 0, 0], index = ["X", "Y", "Z"], 
                               dtype = "float")
            self.translateCoord(center)
            
        # Trying to align all rotation axis vectors with Z.
        # First aligning with X to get 0 on the Y axis.
        newCoords = self.tracks.loc[:, ["Trans_X", "Trans_Y", "Trans_Z"]]
        newCoords.columns = ["X", "Y", "Z"]
        
        newRA = self.data.loc[:, ["RA_uX", "RA_vY", "RA_wZ"]]
        
        transAngles = pd.DataFrame(columns = ["Theta_X", "Theta_Y"],
                                   dtype = "float")
        
        for tp in self.files["TP"] :
            data = newRA.loc[tp]
            
            coord = [data["RA_uX"],
                     data["RA_vY"],
                     data["RA_wZ"]]
            
            theta_x = np.arctan2(coord[1], coord[2])
            transAngles.loc[tp, "Theta_X"] = theta_x
            
            # Applying X rotation
            ycoord = coord[1].copy()
            coord[1] = coord[1]*np.cos(theta_x)-coord[2]*np.sin(theta_x)
            coord[2] = ycoord*np.sin(theta_x)+coord[2]*np.cos(theta_x)
            
            theta_y = np.arctan2(-coord[0], coord[2])
            transAngles.loc[tp, "Theta_Y"] = theta_y
            
            # Applying Y rotation
            xcoord = coord[0].copy()
            coord[0] = coord[0]*np.cos(theta_y)+coord[2]*np.sin(theta_y)
            coord[2] = -xcoord*np.sin(theta_y)+coord[2]*np.cos(theta_y)
            
            newRA.loc[tp] = coord
        
        newRA.columns = ["Aligned_RA_uX", "Aligned_RA_vY", "Aligned_RA_wZ"]
        self.data = pd.concat([self.data, newRA], axis = 1)
        
        for ID in newCoords.index :
            
            coord = [newCoords.loc[ID, "X"],
                     newCoords.loc[ID, "Y"],
                     newCoords.loc[ID, "Z"]]
            
            theta_x = transAngles.loc[self.tracks.loc[ID, "TP"], "Theta_X"]
            transAngles.loc[tp, "Theta_X"] = theta_x
            
            ycoord = coord[1].copy()
            coord[1] = coord[1]*np.cos(theta_x)-coord[2]*np.sin(theta_x)
            coord[2] = ycoord*np.sin(theta_x)+coord[2]*np.cos(theta_x)
            
            theta_y = transAngles.loc[self.tracks.loc[ID, "TP"], "Theta_Y"]
            transAngles.loc[tp, "Theta_Y"] = theta_y
            
            xcoord = coord[0].copy()
            coord[0] = coord[0]*np.cos(theta_y)+coord[2]*np.sin(theta_y)
            coord[2] = -xcoord*np.sin(theta_y)+coord[2]*np.cos(theta_y)
            
            newCoords.loc[ID] = coord
        
        newCoords.columns = ["Aligned_X", "Aligned_Y", "Aligned_Z"]
        self.tracks = pd.concat([self.tracks, newCoords], axis = 1)
        
        self.getVectors(aligned = True)
        
        self.transAngles = transAngles
        
    def computeAngularVelocity(self):
        
        subdf = self.tracks.copy()
        
        angularVelocity = pd.Series(dtype = "float", name = "Angular_Velocity")
        distance = pd.Series(dtype= "float", name = "Distance_rotAxis")
        
        for ID in subdf.index:
            tID = subdf.loc[ID, "TARGET"]
            
            distance[ID] = OAT.euclidDist([0, 0], 
                                          list(subdf.loc[ID, ["Aligned_X",
                                                              "Aligned_Y"]])) 
            
            if type(tID) == str:
                
                # Computing delta t.
                dt = subdf.loc[tID, "TP"]-subdf.loc[ID, "TP"]
                
                # Computing the angle between the point and the X axis.
                y, x = subdf.loc[ID, "Aligned_Y"], subdf.loc[ID, "Aligned_X"]
                theta = np.arctan2(y, x)
                if y < 0 :
                    theta += 2*np.pi
                
                # Computing the angle between the target point and the X axis.
                yTarg, xTarg = subdf.loc[tID, "Aligned_Y"], \
                    subdf.loc[tID, "Aligned_X"]
                targTheta = np.arctan2(yTarg, xTarg)
                if y < 0 :
                    targTheta += 2*np.pi
                
                dTheta = abs(targTheta-theta)
                
                angularVelocity[ID] = dTheta/dt
            
            else :
                
                angularVelocity[ID] = np.nan
        
        self.tracks = pd.concat([self.tracks, angularVelocity.to_frame()], 
                                axis = 1)
        self.tracks = pd.concat([self.tracks, distance.to_frame()], axis = 1)
        
    def exportMatLab(self):
        maxLength = max([self.tracks[self.tracks["TP"] == tp].shape[0] 
                         for tp in self.files["TP"]])
        for tp in self.files["TP"]:
            data = self.tracks[self.tracks["TP"] == tp]
            data = data.loc[:, ["Aligned_X", "Aligned_Y", "Aligned_Z"]]
            dic = {}
            for col in data.columns:
                values = np.concatenate(( data[col].to_numpy(), 
                                          np.zeros((maxLength-data.shape[0])) 
                                         ))
                dic[re.split("_", col)[-1]] = values
            savemat(self.dir["mat"]+"\\aligned_points_"+str(tp)+".mat", dic)
            
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
            
    def showAngularVelocity(self, TP):
        subdf = self.tracks[self.tracks["TP"] == TP].copy()
        fig, axs = plt.subplots(2, 1)
        axs[0].scatter(subdf["Distance_rotAxis"], subdf["Angular_Velocity"])
        axs[0].set_xlabel("Distance from the axis of rotation (pixels)")
        axs[0].set_ylabel("Angular Velocity (rad/tp)")
        axs[0].set_title("Angular Velocity according to the distance from rotation Axis")
        
        axs[1].scatter(subdf["Aligned_X"], subdf["Aligned_Y"], 
                       c = subdf["Angular_Velocity"])
        axs[1].set_xlabel("X")
        axs[1].set_ylabel("Y")
        axs[1].set_title("Spots on the XY plane.")
        
        plt.show()
        plt.close()
                  
if __name__ == "__main__":
    T = OAT(fiji_dir = r"C:\Apps\Fiji.app", wrk_dir = r"D:\Wrk\Datasets\4")
    #T.loadTif()
    #T.getROI()
    #T.getVectors(filtering = False, rescaling = [1, 1, 4])
    #T.computeStats()
    # T.showData()
    # T.animVectors()
    S = OAT(wrk_dir = r"D:\Wrk\Datasets\S3")
    #S.loadTif()
    #T.getROI()
    #S.getVectors(filtering = False)
        