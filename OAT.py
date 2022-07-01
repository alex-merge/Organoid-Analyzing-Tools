# -*- coding: utf-8 -*-
"""
organoid_tracking_tools (OAT) is a set of methods that integrates FIJI's 
Trackmate csv files output to process cell displacement within an organoid.

@author: Alex-932
@version: 0.5
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN, KMeans
from skimage import io
from math import sqrt
import tifffile
import cv2
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from PyVTK import PyVTK

class OAT():
    
    def __init__(self, fiji_dir, wrk_dir = None):
        """
        Initialize the sample analysis by creating the directories needed.

        Parameters
        ----------
        fiji_dir : str
            Directory path for Fiji.app folder.
        sample : str, optional
            Name of the sample. The default are the third first characters of 
            the image files.
        wrk_dir : str, optional
            Working directory i.e. the dir. where the tree will be built. 
            The default is the script directory.

        """
        if  not os.path.exists(fiji_dir):
            raise FileNotFoundError("No such fiji directory")
        # Checking if the user gave a directory.
        if wrk_dir == None :
            root = os.getcwd()
        # If yes, then we save it if it's an actual directory or create it 
        # otherwise.
        else :
            if not os.path.exists(wrk_dir):
                os.makedirs(wrk_dir)
            root = wrk_dir
        self.files = pd.DataFrame(dtype="str")
        self.dir = pd.Series(dtype="str")
        self.dir["fiji"] = fiji_dir
        self.dir["root"] = root
        # Building the directories tree
        self.buildTree()
        self.version = "0.5"      
         
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
            # Looking for images.
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

        """
        # Opening the first tif file.
        file = io.imread(self.dir["tifs"]+"\\"+self.files["tifs"][0])
        # Converting it to a numpy array.
        array = np.array(file)
        # Saving the shapes
        self.VolShape = array.shape
        
    def loadTif(self):
        """
        Loading the Tifs files from the data\3D Organoids directory.

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
        Load segmentation result of Trackmate detection as a dataframe and 
        clean it a little bit. The dataframe is called spots_df

        Parameters
        ----------
        filepath : str
            Path to the file.

        Returns
        -------
        spots_df : pd.DataFrame
            Dataframe where rows represent spots and columns some attributes
            ("QUALITY", "X", "Y", "Z").

        """
        # The csv is imported as a dataframe. It is called raw because there 
        # are unwanted spots from cells that are not in the good organoid.
        spots_df = pd.read_csv(filepath)
        spots_df.drop(index = [0, 1, 2], inplace = True)
        # We remove the 3 first lines as they are redundant labels.
        spots_df.index = spots_df["LABEL"]
        # We set the LABEL column as the dataframe's index.
        for axis in ["X", "Y", "Z"]:
            # We rename the column as it is shorter.
            spots_df.rename(columns = {"POSITION_"+axis:axis},
                                  inplace = True)
        # Selecting usefull columns
        spots_df = spots_df.loc[:,["QUALITY", "X", "Y", "Z"]]
        # Setting every columns as float type
        spots_df = spots_df.astype("float")
        return spots_df
    
    def getSpots(self):
        """
        Creating a dataframe where all spots of all .csv files are present 
        called spots

        """
        # Checking the .csv files and getting the list of names.
        self.importFiles("spots")
        # Creating the spots dataframe which contains info regarding each spots
        # for each image file (or timepoint).
        self.spots = pd.DataFrame(columns=["QUALITY", "X", "Y", "Z",
                                           "FILE"], dtype=("float"))
        #Setting the correct type for the "FILE" and 'CLUSTER' columns.
        self.spots["FILE"] = self.spots["FILE"].astype("str")
        for file in self.files["spots"] :
            reading = OAT.readSpots(self.dir["spots"]+"\\"+file)
            reading["FILE"] = len(reading.index)*[
                re.split("_rs.csv", file)[0]]
            self.spots = pd.concat([self.spots, reading])
    
    def euclid_distance(PointA, PointB):
        """
        Process the euclid distance of 2 given points. Works in 3D as well as
        in 2D. Just make sure that both points have the same amount of 
        coordinates.

        Parameters
        ----------
        PointA : list
            List of the coordinates (2 or 3).
        PointB : list
            List of the coordinates (2 or 3).

        Returns
        -------
        float
            Euclid distance between the 2 given points.

        """
        return sqrt(sum([(PointA[k]-PointB[k])**2 for k in range(len(
            PointA))])) 
    
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
            List that contains the coordinates of the processed centroid.

        """
        centroid = []
        for axis in ["X", "Y", "Z"]:
            centroid.append(spots_df[axis].mean())
        return centroid
    
    def coordTransformation(df, Xratio = 1, Yratio = 1, Zratio = 1):
        """
        Transform the coordinates of the given axis by the given amount. 

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe which contains axis as columns and spots as index.
        Xratio : float
            Coordinates multiplier for the X axis.
        Yratio : float
            Coordinates multiplier for the Y axis.
        Zratio : float
            Coordinates multiplier for the Z axis.

        Returns
        -------
        df : pandas.DataFrame
            Same as the input one but with transformed coordinates.

        """
        out_df = pd.DataFrame(index = df.index, columns = df.columns)        
        out_df["X"] = df.loc[:, "X"]*Xratio
        out_df["Y"] = df.loc[:, "Y"]*Yratio
        out_df["Z"] = df.loc[:, "Z"]*Zratio
        return out_df         
    
    def clusteringEngine(subdf, volume_center, eps = 40, min_samples = 3, 
                         show = False):
        """
        Cluster the spots to get the ones that are the most likely to be part
        of the organoïd.

        Parameters
        ----------
        subdf : pd.DataFrame
            Spots from one frame containing the coordinates in columns.
        volume_center : list 
            center coordinates of the whole volume as follow [X, Y, Z].
        eps : int, optional
            Radius of search for the DBSCAN algorithm. The default is 40.
        min_samples : int , optional
            Min. number of neighbors for the DBSCAN. The default is 3.
        show : bool, optional
            If True, show the histogram of the distance between spots and 
            the center of the volume. The default is False.

        Returns
        -------
        Results : TYPE
            DESCRIPTION.

        """
        # Getting the centroid by randomly subsampling 10 spots, 100 times.
        # This allow us to get the most probable location of the center of the
        # organoïd.
        centroids = [OAT.getCentroid(subdf.sample(10, axis=0)) \
                     for k in range(100)]
        centroids = pd.DataFrame(centroids, columns=["X", "Y", "Z"])
        centroid = [centroids["X"].mean(), centroids["Y"].mean(),
                    centroids["Z"].mean()]
        # Computing the distance between each point and the centroid.
        distance = pd.Series(dtype = "float", name = "DISTANCE")
        for points in subdf.index:
            distance[points] = OAT.euclid_distance(
                list(subdf.loc[points, ["X", "Y", "Z"]]), 
                centroid
                )
        # Clustering the distances.
        identified_clusters = DBSCAN(
            eps=5, min_samples=6).fit_predict(distance.to_frame())
        #Adding the first cluster results to the spots dataframe.
        cluster = pd.Series(identified_clusters, index = subdf.index, 
                            name = "A_CLUSTER")
        # Creating the final dataframe which would be the output of this
        # function.
        Results = cluster.to_frame()
        # Selecting the right cluster (closest to the center).    
        isOrganoid = OAT.selectClusters(pd.concat([subdf, cluster], axis = 1),
                                        volume_center, "A_CLUSTER")
        isOrganoid.rename("A_SELECT", inplace = True)
        # Adding the selection to the final dataframe.
        Results = pd.concat([Results, isOrganoid], axis = 1)
        # Changing the subdf, only keeping the selected cluster and clustering
        # it once again.
        subdf = subdf[isOrganoid]
        identified_clusters = DBSCAN(
            eps=eps, min_samples=min_samples).fit_predict(
                subdf.loc[:,["X", "Y", "Z"]])
        cluster = pd.Series(identified_clusters, index = subdf.index, 
                            name = "F_CLUSTER")
        Results = pd.concat([Results, cluster], axis = 1)
        # Selecting the right cluster once again.
        isOrganoid = OAT.selectClusters(pd.concat([subdf, cluster], axis = 1),
                                        volume_center, "F_CLUSTER")
        isOrganoid.rename("F_SELECT", inplace = True)
        Results = pd.concat([Results, isOrganoid], axis = 1)
        return Results
    
    def selectClusters(subdf, volume_center, cluster_col):
        """
        Search for the cluster that as the highest probability of being the
        organoid cluster. Save the results as booleans in the self.spots 
        dataframe.

        """
        # isOrganoid is a temporary pd.Series that contains booleans values
        # for each spots. Tells if the spot is part of the organoid.
        isOrganoid = pd.Series(dtype = "bool", name = "IS_ORGANOID")
        # Processing for each frame.
        # Retrieving the clusters with their ID and the number of spots
        # within them.
        clusters = subdf[cluster_col].value_counts()
        clustersID = clusters.index
        # Following list holds the distance from the centroid of clusters 
        # to the center of the volume.
        distance2center = []
        # Computing the distance for each clusters.
        for ID in clustersID :
            centroid = OAT.getCentroid(subdf[subdf[cluster_col] == ID])
            distance = OAT.euclid_distance(centroid, volume_center)
            distance2center.append(distance)
        # Converting the list to a pd.Series to add the ID inforamtions.
        distance2center = pd.Series(distance2center, 
                                       index = clustersID)
        # Sorting from the lowest to the greatest distance.
        distance2center.sort_values(ascending = True, inplace = True)
        # Checking if the closest cluster to the center contains more
        # than 10 spots. Choosing it if it is the case.
        selectedCluster = 0
        for idx in range(len(distance2center.index)):
            if clusters[distance2center.index[idx]] >= 10:
                selectedCluster = distance2center.index[idx]
                break
        # Merging the results to those of other clusters within the same
        # file.
        isOrganoid = pd.concat([isOrganoid, 
                                subdf[cluster_col] == selectedCluster])
        return isOrganoid
        
    def getClusters(self, eps = 40, min_samples = 3, df = "spots"):
        """
        Clustering spots for each frame and adding the data to the self.spots
        dataframe.

        """
        if df == "tracks" : 
            if hasattr(self, "tracks"):
                data = self.tracks.loc[:,["X", "Y", "Z", "FILE"]]
        else :
            data = self.spots
        # Saving the center coordinates of the volume.
        volume_center = [self.VolShape[-axis]/2 for axis in range(len(
            self.VolShape))]
        res = pd.DataFrame(dtype = "float")
        for file in self.files.index :
            subdf = data[data["FILE"] == file]
            res = pd.concat([res, OAT.clusteringEngine(
                OAT.coordTransformation(subdf, Zratio = 2),
                volume_center
                )])
        res["F_CLUSTER"].fillna(10, inplace = True)
        res["F_SELECT"].fillna(False, inplace = True)
        if df == "tracks":
            self.tracks = pd.concat([self.tracks, res], axis = 1)
        else :
            self.spots = pd.concat([self.spots, res], axis = 1)
    
    def showSpots(self, filename, ROI = False, save = False, df = "spots"):
        """
        Create and shows a set of 3 scatterplots for each plane.
        Each one represent the spots of the given file, colored by cluster.

        Parameters
        ----------
        filename : str or list
            Name of the image file or list of the name of the image files. 
            Do not include the '.tif'

        Returns
        -------
        Figure

        """
        cmap = plt.cm.get_cmap('tab10')
        if filename == "all" and save:
            for file in self.files.index:
                self.showSpots(file, ROI, save, df)
            return None
        elif type(filename) == str :
            fig, axs = plt.subplots(1, 3, 
                                    figsize=(20, 6), dpi=400)
            planes = [["X","Y"],["X","Z"],["Y","Z"]]
            ROIcol = [[0, 2], [0, 4], [2, 4]]
            if df == "spots":
                subdf = self.spots[self.spots["FILE"] == filename]
            elif df == "tracks":
                subdf = self.tracks[self.tracks["FILE"] == filename]
            # Showing all the 3 planes.
            for idx in range(3):
                color = "b"
                # Coloring the clusters if present
                if "F_SELECT" in subdf.columns :
                    color = subdf["F_CLUSTER"].map(cmap)
                axs[idx].scatter(subdf[planes[idx][0]],
                                 subdf[planes[idx][1]],
                                 c = color)
                axs[idx].set_xlabel(planes[idx][0])
                axs[idx].set_ylabel(planes[idx][1])
                axs[idx].set_title("File : "+filename+", View : "+\
                                   planes[idx][0]+\
                                   "*"+planes[idx][1])
                # If there are cluster selection info, we add a legend 
                # indicating what are the selected cells.
                if "F_SELECT" in subdf.columns :
                    cluster_id = subdf[subdf["F_SELECT"]]["F_CLUSTER"][0]
                    legend = [Line2D([0], [0], marker = 'o', 
                                     color = cmap(cluster_id), 
                                     label = 'Selected spots', 
                                     markerfacecolor = cmap(cluster_id), 
                                     markersize=10)]
                    axs[idx].legend(handles = legend, loc = 'best')
                if hasattr(self, "ROI") and ROI:
                    axs[idx].set_xlim([ self.ROI[ROIcol[idx][0]], 
                                       self.ROI[ROIcol[idx][0]+1] ])
                    axs[idx].set_ylim([ self.ROI[ROIcol[idx][1]], 
                                       self.ROI[ROIcol[idx][1]+1] ])
        elif type(filename) == list :
            fig, axs = plt.subplots(len(filename), 3, 
                                    figsize=(20, 6*len(filename)), dpi=400)
            planes = [["X","Y"],["X","Z"],["Y","Z"]]
            ROIcol = [[0, 2], [0, 4], [2, 4]]
            for fileID in range(len(filename)):
                if df == "spots":
                    subdf = self.spots[self.spots["FILE"] == filename]
                elif df == "tracks":
                    subdf = self.tracks[self.tracks["FILE"] == filename]
                # Showing all the 3 planes.
                for idx in range(3):
                    color = "b"
                    # Coloring the clusters if present
                    if "F_SELECT" in subdf.columns :
                        color = subdf["F_CLUSTER"].map(cmap)
                    axs[fileID, idx].scatter(subdf[planes[idx][0]],
                                     subdf[planes[idx][1]],
                                     c = color)
                    axs[fileID, idx].set_xlabel(planes[idx][0])
                    axs[fileID, idx].set_ylabel(planes[idx][1])
                    axs[fileID, idx].set_title("File : "+filename[fileID]+\
                                               ", View : "+planes[idx][0]+\
                                                   "*"+planes[idx][1])
                    # If there are cluster selection info, we add a legend 
                    # indicating what are the selected cells.
                    if "F_SELECT" in subdf.columns :
                        cluster_id = subdf[subdf["F_SELECT"]]["F_CLUSTER"][0]
                        legend = [Line2D([0], [0], marker = 'o', 
                                         color = cmap(cluster_id), 
                                         label = 'Selected spots', 
                                         markerfacecolor = cmap(cluster_id), 
                                         markersize=10)]
                        axs[fileID, idx].legend(handles = legend, loc = 'best')
                    if hasattr(self, "ROI") and ROI:
                        axs[fileID, idx].set_xlim([ self.ROI[ROIcol[idx][0]], 
                                           self.ROI[ROIcol[idx][0]+1] ])
                        axs[fileID, idx].set_ylim([ self.ROI[ROIcol[idx][1]], 
                                           self.ROI[ROIcol[idx][1]+1] ])
        fig.text(0.08, 0.05, "OAT Version : "+str(self.version))
        if save :
            plt.savefig(self.dir["root"]+"\\debug\\clusters\\"+filename+".png",
                        dpi = 'figure')
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
        if filename == "all":
            for file in self.files.index:
                self.exportSpotsVTK(file, organoid, df)
        elif type(filename) == str:
            if df == "spots":
                subdf = self.spots[self.spots["FILE"] == filename]
            elif df == "tracks":
                subdf = self.tracks[self.tracks["FILE"] == filename]
            if "IS_ORGANOID" in subdf.columns and organoid:
                subdf = subdf[subdf["IS_ORGANOID"]]
                if df == "spots":
                    PyVTK(filename+"_spots_organoid", subdf['X'], subdf['Y'], 
                      subdf['Z'], self.dir["vtk"], "points")
                elif df == "tracks":
                    PyVTK(filename+"_tracks_organoid", subdf['X'], subdf['Y'], 
                      subdf['Z'], self.dir["vtk"], "points")
            else :
                if df == "spots": 
                    PyVTK(filename+"_spots", subdf['X'], subdf['Y'], 
                          subdf['Z'], self.dir["vtk"], "points")
                elif df == "tracks":
                    PyVTK(filename+"_tracks", subdf['X'], subdf['Y'], 
                          subdf['Z'], self.dir["vtk"], "points")
        elif type(filename) == list:
            for file in filename:
                self.exportSpotsVTK(file, organoid, df)              
                
    def selectROI(subdf, std = 15):
        """
        Method to get the most representative value from a pd.Series in the 
        context of the search of the ROI.

        Parameters
        ----------
        subdf : pd.Series
            Series or df column that contain values for a certain category.
        std : int
            Standard deviation threshold.

        Returns
        -------
        float
            Best fitting value for the limit of the ROI.

        """
        col = subdf.name
        # if the standard deviation is very small, then we don't need to 
        # cluster the values.
        if subdf.std() <= std :
            # Hacking the names to makes it works with the next part.
            results = pd.Series(len(subdf.index)*[0], index = subdf.index, 
                                name = col)
        else :
            # Clustering with KMeans as we expect few off points
            results = KMeans(n_clusters = 3).fit_predict(subdf.to_frame())
            results = pd.Series(results, index = subdf.index, name = col)
        # Getting the value of the cluster with the highest number 
        # of values. 
        biggestClusterID = results.value_counts(ascending = False).index[0]
        # Getting the side of the border : if "min" then we remove a
        # a certain amount of the value et reverse for "max".
        extreme = re.split("\_", col)[-1]
        if extreme == "min":
            value = subdf[results == biggestClusterID].min()
            return value-5
        elif extreme == "max":
            value = subdf[results == biggestClusterID].max()
            return value+5
                   
    def getROI(self):
        """
        Processing the limits of the ROI (region of interest).
        The ROI is the cube where the organoid is, whatever the timepoint is.
        These limits are saved in the self.ROI pandas.Series. 
        
        """
        # Importing the spots. (Existence checks are handled by the called 
        # method).
        self.getSpots()
        # Clustering the spots.
        self.getClusters()
        # Temporary list containing the min/max for each axis. Each sublist   
        # represent a file.
        localROI = []
        # Getting the filenames of the processed images.
        filenames = self.spots["FILE"].value_counts().index
        for name in filenames :
            subdf = self.spots[self.spots["FILE"] == name]
            subdf = subdf[subdf["F_SELECT"]]
            localROI.append([subdf["X"].min(), subdf["X"].max(),
                            subdf["Y"].min(), subdf["Y"].max(),
                            subdf["Z"].min(), subdf["Z"].max()])
        # Merging all limits into a dataframe.
        labels = ["X_min", "X_max", "Y_min", "Y_max", "Z_min", "Z_max"]
        self.localROI = pd.DataFrame(localROI, index = filenames,
                                       columns = labels)
        # Searching for the general ROI by clustering on each axis and getting
        # the mean value of the cluster.
        self.ROI = pd.Series(len(labels)*[0], index = labels, name = "ROI")
        # Browsing the columns in the self.localROI dataframe.
        for col in labels :
            # Extracting the column.
            subdf = self.localROI[col]
            self.ROI[col] = OAT.selectROI(subdf)
        # Adding the global ROI in the the local ROI as a checking measure.
        self.localROI = pd.concat([self.localROI, self.ROI.to_frame().T], 
                                  ignore_index = True)
        self.localROI.index = list(filenames)+["ROI"]
        
    def getArea(spot, radius, volShape):
        return [[x for x in range(int(spot["X"])-radius, 
                                  int(spot["X"])+radius+1)],
                [y for y in range(int(spot["Y"])-radius, 
                                  int(spot["Y"])+radius+1)],
                [z for z in range(int(spot["Z"])-radius, 
                                  int(spot["Z"])+radius+1)]]
        
    def denoising(self, file):
        """
        Load a tiff image and set all pixels that are not in the ROI to 0.
        Used in the cleanImage method.

        Parameters
        ----------
        _file : str
            Path to the image.

        Returns
        -------
        _imarray : np.array
            Denoised array of the image.

        """
        # Opening the image.
        image = io.imread(file)
        # Converting into a Numpy array
        imarray = np.array(image)
        # The shapes are in this order Z, Y and X!
        # For each axis, we get the coordinate (1D) of the pixels that needs to 
        # be set to black. 
        X_values = [X for X in range(imarray.shape[2]) if
                     X < self.ROI["X_min"] or X > self.ROI["X_max"]]
        Y_values = [Y for Y in range(imarray.shape[1]) if
                     Y < self.ROI["Y_min"] or Y > self.ROI["Y_max"]]
        Z_values = [Z for Z in range(imarray.shape[0]) if
                     Z < self.ROI["Z_min"] or Z > self.ROI["Z_max"]]
        # Setting the given pixel to 0 on each axis.
        imarray[Z_values,:,:] = 0
        imarray[:,Y_values,:] = 0
        imarray[:,:,X_values] = 0
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
        return imarray 
        
    def cleanImage(self):
        """
        Remove all of the image except for the ROI and proceed to save it in 
        the ./data directory. The output is called "{sample}_tp.tif" where tp 
        mean timepoint.
        The tiff include timepoints and can be used in Trackmate to get tracks. 

        """
        #Creating a temporary array.
        _4Darray = []
        #Adding each 3D array in the list in chronological order.
        for _file in self.files["tifs"]:
            _4Darray.append(self.denoising(self.dir["tifs"]+'\\'+_file))
        #Saving the tif with metadata.
        tifffile.imwrite(self.dir["root"]+'\\data\\'+self.sample+"_tp.tif", 
                         np.array(_4Darray), 
                         imagej=True, metadata={'axes': 'TZYX'})
        
    def readTracks(self):
        """
        Load tracks file ({sample}_re.csv and {sample}_rt.csv) in the 
        data/tracks directory as self.tracks. 
        re stands for raw edges and is the output file called "edge" in 
        Trackmate. 
        rt stands for raw tracks and is the output file called "track" in 
        Trackmate.

        """
        # Importing the files.
        self.importFiles("tracks")
        tracks_df = pd.read_csv(self.tracks_csv)
        edges_df = pd.read_csv(self.edges_csv)
        #Removing the 3 first rows as they are redundant with labels
        tracks_df.drop(index = [0, 1, 2], inplace = True)
        edges_df.drop(index = [0, 1, 2], inplace = True)
        #Setting the correct indexes
        tracks_df.index = tracks_df["LABEL"]
            #For this one, we set the first ID in the label cell to be the 
            #index for the row because it is the starting spot of the edge.
        edges_df.index = [re.split(" ", _lbl)[0] for _lbl in 
                               edges_df["LABEL"]]
        #Renaming some labels in order to be cleaner
        for axis in ["X", "Y", "Z"]:
            tracks_df.rename(columns = {"POSITION_"+axis:axis},
                                  inplace = True)
        edges_df.rename(columns = {"SPOT_TARGET_ID":"TARGET"}, 
                             inplace = True)
        #Keeping the interesting columns 
        tracks_df = tracks_df.loc[:,["TRACK_ID", "QUALITY", "X", "Y", "Z", 
                                     "FRAME"]]
        edges_df = edges_df.loc[:,["TARGET", "EDGE_TIME"]]
        #Setting the dataframes' values type as float
        tracks_df = tracks_df.astype("float")
        tracks_df["TRACK_ID"] = tracks_df["TRACK_ID"].astype("int")
        edges_df = edges_df.astype("float")
        # Setting the correct names for the files
        tracks_df.rename(columns = {"FRAME": "FILE"}, inplace = True)
        tracks_df["FILE"] = [self.files.iloc[int(k)].name \
                             for k in tracks_df["FILE"]]
        #Modifying the "Target" values to be Spot ids
        edges_df["TARGET"] = ["ID"+str(int(_id)) for _id 
                                   in edges_df["TARGET"]]
        self.tracks = tracks_df.join(edges_df)
     
    def buildTimeline(self):
        """
        Method to generate a pandas Series containing the spot order for each
        track and saving it as self.spots_timeline.
    
        Returns
        -------
        None.
    
        """
        #Extracting the column that give the target of a spot (next spot)
        #For each row, the index is the sources and the value is the target 
        _links = self.tracks["TARGET"]
        #Sources who don't have any targets are the ending spots of a track
        #We will construct the list in reverse and reverse it at the end
        _enders = _links[_links.isnull()].index
        #_list is a list of sublists where each sublist is one track
        _list = [[_id] for _id in _enders]
        #Looping until we can't find any backward links (so it's done)
        _unfinished = True
        while _unfinished:
            _unfinished = False
            for _track in _list:
                #_track is the sublist reffered to in the last comment
                try : 
                    #We add the ID of the previous source by looking at the
                    #index of the row where the target id match the last
                    #element of the _track sublist
                    _track.append(_links[_links == _track[-1]].index[0])
                    #There is an established connection, there could be others
                    _unfinished = True
                except :
                    pass
        #Reversing each sublist to have the spots in the timeorder
        for _track in _list :
            _track.reverse()
        #Finally we save the _list as a pandas Series where the indexes are 
        #the track_id for each sequence (values here)
        self.spots_timeline = pd.Series(_list, index = [self.tracks.loc[
                                        idx[0]]["TRACK_ID"] 
                                         for idx in _list])
        
    def processVectors(df):
        """
        Return the 2D or 3D vector between 2 points.
    
        Parameters
        ----------
        df : pd.DataFrame
            Indexes are the IDs, labels are the axis. The first row is the 
            origin and the second one is the destination.
    
        Returns
        -------
        pd.Series
            Indexes are axis, the order is the same as the input one.
    
        """
        _list = [df.iloc[1][_axis] - df.iloc[0][_axis] for _axis in df.columns]
        return pd.Series(_list, index = df.columns, dtype="float") 
    
    def getVectors(self, filtering = False):
        """
        Compute displacement vectors for every spots in the 
        dataframe and add them to it.
        Vectors are computed based on the sequence of the track they're in.
        That means that vectors are computed between 2 following spots of a 
        given track.
        They are saved in the same line as the origin spot that served the
        vector's calculation.
    
        Returns
        -------
        Update self.tracks.
    
        """
        #Creation of the timeline table for every tracks
        self.readTracks()
        if filtering :
            self.getClusters(df="tracks")
            self.tracks = self.tracks[self.tracks["F_SELECT"]]
        self.buildTimeline()
        #Creation of a dataframe that will store vectors as thet are computed.
        #Each row is a vector with its index being the ID of the origin spot
        #of the vector.
        vectors = pd.DataFrame(columns = ["uX", "vY", "wZ"], 
                                 dtype = "float")
        #Browsing the timeline table where _trackidx is the track id.
        for trackidx in self.spots_timeline.index :
            #_list will store every vectors coordinates as a sublist.
            _list = []
            #Now we browse by index the list that contains all the spot ID's in
            #chronological order for a given track (trackidx).
            for spot in range(len(self.spots_timeline[trackidx])-1) :
                #subdf is a subset of the self.tracks Dataframe which 
                #contains the X, Y, Z coordinates as columns of 2 spots. The 
                #spots are ordered that way : origin and the destination.
                subdf = self.tracks.loc[
                    [self.spots_timeline[trackidx][spot], 
                     self.spots_timeline[trackidx][spot+1]
                     ], ["X", "Y", "Z"]]
                #res just retrieve the series we get with the processVectors
                #method.
                res = OAT.processVectors(subdf)
                #We add the coordinates as a sublist in _list. Note that we
                #don't use Series in this case because all the coordinates are 
                #always in the same order (X, Y, Z) in this class.
                _list.append([x for x in res])
            #Adding a null vector for index reasons as the last spot don't have
            #vector.
            _list.append([np.nan]*3)
            #For each track, the lines with the origin spot as index and 
            #its given coordinates are concatenated to the vectors Dataframe.
            vectors = pd.concat([vectors, 
                                 pd.DataFrame(_list, 
                                    index = self.spots_timeline[trackidx],
                                    columns = ["uX", "vY", "wZ"])])
        #We concatenate the vectors Dataframe with self.tracks. As they  
        #have the same index, data (vector coordinates) are added in new 
        #columns.
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
            Timepoint.
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
            #print([ymax-(ymax-ymin)/2, ymax+10])
        elif label == "Y":
            xmax, xmin = subdf["X"].max(), subdf["X"].min()
            ax.set_xlim3d([xmax-(xmax-xmin)/2, xmax+10])
            #print([xmax-(xmax-xmin)/2, xmax+10])
        elif label == "Z":
            zmax, zmin = subdf["Z"].max(), subdf["Z"].min()
            ax.set_zlim3d([zmax-(zmax-zmin)/2, zmax+10])
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
        
    def exportVectorsVTK(self):
        #Retrieving available timepoints and keeping those who have more than 
        #10 spots. That way we remove incorrect vectors. 
        TP_list = [k 
                    for k in self.tracks["EDGE_TIME"].value_counts().index 
                    if self.tracks["EDGE_TIME"].value_counts()[k] >= 10]
        _tracks_list = [k for k in 
                        self.tracks["TRACK_ID"].value_counts().index]
        #Sorting the list.
        TP_list.sort()
        _tracks_list.sort()
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
            for track in _tracks_list :
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
                                    columns = _tracks_list)
        Ypoints = pd.DataFrame(list_Y, index = TP_list, 
                                    columns = _tracks_list)
        Zpoints = pd.DataFrame(list_Z, index = TP_list, 
                                    columns = _tracks_list)
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
            translation = OAT.processVectors(temp_df)
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
                    OAT.processVectors(centroids.iloc[ID-1:ID+1]))
                drift.loc[index] = OAT.euclid_distance(centroids.iloc[ID], 
                                                       centroids.iloc[ID-1])
        if not hasattr(self, "data"):
            self.data = pd.DataFrame(index=self.files["time"])
        centroids.columns = ["cX", "cY", "cZ"]
        self.data = pd.concat([self.data, centroids], axis = 1)
        self.data = pd.concat([self.data, vectors], axis = 1)
        self.data = pd.concat([self.data, drift], axis = 1)
        
    def crossProduct(df):
        #Return the crossproduct of the 2 vectors
        A, B = df.iloc[0], df.iloc[1]
        return pd.Series([(A["vY"]*B["wZ"]-A["wZ"]-B["vY"]),
                          -(A["uX"]*B["wZ"]-A["wZ"]*B["uX"]),
                          (A["uX"]*B["vY"]-A["vY"]*B["uX"])],
                         index = ["uX", "vY", "wZ"], dtype = "float")
        
    def computeRotationAxis(self, drift = True):
        #Compute the rotation axis for each frame or time point
        rotationAxisVectors = pd.DataFrame(columns = ["uX", "vY", "wZ"], 
                                           dtype = "float")
        for file in self.files.index:
            subdf = self.tracks[self.tracks["FILE"] == file]
            crossproduct = pd.DataFrame(columns = ["uX", "vY", "wZ"], 
                                        dtype = "float")
            for iteration in range(1000):
                sample = subdf.sample(2)
                sample = sample.loc[:,["uX", "vY", "wZ"]]
                crossproduct.loc[iteration] = OAT.crossProduct(sample)
                crossproduct.dropna()
            rotationAxisVectors.loc[
                self.files.loc[file, "time"]] = crossproduct.mean()
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
    T.computeStats()
    T.showData()
    T.animVectors()

            
### NO USE AT THE MOMENT ------------------------------------------------------

# def minimizing_func(spots_df, center_coords):
#     x, y, z, r = center_coords
#     _sum = 0
#     for _index in spots_df.index:
#         point = [spots_df.loc[_index][axis] 
#                  for axis in ["X", "Y", "Z"]]
#         _sum += (OAT.euclid_distance(point, [x, y, z])-r)
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

# def selectClusters(self, spots_df):
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
#     volume_center = list(self.volume_shape)
#     volume_center.reverse()
#     for idx in cluster_ids :
#         centroid = OAT.getCentroid(spots_df[spots_df["CLUSTER"] == idx])
#         distance = OAT.euclid_distance(centroid, volume_center)
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
#     identified_clusters = DBSCAN(
#         eps=eps, min_samples=min_samples).fit_predict(
#             subdf.loc[:,["X", "Y", "Z"]])
#     #Adding the cluster results to the spots dataframe.
#     cluster = pd.Series(identified_clusters, index = subdf.index, 
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
#             distance[spot] = OAT.euclid_distance(
#                 list(r_tracks.loc[spot, ["X", "Y", "Z"]]), 
#                 centroid
#                 )
#             tempdf = pd.DataFrame(
#                 [list(r_tracks.loc[spot, ["X", "Y", "Z"]]),
#                  centroid], columns = ["X", "Y", "Z"])
#             res.append(list(OAT.processVectors(tempdf)))
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
        