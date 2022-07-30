# -*- coding: utf-8 -*-
"""
organoid_tracking_tools (OAT) is a set of methods that integrates FIJI's 
Trackmate csv files output to process cell displacement within an organoid.

@author: Alex-932
@version: 0.7
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from skimage import io
import tifffile
import cv2
import time

from modules.utils.clustering import clustering
from modules.tools import tools
from modules.utils.compute import compute
from modules.export import export
from modules.figures import figures

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
        self.version = "0.7"
         
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
        self.dir["avFigs"] = root+'\\output\\figs\\angularVelocity'
        self.dir["anim"] = root+'\\output\\animation'
        self.dir["vtk"] = root+'\\output\\vtk_export'
        self.dir["mat"] = root+'\\output\\matlab_export'
        
        # Creating the directories if they don't already exist.
        for path in self.dir:
            if not os.path.exists(path):
                os.makedirs(path)
                
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
                   
    
        
    def getArea(spot, radius, volShape):
        return [[x for x in range(int(spot["X"])-radius, 
                                  int(spot["X"])+radius+1)],
                [y for y in range(int(spot["Y"])-radius, 
                                  int(spot["Y"])+radius+1)],
                [z for z in range(int(spot["Z"])-radius, 
                                  int(spot["Z"])+radius+1)]]      
        
    
        
    def readTracks(self, rescaling = [1, 1, 1]):
        """
        Load tracks file (tracks.csv and edges.csv) in the 
        \\data\\tracks directory as a DataFrame called self.tracks. 
        
        tracks.csv correspond to the .csv you can get by saving the dataset 
        found in tracks>spots.
        
        tracks.csv correspond to the .csv you can get by saving the dataset 
        found in tracks>edges.

        """
        clock = time.time()
        
        # Importing the files.
        self.importFiles("tracks")
        tracks_df = pd.read_csv(self.tracks_csv)
        edges_df = pd.read_csv(self.edges_csv)
        
        print("# Creating tracks dataframe ...")
        
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
        if not hasattr(self, "files") or "tifs" not in self.files.columns:
            self.loadTif()
        tracks_df.rename(columns = {"FRAME": "FILE"}, inplace = True)
        tracks_df["FILE"] = [self.files.iloc[int(k)].name \
                             for k in tracks_df["FILE"]]
        
        # Modifying the "Target" values to be spots ID.
        edges_df = "ID" + edges_df.astype(int).astype("str")
        
        # Merging the 2 dataframes.
        self.tracks = pd.concat([tracks_df, edges_df], axis = 1)
        
        # Computing the links between spots.
        self.spotsLinks = compute.SpotsLinks()
        
        # Rescaling the coordinates in case we need to.
        self.tracks["X"] = self.tracks["X"]*rescaling[0]
        self.tracks["Y"] = self.tracks["Y"]*rescaling[1]
        self.tracks["Z"] = self.tracks["Z"]*rescaling[2]
        
        # Creating self.data to store informations at time point level.
        self.data = pd.DataFrame(index = self.files["TP"])
        
        print("   Elapsed time :", time.time()-clock, "s")
    
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
            If True, use computeClusters() on tracks dataframe and keep the 
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
        
        clock = time.time()
        
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
            self.computeClusters(df="tracks")
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
        
        if not aligned :
            print("# Computing displacement vectors ...")
        else :
            print("# Computing aligned displacement vectors ...")
        
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
                vectors.loc[spotID] = tools.computeVectors(spotInfo, 
                                                         nextSpotInfo,
                                                         toList = True)
                
            # Adding a null vector as the last spot don't have any vector.
            vectors.loc[self.spotsLinks[trackID][-1]] = 3*[np.nan]
            
        # Merging the vectors dataframe to self.tracks. 
        self.tracks = pd.concat([self.tracks, vectors], axis = 1) 
        
        print("   Elapsed time :", time.time()-clock, "s")
        
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
        clock = time.time()
        print("# Creating animation showing vectors over time ...")        
        
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
            for tp in TP[:-1]:
                
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
        
        print("   Elapsed time :", time.time()-clock, "s")
        
        
    # def exportVectorsVTK(self, TP = "all", tracks = "all"):
    #     """
    #     Export the displacement vectors coordinates in a polydata .vtk format.

    #     Parameters
    #     ----------
    #     TP : int or list, optional
    #         Singe (int) or multiple (list) time points. The default is "all".
    #     tracks : int, optional
    #         Track ID. The default is "all".

    #     Returns
    #     -------
    #     Save the .vtk in \\output\\vtk_export

    #     """
    #     clock = time.time()
    #     print("# Exporting data into a .vtk file ...")
        
    #     # Creating the list of time points to show by the user choice.
    #     if TP == "all":
    #         TP = self.files["TP"]
    #     elif type(TP) == int:
    #         TP = [TP]
        
    #     # Creating the list of tracks that will be exported.
    #     if tracks == "all":
    #         tracks = [k for k in self.tracks["TRACK_ID"].value_counts().index]
    #     elif type(tracks) == int:
    #         tracks = [tracks]
        
    #     # Sorting the list of time point because we want to increment following
    #     # the chronological order. 
    #     tracks.sort()
        
    #     # Saving points coordinates in 3 dataframes, one for each axis. 
    #     # Rows are timepoints and columns are tracks.
    #     list_X, list_Y, list_Z = [], [], []
        
    #     # Filling previous lists with sublists. Each sublist has the 
    #     # corresponding axis coordinates of all points for 1 time point. It's 
    #     # like an image of the cells on the X axis at a certain time point.
    #     for tp in TP :
            
    #         print("Time point : ", tp)
            
    #         # Working with the subDataframe corresponding to the time point.
    #         subdf = self.tracks[self.tracks["TP"] == tp]
    #         list_X.append([])
    #         list_Y.append([])
    #         list_Z.append([])
            
    #         # Adding coordinates values for each track
    #         for track in tracks :
    #             # Retrieving the point data  
    #             pointData = subdf[subdf["TRACK_ID"] == track]
    #             # Checking if the point exist. If it doesn't then we enter no 
    #             # data (represented by a NaN value.) We will deal with these 
    #             # later on.
    #             if pointData.empty :
    #                 list_X[-1].append(np.nan)
    #                 list_Y[-1].append(np.nan)
    #                 list_Z[-1].append(np.nan)
                    
    #             else :
    #                 list_X[-1].append(pointData["X"][0])
    #                 list_Y[-1].append(pointData["Y"][0])
    #                 list_Z[-1].append(pointData["Z"][0])
                    
    #     # Creating the Dataframes that will be used by PyVTK. 
    #     Xpoints = pd.DataFrame(list_X, index = TP, columns = tracks)
    #     Ypoints = pd.DataFrame(list_Y, index = TP, columns = tracks)
    #     Zpoints = pd.DataFrame(list_Z, index = TP, columns = tracks)
        
    #     # Filling the empty values : if the track start later, we fill previous
    #     # positions with the starting one (bfill). We do the same thing the 
    #     # other way when it stop sooner (ffill).
    #     Xpoints.fillna(method = 'ffill', inplace = True)
    #     Ypoints.fillna(method = 'ffill', inplace = True)
    #     Zpoints.fillna(method = 'ffill', inplace = True)
    #     Xpoints.fillna(method = 'bfill', inplace = True)
    #     Ypoints.fillna(method = 'bfill', inplace = True)
    #     Zpoints.fillna(method = 'bfill', inplace = True)
        
    #     # Running PyVTK
    #     PyVTK(self.sample+"_tracks", Xpoints, Ypoints, Zpoints, 
    #           self.dir["vtk"], "polydata")
        
    #     print("   Elapsed time :", time.time()-clock, "s")
        
    # def exportMatLab(self):
    #     maxLength = max([self.tracks[self.tracks["TP"] == tp].shape[0] 
    #                      for tp in self.files["TP"]])
    #     for tp in self.files["TP"]:
    #         data = self.tracks[self.tracks["TP"] == tp]
    #         data = data.loc[:, ["Aligned_X", "Aligned_Y", "Aligned_Z"]]
    #         dic = {}
    #         for col in data.columns:
    #             values = np.concatenate(( data[col].to_numpy(), 
    #                                       np.zeros((maxLength-data.shape[0])) 
    #                                      ))
    #             dic[re.split("_", col)[-1]] = values
    #         savemat(self.dir["mat"]+"\\aligned_points_"+str(tp)+".mat", dic)
            
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
            
    
        
    
        
    def AngularVelocity(self, show = True):
        
        TranslatedCoords = compute.translateCoord(self.tracks)
        self.tracks = pd.concat([self.tracks, TranslatedCoords], axis = 1)
        
        data = compute.computeRotationAxis(self.tracks)
        
        AlignedCoords, newRA = compute.alignRotAxis(self.tracks, data)
        self.tracks = pd.concat([self.tracks, AlignedCoords], axis = 1)
        data = pd.concat([data, newRA], axis = 1)
        
        self.getVectors(aligned = True)
        
        velocityByTP, velocityByCell = compute.computeAngularVelocity(
            self.tracks, data)
        data = pd.concat([data, velocityByTP], axis = 1)
        self.tracks = pd.concat([self.tracks, velocityByCell], axis = 1)
        
        self.data = data
        
        if show:
            figures.AngularVelocity(self.tracks, "all")
        
        
                  
if __name__ == "__main__":
    T = OAT(fiji_dir = r"C:\Apps\Fiji.app", wrk_dir = r"D:\Wrk\Datasets\4")
    #T.loadTif()
    #T.getROI()
    #T.getVectors(filtering = False, rescaling = [1, 1, 4])
    #T.computeStats()
    # T.showData()
    # T.animVectors()
    S = OAT(wrk_dir = r"D:\Wrk\Datasets\S5")
    #S.getVectors()
    #S.alignRotAxis()
    #S.computeAngularVelocity()
    #S.loadTif()
    #T.getROI()
    #S.getVectors(filtering = False)
        