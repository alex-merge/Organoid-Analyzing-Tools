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
        