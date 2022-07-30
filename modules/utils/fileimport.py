# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 09:53:34 2022

@author: Alex
"""

import pandas as pd
import os
from modules.utils.compute import *

class fileimport():
    
    def readTracks(filepath, rescaling = [1, 1, 1]):
        """
        Load tracks file (tracks.csv and edges.csv) in the 
        \\data\\tracks directory as a DataFrame called self.tracks. 
        
        tracks.csv correspond to the .csv you can get by saving the dataset 
        found in tracks>spots.
        
        tracks.csv correspond to the .csv you can get by saving the dataset 
        found in tracks>edges.

        """
        
        # Importing the files.
        files = [filepath+"\\"+file for file in os.lisdir(filepath)]
        
        attributes = []
        dataframes = []
        
        for file in files:
            stream = pd.read_csv(filepath)
            dataframes.append(stream)
            
            ## 1 is the edge dataframe and 2 is the tracks dataframe.
            if "SPOT_SOURCE_ID" in stream.columns :
                attributes.append(1)
                edges_df = stream
                
            elif "QUALITY" in stream.columns :
                attributes.append(2)
                tracks_df = stream
                
            else :
                attributes.append(0)
        
        ## If the sum is different than 3, we have either to many files or 
        ## missing files.
        if sum(attributes) != 3:
            raise IOError("Missing or redundant files")
            
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
        
        # Modifying the "Target" values to be spots ID.
        edges_df = "ID" + edges_df.astype(int).astype("str")
        
        # Merging the 2 dataframes.
        tracks = pd.concat([tracks_df, edges_df], axis = 1)
        
        # Rescaling the coordinates in case we need to.
        tracks["X"] = tracks["X"]*rescaling[0]
        tracks["Y"] = tracks["Y"]*rescaling[1]
        tracks["Z"] = tracks["Z"]*rescaling[2]
        
        return tracks
        