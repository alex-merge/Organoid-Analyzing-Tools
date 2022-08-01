# -*- coding: utf-8 -*-
"""
Vector based analyzing components of OAT.

@author: Alex-932
@version: 0.7
"""

import os
import pandas as pd
import time
from modules.utils.filemanager import *
from modules.utils.clustering import *
from modules.utils.compute import *
from modules.utils.tools import *

class vectors():
    
    def read_tracks(dirpath, rescaling = [1, 1, 1]):
        """
        Load tracks file (tracks.csv and edges.csv) in the 
        \\data\\tracks directory as a DataFrame called df. 
        
        tracks.csv correspond to the .csv you can get by saving the dataset 
        found in tracks>spots.
        
        tracks.csv correspond to the .csv you can get by saving the dataset 
        found in tracks>edges.

        """
        
        ## Importing the files.
        files = filemanager.search_file(dirpath, "csv")
        
        attributes = []
        dataframes = []
        
        for filepath in files:
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
                                     "TP"]]
        edges_df = edges_df.loc[:,"TARGET"]
        
        # Setting the dataframes' values type
        tracks_df = tracks_df.astype("float")
        tracks_df[["TRACK_ID", "TP"]] = tracks_df[["TRACK_ID", "TP"]
                                                  ].astype("int")
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
    
    def full_analysis(dirpath, rescaling = [1, 1, 1]):
        
        start_time = time.time()
        
        ## Loading vectors.
        step_time = time.time()
        print("Opening files ...", end = " ")
        df = vectors.read_tracks(dirpath)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        
        ## Computing vectors.
        step_time = time.time()
        print("Computing vectors ...", end = " ")
        df = compute.vectors(df)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        
        ## Translating coords to the center ([0, 0, 0])
        step_time = time.time()
        print("Translating coordinates to the center ...", end = " ")
        df = compute.translation(df)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        
        ## Computing the axis of rotation vectors.
        step_time = time.time()
        print("Computing the axis of rotations ...", end = " ")
        data = compute.rotation_axis(df)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        
        ## Aligning axis of rotation with the Z axis.
        step_time = time.time()
        print("Aligning rotation axis and Z axis ...", end = " ")
        df, data = compute.alignment(df, data)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        
        ## Preparing the dataframe to compute the aligned vectors.
        aligned_df = df.loc[:,["TRACK_ID", "Aligned_X", "Aligned_Y", 
                               "Aligned_Z", "TARGET"]].copy()
        aligned_df.rename(columns = {"Aligned_X": "X", 
                                     "Aligned_Y": "Y",
                                     "Aligned_Z": "Z"}, 
                          inplace = True)
        
        ## Computing aligned displacements vectors.
        step_time = time.time()
        print("Computing aligned vectors ...", end = " ")
        aligned_vectors = compute.vectors(aligned_df, inplace = False)
        aligned_vectors.rename(columns = {"uX": "Aligned_uX",
                                          "vY": "Aligned_vY",
                                          "wZ": "Aligned_wZ"},
                               inplace = True)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        
        ## Merging results.
        df = pd.concat([df, aligned_vectors], axis = 1)
        
        ## Computing angular velocity.
        step_time = time.time()
        print("Computing angular velocity ...", end = " ")
        df, data = compute.angular_velocity(df, data)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        print("")
        print("Analysis done ! Total time :", 
              str(round(time.time()-start_time, 2))+"s")
        
        ## Returning datasets.
        return df, data

        
        