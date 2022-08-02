# -*- coding: utf-8 -*-
"""
Vector based analyzing components of OAT.

@author: Alex-932
@version: 0.7
"""

import os
import pandas as pd
import time

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

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
    
    def full_analysis(dirpath, rescaling = [1, 1, 1], filtering = False):
        
        start_time = time.time()
        
        ## Loading vectors.
        step_time = time.time()
        print("Opening files ...", end = " ")
        df = vectors.read_tracks(dirpath)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        
        ## Computing vectors.
        step_time = time.time()
        print("Computing displacement vectors ...", end = " ")
        df = compute.vectors(df, filtering = filtering)
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
        print("Computing aligned displacement vectors ...", end = " ")
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
    
    def show_vectors(df, TP, data = None, dim = "xyz", 
                     show = True, savepath = None,
                     show_centroid = True, color_clusters = True, 
                     show_rot_axis = True):
        
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
        
        ## Setting the projection if 3D figure.
        if len(dim) < 3 :
            ax = fig.add_subplot(111)
        else :
            ax = fig.add_subplot(111, projection = "3d")
        
        ## Getting the order of axis wanted and in upper case.
        axis_order = [axis.upper() for axis in list(dim)]
        
        ## Getting the vectors axis in the same order and adding them to the
        ## axis_order list.
        rel_vect_axis = {"X": "uX", "Y": "vY", "Z": "wZ"}
        quiver_col = axis_order + [rel_vect_axis[axis] for axis in axis_order]
        
        if color_clusters and "F_SELECT" in subdf.columns :
            t_df = subdf[subdf["F_SELECT"]]
            f_df = subdf[subdf["F_SELECT"] == False]
            
            ax.quiver(t_df[quiver_col[0]], t_df[quiver_col[1]], 
                      t_df[quiver_col[2]], t_df[quiver_col[3]], 
                      t_df[quiver_col[4]], t_df[quiver_col[5]],
                      color = "green")
            ax.quiver(f_df[quiver_col[0]], f_df[quiver_col[1]], 
                      f_df[quiver_col[2]], f_df[quiver_col[3]], 
                      f_df[quiver_col[4]], f_df[quiver_col[5]],
                      color = "orange")
            
            legend.append(Line2D([0], [0], marker = ('>'), 
                                 color = "green", 
                                 label = 'Selected vectors',
                                 markerfacecolor = "green", 
                                 markersize=7, ls = ''))
            
        else :
            ax.quiver(subdf[quiver_col[0]], subdf[quiver_col[1]], 
                      subdf[quiver_col[2]], subdf[quiver_col[3]], 
                      subdf[quiver_col[4]], subdf[quiver_col[5]],
                      color = "black")
        
        ## Showing raw centroid and more if available/wanted.
        if show_centroid :
            if color_clusters and "F_SELECT" in subdf.columns :
                cx, cy, cz = tools.get_centroid(subdf[subdf["F_SELECT"]])
                
                ax.scatter(cx, cy, cz, c="navy", marker = "^", s = 50)
                
                ## Adding the legend.
                legend.append(Line2D([0], [0], marker = '^', 
                                     color = "navy", 
                                     label = "Cluster's centroid",
                                     markerfacecolor = "navy", 
                                     markersize=7, ls = ''))
            
            ## Showing raw centroid.
            cx, cy, cz = tools.get_centroid(subdf)
            label_name = "Raw centroid"
                
            ax.scatter(cx, cy, cz, c="red", marker = "^", s = 50)
            
            legend.append(Line2D([0], [0], marker = '^', 
                                 color = "red", 
                                 label = label_name,
                                 markerfacecolor = "red", 
                                 markersize=7, ls = ''))
        
        if show_rot_axis :
            pass
        
        ax.set_title("Displacement vectors for time point "+str(TP), 
                     fontsize = 15)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        ax.legend(handles = legend, loc = 'best')
        
        plt.show()
        

        
        