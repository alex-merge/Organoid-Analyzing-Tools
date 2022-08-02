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
from modules.utils.figures import *

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
        
        # Merging coordinates into an array.
        tracks_df[list("XYZ")] = tracks_df[list("XYZ")].astype("float")
        coords = pd.Series([arr for arr in tracks_df[list("XYZ")].to_numpy()],
                            index = tracks_df.index, name = "COORD")
        tracks_df = pd.concat([tracks_df, coords], axis = 1)
        
        # Keeping the interesting columns 
        tracks_df = tracks_df.loc[:,["TRACK_ID", "QUALITY", "COORD", 
                                     "TP"]]
        edges_df = edges_df.loc[:,"TARGET"]
        
        # Setting the dataframes' values type
        tracks_df[["QUALITY", "TP"]] = tracks_df[["QUALITY", "TP"]
                                                 ].astype("float")
        tracks_df[["TRACK_ID", "TP"]] = tracks_df[["TRACK_ID", "TP"]
                                                  ].astype("int")
        edges_df = edges_df.astype("float")
        
        # Modifying the "Target" values to be spots ID.
        edges_df = "ID" + edges_df.astype(int).astype("str")
        
        # Merging the 2 dataframes.
        tracks = pd.concat([tracks_df, edges_df], axis = 1)
        
        # Rescaling the coordinates in case we need to.
        tracks["COORD"] = [arr*np.array(rescaling) for arr in tracks["COORD"]]
        
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
        
        ## Computing drift.
        step_time = time.time()
        print("Computing drift ...", end = " ")
        data = compute.drift(df)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        
        ## Computing volume and radius.
        step_time = time.time()
        print("Computing volume and radius ...", end = " ")
        data = pd.concat([data, compute.volume(df)], axis = 1)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        
        ## Translating coords to the center ([0, 0, 0])
        step_time = time.time()
        print("Translating coordinates to the center ...", end = " ")
        df = compute.translation(df)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        
        ## Computing the axis of rotation vectors.
        step_time = time.time()
        print("Computing the axis of rotations ...", end = " ")
        data = pd.concat([data, compute.rotation_axis(df)], axis = 1)
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
        
        ## Coloring or not the vectors according to their cluster.
        if color_clusters and "F_SELECT" in subdf.columns :

            ax, new_legend = figures.add_3Dvectors(
                ax, subdf[subdf["F_SELECT"]], quiver_col,
                label = "Selected vectors", color = "green")
            legend.append(new_legend)
            
            ax, new_legend = figures.add_3Dvectors(
                ax, subdf[subdf["F_SELECT"] == False], quiver_col,
                label = "Unselected vectors", color = "orange")
            legend.append(new_legend)
        
        else :
            ax = figures.add_3Dvectors(ax, subdf, quiver_col)
        
        ## Showing raw centroid and more if available/wanted.
        if show_centroid :
            
            ax, new_legend = figures.add_spots(ax, subdf, label = "Raw centroid",
                                               marker = "^", color = "red") 
            
            if color_clusters and "F_SELECT" in subdf.columns :

                ax, new_legend = figures.add_spots(
                    ax, subdf[subdf["F_SELECT"]], label = "Cluster's centroid", 
                    marker = "^", color = "navy") 
        
        if show_rot_axis and data is not None and "radius" in data.columns :
            
            RA_vect = data.loc[TP, ["RA_uX", "RA_vY", "RA_wZ"]].tolist()
            
            if color_clusters and "F_SELECT" in subdf.columns :
                cx, cy, cz = tools.get_centroid(subdf[subdf["F_SELECT"]])
            
            else :
                cx, cy, cz = tools.get_centroid(subdf)
                
            vect_scale = ((0.5*data.loc[TP, "radius"])/
                          (tools.euclid_distance(RA_vect, [0, 0, 0])))
            
            ax.quiver(cx, cy, cz, RA_vect[0], RA_vect[1], RA_vect[2], 
                      color = "orange", length = vect_scale)
            ax.quiver(cx, cy, cz, -RA_vect[0], -RA_vect[1], -RA_vect[2], 
                      color = "orange", length = vect_scale)
            
            legend.append(Line2D([0], [0], marker = ('>'), 
                                 color = "orange", 
                                 label = 'Rotation axis',
                                 markerfacecolor = "orange", 
                                 markersize=7, ls = ''))
        
                
            
                
                
        ax.set_title("Displacement vectors for time point "+str(TP), 
                     fontsize = 15)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        ax.legend(handles = legend, loc = 'best')
        
        plt.show()
        
        