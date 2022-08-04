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
import seaborn as sns

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
        
        ## Computing aligned displacements vectors.
        step_time = time.time()
        print("Computing aligned displacement vectors ...", end = " ")
        df = compute.vectors(df, coord_column = "ALIGNED_COORD",
                             vect_column = "ALIGNED_DISP_VECT")

        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")

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
    
    def show_vectors(df, TP, data = None, mode = "Default",
                     show = True, savepath = None,
                     show_centroid = True, color_clusters = True, 
                     show_rot_axis = True):
        
        ## Setting which time points to display.
        if not isinstance(TP, int) :
            raise TypeError("TP must be int")
        
            
        if mode == "Default":
            coord_column = "COORD"
            vect_column = "DISP_VECT"
            RA_column = "RA_VECT"
            
        if mode == "Translated":
            coord_column = "TRANS_COORD"
            vect_column = "DISP_VECT"
            RA_column = "RA_VECT"
            
        if mode == "Aligned":
            coord_column = "ALIGNED_COORD"
            vect_column = "ALIGNED_DISP_VECT"
            RA_column = "ALIGNED_RA_VECT"
        
        ## Checking if we can actually show the clusters.
        if color_clusters and "F_SELECT" not in df.columns:
            color_clusters = False
            
        # ## Same with the centroids.
        # if show_centroid and (data is None or "Raw_centroid" not in data.columns
        #                       or (color_clusters and "Cluster_centroid" not
        #                           in data.columns)):
        #     show_centroid = False
            
        ## Same with the rotation axis.
        if show_rot_axis and (data is None or RA_column not in data.columns):
            show_rot_axis = False
            
        ## Creating the figure and setting global parameters.
        fig = plt.figure(figsize=(10, 7), dpi = 400)
        plt.style.use("seaborn-paper")
        plt.rcParams.update({'font.family':'Montserrat'})
        
        legend = []
        
        subdf = df[df["TP"] == TP].copy()
        
        ax = fig.add_subplot(111, projection = "3d")
        
        ## Showing displacements vectors xith or without cluster coloration.
        if color_clusters:
            ax, nlegend = figures.add_3Dvectors(ax, subdf[subdf["F_SELECT"]],
                                                coord_column, vect_column,
                                                label = "Selected displacement vectors",
                                                color = "green")
            legend.append(nlegend)
            
            ax, nlegend = figures.add_3Dvectors(ax, 
                                                subdf[subdf["F_SELECT"] == False],
                                                coord_column, vect_column,
                                                label = "Unselected displacement vectors",
                                                color = "orange")
            legend.append(nlegend)
            
        else :
            ax, nlegend = figures.add_3Dvectors(ax, subdf, coord_column,
                                                vect_column, 
                                                label = "Displacement vectors")
            legend.append(nlegend)
            
        ## Showing centroid
        if show_centroid :
            if color_clusters:
                ax, nlegend = figures.add_spots(ax, subdf[subdf["F_SELECT"]], 
                                                coord_column, 
                                                label = "Cluster centroid")
                legend.append(nlegend)
            
            ax, nlegend = figures.add_spots(ax, subdf, 
                                            coord_column, 
                                            label = "Raw centroid", 
                                            color = "red")
            legend.append(nlegend)
            
        ## Showing rotation axis.
        if show_rot_axis:
            if color_clusters:
                centroid = tools.get_centroid(subdf[subdf["F_SELECT"]], 
                                              coord_column)
            else:
                centroid = tools.get_centroid(subdf, coord_column)
                
            RA_vect = data.loc[TP, RA_column]/np.linalg.norm(
                                                data.loc[TP, RA_column])
            
            ax.quiver(centroid[0], centroid[1], centroid[2],
                      RA_vect[0], RA_vect[1], RA_vect[2],
                      color = "dodgerblue", length = 20)
            ax.quiver(centroid[0], centroid[1], centroid[2],
                      -RA_vect[0], -RA_vect[1], -RA_vect[2],
                      color = "dodgerblue", length = 20)
            
            legend.append(Line2D([0], [0], marker = "", color = "dodgerblue", 
                                 label = "Rotation Axis", 
                                 markerfacecolor = "dodgerblue", 
                                 markersize = 7))
        
        ax.legend(handles = legend, loc = 'best')
        
        ax.set_title("Displacement vectors for time point "+str(TP), 
                     fontsize = 15)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        plt.show()
        plt.close()

        
    def show_angular_velocity(df, data, show = True, savepath = None):
        
        if not set(["Mean_AV", "Std_AV", "volume", "Drift_distance"]).issubset(
                set(data.columns)) or not "ALIGNED_COORD" in df.columns:
            raise KeyError("Missing informations to proceed")
            
        fig, axs = plt.subplots(2, 2, figsize = (16, 9), dpi = 400)
        
        ## Plotting AV over time.
        sns.lineplot(ax = axs[0, 0], data = df, x = "TP", y = "ANG_VELOCITY",
                     ci='sd', err_style='bars', marker = "o")
        sns.despine()
        axs[0, 0].set(xlabel = 'Time point', ylabel = 'Angular velocity',
                      title = "Cell angular velocity over time")
        
        ## Plotting volume over time.
        sns.lineplot(ax = axs[1, 0], x = data.index, y = data["volume"], 
                     marker = "o")
        sns.despine()
        axs[1, 0].set(xlabel = 'Time point', ylabel = 'Volume (px³)',
                      title = 'Organoïd volume over time')
        
        ## Plotting angular velocity over distance to rotation axis.
        distance = [np.linalg.norm(arr[:2]) for arr in df["ALIGNED_COORD"]]
        hue = (df["ANG_VELOCITY"]/(df["ANG_VELOCITY"].mean()))
        hue.name = 'Relative to the average angular velocity'
        
        sns.scatterplot(ax = axs[0, 1], x = distance, y = df["ANG_VELOCITY"],
                        hue = hue)
        sns.despine()
        axs[0, 1].set(xlabel = 'Distance to rotation axis (px)', 
                      ylabel = 'Angular velocity',
                      title = 'Angular velocity relative to the distance from the rotation axis')
        
        ## Plotting drift over time.
        y_pt = [data.iloc[:k+1]["Drift_distance"].sum() for k in data.index]
        
        sns.lineplot(ax = axs[1, 1], x = data.index, y = y_pt, marker = "o")
        sns.despine()
        axs[1, 1].set(xlabel = 'Time point', ylabel = 'Travelled distance (px)',
                      title = 'Sum of the travelled distance over time')
        
        if savepath is not None:
            plt.savefig(savepath)
        
        if show:
            plt.show()

        plt.close()
        
        # ## Plotting mean velocity for each point.
        # subdf = df.loc[[k for k in df.index 
        #                if df.loc[k, "ALIGNED_COORD"][-1] >= 0]]
        # x_pt = np.array(subdf["ALIGNED_COORD"].tolist())[:, 0]
        # y_pt = np.array(subdf["ALIGNED_COORD"].tolist())[:, 1]
        
        # sns.scatterplot(ax = axs[0, 1], x = x_pt, y = y_pt, 
        #                 hue = subdf["ANG_VELOCITY"])
        # axs[0, 1].set(xlabel = 'x', ylabel = 'y')
        