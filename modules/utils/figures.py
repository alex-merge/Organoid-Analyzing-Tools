# -*- coding: utf-8 -*-
"""
Plotting figures methods for OAT.

@author: Alex-932
@version: 0.7
"""
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns

from modules.utils.tools import *

class figures():
    
    def add_centroid(ax, df, coord_column = "COORD", label = None,
                     marker = "o", color = "navy", size = 50):
        
        cx, cy, cz = tools.get_centroid(df, coord_column)
        
        ax.scatter(cx, cy, cz, c = color, marker = marker, s = size)
        
        if label is not None :
            legend = Line2D([0], [0], marker = marker, color = color, 
                            label = label, markerfacecolor = color, 
                            markersize = 7, ls = '')
            
            return ax, legend
        
        return ax
    
    def add_3Dvectors(ax, df, coord_column = "COORD", 
                      vect_column = "DISP_VECT", label = None, marker = ">", 
                      color = "black", length = None, c_column = None):
        
        coords = np.array(df[coord_column].tolist())
        vectors = np.array(df[vect_column].tolist())
        
        if length is not None :
            if c_column is None :
                ax.quiver(coords[:, 0], coords[:, 1], coords[:, 2],
                          vectors[:, 0], vectors[:, 1], vectors[:, 2],
                          color = color, length = length)
            else :
                ax.quiver(coords[:, 0], coords[:, 1], coords[:, 2],
                          vectors[:, 0], vectors[:, 1], vectors[:, 2],
                          colors = plt.cm.tab10(df[c_column]), 
                          length = length)
        else :
            if c_column is None :
                ax.quiver(coords[:, 0], coords[:, 1], coords[:, 2],
                          vectors[:, 0], vectors[:, 1], vectors[:, 2],
                          color = color)
            else :
                ax.quiver(coords[:, 0], coords[:, 1], coords[:, 2],
                          vectors[:, 0], vectors[:, 1], vectors[:, 2],
                          colors = plt.cm.tab10(df[c_column]))
            
        if label is not None :
            legend = Line2D([0], [0], marker = marker, color = color, 
                            label = label, markerfacecolor = color, 
                            markersize = 7, ls = '')
            return ax, legend
        
        return ax
    
    def add_spots(ax, df, coord_column, label = None, marker = "o", 
                  color = "navy", size = 20):
        
        coords = np.array(df[coord_column].tolist())
        
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c = color, 
                   s = size)
        
        if label is not None :
            legend = Line2D([0], [0], marker = marker, color = color, 
                            label = label, markerfacecolor = color, 
                            markersize = 7, ls = '')
            return ax, legend
        
        return ax
        
    def show_clustering_centroids(df, TP, show = True, savepath = None):
        
        ## Checking TP
        if not isinstance(TP, int) :
            raise TypeError("TP must be int")
        
        fig, axs = plt.subplots(1, 3, figsize = (16, 7), dpi = 400)
        plt.style.use("seaborn-paper")
        plt.rcParams.update({'font.family':'Montserrat'})
        
        subdf = df[df["TP"] == TP]
        arr = np.array(subdf["COORD"].tolist())
        
        ## Creating a list to hold which axis to draw at each frame.
        axis_order = [[0, 1], [0, 2], [1, 2]]
        label_dict = {0: "x", 1: "y", 2: "z"}
        
        for col in range(3):
            ## Plotting the centroids.
            axs[col].scatter(arr[:, axis_order[col][0] ], 
                             arr[:, axis_order[col][1] ],
                             c = "navy")
            
            ## Plotting the median centroid.
            axs[col].scatter(np.median( arr[:, axis_order[col][0] ] ),
                             np.median( arr[:, axis_order[col][1] ] ),
                             c = "red", marker = "+")
            
            axs[col].set_xlabel(label_dict[ axis_order[col][0] ])
            axs[col].set_ylabel(label_dict[ axis_order[col][1] ])
            
        ## Creating the legend.
        legend = [Line2D([0], [0], marker = "o", color = "red", 
                         label = "Median centroid", markerfacecolor = "red", 
                         markersize = 7, ls = '')]
        
        ## Adding the legend and title to the figure.
        plt.legend(handles = legend, loc = 'best')
        fig.suptitle("Centroids computed from subsampling the dataset")
        
        if savepath is not None:
            plt.savefig(savepath, dpi = 400)
        
        if show:
            plt.show()
            
        plt.close()
        
    def show_clustering_distance(df, TP = "all", bins = 30, show = True, 
                                 savepath = None, cmap = 'tab10'):
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
        ## Setting the cmap.
        cmap = plt.cm.get_cmap(cmap)
        
        ## Checking TP
        if not isinstance(TP, int) :
            raise TypeError("TP must be int")
        
        ## Retrieving the data we will need.
        data = df[df["TP"] == TP].copy()
        
        fig, ax = plt.subplots(figsize = (10, 5), dpi = 400)
        plt.style.use("seaborn-paper")
        plt.rcParams.update({'font.family':'Montserrat'})
        
        legend = []
        
        ## Showing cluster data if available.
        if "A_CLUSTER" in data.columns :
            ## Setting the colors.
            data["Color"] = (data["A_CLUSTER"]+1).map(cmap)

            ## Iterating over clusters ID.
            for cluster in data["A_CLUSTER"].unique():
                
                ## Getting the rows for 1 cluster.
                subdata = data[data["A_CLUSTER"] == cluster]

                ## Plotting the histogram with colors.
                ax.hist(subdata["DISTANCE_CENTROID"], 
                        color = subdata["Color"][0], 
                        bins = bins, edgecolor = "white")
                
                ## Adding the legend.
                is_selected = bool(subdata.at[subdata.index[0], "A_SELECT"])
                label = "Cluster ID: "+str(cluster)+", "+\
                    (is_selected)*"selected"+(not is_selected)*"unselected" 
                    
                legend.append(Patch(facecolor = subdata["Color"][0], 
                                    edgecolor = subdata["Color"][0],
                                    label= label))
        else :
            ## Plotting the histogram without color presets.
            ax.hist(data["DISTANCE_CENTROID"], 
                    bins = bins, edgecolor = "white")
        
        ## Labelling axis and the figure. Drawing legends
        ax.set_xlabel("Distance (px)")
        ax.set_ylabel("Number of spots")
        fig.suptitle("Spots relative to the distance from the centroid")
        plt.legend(handles = legend, loc = 'best')
        
        if show :
            plt.show()
        if not savepath is None :
            plt.savefig(savepath, dpi = 400)
        
        plt.close()
            
    def show_spots(df, TP, coord_column = "COORD", color = "navy", show = True, 
                   savepath = None, show_centroids = True, 
                   color_clusters = True):
        """
        Create a figure showing spots for a given time point.
        The figure can be saved. See parameters below.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing coordinates and time points (X, Y, Z, TP).
        TP : int
            Time point to show the spots.
        show : bool, optional
            Show the figure. The default is True.
        save : bool, optional
            Save the figure, need savepath to work. The default is False.
        savepath : str, optional
            Full path of the output figure. The default is None.
        show_centroids : bool, optional
            Add the centroid to the figure. The default is True.
        color_clusters : bool, optional
            Add cluster information to the figure. The default is True.

        """
        
        ## Checking TP
        if not isinstance(TP, int) :
            raise TypeError("TP must be int")
            
        ## Creating the figure.
        fig = plt.figure(figsize=(10, 7), dpi = 400)
        plt.style.use("seaborn-paper")
        plt.rcParams.update({'font.family':'Montserrat'})
        
        legend = []
            
        subdf = df[df["TP"] == TP].copy()
        
        ax = fig.add_subplot(111, projection = "3d")
        
        if color_clusters and "F_SELECT" in subdf.columns:
            t_df = subdf[subdf["F_SELECT"]]
            
            ax, nlegend = figures.add_spots(ax, 
                                            subdf[subdf["F_SELECT"]], 
                                            coord_column, 
                                            label = "Selected spots", 
                                            color = "green",
                                            size = 50)
            legend.append(nlegend)
            
            ax, nlegend = figures.add_spots(ax, 
                                            subdf[subdf["F_SELECT"] == False], 
                                            coord_column, 
                                            label = "Unselected spots", 
                                            color = "orange",
                                            size = 20)
            legend.append(nlegend)
            
        else :
            ax = figures.add_spots(ax, subdf, coord_column, color = color)
        
        if show_centroids :
            
            ax, nlegend = figures.add_centroid(ax, subdf, coord_column, 
                                               label = "Raw centroid",
                                               marker = "^",
                                               color = "red",
                                               size = 50)
            legend.append(nlegend)
            
            if color_clusters and "F_SELECT" in subdf.columns:
                ax, nlegend = figures.add_centroid(ax, 
                                                   subdf[subdf["F_SELECT"]], 
                                                   coord_column, 
                                                   label = "Cluster centroid",
                                                   marker = "^",
                                                   color = "navy",
                                                   size = 50)
                legend.append(nlegend)
        
        ax.set_title("Detected spots for time point #"+str(TP), fontsize = 15)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        ax.legend(handles = legend, loc = 'best')
        
        if show :
            plt.show()
            
        if savepath is not None :
            plt.savefig(savepath, dpi = 400)
            
        plt.close()
        
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
                ax, nlegend = figures.add_centroid(ax, subdf[subdf["F_SELECT"]], 
                                                   coord_column, 
                                                   label = "Cluster centroid")
                legend.append(nlegend)
            
            ax, nlegend = figures.add_centroid(ax, subdf, 
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
            # ax.quiver(centroid[0], centroid[1], centroid[2],
            #           -RA_vect[0], -RA_vect[1], -RA_vect[2],
            #           color = "dodgerblue", length = 20)
            
            legend.append(Line2D([0], [0], marker = "", color = "dodgerblue", 
                                 label = "Rotation Axis", 
                                 markerfacecolor = "dodgerblue", 
                                 markersize = 7))
        
        ax.legend(handles = legend, loc = 'best')
        
        ax.set_title("Displacement vectors for time point #"+str(TP), 
                     fontsize = 15)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        fig.tight_layout()
        
        if not savepath is None:
            plt.savefig(savepath)
        
        plt.show()
        #plt.close()

        
    def show_angular_velocity(df, data, show = True, savepath = None):
        
        if not set(["Mean_AV", "Std_AV", "volume", "Drift_distance"]).issubset(
                set(data.columns)) or not "ALIGNED_COORD" in df.columns:
            raise KeyError("Missing informations to proceed")
            
        plt.style.use("seaborn-paper")
        plt.rcParams.update({'font.family':'Montserrat'})
            
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
        
        fig.tight_layout()
        
        if savepath is not None:
            plt.savefig(savepath)
        
        if show:
            plt.show()

        plt.close()
        
    def show_rotation_axis(data):
        fig = plt.figure(figsize = (16, 9), dpi = 400)
        ax = fig.add_subplot(111, projection = "3d")
        
        plt.style.use("seaborn-paper")
        plt.rcParams.update({'font.family':'Montserrat'})
        
        RA = [k/(np.linalg.norm(k)) for k in 
              data["RA_VECT"] if isinstance(k, np.ndarray)]
        arr = np.array(RA)
        origin = [0]*len(RA)
        
        ax.quiver(origin, origin, origin, arr[:, 0], arr[:, 1], arr[:, 2], 
                  color = "navy")
        
        ax.set_title("Rotation axis vectors", 
                      fontsize = 15)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        fig.tight_layout()
        
        plt.show()
        plt.close()
        
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
