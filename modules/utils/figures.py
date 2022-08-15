# -*- coding: utf-8 -*-
"""
Plotting figures methods for OAT.

@author: Alex-932
@version: 0.7
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns

from modules.utils.tools import *

class figures():
    
    def show_data(df, TP, data = None, mode = "default",
                  show = True, savepath = None, close = True,
                  show_centroid = True, color_clusters = True, 
                  show_rot_axis = True, show_vectors = True):
        """
        Generate a 3D figure that can show the spots, their displacement 
        vectors, the centroid and the rotation axis. 
        The figure can also show the clustering results by setting colors to
        clusters.
        It comes with a several on/off switches to easily get what you want.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe which contain the following columns :
                - COORD
                - TP
            Those columns are optional :
            ALIGNED_COORD, CENTRD_COORD, ALIGNED_DISP_VECT, F_SELECT.
        TP : int
            Time point.
        data : pandas.DataFrame, optinal
            Dataframe with the following optional columns :
            RA_VECT, ALIGNED_RA_VECT, CENTROID, CLUST_CENTROID
        mode : str, optional
            Group of data to show. 
            Options are :
                - "default" (by default): raw coordinates and displacement 
                vectors.
                - "centered" : centered coordinates, raw displacement vectors.
                - "aligned" : aligned data (coordiates, vectors).
        show : bool, optional
            If True, show the figure. The default is True.
        savepath : str, optional
            Fullpath of the figure. The default is None.
        close : bool, optional
            If True, close the figure. If False, allow interactions.
            The default is True.
        show_centroid : bool, optional
            If True, try to show the centroid. The default is True.
        color_clusters : bool, optional
            If True, try to color data according to clusters selection. 
            The default is True.
        show_rot_axis : bool, optional
            If True, try to draw the rotation axis. The default is True.
        show_vectors : bool, optional
            If True, try to draw displacement vectors. The default is True.

        """
        modes = {"default": ["COORD", "DISP_VECT", "RA_VECT"],
                 "centered": ["CENTRD_COORD", "DISP_VECT", "RA_VECT"],
                 "aligned": ["ALIGNED_COORD", "ALIGNED_DISP_VECT", "ALIGNED_RA_VECT"]}
        
        ## Checking if all data are available
        if not set(modes[mode][:2]).issubset(set(df.columns)) :
            raise KeyError("Missing informations to proceed")
        if not isinstance(TP, int) or TP not in df["TP"].unique() :
            raise TypeError("Invalid time point : must be int and must exist")
        if color_clusters and "F_SELECT" not in df.columns:
            print("Warning : No cluster data found, proceeding without them")
            color_clusters = False
        if show_vectors and modes[mode][1] not in df.columns:
            print("Warning : No vectors data found, proceeding without them")
            show_vectors = False
        if show_rot_axis and (not show_vectors or mode != "aligned" or \
            data is None or modes[mode][2] not in data.columns):
            print("Warning : No rotation axis data found, proceeding without them")
            show_rot_axis = False
        if show_centroid and (data is None or ("CENTROID" not in data.columns and
                              "CLUST_CENTROID" not in data.columns)):
            print("Warning : No centroïd data found, proceeding without them")
            show_centroid = False 
            
        ## Creating the figure and setting some theme and font for it
        plt.style.use("seaborn-paper"); plt.rcParams.update({'font.family':'Montserrat'})
        fig, legend = plt.figure(figsize = (10, 7), dpi = 400), []
        
        ax = fig.add_subplot(111, projection = "3d")
        ax.set(xlabel = "x", ylabel = "y", zlabel = "z",
               title = "Displacement vectors for time point #"+str(TP))
        
        subdf = df[df["TP"] == TP].copy()
        coords = np.array( subdf[ modes[mode][0] ].tolist() )
        vect = np.array( subdf[ modes[mode][1] ].tolist() )
        
        ## Setting the colors and legend in case clusters are shown
        if color_clusters:
            colors = pd.Series(["green"*val+"orange"*(val == 0) for val in subdf["F_SELECT"]],
                               index = subdf.index)
            legend.append(Line2D([0], [0], marker = ">", color = "green", 
                                 label = "Selected vectors", ls = ''))
            legend.append(Line2D([0], [0], marker = ">", color = "orange", 
                                 label = "Unselected vectors", ls = ''))
        else :
            colors = "navy"
        
        ## Plotting spots
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color = colors)   
         
        ## Plotting displacement vectors
        if show_vectors :
            ax.quiver(coords[:, 0], coords[:, 1], coords[:, 2],
                      vect[:, 0], vect[:, 1], vect[:, 2],
                      color = colors)
      
        ## Adding the centroid if wanted
        if show_centroid:
            try :
                centroid = data.loc[TP, "CENTROID"]
                ax.scatter(centroid[0], centroid[1], centroid[2], color = "red",
                           marker = "^")
                legend.append(Line2D([0], [0], marker = "^", color = "red", 
                                     label = "Centroid", ls = ''))
            except :
                None
            try :
                centroid = data.loc[TP, "CLUST_CENTROID"]
                ax.scatter(centroid[0], centroid[1], centroid[2], color = "green",
                           marker = "^")
                legend.append(Line2D([0], [0], marker = "^", color = "green", 
                                     label = "CLuster's centroid", ls = ''))
            except :
                None
                
            ## Showing rotation axis
            if show_rot_axis:
                RA_vect = data.loc[TP, modes[mode][2]]
                ax.quiver(centroid[0], centroid[1], centroid[2],
                          RA_vect[0], RA_vect[1], RA_vect[2],
                          color = "dodgerblue", length = 20, pivot = "middle")
                legend.append(Line2D([0], [0], marker = ">", color = "dodgerblue", 
                                     label = "Rotation axis", ls = ''))
        
        ax.legend(handles = legend, loc = 'best')
        fig.tight_layout()
        
        if not savepath is None:
            plt.savefig(savepath)
        if show:
            plt.show()
        if close:
            plt.close()

        
    def show_angular_velocity(df, data, show = True, savepath = None):
        """
        Produce a result figure showing informations relative to the angular 
        velocity over available time points.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe which contain the following columns :
                - ALIGNED_COORD
                - AV_RAD
                - TP
        data : pandas.DataFrame
            Dataframe which contains the following column :
                - DRIFT
                - VOLUME
        show : bool, optional
            If True, show the figure. The default is True.
        savepath : str, optional
            If provided, save the figure in the fullpath. The default is None.

        """
        ## Checking if all informations are available
        if not set(["VOLUME", "DRIFT"]).issubset(set(data.columns)) \
            or not set(["ALIGNED_COORD", "TP", "AV_RAD"]).issubset(set(df.columns)):
            raise KeyError("Missing informations to proceed")
            
        ## Creating the figure and setting some theme and font for it
        plt.style.use("seaborn-paper"); plt.rcParams.update({'font.family':'Montserrat'})
        fig, axs = plt.subplots(2, 2, figsize = (16, 9), dpi = 400)
        
        ## Plotting angular velocity over time
        sns.lineplot(ax = axs[0, 0], data = df, x = "TP", y = "AV_RAD",
                     ci='sd', err_style='bars', marker = "o")
        sns.despine()
        axs[0, 0].set(xlabel = 'Time point', ylabel = 'Angular velocity',
                      title = "Cell angular velocity over time")
        
        ## Plotting volume over time
        volume_evol = [(data["VOLUME"][k+1]-data["VOLUME"][k])/data["VOLUME"][k]*100
                       for k in range(len(data.index)-1)]
        sns.lineplot(ax = axs[1, 0], x = data.index[:-1], y = volume_evol, marker = "o")
        sns.despine()
        axs[1, 0].set(xlabel = 'Time point', ylabel = 'Volume evolution (%)',
                      title = 'Organoïd volume evolution between time points')
        
        ## Plotting angular velocity over distance to rotation axis
        distance = [np.linalg.norm(arr[:2]) for arr in df["ALIGNED_COORD"]]
        hue = (df["AV_RAD"]/df["AV_RAD"].mean())*100
        hue.name = 'Relative to the average angular velocity'
        
        sns.scatterplot(ax = axs[0, 1], x = distance, y = df["AV_RAD"], hue = hue)
        sns.despine()
        axs[0, 1].set(xlabel = 'Distance to rotation axis (px)', 
                      ylabel = 'Angular velocity',
                      title = 'Angular velocity relative to the distance from the rotation axis')
        
        ## Plotting drift evolution between time points
        drift_evol = [(data["DRIFT"][k+1]-data["DRIFT"][k])/data["DRIFT"][k]*100
                       for k in range(len(data.index)-1)]
        sns.lineplot(ax = axs[1, 1], x = data.index[:-1], y = drift_evol, marker = "o")
        sns.despine()
        axs[1, 1].set(xlabel = 'Time point', ylabel = 'Travelled distance evolution (%)',
                      title = 'Travelled distance evolution between time points')
        
        fig.tight_layout()
        
        if savepath is not None:
            plt.savefig(savepath)
        if show:
            plt.show()

        plt.close()
        
        
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
        
    def show_angular_velocity_by_cell(df, TP, threshold = 15, data = None, 
                                      show = True, savepath = None):
        plt.style.use("seaborn-paper")
        plt.rcParams.update({'font.family':'Montserrat'})
        
        cmap = plt.cm.get_cmap("viridis")
        
        arr = np.array(df[df["TP"] == TP]["ALIGNED_COORD"].tolist())
        print(arr)
        if np.isnan(arr).any() :
            raise ValueError("Unable to show as there are NaN values")
        
        fig, axs = plt.subplots(1, figsize = (16, 9), dpi = 400)
        
        color = ["red"*(k < threshold)+"green"*(k >= threshold) for k in 
                 df[df["TP"] == TP]["ANG_VELOCITY_DEG"]]
        
        axs.scatter(arr[:, 0], arr[:, 1], c = color)
        
        axs.set_xlabel("x")
        axs.set_ylabel("y")
        
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
