# -*- coding: utf-8 -*-
"""
Figures drawing methods for OAT.

@author: alex-merge
@version: 0.8
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns



class figures():
    """
    Set of methods to create figures from OAT results.
    """
    
    def show_data(df, TP, data = None, mode = "default",
                  viewing_angles = (30, 60),
                  show = True, savepath = None, close = True,
                  show_centroid = True, color_clusters = True, 
                  show_rot_axis = True, show_vectors = True,
                  show_av = True):
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
        viewing_angles : tuple, optional
            Viewing angles for the 3D graph (elevation, azimuth).
            The default is (30, 60).
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
        modes = {"default": ["COORD", "DISP_VECT", "RA_VECT", "CENTROID"],
                 "centered": ["CENTRD_COORD", "DISP_VECT", "RA_VECT", "CENTRD_CENTROID"],
                 "aligned": ["ALIGNED_COORD", "ALIGNED_DISP_VECT", "ALIGNED_RA_VECT", "CENTRD_CENTROID"]}
        
        ## Checking if all data are available
        if not set(modes[mode][:2]).issubset(set(df.columns)) :
            raise KeyError("Missing informations to proceed")
        if not isinstance(TP, int) or TP not in df["TP"].unique() :
            raise TypeError("Invalid time point : must be int and must exist")
        if color_clusters and "CLUSTER_SELECT" not in df.columns:
            print("Warning : No cluster data found, proceeding without them")
            color_clusters = False
        if show_vectors and modes[mode][1] not in df.columns:
            print("Warning : No vectors data found, proceeding without them")
            show_vectors = False
        if show_rot_axis and (not show_vectors or \
            data is None or modes[mode][2] not in data.columns):
            print("Warning : No rotation axis data found, proceeding without them")
            show_rot_axis = False
        if show_centroid and (data is None or ("CENTROID" not in data.columns and
                              "CLUST_CENTROID" not in data.columns)):
            print("Warning : No centroïd data found, proceeding without them")
            show_centroid = False 
            
        ## Creating the figure and setting some theme and font for it
        plt.style.use("seaborn-paper"); plt.rcParams.update({'font.family':'Montserrat'})
        fig, legend = plt.figure(figsize = (10, 7), dpi = 400, facecolor = "white"), []
        
        ax = fig.add_subplot(111, projection = "3d")
        ax.set(xlabel = "x", ylabel = "y", zlabel = "z",
               title = "Displacement vectors for time point #"+str(TP))
        
        subdf = df[df["TP"] == TP].copy()
        coords = np.array( subdf[ modes[mode][0] ].tolist() )
        vect = np.array( subdf[ modes[mode][1] ].tolist() )
        
        ## Setting the colors and legend in case clusters are shown
        if color_clusters:
            colors = pd.Series(["green"*val+"orange"*(val == 0) for val in subdf["CLUSTER_SELECT"]],
                               index = subdf.index)
            # legend.append(Line2D([0], [0], marker = ">", color = "green", 
            #                      label = "Selected vectors", ls = ''))
            # legend.append(Line2D([0], [0], marker = ">", color = "orange", 
            #                      label = "Unselected vectors", ls = ''))
        else :
            colors = "navy"
        
        ## Plotting spots
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color = colors,
                   label = "Spots")   
         
        ## Plotting displacement vectors
        if show_vectors :
            if isinstance(colors, pd.Series):
                ax.quiver(coords[:, 0], coords[:, 1], coords[:, 2],
                          vect[:, 0], vect[:, 1], vect[:, 2],
                          color = colors, 
                          label = ["Selected", "Unselected"])
            elif isinstance(colors, str):
                ax.quiver(coords[:, 0], coords[:, 1], coords[:, 2],
                          vect[:, 0], vect[:, 1], vect[:, 2],
                          color = colors, 
                          label = "Displacement vectors")
      
        ## Adding the centroid if wanted
        if show_centroid:
            try :
                centroid = data.loc[TP, "CLUST_CENTROID"]
                ax.scatter(centroid[0], centroid[1], centroid[2], color = "green",
                           marker = "^", label = "Cluster's centroid")
            except :
                None
            try :
                centroid = data.loc[TP, modes[mode][3]]
                ax.scatter(centroid[0], centroid[1], centroid[2], color = "red",
                           marker = "^", label = "Centroid")
                
            except :
                None
                 
            ## Showing rotation axis
            if show_rot_axis:
                RA_vect = data.loc[TP, modes[mode][2]]
                ax.quiver(centroid[0], centroid[1], centroid[2],
                          RA_vect[0], RA_vect[1], RA_vect[2],
                          color = "dodgerblue", length = 20, pivot = "middle",
                          label = "Rotation axis")
        
        ## Setting the viewing angles
        elev, azimuth = viewing_angles
        ax.view_init(elev, azimuth)
        
        ax.legend()
        fig.tight_layout()
        
        if not savepath is None:
            plt.savefig(savepath)
        if show:
            plt.show()
        if close:
            plt.close()
            
    def show_data_2D(df, TP, data = None, axis = None, mode = "default",
                     show = True, savepath = None, close = True,
                     show_centroid = True, color_clusters = True, 
                     show_rot_axis = True, show_vectors = True,
                     show_av = True):
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
        modes = {"default": ["COORD", "DISP_VECT", "RA_VECT", "CENTROID"],
                 "centered": ["CENTRD_COORD", "DISP_VECT", "RA_VECT", "CENTRD_CENTROID"],
                 "aligned": ["ALIGNED_COORD", "ALIGNED_DISP_VECT", "ALIGNED_RA_VECT", "CENTRD_CENTROID"]}
        
        ## Checking if all data are available
        if not set(modes[mode][:2]).issubset(set(df.columns)) :
            raise KeyError("Missing informations to proceed")
        if not isinstance(TP, int) or TP not in df["TP"].unique() :
            raise TypeError("Invalid time point : must be int and must exist")
        if color_clusters and "CLUSTER_SELECT" not in df.columns:
            print("Warning : No cluster data found, proceeding without them")
            color_clusters = False
        if show_vectors and modes[mode][1] not in df.columns:
            print("Warning : No vectors data found, proceeding without them")
            show_vectors = False
        if show_rot_axis and (not show_vectors or \
            data is None or modes[mode][2] not in data.columns):
            print("Warning : No rotation axis data found, proceeding without them")
            show_rot_axis = False
        if show_centroid and (data is None or ("CENTROID" not in data.columns and
                              "CLUST_CENTROID" not in data.columns)):
            print("Warning : No centroïd data found, proceeding without them")
            show_centroid = False 
        if axis is None or len(axis) != 2 or len([k for k in list(axis.upper())
            if k in ["X", "Y", "Z"]]):
            raise ValueError("Axis must be 'XY', 'XZ' or 'YZ'")
            
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
            colors = pd.Series(["green"*val+"orange"*(val == 0) for val in subdf["CLUSTER_SELECT"]],
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
                centroid = data.loc[TP, "CLUST_CENTROID"]
                ax.scatter(centroid[0], centroid[1], centroid[2], color = "green",
                           marker = "^")
                legend.append(Line2D([0], [0], marker = "^", color = "green", 
                                     label = "Cluster's centroid", ls = ''))
            except :
                None
            try :
                centroid = data.loc[TP, modes[mode][3]]
                ax.scatter(centroid[0], centroid[1], centroid[2], color = "red",
                           marker = "^")
                legend.append(Line2D([0], [0], marker = "^", color = "red", 
                                     label = "Centroid", ls = ''))
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
        fig, axs = plt.subplots(2, 2, figsize = (16, 9), dpi = 400, facecolor = "white")
        
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
        drift_evol = []
        for k in range(len(data.index)-1):
            if data["DRIFT"][k] != 0 :
                drift_evol.append(
                    (data["DRIFT"][k+1]-data["DRIFT"][k])/data["DRIFT"][k]*100
                    )
            
            else : 
                drift_evol.append(0)
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
        
    def SSA(df, TP, variable_column = "AV_RAD", groups = None, window_size = None, 
            figtitle = None, show = True, savepath = None):
        """
        Produce a Singular Spectrum Analysis on the wanted variable using
        the implementation of it in pyts.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the given variable column.
        TP : int
            Time point .
        variable_column : TYPE, optional
            DESCRIPTION. The default is "AV_RAD".
        groups : TYPE, optional
            DESCRIPTION. The default is None.
        window_size : TYPE, optional
            DESCRIPTION. The default is None.
        figtitle : TYPE, optional
            DESCRIPTION. The default is None.
        show : TYPE, optional
            DESCRIPTION. The default is True.
        savepath : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        ## Putting the import here that way, it is not strictly required
        from pyts.decomposition import SingularSpectrumAnalysis
        
        ## Creating a dataframe to store the information the way SSA wants them
        ssa_df = pd.DataFrame(dtype = "float")
        
        time_points = df["TP"].unique().tolist()
        time_points.sort()
        
        ## Iterating over time points
        for tp in time_points:
            series = df[df["TP"] == tp][variable_column]
            series.name = tp
            series.index = list(range(len(series.index)))
            
            ## Checking if the series only contains nan values
            if not series.dropna().empty :
                ssa_df = pd.concat([ssa_df, series], axis = 1)
                
        arr = ssa_df.to_numpy()
        
        if window_size is None:
            window_size = int(arr.shape[1]/2)
        
        SSA = SingularSpectrumAnalysis(window_size, groups)
        res = SSA.fit_transform(arr)
        
        ## Creating the figure and setting some theme and font for it
        plt.style.use("seaborn-paper"); plt.rcParams.update({'font.family':'Montserrat'})
        fig, axs = plt.subplots(1, 2, figsize = (16, 9), dpi = 400)

        ## Plotting original data
        axs[0].plot(arr[0], 'o-', label='Original')
        axs[0].legend(loc='best', fontsize=14)
        
        ## Plotting SSA results
        for i in range(res.shape[1]):
            axs[1].plot(res[0, i], 'o--', label='SSA {0}'.format(i + 1))
        axs[1].legend(loc='best', fontsize=14)
        
        fig.suptitle(figtitle, fontsize=20)
        
        plt.tight_layout()
        #plt.subplots_adjust(top=0.88)
        
        if savepath is not None:
            plt.savefig(savepath)
        if show:
            plt.show()
        
    def show_angular_velocity_by_cell(df, TP, threshold = 15, data = None, 
                                      show = True, savepath = None):
        
        plt.style.use("seaborn-paper"); plt.rcParams.update({'font.family':'Montserrat'})
        
        cmap = plt.cm.get_cmap("viridis")
        
        arr = np.array(df[df["TP"] == TP]["ALIGNED_COORD"].tolist())
        print(arr)
        if np.isnan(arr).any() :
            raise ValueError("Unable to show as there are NaN values")
        
        fig, axs = plt.subplots(1, figsize = (16, 9), dpi = 400, facecolor = "white")
        
        color = ["red"*(k < threshold)+"green"*(k >= threshold) for k in 
                 df[df["TP"] == TP]["AV_RAD"]]
        
        axs.scatter(arr[:, 0], arr[:, 1], c = color)
        
        axs.set_xlabel("x")
        axs.set_ylabel("y")
        
        plt.show()
        plt.close()

    def create_timelapse(df, data, savepath, fps = 1, mode = "default"):
        """
        Generate an animation from the figures generated with 
        figures.show_data() method.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing spots informations.
        data : pandas.DataFrame
            Dataframe containing time points informations.
        savepath : str
            Folder path to the animation figures and file.
        fps : int, optional
            FPS number for the animation. The default is 1.
        mode : str, optional
            Similar modes as show_data():
                - default
                - centered
                - aligned
            The default is "default".

        """
        ## Importing openCV
        import cv2
        
        TP = df["TP"].unique().tolist()
        TP.sort()
        filepaths, img_list = [], []
        
        animation = cv2.VideoWriter(savepath+"\\Animation.avi", 0, 
                                    cv2.VideoWriter_fourcc(*'DIVX'), fps = fps,
                                    frameSize = (4000, 2800))
        
        for tp in TP[:-1]:
            filepaths.append(savepath+"\\fig_TP{}.png".format(tp))
            figures.show_data(df, tp, data, mode = mode, show = False, 
                              savepath = filepaths[-1])
            
            img = cv2.imread(filepaths[-1])
            animation.write(img)
            
        animation.release()
        
    def create_rotation(df, data, TP, savepath, fps = 1, mode = "default",
                        nb_view = 12):
        """
        Generate an animation from the figures generated with 
        figures.show_data() method.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing spots informations.
        data : pandas.DataFrame
            Dataframe containing time points informations.
        savepath : str
            Folder path to the animation figures and file.
        fps : int, optional
            FPS number for the animation. The default is 1.
        mode : str, optional
            Similar modes as show_data():
                - default
                - centered
                - aligned
            The default is "default".

        """
        ## Importing openCV
        import cv2
        
        filepaths, img_list = [], []
        
        animation = cv2.VideoWriter(savepath+"\\Animation.avi", 0, 
                                    cv2.VideoWriter_fourcc(*'DIVX'), fps = fps,
                                    frameSize = (4000, 2800))
        
        angles = list(range(0, 360, int(360/nb_view)))
        
        for az in angles:
            filepaths.append(savepath+"\\fig_az{}.png".format(az))
            figures.show_data(df, TP, data, mode = mode, show = False, 
                              savepath = filepaths[-1], 
                              viewing_angles = (30, az))
            
            img = cv2.imread(filepaths[-1])
            animation.write(img)
            
        animation.release()