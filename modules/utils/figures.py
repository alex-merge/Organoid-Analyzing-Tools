# -*- coding: utf-8 -*-
"""
Plotting figures methods for OAT.

@author: Alex-932
@version: 0.7
"""
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from modules.utils.tools import *

class figures():
    
    def add_spots(ax, df, coord_column = "COORD", label = None, marker = "o", 
                  color = "navy", size = 50):
        
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
                    color = "black", length = None):
        
        coords = np.array(df[coord_column].tolist())
        vectors = np.array(df[vect_column].tolist())
        
        if length is not None :
            ax.quiver(coords[:, 0], coords[:, 1], coords[:, 2],
                      vectors[:, 0], vectors[:, 1], vectors[:, 2],
                      color = color, length = length)
        
        else :
            ax.quiver(coords[:, 0], coords[:, 1], coords[:, 2],
                      vectors[:, 0], vectors[:, 1], vectors[:, 2],
                      color = color)
            
        if label is not None :
            legend = Line2D([0], [0], marker = marker, color = color, 
                                 label = label, markerfacecolor = color, 
                                 markersize = 7, ls = '')
            return ax, legend
        
        return ax
    
        
        
        
    
    def showDistances(df, TP = "all", bins = 30, show = True, savepath = None, 
                      cmap = 'tab10'):
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
        # Setting the cmap.
        cmap = plt.cm.get_cmap(cmap)
        
        # Setting TP variable according to the user choice.
        if type(TP) in [int, float] :
            TP = [int(TP)]
            
        elif TP == "all" :
            TP = df["TP"].unique().tolist()
        
        for tp in TP :
            # Retrieving the data we will need.
            data = df[df["TP"] == tp].copy()
            fig, ax = plt.subplots(figsize = (20, 8), dpi = 400)
            
            # Showing cluster data if available.
            if "A_CLUSTER" in data.columns :
                # Setting the colors.
                data["Color"] = data.loc[:,"A_CLUSTER"].map(cmap)
                
                # Iterating over clusters ID.
                for cluster in data["A_CLUSTER"
                                    ].value_counts(ascending = True).index :
                    
                    # Getting the rows for 1 cluster.
                    subdata = data[data["A_CLUSTER"] == cluster]
                    
                    # Plotting the histogram with colors.
                    ax.hist(subdata["Distance"], color = subdata["Color"][0], 
                            bins = bins, edgecolor = "white")
            else :
                # Plotting the histogram without color presets.
                ax.hist(data["Distance"], bins = bins, edgecolor = "white")
            
            # Labelling axis and the figure.
            ax.set_xlabel("Distance (in pixels)")
            ax.set_ylabel("Number of spots")
            ax.set_title("Spots by the distance from the centroid")
            
            if show :
                plt.show()
            if not savepath is None :
                plt.savefig(savepath, dpi = 400)
            
            plt.close()
            
    def showCentroids(self, TP = "all", figsize = (20, 8), dpi = 400, 
                      show = True, save = False):
        pass
            
    def showVectors(self, TP, df = "default", angles = None, lim = None,
                    rotAxis = True, show = True, label = "3D",
                    save = False, cellVoxels = False, vectorColor = "black"):
        """
        Create a figure with a representation of the vector field. The figure 
        is then saved.
    
        Parameters
        ----------
        TP : float
            Time point.
        df : str, optional
            Select the data to show:
            - default : raw vectors.
            - translated : translated vectors if computed.
            - aligned : translated and the rotation axis is the Z axis.
        angles : tuple, optional
            Viewing angle as follow (azimuth, elevation). The default is None.
        lim : list, optional
            Limits for the axis. Format is as follow : 
                [[xmin, xmax], [ymin, ymax], [zmin, zmax]] 
            The default is None.    
        rotAxis : bool, optional
            If True, show the rotation axis if available. The default is True.
        show : bool, optional
            If True, show the figure. Default is True.
        label : str, optional
            Name of the representation. The default is "3D".
        save : bool, optional
            If True, save the figures in \\output\\figs\\vectors.
        cellVoxels : bool, optional
            Computationally heavy, use with caution !
            If True, show the cells as voxels. Voxels are obtained using the
            getCellVoxels().
        vectorColor : str, optional
            Set the color of the vectors. The default is black.
    
        """
        # Initializing the figure and its unique axes.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Normal vectors, directly computed from trackmate.
        subdf = self.tracks[self.tracks["TP"] == TP].copy()
        
        if df == "default" :
            if np.isin(["RA_uX", "Cent_X"], self.data.columns).all()\
                and rotAxis : 
                RA = self.data.loc[TP].copy()
                RA.loc[["Cent_X", "Cent_Y", "Cent_Z"]] = [0, 0, 0]
        
        # Using translated coordinates if desired and available.
        if df == "translated" and "Trans_X" in subdf.columns :
            subdf = subdf.drop(columns = ["X", "Y", "Z"])
            subdf.rename(columns = {"Trans_X": "X",
                                    "Trans_Y": "Y",
                                    "Trans_Z": "Z"},
                         inplace = True)
            
            # Preparing rotation axis data if available and wanted.
            if "RA_uX" in self.data.columns and rotAxis : 
                RA = self.data.loc[TP].copy()
                RA.loc[["Cent_X", "Cent_Y", "Cent_Z"]] = [0, 0, 0]
                
        # Using aligned coordinates if desired and available.    
        if df == "aligned" and "Aligned_X" in subdf.columns :
            subdf = subdf.drop(columns = ["X", "Y", "Z", "uX", "vY", "wZ"])
            subdf.rename(columns = {"Aligned_X": "X", "Aligned_Y": "Y",
                                    "Aligned_Z": "Z", "Aligned_uX": "uX",
                                    "Aligned_vY": "vY", "Aligned_wZ": "wZ"},
                         inplace = True)
            
            if "Aligned_RA_uX" in self.data.columns and rotAxis :
                RA = self.data.loc[TP].copy()
                RA.loc[["Cent_X", "Cent_Y", "Cent_Z"]] = [0, 0, 0]
                RA = RA.drop(index = ["RA_uX", "RA_vY", "RA_wZ"])
                RA.rename(index = {"Aligned_RA_uX": "RA_uX",
                                   "Aligned_RA_vY": "RA_vY",
                                   "Aligned_RA_wZ": "RA_wZ"},
                          inplace = True)
           
        # Plotting the vector field according to the user choice.
        ax.quiver(subdf["X"], subdf["Y"], subdf["Z"], 
                  subdf["uX"], subdf["vY"], subdf["wZ"],
                  color = vectorColor)
        
        # Plotting the axis of rotation if desired and available.
        if rotAxis and "Cent_X" in self.data.columns and \
            "RA_uX" in self.data.columns:
            ax.quiver(RA["Cent_X"], RA["Cent_Y"], RA["Cent_Z"], 
                      RA["RA_uX"], RA["RA_vY"], RA["RA_wZ"],
                      color = "red", length = 5, pivot = "middle")
            
        # Showing cell voxels if so desired.
        if hasattr(self, "cellArray") and cellVoxels and label == "3D" and \
            TP in self.cellArray.index :
                
            ax.voxels(self.cellArray[TP], shade = True)
            
        # Labeling axis
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        # Giving a title to the figure (renamed further if an angle is 
        # provided)
        ax.set_title("Time point : "+str(TP))
        
        # Setting limits to axis.
        # If nothing has been provided : limit are set only for special views.
        # -> The first front half is shown when "2D" angling.
        if lim == None :
            if angles == (0, 90):
                ymax, ymin = subdf["Y"].max(), subdf["Y"].min()
                ax.set_ylim3d([ymax-(ymax-ymin)/2, ymax+10])
                ax.set_yticks([])
    
            elif angles == (0, 0):
                xmax, xmin = subdf["X"].max(), subdf["X"].min()
                ax.set_xlim3d([xmax-(xmax-xmin)/2, xmax+10])
                ax.set_xticks([])
    
            elif angles == (90, 0):
                zmax, zmin = subdf["Z"].max(), subdf["Z"].min()
                ax.set_zlim3d([zmax-(zmax-zmin)/2, zmax+10])
                ax.set_zticks([])
                
            else :
                ymax, ymin = subdf["Y"].max(), subdf["Y"].min()
                xmax, xmin = subdf["X"].max(), subdf["X"].min()
                zmax, zmin = subdf["Z"].max(), subdf["Z"].min()
                ax.set_ylim3d([ymin-10, ymax+10])
                ax.set_xlim3d([xmin-10, xmax+10])
                ax.set_zlim3d([zmin-10, zmax+10])
                
        # If limits are provided.        
        else :
            ax.set_xlim3d(lim[0])
            ax.set_ylim3d(lim[1])
            ax.set_zlim3d(lim[2])
            
        # Setting the viewing angle if provided and renaming the figure to
        # include the angle information
        if type(angles) == tuple:
            ax.view_init(angles[0],angles[1])
            ax.set_title("Timepoint : "+str(TP)+', Angle : ('+str(angles[0])+\
                         ","+str(angles[1])+")")
                
        if show :
            plt.show()
            
        # Saving the figure as an image. Name template is as follow :
        # {name of the file (argument given in _init_)}_vf_({TP})_{label}.png
        if save :
            plt.savefig(self.dir["vectorsFigs"]+"\\"+self.sample+\
                        "_vf_("+str(TP)+")_"+label+".png", dpi=400)
                
        plt.close(fig)
            
    def showOrgVolume(data, show = True, savepath = None):
        
        # if not "volume" in data.columns:
        #     self.computeConvexHull()
        
        print("# Plotting the organoid volume over time ... ")    
        
        fig, axs = plt.subplots(dpi = 400)
        
        axs.plot(data.index, data["volume"])
        axs.set_xlabel("Time point")
        axs.set_ylabel("Volume")
        axs.set_title("Volume of the organoid over time")
        
        if not savepath is None :
            plt.savefig(savepath, dpi = 400)
            
        if show :
            plt.show()
            
        plt.close()
    
    def AngularVelocity(df, TP, show = True, savepath = None):
        """
        Plot the angular velociy for the given time point.

        Parameters
        ----------
        TP : int or str.
            Time point (int) or all timepoints ("all") as a violinplot.

        """
        
        # Checking if all computing has been done.
        # if not hasattr(self, "tracks") :
        #     self.getVectors()
        # if not "Angular_Velocity" in df.columns :
        #     self.computeAngularVelocity()
        
        # If only one time point.
        if type(TP) == int:
            
            print("# Plotting angular velocity for time point ", TP, " ... ")
            
            # Subsampling the dataset.
            subdf = df[df["TP"] == TP].copy()
            
            fig, axs = plt.subplots(2, 1, figsize = (15, 13), dpi = 400)
            
            # Creating the first panel.
            axs[0].scatter(subdf["Distance_rotAxis"], subdf["Angular_Velocity"])
            axs[0].set_xlabel("Distance from the axis of rotation (pixels)")
            axs[0].set_ylabel("Angular Velocity (rad/tp)")
            axs[0].set_title("Angular Velocity according to the distance from rotation Axis")
            
            # Creating the seconf panel.
            P1 = axs[1].scatter(subdf["Aligned_X"], subdf["Aligned_Y"], 
                                c = subdf["Angular_Velocity"], 
                                cmap = "RdYlGn_r")
            fig.colorbar(P1, ax = axs[1])
            axs[1].set_xlabel("X")
            axs[1].set_ylabel("Y")
            axs[1].set_title("Spots on the XY plane.")
        
        # If all the time points then violin plot.
        elif type(TP) == str and TP == "all":
            
            print("# Plotting mean angular velocity over time ...")
            
            fig, axs = plt.subplots(dpi = 400)
            sns.violinplot(ax = axs,
                           x = df["TP"].astype("int"), 
                           y = df["Angular_Velocity"])
            
        # Showing, saving and closing the figure
        if not savepath is None :
            plt.savefig(savepath, dpi = 400)
        
        if show :
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
