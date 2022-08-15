# -*- coding: utf-8 -*-
"""
Compute methods for OAT.

@author: Alex-932
@version: 0.7
"""

import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import cv2
from skimage import io
import numpy as np

from modules.utils.tools import *
from modules.utils.clustering import *

class compute():
    
    def vectors(df, coord_column = "COORD", vect_column = "DISP_VECT",
                filtering = False, center = None, inplace = True):
        """
        Compute displacement vectors for every spots in the tracks dataframe 
        and add them to it.
        
        Vectors are computed based on the sequence of the track they're in.
        That means that vectors are computed between 2 following spots of a 
        given track.
        
        They are saved in the same line as the origin spot of the vector.
        
        Parameters
        ----------
        filtering : bool, optional
            If True, use computeClusters() on tracks dataframe and keep the 
            selected ones (F_SELECT = True).
    
        """
        ## Creating a pd.Series to store vectors.
        vectors = pd.Series(dtype = "object", name = vect_column)
    
        for ID in df.index :
            try :
                vectors.loc[ID] = (df.loc[df.loc[ID, "TARGET"], coord_column]-
                                   df.loc[ID, coord_column])
            except :
                vectors.loc[ID] = np.nan
                
        if filtering :
            if center is None :
                center = tools.get_centroid(df, coord_column)
            
            df, clustCent = clustering.compute_clusters(df, center, coord_column)
        
        if inplace :
            return pd.concat([df, vectors], axis = 1)
        
        return vectors 
    
    def drift(df, coord_column = "COORD"):
        """
        Compute the drift of the organoid between time points.

        """
        
        TP = df["TP"].unique().tolist()
        TP.sort()
        
        raw_centroid = pd.Series(index = TP, name = "RAW_CENT", 
                                 dtype = "object")
        drift_distance = pd.Series(index = TP, name = "DRIFT", 
                                 dtype = "float")
        drift_vector = pd.Series(index = TP, name = "DRIFT_VECT", 
                                 dtype = "object")
        
        if "F_SELECT" in df.columns :
            clust_centroid = pd.Series(index = TP, name = "CLUST_CENT", 
                                       dtype = "object")
        
        ## Iterating over time point.
        for tp in TP:
            
            subdf = df[df["TP"] == tp]

            ## Getting the centroid.
            raw_centroid[tp] = tools.get_centroid(subdf, coord_column)
            
            try :
                clust_centroid[tp] = tools.get_centroid(subdf[subdf["F_SELECT"]],
                                                        coord_column)
            
            except :
                pass
            
            # If we're not at the first file, we can compute the drift vector 
            # between the centroids from the n-1 and n time point.
            # The drift vector is saved with the n-1 time point index. 
            if tp >= 1 :
                
                try : 
                    drift_vector[tp-1] = clust_centroid[tp]-clust_centroid[tp-1]
                    
                
                except :
                    drift_vector[tp-1] = raw_centroid[tp]-raw_centroid[tp-1]
                
                drift_distance[tp-1] = np.linalg.norm(drift_vector[tp-1])
        
        drift_distance = round(drift_distance, 2)
        
        try :
            data = pd.concat([raw_centroid, clust_centroid, drift_distance,
                              drift_vector], 
                             axis = 1)
        
        except : 
            data = pd.concat([raw_centroid, drift_distance, drift_vector], 
                             axis = 1)
        return data
    
        
    def volume(df, coord_column = "COORD"):
        """
        Use the Convex Hull algorithm of scipy to get the cells that are 
        forming the outershell as well as the volume of the organoid. 

        """       
        data = pd.DataFrame(index = df["TP"].unique().tolist())
        
        # Creating 2 buffer lists.
        volume, spots = [], []
        
        # Iterating over files.
        for tp in df["TP"].unique().tolist():
            
            # Getting the sub dataframe.
            subdf = df[df["TP"] == tp]
            
            # Using the Convex Hull algorithm.
            hull = ConvexHull(np.array(subdf["COORD"].tolist()))
            
            # Saving the volume and the spots that are the outershell.
            volume.append(hull.volume) 
            spots += list(subdf.iloc[hull.vertices.tolist()].index)
            
        # Setting the bool value for the question : is this spot part of the 
        # outershell ?
        isHull = pd.Series(name = "isHull", dtype="bool")
        for idx in df.index :
            if idx in spots:
                isHull.loc[idx] = True
            else :
                isHull.loc[idx] = False
                
        # Merging the bool isHull to df.
        df = pd.concat([df, isHull], axis = 1)
        
        # Converting the volume list to a Series to add time point informations
        volume = pd.Series(volume, index = df["TP"].unique().tolist(), 
                           name = "VOLUME", dtype = "float")
        
        # Adding volume and mean radius to data
        data = pd.concat([data, volume], axis = 1)
        data["RADIUS"] = [(3*V/(4*np.pi))**(1/3) for V in data["VOLUME"]]
        
        return data
        
        
    def translation(df, data = None, coord_column = "COORD", 
                    destination = [0, 0, 0], inplace = True):
        
        dest = pd.Series([np.array(destination)]*len(df["TP"].unique()), 
                         index = df["TP"].unique())
        
        if data is not None :
            column = "RAW_CENT"
            if "CLUST_CENT" in data.columns :
                column = "CLUST_CENT"
                
            translation = data[column] - dest
        
        else :
            translation = pd.Series(dtype = "object")
            
            for tp in df["TP"].unique():
                translation.loc[tp] = (
                    tools.get_centroid(df[df["TP"] == tp], coord_column) -
                    dest.loc[tp])
            
        f_translation = pd.Series(
            [translation.loc[df.loc[ID, "TP"]] for ID in df.index], 
            index = df.index)
        
        if inplace:
            df["TRANS_COORD"] = df[coord_column] - f_translation
            return df

        return df[coord_column] + f_translation
        
    
    def rotation_axis(df, data = None, vect_column = "DISP_VECT"):
        """
        Compute the rotation axis of the dataset, at each time point.
        Update data with the colinear vectors of the rotation axis.

        """
        if data is None :
            data = pd.DataFrame(index = df["TP"].unique().tolist())
            
        # Creating a dataframe to store both vectors forming the PCA plane as
        # well as the crossproduct of those 2.
        pca_vectors = pd.DataFrame(columns = ["V1", "V2", "RA_VECT"], 
                                        dtype = "object")
        
        # Iterating over time points.
        for tp in df["TP"].unique().tolist():
            subdf = df[df["TP"] == tp][vect_column]
            subdf = subdf.dropna()
            
            ## Checking if the dataframe is not empty.
            if not subdf.empty:   
                
                arr = np.array(subdf.tolist())
                
                pca = PCA(n_components = 2)
                pca.fit(arr)
                
                V1 = pca.components_[0]
                V2 = pca.components_[1]
                
                # Computing the crossproduct.
                RA = np.cross(V1, V2)
                
                # Saving coordinates to the dataframe.
                pca_vectors.loc[tp] = [V1, V2, RA]
        
        ## Adding the PCA vectors to data 
        data = pd.concat([data, pca_vectors], axis = 1)
        
        return data
        
    def alignment(df, data, inplace = True):
        """
        Rotate the points of df to get the axis of rotation aligned 
        with the Z axis. New coordinates are saved in df in 
        "Aligned_..." columns.

        """

        newCoords = df.loc[:, "TRANS_COORD"].copy()
        
        newRA = data.loc[:, "RA_VECT"].copy()
        
        new_Disp = df.loc[:, "DISP_VECT"].copy()
        
        rot_angles = pd.DataFrame(columns = ["Theta_X", "Theta_Y"],
                                   dtype = "float")
        
        for tp in df["TP"].unique() :
            
            coord = newRA.loc[tp]
            
            if not isinstance(coord, np.ndarray) and tp != 0:
                coord = newRA.loc[tp-1]
            
            theta_x = np.arctan2(coord[1], coord[2])#%np.pi
            rot_angles.loc[tp, "Theta_X"] = theta_x
            
            # Applying X rotation
            rot_x = np.array([[1, 0, 0],
                              [0, np.cos(theta_x), -np.sin(theta_x)],
                              [0, np.sin(theta_x), np.cos(theta_x)]])
            
            coord = np.matmul(rot_x, coord)
            
            theta_y = -np.arctan2(coord[0], coord[2])#%np.pi
            rot_angles.loc[tp, "Theta_Y"] = theta_y
            
            # Applying Y rotation
            rot_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                              [0, 1, 0],
                              [-np.sin(theta_y), 0, np.cos(theta_y)]])
            
            coord = np.matmul(rot_y, coord)
            
            newRA.loc[tp] = coord
        
        newRA.name = "ALIGNED_RA_VECT"
        
        for ID in newCoords.index :
            
            coord = newCoords.loc[ID]            
            
            tp = df.loc[ID, "TP"]
            
            ## Rotation on the x axis
            theta_x = rot_angles.loc[tp, "Theta_X"]
            
            rot_x = np.array([[1, 0, 0],
                              [0, np.cos(theta_x), -np.sin(theta_x)],
                              [0, np.sin(theta_x), np.cos(theta_x)]])
            
            coord = np.matmul(rot_x, coord)
            
            
            theta_y = rot_angles.loc[tp, "Theta_Y"]
            
            ## Rotation on the y axis
            rot_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                              [0, 1, 0],
                              [-np.sin(theta_y), 0, np.cos(theta_y)]])
            
            coord = np.matmul(rot_y, coord)
            
            ## Rotating the displacement vectors
            disp = new_Disp.loc[ID]
            if isinstance(disp, np.ndarray):
                disp = np.matmul(rot_x, disp)
                disp = np.matmul(rot_y, disp)
            
            else:
                disp = np.nan
            
            newCoords.loc[ID] = coord
            new_Disp.loc[ID] = disp
        
        newCoords.name = "ALIGNED_COORD"
        new_Disp.name = "ALIGNED_DISP_VECT"

        if inplace :
            return (pd.concat([df, newCoords, new_Disp], axis = 1), 
                    pd.concat([data, newRA, rot_angles], axis = 1))
        
        else :
            return (newCoords, 
                    pd.concat([newRA, rot_angles], axis = 1))
        
    def angular_velocity(df, data, inplace = True):

        av_res = pd.DataFrame(index = df.index, 
                              columns = ["R_VECT", "R", "AV_VECT", 
                                         "AV_RAD", "AV_DEG"], 
                              dtype = "object")
        
        for ID in df.index:
            V1, V2 = data.loc[ df.loc[ID, "TP"], ["V1", "V2"]]
            coord, displ = df.loc[ID, ["COORD", "DISP_VECT"]]
            
            if not np.isnan(displ).any() and \
                not np.equal(displ, np.zeros(3)).all():
                r_vect = np.dot(coord, V1)*V1 + np.dot(coord, V2)*V2
                r = np.linalg.norm(r_vect)
    

                AV_vect = np.cross(r_vect, displ)/(r**2)
                AV = np.linalg.norm(AV_vect)
    
                av_res.loc[ID] = pd.Series(
                    [np.round(r_vect, 3), round(r, 3), np.round(AV_vect, 3), 
                     round(AV, 3), round(AV*180/np.pi, 3)], 
                    name = ID, index = av_res.columns, dtype = "object")
        
        velocityByTP = pd.DataFrame(columns = ["Mean_AV", "Std_AV"],
                                    index = data.index, dtype = "float")
        
        for tp in velocityByTP.index:
            indexes = df[df["TP"] == tp].index
            subdf = av_res.loc[indexes]
            
            av_mean = subdf["AV_RAD"].mean()
            av_std = subdf["AV_RAD"].std()
            
            velocityByTP.loc[tp] = [av_mean, av_std]
        
        if inplace :
            return (pd.concat([df, av_res], axis = 1),
                    pd.concat([data, velocityByTP], axis = 1))
        
        else :
            return velocityByTP, av_res
        
    def voxels(df, filepath, offset = 10, outerThres = 0.9):
        """
        Compute voxels of cells. For each time point, the corresponding image
        is loaded as an array. We get a threshold by getting the minimal pixel
        value for the pixels that are at spots coordinates, for a given 
        time point. 

        Parameters
        ----------
        TP : int or list, optional
            Time points to compute. The default is "all".
        offset : int, optional
            Added value to the min and max to make sure everything is inside. 
            The default is 10.
        outerThres : float, optional
            Threshold determining that a pixels is inside the voxel and need to
            be set to 0. Used to improve plotting speed. The default is 0.9.

        """
        # Opening the image.
        image = io.imread(filepath)
        
        # Converting into a Numpy array. The shape is in this order : Z, Y, X.
        imarray = np.array(image)
        
        # Getting the minimal value of pixels at spots coordinates.
        df = df.astype("int")
        values = imarray[df["Z"], df["Y"], df["X"]].tolist()
        minimal = min(values)
        
        # Setting the outside as 0 and the inside as 1.
        imarray[imarray < (minimal-offset)] = 0
        imarray[imarray >= (minimal-offset)] = 1
        
        # Transposing the array that way it is oriented the same as the 
        # other objects (spots, vectors).
        imarray = np.transpose(imarray, (2, 1, 0))
        
        # Setting the inner pixels to 0 as well to only get the outer 
        # shell.
        toChange = []
        for x in range(1, imarray.shape[0]-1):
            for y in range(1, imarray.shape[1]-1):
                for z in range(1, imarray.shape[2]-1):
                    
                    # Getting the 3*3 square array centered on (x,y,z). 
                    neighbors = imarray[x-1:x+2, y-1:y+2, z-1:z+2]
                    
                    # Summing the values and if the number of ones is 
                    # greater than the threshold, saving the pixel coord.
                    if neighbors.sum()/27 >= outerThres :
                        toChange.append([x, y, z])
        
        # Setting the values of the selected pixels to 0.
        for coord in toChange:
            imarray[coord[0], coord[1], coord[2]] = 0
    
        return imarray