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
    
    def links(df):
        """
        Generate a Series containing , for each track (row), the list of 
        spots ID that they are composed of, in chronological order.
        The Series is called self.spotsLinks.

        """
        
        # Getting the target for each spots.
        links = df["TARGET"]
        
        # Every Nan means that the spots has no target, so it is the last one.
        # We will build the lists by going backward.
        enders = links[links.isnull()].index
        
        # Creating the final list and creating the sublists with the enders ID.
        spotsLinks = [[ID] for ID in enders]
        
        # Looping the spots until no backward link is established.
        unfinished = True
        
        while unfinished:
            unfinished = False
            
            for track in spotsLinks:
                # Trying to add the ID of the previous spots from the last spot
                # in the sublist.
                try : 
                    track.append(links[links == track[-1]].index[0])
                    # If it works, then there is a connection.
                    unfinished = True
                except :
                    pass
        
        # Reversing each sublist.
        for track in spotsLinks :
            track.reverse()
        
        # Saving the info.
        spotsLinks = pd.Series(spotsLinks, 
                               index = [df.loc[idx[0]]["TRACK_ID"] 
                                        for idx in spotsLinks])
    
        return spotsLinks
    
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
        aligned : bool, optional
            If True, compute displacement vectors for all aligned coordinates.
            See self.alignRotAxis().
    
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
            
            df, clustCent = clustering.compute_clusters(df, 
                                                        center, 
                                                        coord_column)
            # selected = df[df["F_SELECT"]].index
            # vectors = vectors.loc[selected]
        
        if inplace :
            return pd.concat([df, vectors], axis = 1)
        
        else :
            return vectors
    
    
    
    def drift(df, coord_column = "COORD"):
        """
        Compute the drift of the organoid between time points.

        """
        
        TP = df["TP"].unique().tolist()
        TP.sort()
        
        #data = pd.DataFrame(index = TP)
        
        raw_centroid = pd.Series(index = TP, name = "Raw_centroid", 
                                 dtype = "object")
        drift_distance = pd.Series(index = TP, name = "Drift_distance", 
                                 dtype = "float")
        drift_vector = pd.Series(index = TP, name = "Drift_vector", 
                                 dtype = "object")
        
        if "F_SELECT" in df.columns :
            clust_centroid = pd.Series(index = TP, name = "Cluster_centroid", 
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
                           name = "volume", dtype = "float")
        
        # Adding volume and mean radius to data
        data = pd.concat([data, volume], axis = 1)
        data["radius"] = [(3*V/(4*np.pi))**(1/3) for V in data["volume"]]
        
        return data
        
    def translation(df, coord_column = "COORD",
                    destination = [0, 0, 0], inplace = True):
        """
        Translate the coordinates of all points within df to get the
        centroid of the organoid at [0, 0, 0].
        The translated coordinates are added to df.

        """
        # Setting the center coordinates.
        center = np.array(destination)
        
        # Creating a DataFrame to store the translated coordinates.
        coords = pd.Series(name = "TRANS_COORD", 
                           dtype = "object")
        
        # Iterating over time points.
        for tp in df["TP"].unique().tolist():
            
            # Getting the wanted rows.
            subdf = df[df["TP"] == tp]
            
            # Computing the centroid as well as the translation between the 
            # centroid and the center coordinates.
            centroid = tools.get_centroid(subdf, coord_column)
            translation = center-centroid
                
            new_coords = [subdf.loc[ID, coord_column]+translation
                           for ID in subdf.index]
            
            # Adding the translated coordinates to the DataFrame.
            coords = pd.concat([coords, 
                                pd.Series(new_coords, index = subdf.index, 
                                             dtype = "object", 
                                             name = "TRANS_COORD")], axis = 0)
            
        if inplace :
            return pd.concat([df, coords], axis = 1)
        
        else : 
            return coords
    
    def rotation_axis(df, vect_column = "DISP_VECT"):
        """
        Compute the rotation axis of the dataset, at each time point.
        Update data with the colinear vectors of the rotation axis.

        """
        
        data = pd.DataFrame(index = df["TP"].unique().tolist())
            
        # Creating a dataframe to store both vectors forming the PCA plane as
        # well as the crossproduct of those 2.
        componentVectors = pd.DataFrame(columns = ["V1", "V2", "RA_VECT"], 
                                        dtype = "object")
        
        # aligned_vectors = pd.Series(dtype = "object",
        #                             name = "ALIGNED_DISP_VECT")
        
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
                
                # ## Getting the projection of the displacement vectors on the
                # ## pca space (2D)
                # vect_2D_arr = pca.transform(arr)
                
                # ## Resizing into a 3D array.
                # vect_3D_arr = np.zeros((vect_2D_arr.shape[0], 3))
                # vect_3D_arr[:, :2] = vect_2D_arr
                
                # ## Transforming it into a series with correct index.
                # tmp_vectors = pd.Series([vect for vect in vect_3D_arr], 
                #                         dtype = "object",
                #                         name = "ALIGNED_DISP_VECT",
                #                         index = subdf.index)
                # aligned_vectors = pd.concat([aligned_vectors, tmp_vectors],
                #                             axis = 0)
                
                # Computing the crossproduct.
                RA = np.cross(V1, V2)
                # if RA[2] < 0:
                #     RA = -RA
                
                # Saving coordinates to th dataframe.
                componentVectors.loc[tp] = [V1, V2, RA]
        
        ## Merging componentVectors with data and adding computed aligned 
        ## displacement vectors to df.
        data = pd.concat([data, componentVectors], axis = 1)
        # df = pd.concat([df, aligned_vectors], axis = 1)
        
        return df, data
        
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
        
        for tp in df["TP"].unique().tolist() :
            
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
        
        subdf = df.copy()
        
        angular_velocity = pd.Series(dtype = "float", name = "ANG_VELOCITY")
        distance = pd.Series(dtype= "float", name = "DISTANCE_TO_RA")
        
        for ID in subdf.index:
            coord = subdf.loc[ID, "ALIGNED_COORD"]

            distance.loc[ID] = np.linalg.norm(coord[:2])
            disp = subdf.loc[ID, "ALIGNED_DISP_VECT"]
            RA_vect = data.loc[subdf.loc[ID, "TP"], "ALIGNED_RA_VECT"]
            
            if isinstance(disp, np.ndarray) and \
                isinstance(RA_vect, np.ndarray):
                velocity = np.linalg.norm(np.cross(disp, RA_vect))/(distance[ID])
            else:
                velocity = np.nan
            
            angular_velocity.loc[ID] = velocity
        
        velocityByCell = pd.concat([angular_velocity.to_frame(),
                                    distance.to_frame(), df["TP"].to_frame()],
                                   axis = 1)
        velocityByCell["ANG_VELOCITY"] = round(velocityByCell["ANG_VELOCITY"], 
                                               3)
        
        velocityByTP = pd.DataFrame(columns = ["Mean_AV", "Std_AV"],
                              index = data.index, dtype = "float")
        
        for tp in velocityByTP.index:
            subdf = velocityByCell[velocityByCell["TP"] == tp].copy()
            mean = subdf["ANG_VELOCITY"].mean()
            std = subdf["ANG_VELOCITY"].std()
            
            velocityByTP.loc[tp] = [mean, std]
            
        velocityByCell.drop(columns = "TP", inplace = True)
        
        if inplace :
            return (pd.concat([df, velocityByCell], axis = 1),
                    pd.concat([data, velocityByTP], axis = 1))
        
        else :
            return velocityByTP, velocityByCell
        
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