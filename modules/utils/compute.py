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

from modules.utils.centroid import centroid
from modules.utils.clustering import clustering

class compute():
    
    def vectors(df, coord_column = "COORD", vect_column = "DISP_VECT",
                filtering = False, inplace = True):
        """
        Compute displacement vectors for every spots in the tracks dataframe 
        and add them to it.
        
        Vectors are computed based on the sequence of the track they're in.
        That means that vectors are computed between 2 following spots of a 
        given track.
        
        They are saved in the same line as the origin spot of the vector.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing spot informations. It must contains the 
            following columns :
            - TP : time points
            - COORD : coordinates as an array for each point.
        coord_column : str, optional
            Name of the column containing the coordinates.
        vect_column : str, optional
            Name of the column where displacement vectors will be saved.
        filtering : bool, optional
            If True, select the spots by clustering them and selecting the one
            that it most likely to be the organoïd.
        inplace : bool, optional
            If True, add the displacement vectors column to the input dataframe
            and returns it. If False, return the displacement vectors as a 
            pandas.Series.
            
        Returns
        -------
        pd.DataFrame or pd.Series
            Dataframe if inplace is True, Series if not.
            
        """
        ## Creating a pd.Series to store vectors
        vectors = pd.Series(dtype = "object", name = vect_column)
        
        ## Iterating over all spots
        for ID in df.index :
            ## Computing the difference between the target coordinates and the
            ## origin coordinates
            try :
                vectors.loc[ID] = (df.loc[df.loc[ID, "TARGET"], coord_column]-
                                   df.loc[ID, coord_column])
            
            ## If not able, then there is no target so set it to nan
            except :
                vectors.loc[ID] = np.nan
        
        ## Adding clustering informations if wanted
        if filtering :
            df = clustering.compute_clusters(df, coord_column)
        
        ## Return the results
        if inplace :
            return pd.concat([df, vectors], axis = 1)
        return vectors 
    
    def drift(df, coord_column = "COORD"):
        """
        Compute the drift of the organoid between time points.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing spot informations. It must contains the 
            following columns :
            - TP : time points
            - COORD : coordinates as an array for each point.
        coord_column : str, optional
            Name of the column containing the coordinates.
            
        Returns
        -------
        pd.DataFrame
            Dataframe with all drift results.

        """
        ## Retrieving time points and sorting them
        TP = df["TP"].unique().tolist()
        TP.sort()
        
        drift_df = pd.DataFrame(index = TP, 
                                columns = ["CENTROID", "CLUST_CENTROID", 
                                           "DRIFT", "DRIFT_VECT"],
                                dtype = "object") 
        
        ## Iterating over time point
        for tp in TP:
            
            subdf = df[df["TP"] == tp]

            ## Getting the gradient centroid
            drift_df.loc[tp, "CENTROID"] = centroid.compute_gradient_centroid(subdf, coord_column)
            ## Trying to compute the cluster's centroid
            try :
                drift_df.loc[tp, "CLUST_CENTROID"] = centroid.compute_mean_centroid(
                    subdf[subdf["CLUSTER_SELECT"]], coord_column)
            except :
                None
            
            ## Computing the drift starting at tp > 1
            if tp >= 1 :
                if not drift_df["CLUST_CENTROID"].dropna().empty : 
                    drift_df.loc[tp-1, "DRIFT_VECT"] = (
                        drift_df.loc[tp, "CLUST_CENTROID"] - drift_df.loc[tp-1, "CLUST_CENTROID"])
                    
                else :
                    drift_df.loc[tp-1, "DRIFT_VECT"] = (
                        drift_df.loc[tp, "CENTROID"]-drift_df.loc[tp-1, "CENTROID"])
                
                ## Computing the drift distance
                drift_df.loc[tp-1, "DRIFT"] = round(
                    np.linalg.norm(drift_df.loc[tp-1, "DRIFT_VECT"]), 2)
        
        return drift_df
    
        
    def volume(df, coord_column = "COORD"):
        """
        Use the convex hull algorithm to compute the volume of the organoid at 
        each time point .
        
        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing informations for each spots. It must contains 
            the following columns :
                - TP
                - COORD : column with coordinates as array.
        coord_column : str, optional
            Name of the column which contains the coordinates.
            
        Returns
        -------
        pandas.DataFrame
            Dataframe containing the volume and mean radius per time points. 

        """
        TP = df["TP"].unique().tolist()
        data = pd.DataFrame(index = TP, columns = ["VOLUME"], dtype = "float")
        
        ## Iterating over time points
        for tp in TP:
            
            ## Getting the sub dataframe
            subdf = df[df["TP"] == tp]
            
            ## Using the Convex Hull algorithm
            hull = ConvexHull(np.array(subdf["COORD"].tolist()))
            
            ## Saving the volume info
            data.loc[tp, "VOLUME"] = round(hull.volume, 2) 
                
        ## Adding the mean radius to data
        data["RADIUS"] = [round((3*V/(4*np.pi))**(1/3), 2) for V in data["VOLUME"]]
        
        return data
        
        
    def translation(df, data = None, coord_column = "COORD", 
                    destination = np.zeros(3), inplace = True):
        """
        Translating coordinates of each spots to get the centroid to the 
        destination.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing informations for each spots. It must contains 
            the following columns :
                - TP
                - COORD : column with coordinates as array.
        data : pandas.DataFrame, optional
            Dataframe containing informations for each time points.
            If provided, it must contains one of the following columns :
                - CENTROID
                - CLUST_CENTROID
            The default is None.
        coord_column : str, optional
            Name of the column which contains the coordinates . 
            The default is "COORD".
        destination : np.ndarray, optional
            Destination of the centroid. The default is (0, 0, 0).
        inplace : bool, optional
            If True, add the displacement vectors column to the input dataframe
            and returns it. If False, return the displacement vectors as a 
            pandas.Series.
            The default is True.

        Returns
        -------
        pandas.DataFrame or pandas.Series
            Input dataframe containing the "CENTRD_COORD". Or, Series containing
            the translated coordinates.

        """
        ## Creating a Series of the same length as the dataframe, 
        ## which contains the destination of the centroid
        dest = pd.Series([destination]*len(df["TP"].unique()), 
                         index = df["TP"].unique())
        
        ## If data is provided, using the more precise centroid
        if data is not None :
            column = "CENTROID"
            if "CLUST_CENTROID" in data.columns and not data["CLUST_CENTROID"].dropna().empty:
                column = "CLUST_CENTROID"
            ## Translation is the vectors from the destination to the centroid,
            ## for each time points
            translation = data[column] - dest
        
        ## Else, computing the gradient centroid
        else :
            translation = pd.Series(dtype = "object")
            
            for tp in df["TP"].unique():
                translation.loc[tp] = (
                    centroid.compute_gradient_centroid(df[df["TP"] == tp], coord_column) -
                    dest.loc[tp])
        
        ## Creating a series which contains the translation for each points 
        ## according to their time points
        f_translation = pd.Series(
            [translation.loc[df.loc[ID, "TP"]] for ID in df.index], 
            index = df.index)
        
        ## Computing the translation and returning as desired
        if inplace:
            df["CENTRD_COORD"] = df[coord_column] - f_translation
            return df
        return df[coord_column] + f_translation
        
    
    def rotation_axis(df, data = None, vect_column = "DISP_VECT"):
        """
        Compute the rotation axis of the dataset, at each time point.
        Update data with the rotaion axis vectors as well as the PCA plane 
        directory vectors.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing informations for each spots. It must contains 
            the following columns :
                - TP
                - DISP_VECT
        data : pandas.DataFrame, optional
            Dataframe containing informations for each time points.
            The default is None.
        vect_column : str, optional
            Name of the column containing the displacement vectors.
            The default is "DISP_VECT".
            
        Returns
        -------
        pandas.DataFrame
            Dataframe which contains informations for each time points.
            Update data if provided.
                

        """
        ## Creating data if not provided
        if data is None :
            data = pd.DataFrame(index = df["TP"].unique().tolist())
            
        ## Creating a dataframe to store both vectors forming the PCA plane as
        ## well as the cross product of those 2
        pca_vectors = pd.DataFrame(columns = ["V1", "V2", "RA_VECT"], 
                                        dtype = "object")
        
        ## Iterating over time points.
        for tp in df["TP"].unique().tolist():
            subdf = df[df["TP"] == tp][vect_column]
            subdf = subdf.dropna()
            
            ## Checking if the dataframe is not empty.
            if not subdf.empty:   
                ## Converting displacement vectors as a big array where each
                ## row is a vector and each column is the coordinate for a 
                ## given axis
                arr = np.array(subdf.tolist())
                
                pca = PCA(n_components = 2)
                pca.fit(arr)
                
                ## Getting the 2 vectors forming the PCA plane
                V1 = pca.components_[0]
                V2 = pca.components_[1]
                
                ## Computing the cross product
                RA = np.cross(V1, V2)
                
                ## Saving coordinates to the dataframe
                pca_vectors.loc[tp] = [V1, V2, RA]
        
        ## Adding the PCA vectors to data 
        data = pd.concat([data, pca_vectors], axis = 1)
        
        return data
        
    def alignment(df, data, inplace = True):
        """
        Rotate the points of df to get the axis of rotation aligned 
        with the Z axis. New coordinates are saved in df in 
        "Aligned_..." columns.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing informations for each spots. It must contains 
            the following columns :
                - TP
                - CENTRD_COORD
                - DISP_VECT
        data : pandas.DataFrame
            Dataframe containing informations for each time points. It must 
            contains the following columns :
                - RA_VECT
            The default is None.
        inplace : bool, optional
            If True, add the displacement vectors column to the input dataframe
            and returns it. If False, return the displacement vectors as a 
            pandas.Series.
            The default is True.

        """
        ## Retrieving required informations
        newCoords = df.loc[:, "CENTRD_COORD"].copy()
        newRA = data.loc[:, "RA_VECT"].copy()
        new_Disp = df.loc[:, "DISP_VECT"].copy()
        
        ## Creating a dataframe to hold rotation angle for each time point
        rot_angles = pd.DataFrame(columns = ["Theta_X", "Theta_Y"],
                                   dtype = "float")
        
        ## Iterating over time points
        for tp in df["TP"].unique() :
            ## Retrieving coordinates for the given time point
            coord = newRA.loc[tp]
            
            # if not isinstance(coord, np.ndarray) and tp != 0:
            #     coord = newRA.loc[tp-1]
            
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
        
        velocityByTP = pd.DataFrame(columns = ["MEAN_AV", "STD_AV"],
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