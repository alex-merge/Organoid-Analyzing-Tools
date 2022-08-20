# -*- coding: utf-8 -*-
"""
Compute methods for OAT.

@author: alex-merge
@version: 0.7
"""

import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
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
        
        
    def translation(df, data, coord_column = "COORD", 
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
        data : pandas.DataFrame
            Dataframe containing informations for each time points.
            It must contains one of the following columns :
                - CENTROID
                - CLUST_CENTROID
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
        data["CENTRD_CENTROID"] = pd.Series([destination]*len(df["TP"].unique()), 
                                            index = df["TP"].unique())
        
        ## Retrieving info for the more detailled column available
        if "CLUST_CENTROID" in data.columns and not data["CLUST_CENTROID"].dropna().empty:
            column = "CLUST_CENTROID"
            translation = data[column] - data["CENTRD_CENTROID"]
            
        elif "CENTROID" in data.columns and not data["CENTROID"].dropna().empty:
            column = "CENTROID"
            translation = data[column] - data["CENTRD_CENTROID"]
            
        ## Else, computing the gradient centroid
        else :
            translation = pd.Series(dtype = "object")
            
            for tp in df["TP"].unique():
                translation.loc[tp] = (
                    centroid.compute_gradient_centroid(df[df["TP"] == tp], coord_column) -
                    data["CENTRD_CENTROID"].loc[tp])
        
        ## Creating a series which contains the translation for each points 
        ## according to their time points
        f_translation = pd.Series(
            [translation.loc[df.loc[ID, "TP"]] for ID in df.index], 
            index = df.index)
        
        ## Computing the translation and returning as desired
        if inplace:
            df["CENTRD_COORD"] = df[coord_column] - f_translation
            return df, data
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
        
    def alignment(df, data, coord_column = "CENTRD_COORD", 
                  vect_column = "DISP_VECT", inplace = True):
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
        coord_column : str, optional
            Name of the column containing the coordinates.
            The default is "CENTRD_COORD".
        vect_column : str, optional
            Name of the column containing the displacement vectors.
            The default is "DISP_VECT".
        inplace : bool, optional
            If True, add the displacement vectors column to the input dataframe
            and returns it. If False, return the displacement vectors as a 
            pandas.Series.
            The default is True.
            
        Returns
        -------
        pandas.DataFrame
            If inplace : return both input dataframes with computed coordinates.
            If not inplace : return coordinates as a Series and angles as a 
            dataframe.

        """
        ## Retrieving required informations
        new_coords = df.loc[:, coord_column].copy()
        new_RA = data.loc[:, "RA_VECT"].copy()
        new_disp = df.loc[:, vect_column].copy()
        
        ## Creating a dataframe to hold rotation angle for each time point
        rot_angles = pd.DataFrame(columns = ["Theta_X", "Theta_Y"],
                                   dtype = "float")
        
        ## Iterating over time points
        for tp in df["TP"].unique() :
            ## Retrieving vector for the given time point
            coord = new_RA.loc[tp]
            
            ## Use the last known rotation axis 
            if not isinstance(coord, np.ndarray) and tp != 0:
                coord = new_RA.loc[tp-1]
            
            ## Computing rotation angle on the X axis
            theta_x = np.arctan2(coord[1], coord[2])
            rot_angles.loc[tp, "Theta_X"] = theta_x
            
            ## Applying X rotation
            rot_x = np.array([[1, 0, 0],
                              [0, np.cos(theta_x), -np.sin(theta_x)],
                              [0, np.sin(theta_x), np.cos(theta_x)]])
            
            coord = np.matmul(rot_x, coord)
            
            ## Computing rotation angle on the Y axis
            theta_y = -np.arctan2(coord[0], coord[2])
            rot_angles.loc[tp, "Theta_Y"] = theta_y
            
            ## Applying Y rotation
            rot_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                              [0, 1, 0],
                              [-np.sin(theta_y), 0, np.cos(theta_y)]])
            
            coord = np.matmul(rot_y, coord)
            
            new_RA.loc[tp] = coord
        
        new_RA.name = "ALIGNED_RA_VECT"
        
        ## Iterating over time points
        for ID in new_coords.index :
            ## Retrieving the coordinates of the spot
            coord = new_coords.loc[ID]            
            
            tp = df.loc[ID, "TP"]
            
            ## Rotation on the x axis
            theta_x = rot_angles.loc[tp, "Theta_X"]
            rot_x = np.array([[1, 0, 0],
                              [0, np.cos(theta_x), -np.sin(theta_x)],
                              [0, np.sin(theta_x), np.cos(theta_x)]])
            
            coord = np.matmul(rot_x, coord)
            
            ## Rotation on the y axis
            theta_y = rot_angles.loc[tp, "Theta_Y"]
            rot_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                              [0, 1, 0],
                              [-np.sin(theta_y), 0, np.cos(theta_y)]])
            
            coord = np.matmul(rot_y, coord)
            
            ## Rotating the displacement vectors
            disp = new_disp.loc[ID]
            if isinstance(disp, np.ndarray):
                disp = np.matmul(rot_x, disp)
                disp = np.matmul(rot_y, disp)
            else:
                disp = np.nan
            
            ## Replacing the old coordinates
            new_coords.loc[ID] = coord
            new_disp.loc[ID] = disp
        
        new_coords.name = "ALIGNED_COORD"
        new_disp.name = "ALIGNED_DISP_VECT"

        if inplace :
            return (pd.concat([df, new_coords, new_disp], axis = 1), 
                    pd.concat([data, new_RA, rot_angles], axis = 1)) 
        return (new_coords, pd.concat([new_RA, rot_angles], axis = 1))
        
    def angular_velocity(df, data, coord_column = "COORD", 
                         vect_column = "DISP_VECT", inplace = True):
        """
        Computing the angular velocity for each spot using the displacement 
        vector and the R vectorwhich is the vector from the 
        rotation axis to the spot coordinates. The vector is perpendicular to
        the rotation axis.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing informations for each spots. It must contains 
            the following columns :
                - TP
                - COORD
                - DISP_VECT
        data : pandas.DataFrame
            Dataframe containing informations for each time points. It must 
            contains the following columns :
                - RA_VECT
        coord_column : str, optional
            Name of the column containing the coordinates.
            The default is "COORD".
        vect_column : str, optional
            Name of the column containing the displacement vectors.
            The default is "DISP_VECT".
        inplace : bool, optional
            If True, add the displacement vectors column to the input dataframe
            and returns it. If False, return the displacement vectors as a 
            pandas.Series.
            The default is True.

        Returns
        -------
        pandas.DataFrame
            If inplace : return the input dataframes with computed informations.
            If not inplace : return dataframes with computed informations.

        """

        av_res = pd.DataFrame(index = df.index, 
                              columns = ["R_VECT", "R", "AV_VECT", 
                                         "AV_RAD", "AV_DEG"], 
                              dtype = "float")
        av_res[["R_VECT", "AV_VECT"]] = av_res[["R_VECT", "AV_VECT"]].astype("object")
        
        ## Iterating over spot IDs
        for ID in df.index:
            ## Retrieving directory vectors of the PCA plane
            V1, V2 = data.loc[ df.loc[ID, "TP"], ["V1", "V2"]]
            ## Retrieving the coordinates and displacement vectors
            coord, displ = df.loc[ID, [coord_column, vect_column]]
            
            ## Checking if the displacement vector exist
            if not np.isnan(displ).any() and \
                not np.equal(displ, np.zeros(3)).all():
                ## Computing the R vector
                r_vect = np.dot(coord, V1)*V1 + np.dot(coord, V2)*V2
                r = np.linalg.norm(r_vect)
    
                ## Computing the angular velocity
                AV_vect = np.cross(r_vect, displ)/(r**2)
                AV = np.linalg.norm(AV_vect)
    
                ## Saving the information to the temporary dataframe
                av_res.loc[ID] = pd.Series(
                    [np.round(r_vect, 3), round(r, 3), np.round(AV_vect, 3), 
                     round(AV, 3), round(AV*180/np.pi, 3)], 
                    name = ID, index = av_res.columns, dtype = "object")
        
        ## velocity_TP hold the informations at the time point level (mean, std
        ## variation)
        velocity_TP = pd.DataFrame(columns = ["MEAN_AV", "STD_AV"],
                                    index = data.index, dtype = "float")
        
        ## Iterating over time points
        for tp in velocity_TP.index:
            ## Retrieving data realted to the given time point
            indexes = df[df["TP"] == tp].index
            subdf = av_res.loc[indexes]
            
            ## Computing mean and std deviation of the angular velocity
            av_mean = subdf["AV_RAD"].mean()
            av_std = subdf["AV_RAD"].std()
            
            ## Saving data to the dataframe
            velocity_TP.loc[tp] = [av_mean, av_std]
        
        if inplace :
            return (pd.concat([df, av_res], axis = 1),
                    pd.concat([data, velocity_TP], axis = 1))
        return velocity_TP, av_res