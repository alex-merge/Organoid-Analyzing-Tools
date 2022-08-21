# -*- coding: utf-8 -*-
"""
Clustering methods for OAT.

@author: alex-merge
@version: 0.8
"""

import pandas as pd
import numpy as np
import re

from sklearn.cluster import DBSCAN
from modules.utils.centroid import centroid

class clustering():
    """
    Set of methods to cluster spots for OAT.
    """
    
    def clusters_on_distance(df, coord_column, centroid_method = "gradient"):
        """
        Cluster the spots based on the distance from the centroid.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe which contain the coordinates of the spots.
        coord_column : str
            Name of the column which contains the coordinates.
        centroid_method : str, optional
            Method to compute the centroid. Available options are :
                - mean : the mean of spots coordinates on each axis.
                - gradient : gradient slope to reduce the sum of the distances.
                - sampled : random subsampling and median of the mean centroid
                            computed.
            The default is "gradient".

        Returns
        -------
        cluster : pandas.Series
            Series containing the cluster id for each spots.

        """
        ## Computing the centroid according to the method chosen
        if centroid_method == "mean":
            cent_coords = centroid.compute_mean_centroid(df, 
                                                         coord_column)
        elif centroid_method == "gradient":
            cent_coords = centroid.compute_gradient_centroid(df, 
                                                             coord_column)
        elif centroid_method == "sampled":
            cent_coords = centroid.compute_sampled_centroid(df, 
                                                            coord_column)
        
        ## Computing the distances between the centroid and the spots    
        distances = pd.Series([ np.linalg.norm(coords-cent_coords) 
                               for coords in df[coord_column] ], 
                              index = df.index, name = "Distances")
        
        ## Clustering the distances and saving it as a pd.Series
        cluster = DBSCAN(eps=5, min_samples=6).fit_predict(distances.to_frame())
        cluster = pd.Series(cluster, index = df.index, 
                            name = "DIST_CLUSTER", dtype = "int")
        
        return cluster
        
        
    def compute_clusters(df, coord_column, eps = 7, min_samples = 1,
                         clustering_on_distance = True,
                         centroid_method = "gradient",
                         inplace = True):
        """
        Main method to cluster the spots and try to assess which spots are from
        the organoid and which one are not.

        Parameters
        ----------
        df : pandas.DatFrame
            Dataframe containing the following columns:
                - TP
                - coordinates column
        coord_column : str
            Name of the column containing coordinates.
        eps : int, optional
            Range of the search. See DBSCAN method in scikit-learn. 
            The default is 7.
        min_samples : int, optional
            Number of neighbors. See DBSCAN method in scikit-learn.
            The default is 1.
        clustering_on_distance : bool, optional
            If True, run a clustering based on the distance from the centroid. 
            The default is True.
        centroid_method : str, optional
            Method to compute the centroid. Available options are :
                - mean : the mean of spots coordinates on each axis.
                - gradient : gradient slope to reduce the sum of the distances.
                - sampled : random subsampling and median of the mean centroid
                            computed.
            The default is "gradient".
        inplace : bool, optional
            If True, return the input dataframe with cluster data added to it. 
            The default is True.

        Returns
        -------
        pandas.DataFrame
            Dataframe containing clustering data (raw clustering, final 
            clustering and cluster selection).

        """
        ## Filtering coordinates to remove nan values
        filtered_df = df[[coord_column, "TP"]].dropna()
        
        ## Getting time points and creating the result dataframe
        TP = filtered_df["TP"].unique()
        clusters_res = pd.DataFrame(np.zeros((df.shape[0], 2)),
                                    columns = ["DIST_CLUSTER", "MAIN_CLUSTER"],
                                    index = df.index, dtype = "int")
        
        ## Iterating over time points
        for tp in TP:
            sub_df = filtered_df[filtered_df["TP"] == tp]
            
            ## Clustering on distance if set to
            if clustering_on_distance:
                res = clustering.clusters_on_distance(sub_df, coord_column)
                for ID in res.index :
                    clusters_res.loc[ID, "DIST_CLUSTER"] = int(res.loc[ID])
            
            ## Converting the coordinates column into an array
            coord_arr = np.array(sub_df[coord_column].tolist())
            
            ## Running DBSCAN algorithm on the coordinates
            cluster = DBSCAN(eps = 7, min_samples = 1).fit_predict(coord_arr)
            cluster = pd.Series(cluster, index = sub_df.index, 
                                name = "MAIN_CLUSTER", dtype = "int")

            ## Saving cluster informations
            clusters_res.loc[cluster.index, "MAIN_CLUSTER"] = [
                int(cluster.loc[ID]) for ID in cluster.index]
        
        ## Adding the cluster results to get the final clusters    
        clusters_res["FINAL_CLUSTER"] = clusters_res["MAIN_CLUSTER"]+clusters_res["DIST_CLUSTER"]
        
        ## Get the ID of the cluster which have the most points
        s_clust_ID = clusters_res["FINAL_CLUSTER"].value_counts().index[0]
        clusters_res["CLUSTER_SELECT"] = [False]*len(clusters_res.index)
        
        ## In the CLUSTER_SELECT column, setting all spots to False then, 
        ## setting the selected spots to True 
        selected_id = clusters_res[clusters_res["FINAL_CLUSTER"] == s_clust_ID].index
        clusters_res.loc[selected_id, "CLUSTER_SELECT"] = [True]*len(selected_id)
        
        if inplace :
            return pd.concat([df, clusters_res], axis = 1)
        return clusters_res
        
        
    def select_ROI(df, std = 15, eps = 2, min_samples = 3, offset = 5):
        """
        Method to get the most representative value from a pd.Series in the 
        context of the search of the ROI.

        Parameters
        ----------
        subdf : pd.Series
            Series or df column that contain values for a certain category.
        std : int, optional
            Standard deviation threshold.
            The default is 15.
        eps : int, optional
            Range of the search. See DBSCAN method in scikit-learn. 
            The default is 2.
        min_samples : int, optional
            Number of neighbors. See DBSCAN method in scikit-learn.
            The default is 3.
        offset : float, optional
            Value to add or retrieve to the max or the min value to take into 
            account the size of the real object. 
            The default is 5.        

        Returns
        -------
        float
            Best fitting value for the limit of the ROI.

        """
        df.dropna(inplace = True)
        
        ## If the standard deviation is small, we don't need to cluster, all
        ## spots are given the same clusterID (0)
        if df.std() <= std :
            results = pd.Series(len(df.index)*[0], 
                                index = df.index, 
                                name = df.name)
            
        ## Else, clustering with DBSCAN to get the most representative values
        else :
            results = DBSCAN(eps = eps, min_samples = min_samples).fit_predict(
                df.to_frame())
            results = pd.Series(results, 
                                index = df.index, 
                                name = df.name)
            
        ## Getting the biggest cluster ID
        biggestClusterID = results.value_counts(ascending = False).index[0]
        
        ## Getting the side of the limit (min or max)
        extreme = re.split("\_", df.name)[-1]
        
        ## Returning the limit value +/- an offset depending 
        ## on the side (min/max).
        if extreme == "min":
            value = df[results == biggestClusterID].min()
            return value-offset
        elif extreme == "max":
            value = df[results == biggestClusterID].max()
            return value+offset