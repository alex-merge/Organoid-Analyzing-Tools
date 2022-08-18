# -*- coding: utf-8 -*-
"""
Clustering methods for OAT.

@author: Alex-932
@version: 0.7
"""

import pandas as pd
import numpy as np
import re

from sklearn.cluster import DBSCAN
from modules.utils.centroid import centroid

class clustering():
    
    def clusters_on_distance(df, coord_column, centroid_method = "gradient"):
        
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
        ## Filtering coordinates to remove nan values
        filtered_df = df[[coord_column, "TP"]].dropna()
        
        TP = filtered_df["TP"].unique()
        clusters_res = pd.DataFrame(np.zeros((df.shape[0], 2)),
                                    columns = ["DIST_CLUSTER", "MAIN_CLUSTER"],
                                    index = df.index, dtype = "int")
        
        for tp in TP:
            sub_df = filtered_df[filtered_df["TP"] == tp]
            
            if clustering_on_distance:
                res = clustering.clusters_on_distance(sub_df, coord_column)
                for ID in res.index :
                    clusters_res.loc[ID, "DIST_CLUSTER"] = int(res.loc[ID])
                
            coord_arr = np.array(sub_df[coord_column].tolist())
            
            cluster = DBSCAN(eps = 7, min_samples = 1).fit_predict(coord_arr)
            cluster = pd.Series(cluster, index = sub_df.index, 
                                name = "MAIN_CLUSTER", dtype = "int")
            
            for ID in cluster.index:
                clusters_res.loc[ID, "MAIN_CLUSTER"] = int(cluster.loc[ID])
            
        clusters_res["FINAL_CLUSTER"] = clusters_res["MAIN_CLUSTER"]+clusters_res["DIST_CLUSTER"]
        
        s_clust_ID = clusters_res["FINAL_CLUSTER"].value_counts().index[0]
        clusters_res["CLUSTER_SELECT"] = [False]*len(clusters_res.index)
        
        selected_id = clusters_res[clusters_res["FINAL_CLUSTER"] == s_clust_ID].index
        clusters_res.loc[selected_id, "CLUSTER_SELECT"] = [True]*len(selected_id)
        
        if inplace :
            return pd.concat([df, clusters_res], axis = 1)
        return clusters_res
        
        
    def select_ROI(subdf, std = 15, eps = 2, min_samples = 3, offset = 5):
        """
        Method to get the most representative value from a pd.Series in the 
        context of the search of the ROI.

        Parameters
        ----------
        subdf : pd.Series
            Series or df column that contain values for a certain category.
        std : int
            Standard deviation threshold.
        eps : int, optional
            Radius of search for the DBSCAN algorithm. The default is 2.
        min_samples : int , optional
            Min. number of neighbors for DBSCAN. The default is 3.
        offset : float, optional
            Value to add or retrieve to the max or the min value to take into 
            account the size of the real object. 
            The default is 5.        

        Returns
        -------
        float
            Best fitting value for the limit of the ROI.

        """
        subdf.dropna(inplace = True)
        # If the standard deviation is small, we don't need to cluster, all
        # spots are given the same clusterID (0).
        if subdf.std() <= std :
            results = pd.Series(len(subdf.index)*[0], 
                                index = subdf.index, 
                                name = subdf.name)
            
        # Else, clustering with DBSCAN to get the most representative values.
        else :
            results = DBSCAN(eps = eps, min_samples = min_samples).fit_predict(
                subdf.to_frame())
            results = pd.Series(results, 
                                index = subdf.index, 
                                name = subdf.name)
            
        # Getting the biggest cluster ID.
        biggestClusterID = results.value_counts(ascending = False).index[0]
        
        # Getting the side of the limit (min or max)
        extreme = re.split("\_", subdf.name)[-1]
        
        # Returning the limit value +/- an offset depending 
        # on the side (min/max).
        if extreme == "min":
            value = subdf[results == biggestClusterID].min()
            return value-offset
        elif extreme == "max":
            value = subdf[results == biggestClusterID].max()
            return value+offset