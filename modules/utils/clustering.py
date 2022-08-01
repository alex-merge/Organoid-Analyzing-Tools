# -*- coding: utf-8 -*-
"""
Clustering methods for OAT.

@author: Alex-932
@version: 0.7
"""

import pandas as pd
import re
from sklearn.cluster import DBSCAN
from modules.utils.tools import tools

class clustering():
    
    def select_cluster(subdf, center, column, threshold = 10):
        """
        Search for the cluster that is the most likely of being the
        organoid cluster.
        
        The algorithm compute the centroid of each cluster.
        
        Then it computes the distance between each centroid and the center of 
        the volume.
        
        Finally, it returns the cluster that is the closest to the center and
        that contains more than {threshold} spots.
    
        Parameters
        ----------
        subdf : pd.DataFrame
            Dataframe that contains spots coordinates as well as clustering
            results.
        center : list 
            Center coordinates of the whole volume as follow [X, Y, Z].
        column : str
            Name of the column that contains cluster IDs.
        threshold : int, optional
            Number of spots a cluster must have to be selected. The default is
            10.
    
        Returns
        -------
        selected : pd.Series
            Index is the ID of the spots, values are booleans.
            
    
        """
        # Retrieving the ID of the clusters as well as the number of spots they
        # contains.
        clustersInfo = subdf[column].value_counts()
        clustersID = clustersInfo.index
        
        # Creating a Series to store the distances between the centroid and the
        # center.
        dist = pd.Series(dtype = "float")
        
        # Computing the distance for each clusters.
        for ID in clustersID :
            centroid = tools.get_centroid(subdf[subdf[column] == ID])
            distance = tools.euclid_distance(centroid, center)
            dist.loc[ID] = distance
        
        # Sorting from the lowest to the greatest distance.
        dist.sort_values(ascending = True, inplace = True)
        
        # Going through the closest to farthest cluster until it contains more 
        # than {threshold} spots. If there are no cluster that meets both 
        # conditions, we take the first one.
        selectedClusterID = 0
        for idx in range(len(dist.index)):
            if clustersInfo[dist.index[idx]] >= threshold:
                selectedClusterID = dist.index[idx]
                break
        
        # Returning the selection result as a pd.Series
        return subdf[column] == selectedClusterID      
    
    def clustering_core(df, center, cIter = 100, cSample = 10, 
                         eps = 40, min_samples = 3, threshold = 10):
        """
        Cluster and select the spots that are more likely to be part of the
        organoid.
        
        First, the algorithm compute the centroid of the organoid by taking 
        {cSample} random spots and computing their centroid. 
        It's repeated {cIter} times then it takes the average centroid.
        
        A DBSCAN is then runned on the distance between the spots and the 
        centroid as we expect a spike at a certain distance given all spots 
        that are part of the organoid should be at the same distance.
        DBSCAN give the spots for each spikes and we select the right spike by 
        taking the one that is closer to the centroid.
        
        A second DBSCAN is runned on the spots of the selected spikes to 
        separate the ones that are close but not part of the organoid. The 
        cluster is also selected by the selectCluster method.  
    
        Parameters
        ----------
        df : pd.DataFrame
            Spots to cluster. Same formatting as self.spots expected.
        center : list 
            Center coordinates of the whole volume as follow [X, Y, Z].
        cIter :  int, optional
            cIter number for the centroid location. The default is 100.
        cSample : int, optional
            Number of spots to compute the centroid. The default is 10.
        eps : int, optional
            Radius of search for the 2nd DBSCAN algorithm. The default is 40.
        min_samples : int , optional
            Min. number of neighbors for the 2nd DBSCAN. The default is 3.
        threshold : int, optional
            Number of spots a cluster must have to be selected. The default is
            10.
    
        Returns
        -------
        Results : pd.DataFrame
            Dataframe where index are the spots ID and columns are :
                A_CLUSTER : Clusters ID (int) for the first clustering step.
                A_SELECT : Selected spots for the first clustering (bool).
                F_CLUSTER : Clusters ID (int) for the 2nd clustering step.
                F_SELECT : Selected spots for the second clustering (bool).
    
        """
        # Getting the centroid.
        centroids = [tools.get_centroid(df.sample(cSample, axis=0)) \
                     for k in range(cIter)]
        centroids = pd.DataFrame(centroids, columns=["X", "Y", "Z"])
        centroid = [centroids["X"].median(), 
                    centroids["Y"].median(),
                    centroids["Z"].median()]
        
        # Computing the distance between each point and the centroid.
        distance = pd.Series(dtype = "float", name = "DISTANCE")
        for points in df.index:
            distance[points] = tools.euclid_distance(
                list(df.loc[points, ["X", "Y", "Z"]]), 
                centroid)
        
        # Clustering the distances and saving it as a pd.Series.
        cluster = DBSCAN(eps=5, min_samples=6).fit_predict(distance.to_frame())
        cluster = pd.Series(cluster, index = df.index, name = "A_CLUSTER")        
        
        # Creating the final dataframe with the first clustering results.
        Results = cluster.to_frame()
        
        # Selecting the cluster based on the clustering.selectCluster method.    
        selected = clustering.select_cluster(pd.concat([df, cluster], 
                                                       axis = 1),
                                             center, column = "A_CLUSTER",
                                             threshold = threshold)
        selected.name = "A_SELECT"
        
        # Adding the selection results to Results dataframe.
        Results = pd.concat([Results, selected], axis = 1)
        
        # Keeping the selected spots for the next clustering step.
        subdf = df[selected].loc[:,["X", "Y", "Z"]]
        
        # Clustering the spots.
        cluster = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(subdf)
        cluster = pd.Series(cluster, index = subdf.index, name = "F_CLUSTER")
        
        # Merging the clusters ID to the Results dataframe. 
        Results = pd.concat([Results, cluster], axis = 1)
        
        # Selecting the right cluster once again using the same method.
        selected = clustering.select_cluster(pd.concat([subdf, cluster], 
                                                       axis = 1),
                                             center, column = "F_CLUSTER", 
                                             threshold = threshold)
        selected.name = "F_SELECT"
        
        # Merging the selection to the Results dataframe.
        Results = pd.concat([Results, selected], axis = 1)
        
        # Filling the NaN values in the 2nd clustering results as some spots
        # were not computed.
        Results["F_CLUSTER"].fillna(100, inplace = True)
        Results["F_SELECT"].fillna(False, inplace = True)
        
        return Results, distance, centroids
        
    def compute_clusters(df, center, eps = 40, min_samples = 3, 
                        cIter = 1000, cSample = 10, threshold = 10, 
                        rescaling = [1, 1, 1], inplace = True):
        """
        Clustering the spots for each frame using the .clusteringEngine() 
        method.
    
        Parameters
        ----------
        df : str, optional
            Name of the dataframe. The default is "spots".
            It can be either "spots" or "tracks".
        eps : int, optional
            See .clusteringEngine() method. The default is 40.
        min_samples : int , optional
            See .clusteringEngine() method. The default is 3.
        cIter :  int, optional
            See .clusteringEngine() method. The default is 1000.
        cSample : int, optional
            See .clusteringEngine() method. The default is 10.
        threshold : int, optional
            See .clusteringEngine() method. The default is 10.
        rescaling : list, otpional
            Rescale the spots coordinates on each axis by the given value :
                [Xratio, Yratio, Zratio].
            The default is [1, 1, 1].
    
        """
        
        # Clustering every spots, frame by frame and adding the results to the
        # res temporary datafame.
        Results = pd.DataFrame(dtype = "float")
        
        # Creating a dataframe to store informations about the distance (see
        # clusteringEngine). 
        clustDist = pd.DataFrame(columns = ["Distance", "TP"], 
                                 dtype = "object")
        
        # Creating a dataframe to save the centroids coordinates that have been
        # computed (for debug reasons).
        clustCent = pd.DataFrame(columns = ["X", "Y", "Z", "TP"], 
                                 dtype = "object")
        
        for tp in df["TP"].unique().tolist():
            subdf = df[df["TP"] == tp]
            subdf = tools.reScaling(subdf, ratio = rescaling)
            clusterResults, dist, cent = clustering.clustering_core(subdf, 
                                         center, cIter, cSample, eps, 
                                         min_samples, threshold)
            
            dist = dist.to_frame()
            dist.columns = ["Distance"]
            
            # Adding time points info to the distance and centroid dataframes.
            dist["TP"] = [tp]*dist.shape[0]
            cent["TP"] = [tp]*cent.shape[0]
            
            Results = pd.concat([Results, clusterResults])
            clustDist = pd.concat([clustDist, dist])
            
        if inplace :    
            # Adding the cluster infos to the dataframe. 
            df = pd.concat([df, Results], axis = 1)
            return df
        
        else :
            return Results
        
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