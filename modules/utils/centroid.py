# -*- coding: utf-8 -*-
"""
Methods to compute the centroid from a dataset.

@author: alex-merge
@version: 0.8
"""
import numpy as np
from random import randrange

class centroid():
    """
    Set of methods to compute the centroid for OAT.
    """
    
    def compute_distance_sum(point, arr):
        """
        Return the sum of the distance between all spots and the point. 

        Parameters
        ----------
        point : np.ndarray
            Point to compute the distance from.
        arr : np.ndarray
            Array containing all the spots coordinates.
            Each row contains the spot coordinates.
            Each column contains the coordinates on a given axis.

        Returns
        -------
        float
            Sum of the distances.

        """
        return sum([np.linalg.norm( arr[k, :]-point ) for k in range(arr.shape[0])])
    
    
    def compute_gradient_centroid(df, coord_column, searching_spd = 1, 
                                  threshold = 0.05, iteration = 100):
        """
        Compute the centroid of the given coordinates using a gradient slope.
        
        First we take a random point, compute its distance from all spots.
        Then we compute the variation of this distance on each axis and go 
        where the distance decrease.
        
        This is reiterated several times to make sure we get the global minima.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe which contain the coordinates of the spots.
        coord_column : str
            Name of the column which contains the coordinates.
        searching_spd : int, optional
            Speed at which the coordinates goes down the gradient. 
            The default is 1.
        threshold : float, optional
            Stop the search if the distance decrease if less than the threshold. 
            The default is 0.05.
        iteration : int, optional
            Number of iterations. The default is 100.

        Returns
        -------
        cent_coord : numpy.ndarray
            Coordinates of the centroid.

        """
        ## Creating a list to store results as follow :
        ##  [ [sum of the distance, coordinates array of the point], ... ]
        computed_coords = []
        
        ## delta on each axis
        h = [np.array([0.001, 0, 0]), np.array([0, 0.001, 0]), 
                     np.array([0, 0, 0.001])]
        
        arr = np.array(df[coord_column].dropna().tolist())
        
        ## Looping over the number of iteration wanted
        for step in range(iteration):
            
            ## Randomly selecting a point within the min/max on each axis
            point = np.array([randrange(int(arr[:, 0].min()), int(arr[:, 0].max())),
                              randrange(int(arr[:, 1].min()), int(arr[:, 1].max())),
                              randrange(int(arr[:, 2].min()), int(arr[:, 2].max()))])
            
            """ 
            Looping while the distance between 2 computed distances is less
            than the threshold or if the last computed sum is equal to one of the
            already computed sums.
            """
            while (len(computed_coords) >= 3 and 
                abs(computed_coords[-1][0] - computed_coords[-2][0]) >= threshold \
                and computed_coords[-1][0] != computed_coords[-3][0]) \
                or len(computed_coords) < 3:
                
                distance = centroid.compute_distance_sum(point, arr)
                computed_coords.append([distance, point])
                
                ## Getting variation of the distance by slightly varying coordinates 
                delta_dist = [distance - centroid.compute_distance_sum(point+h[axis], 
                                                                       arr)
                              for axis in range(3)]
                
                point += np.array([searching_spd*int(round((k/abs(k)), 0)) 
                                   for k in delta_dist])
        
        ## Getting the minimum then the coordinates where this minimum is achieved.
        minimum = min([value[0] for value in computed_coords])
        cent_coord = [value[1] for value in computed_coords if value[0] == minimum][0]
        
        return cent_coord
    
    
    def compute_mean_centroid(df, coord_column):
        """
        Simply compute the mean value on each axis and return the coordinates.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe where coordinates are stored as array in the 
            coord_column.
        coord_column : str
            Name of the column containing the coordinates.

        Returns
        -------
        cent_coord : numpy.ndarray
            Coordinates of the centroid as an array.

        """
        ## Converting the coordinates in a big array where rows are points 
        ## coordinates and columns are the coordinates on a given axis.
        arr = np.array(df[coord_column].dropna().tolist())
        
        ## Computing the mean by columns.
        cent_coord = np.array([arr[:, ax].mean() for ax in range(arr.shape[1])])
        
        return cent_coord
    
    
    def compute_sampled_centroid(df, coord_column, n_sample = 10, iteration = 100):
        """
        Compute the mean centroid for a random sample of the dataframe, several 
        times depending on the number of wanted iteration..
        The size of the random sample (i.e. the number of rows) is set by
        n_sample.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe where coordinates are stored as array in the 
            coord_column.
        coord_column : str
            Name of the column containing the coordinates.
        n_sample : int
            Number of sample by random sampling.
        iteration : int
            Number of subsampled centroid to compute.

        Returns
        -------
        numpy.ndarray
            Coordinates of the centroid as an array.

        """
        centroids = [centroid.compute_mean_centroid(df.sample(n_sample, axis = 0), 
                                                    coord_column)
                     for i in range(iteration)]
        centroids_arr = np.array(centroids)
        
        return np.array([np.mean(centroids_arr[:, k]) 
                         for k in range(centroids_arr.shape[1])])
