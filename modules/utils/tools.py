# -*- coding: utf-8 -*-
"""
Tools methods for OAT.

@author: Alex-932
@version: 0.7
"""

import pandas as pd
import numpy as np

class tools():
    
    def euclid_distance(PointA, PointB):
        """
        Return the euclid distance between PointA and PointB. 
        Works in 3D as well as in 2D. 
        Just make sure that both points have the same dimension.

        Parameters
        ----------
        PointA : list
            List of the coordinates (length = dimension).
        PointB : list
            List of the coordinates (length = dimension).

        Returns
        -------
        float
            Euclid distance between the 2 given points.

        """
        # Computing (xa-xb)², (ya-yb)², ...
        sqDist = [(PointA[k]-PointB[k])**2 for k in range(len(PointA))]
        
        return np.sqrt(sum(sqDist))
    
    def reScaling(df, ratio = [1, 1, 1]):
        """
        Rescale the coordinates of the given axis by the given ratio. 
    
        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe which contains axis as columns and spots as index.
        ratio : list
            Coordinates multiplier for each axis : [Xratio, Yratio, Zratio].
    
        Returns
        -------
        df : pandas.DataFrame
            Same as the input one but with rescaled coordinates.
    
        """
        out_df = pd.DataFrame(index = df.index, columns = df.columns)        
        out_df["X"] = df.loc[:, "X"]*ratio[0]
        out_df["Y"] = df.loc[:, "Y"]*ratio[1]
        out_df["Z"] = df.loc[:, "Z"]*ratio[2]
        
        return out_df
    
    def get_centroid(df, column):
        """
        Process the "euclid" centroid from a set of points. Get the centroid by
        getting the mean value on each axis.
    
        Parameters
        ----------
        df : pd.DataFrame
            Whole dataframe containing the given column.
        column : str
            columns containing coordinates arrays of the points.
    
        Returns
        -------
        centroid : np.array
            Array that contains the coordinates of the computed centroid.
    
        """
        centroid = []
        
        data = np.array(df[column].tolist())
        
        for axis in range(data.shape[1]):
            centroid.append(data[:,axis].mean())
        
        return np.array(centroid)
    
    def cross_product(df):
        """
        Compute the cross product of 2 vectors.

        Parameters
        ----------
        df : pd.DataFrame
            Each row is a vector and columns are ["uX", "vY", "wZ"].

        Returns
        -------
        pd.Series
            Series where index are ["uX", "vY", "wZ"].

        """
        # Retrieve the vectors as Series from the DataFrame.
        A, B = df.iloc[0], df.iloc[1]
        
        return pd.Series([(A["vY"]*B["wZ"]-A["wZ"]*B["vY"]),
                          -(A["uX"]*B["wZ"]-A["wZ"]*B["uX"]),
                          (A["uX"]*B["vY"]-A["vY"]*B["uX"])],
                         index = ["uX", "vY", "wZ"], dtype = "float")
    
    def displacement_vector(PtA, PtB, toList = False):
        """
        Return the 2D or 3D displacement vector between 2 points.
        The vector is oriented from PtA to PtB. 
        Both input must have the same dimension.
        
        Parameters
        ----------
        PtA : list or pd.Series
            List or Series containing the coordinates values. For example :
            [X, Y, Z].
        PtB : list or pd.Series
            List or Series containing the coordinates values. For example :
            [X, Y, Z].
        toList : bool, optional
            Force to save the vectors coordinates as a list.
        
        Returns
        -------
        list or pd.Series
            Return the coordinates of the vector in the same format as PtA.
    
        """
        vect = [PtB[axis] - PtA[axis] for axis in range(len(PtA))]
        
        if type(PtA) == list or toList:
            return vect
        else :
            return pd.Series(vect, index = PtA.index, dtype="float") 