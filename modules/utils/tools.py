# -*- coding: utf-8 -*-
"""
Tools methods for OAT.

@author: Alex-932
@version: 0.7
"""

import pandas as pd
import numpy as np

class tools():
    
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
    