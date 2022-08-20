# -*- coding: utf-8 -*-
"""
Export methods for OAT.

@author: alex-merge
@version: 0.7
"""

import numpy as np
import pandas as pd

class export():
    """
    Set of methods to export OAT results.
    """
    
    def to_csv(df, savepath):
        """
        Save the given dataframe as a .csv, at the given path.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to save.
        savepath : str
            Full path of the output file.

        """
        outdf = df.copy()
        
        ## Getting names from the column containing arrays.
        is_array = [name for name in outdf.columns 
                    if isinstance(outdf[name][0], np.ndarray)]
        
        ## For each one, splitting the array into 3 columns 
        ## (one for each axis).
        for column in is_array:
            value_list = outdf[column].to_numpy()
            
            ## Replacing nan values with array([0, 0, 0]).
            for k in range(len(value_list)):
                if not isinstance(value_list[k], np.ndarray):
                    value_list[k] = np.array([np.nan]*3)
                    
            outdf[column] = value_list
            
            ## creating an array and the a dataframe out of the column.
            arr = np.array(outdf[column].tolist())
            splitted_values = pd.DataFrame(arr, index = outdf.index, 
                                           columns = [column+"_"+axis 
                                                      for axis in list("XYZ")],
                                           dtype = "float"
                                           )
            
            ## Merging the new dataframe and removing the old column.
            outdf = pd.concat([outdf, splitted_values], axis = 1)
            outdf.drop(columns = column, inplace = True)
        
        ## Saving the .csv at the given path.
        outdf.to_csv(savepath)
    
    def to_vtk_polydata(df, savepath, column_name = "COORD",
                        clusters_only = False):
        """
        Export the given coord column in the given dataframe 
        to a points type .vtk for visualization in paraview.
        Create trajectories for each tracks available.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe which contains coordinates columns.
        savepath : str
            Fullpath for the save file.
        column_name : str, optional
            Name of the column to export. The default is "COORD".
        clusters_only : bool, optional
            If True, export the spots that have been selected by the clustering.
            The default is False.

        """
        subdf = df.copy()
        
        """
        Preparing data.
        """
        ## Selecting spots if available and if the user wants it
        if clusters_only and "CLUSTER_SELECT" in subdf.columns :
            subdf = subdf[subdf["IS_ORGANOID"]]
                 
        ## Removing columns (tracks) containing nan values
        subdf.dropna(axis = "columns", inplace = True)
        
        ## Getting the number of columns i.e. the number of different tracks
        Tracks = subdf["TRACK_ID"].unique()
        
        ## Getting the number of points per tracks
        TP = subdf["TP"].unique()
        
        x_df = pd.DataFrame(dtype = "float", columns = Tracks, index = TP)
        y_df = pd.DataFrame(dtype = "float", columns = Tracks, index = TP)
        z_df = pd.DataFrame(dtype = "float", columns = Tracks, index = TP)
        
        ## Iterating over time points
        for tp in TP:
            for track in Tracks:
                subset = subdf[subdf["TP"] == tp]
                subset = subset[subset["TRACK_ID"] == track]
                
                ## Checking if the subset is empty
                if not subset.empty:
                    x_df.loc[tp, track] = subset[column_name][0][0]
                    y_df.loc[tp, track] = subset[column_name][0][1]
                    z_df.loc[tp, track] = subset[column_name][0][2]
                
                ## Otherwise, filling with nan values
                else :
                    x_df.loc[tp, track] = np.nan
                    y_df.loc[tp, track] = np.nan
                    z_df.loc[tp, track] = np.nan
        
        ## Replacing nan values with the last known position.
        x_df.fillna(method = "ffill", inplace = True)
        x_df.fillna(method = "bfill", inplace = True)
        y_df.fillna(method = "ffill", inplace = True)
        y_df.fillna(method = "bfill", inplace = True)
        z_df.fillna(method = "ffill", inplace = True)
        z_df.fillna(method = "bfill", inplace = True)        
        
        """
        Writing the file.
        """
        stream = open(savepath, "w")
        
        ## Writing the file specifications
        stream.write("# vtk DataFile Version 2.0\n")
        stream.write("PIV3D Trajectories\n")
        stream.write("ASCII\n")
        stream.write("DATASET POLYDATA\n")
        stream.write("POINTS "+str(len(Tracks)*len(TP))+" double\n")
        
        ## For every tracks and every timepoint (row), we write the coordinates 
        ## of the given point in the file
        for pt in Tracks:
            for tp in TP:
                X = x_df.loc[tp, pt] 
                Y = y_df.loc[tp, pt]                                
                Z = z_df.loc[tp, pt]
                stream.write(str(X)+" "+str(Y)+" "+str(Z)+"\n")
                    
        ## Writing the number of lines and the number of points
        stream.write("LINES "+str(len(Tracks))+" "+str((len(TP)+1)*len(Tracks))+"\n")
        stream.write("\n")
        
        ## Writing some more informations
        idx = 0;
        for pt in range(len(Tracks)):
            stream.write(str(len(TP))+" \n")
            for tp in range(len(TP)):
                stream.write(str(idx)+" \n")
                idx += 1
            stream.write("\n")

        stream.write("POINT_DATA "+str(len(Tracks)*len(TP))+"\n")
        stream.write("SCALARS index int 1\n")
        stream.write("LOOKUP_TABLE default\n")
        stream.write("\n")
        
        for pt in range(len(Tracks)):
             for tp in range(len(TP)):
                stream.write(str(tp)+" \n")
                
        ## Closing the file and saving it       
        stream.close()

        
    def toVTKpoints(df, savepath):
        """
        Writer to convert a series of coordinates into a points .vtk file.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe which contains coordinates columns.
        savepath : str
            Fullpath for the save file.

        """
        ## Setting the name of the file and opening it
        stream = open(savepath, "w")
        
        ## Getting the number of columns i.e. the number of different tracks
        points_nb = len(df['X'].index)
        stream.write("# vtk DataFile Version 2.0\n")
        stream.write("PIV3D Points\n")
        stream.write("ASCII\n")
        stream.write("DATASET POLYDATA\n")
        stream.write("POINTS "+str(points_nb)+" double\n")
        
        ## For every tracks and every timepoint (row), we write the coordinates 
        ## of the given point in the file
        for pt in range(points_nb):
            stream.write(str(df['X'].iloc[pt])+" "+\
                       str(df['Y'].iloc[pt])+" "+\
                       str(df['Z'].iloc[pt])+"\n")
                
        ## Closing the file and saving it.
        stream.close()        