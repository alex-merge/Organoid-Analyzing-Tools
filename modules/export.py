# -*- coding: utf-8 -*-
"""
Export methods for OAT.

@author: Alex-932
@version: 0.7
"""

import numpy as np
from scipy.io import savemat
import re
import pandas as pd

class export():
    
    def toCSV(df, savepath):
        """
        Save the given dataframe as a .csv at the given path.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to save.
        savepath : str
            Full path of the output file.

        """
        df.to_csv(savepath)
    
    def toMatLab(df, savepath):
        maxLength = max([df[df["TP"] == tp].shape[0] 
                         for tp in df["TP"].unique().tolist()])
        for tp in df["TP"].unique().tolist():
            data = df[df["TP"] == tp]
            data = data.loc[:, ["Aligned_X", "Aligned_Y", "Aligned_Z"]]
            dic = {}
            for col in data.columns:
                values = np.concatenate(( data[col].to_numpy(), 
                                          np.zeros((maxLength-data.shape[0])) 
                                         ))
                dic[re.split("_", col)[-1]] = values
            savemat(savepath, dic)
    
    def toVTKpolydata(df, TP, savepath, organoid = False):
    #def exportSpotsVTK(self, filename, , df = "spots"):
        """
        Export the given files to a points type .vtk for visualization in 
        paraview.

        Parameters
        ----------
        filename : str or list
            Name of the image file or list of the name of the image files. 
            Do not include the '.tif'

        organoid : bool, optional (default is False)
            True : Export only the spots that are supposed to be in the 
            organoid.        

        """
        # Exporting all spots, frame by frame, by calling back the method.
        if TP == "all":
            for tp in df["TP"].unique().tolist():
                export.toVTKpolydata(df, tp, organoid)
        
        #Exporting the spots from the given frames, by calling back the method.
        elif type(TP) == list:
            for tp in TP:
                export.toVTKpolydata(df, tp, organoid)
        
        # Actual export of the spots for the given file.
        elif type(TP) == int:

            subdf = df[df["TP"] == tp]
                
            # Selecting spots if available and if the user wants it.
            if "F_SELECT" in subdf.columns and organoid:
                subdf = subdf[subdf["IS_ORGANOID"]]
                
        """
        Writer to convert a series of coordinates into a polydata .vtk file.

        """

        file = open(savepath, "w")
        
        # Removing columns (tracks) containing nan values.
        subdf['X'].dropna(axis = "columns", inplace = True)
        subdf['Y'].dropna(axis = "columns", inplace = True)
        subdf['Z'].dropna(axis = "columns", inplace = True)
        
        #Getting the number of columns i.e. the number of different tracks.
        points_nb = len(subdf['X'].columns)
        
        #Getting the number of points per tracks.
        tp_nb = len(subdf['X'].index)
        
        #Writing the file specifications
        file.write("# vtk DataFile Version 2.0\n")
        file.write("PIV3D Trajectories\n")
        file.write("ASCII\n")
        file.write("DATASET POLYDATA\n")
        file.write("POINTS "+str(points_nb*tp_nb)+" double\n")
        
        #For every tracks and every timepoint (row), we write the coordinates 
        #of the given point in the file.
        for pt in range(points_nb):
            for tp in range(tp_nb):
                X = subdf['X'].iloc[tp, pt] 
                Y = subdf['Y'].iloc[tp, pt]                                
                Z = subdf['Z'].iloc[tp, pt]
                file.write(str(X)+" "+str(Y)+" "+str(Z)+"\n")
                    
        #Writing the number of lines and the number of points.
        file.write("LINES "+str(points_nb)+" "+str((tp_nb+1)*points_nb)+"\n" )
        file.write("\n")
        
        #Writing some more informations
        idx = 0;
        for pt in range(points_nb):
            file.write(str(tp_nb)+" \n")
            for tp in range(tp_nb):
                file.write(str(idx)+" \n")
                idx += 1
            file.write("\n")

        file.write("POINT_DATA "+str(points_nb*tp_nb)+"\n")
        file.write("SCALARS index int 1\n")
        file.write("LOOKUP_TABLE default\n")
        file.write("\n")
        for pt in range(points_nb):
             for tp in range(tp_nb):
                file.write(str(tp)+" \n")
                
        #Closing the file and saving it.        
        file.close()
        
    def toVTKpoints(df, savepath):
        """
        Writer to convert a series of coordinates into a points .vtk file.

        """
        #Setting the name of the file and opening it.
        file = open(savepath, "w")
        
        #Getting the number of columns i.e. the number of different tracks.
        points_nb = len(df['X'].index)
        file.write("# vtk DataFile Version 2.0\n")
        file.write("PIV3D Trajectories\n")
        file.write("ASCII\n")
        file.write("DATASET POLYDATA\n")
        file.write("POINTS "+str(points_nb)+" double\n")
        
        #For every tracks and every timepoint (row), we write the coordinates 
        #of the given point in the file.
        for pt in range(points_nb):
            file.write(str(df['X'].iloc[pt])+" "+\
                       str(df['Y'].iloc[pt])+" "+\
                       str(df['Z'].iloc[pt])+"\n")
        #Closing the file and saving it.
        file.close()
        