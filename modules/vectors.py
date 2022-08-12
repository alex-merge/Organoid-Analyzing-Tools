# -*- coding: utf-8 -*-
"""
Vector based analyzing components of OAT.

@author: Alex-932
@version: 0.7
"""

import os
import pandas as pd
import time

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import seaborn as sns

from modules.utils.filemanager import *
from modules.utils.clustering import *
from modules.utils.compute import *
from modules.utils.tools import *
from modules.utils.figures import *

class vectors():
    
    def read_tracks(dirpath, rescaling = [1, 1, 1]):
        """
        Load tracks file (tracks.csv and edges.csv) in the 
        \\data\\tracks directory as a DataFrame called df. 
        
        tracks.csv correspond to the .csv you can get by saving the dataset 
        found in tracks>spots.
        
        tracks.csv correspond to the .csv you can get by saving the dataset 
        found in tracks>edges.

        """
        
        ## Importing the files.
        files = filemanager.search_file(dirpath, "csv")
        
        attributes = []
        dataframes = []
        
        for filepath in files:
            stream = pd.read_csv(filepath)
            dataframes.append(stream)
            
            ## 1 is the edge dataframe and 2 is the tracks dataframe.
            if "SPOT_SOURCE_ID" in stream.columns :
                attributes.append(1)
                edges_df = stream
                
            elif "QUALITY" in stream.columns :
                attributes.append(2)
                tracks_df = stream
                
            else :
                attributes.append(0)
        
        ## If the sum is different than 3, we have either to many files or 
        ## missing files.
        if sum(attributes) != 3:
            raise IOError("Missing or redundant files")
            
        # Removing the 3 first rows as they are redundant with the labels
        tracks_df.drop(index = [0, 1, 2], inplace = True)
        edges_df.drop(index = [0, 1, 2], inplace = True)
        
        # Setting the spots ID as index in both dataframes.
        tracks_df.index = "ID"+tracks_df["ID"].astype("str")
        edges_df.index = "ID"+edges_df["SPOT_SOURCE_ID"].astype("str")
        
        # Renaming some labels in order to be cleaner
        for axis in ["X", "Y", "Z"]:
            tracks_df.rename(columns = {"POSITION_"+axis:axis},
                                  inplace = True)
        edges_df.rename(columns = {"SPOT_TARGET_ID":"TARGET"}, 
                        inplace = True)
        tracks_df.rename(columns = {"POSITION_T":"TP"}, inplace = True)
        
        # Merging coordinates into an array.
        tracks_df[list("XYZ")] = tracks_df[list("XYZ")].astype("float")
        coords = pd.Series([arr for arr in tracks_df[list("XYZ")].to_numpy()],
                            index = tracks_df.index, name = "COORD")
        tracks_df = pd.concat([tracks_df, coords], axis = 1)
        
        # Keeping the interesting columns 
        tracks_df = tracks_df.loc[:,["TRACK_ID", "QUALITY", "COORD", 
                                     "TP"]]
        edges_df = edges_df.loc[:,"TARGET"]
        
        # Setting the dataframes' values type
        tracks_df[["QUALITY", "TP"]] = tracks_df[["QUALITY", "TP"]
                                                 ].astype("float")
        tracks_df[["TRACK_ID", "TP"]] = tracks_df[["TRACK_ID", "TP"]
                                                  ].astype("int")
        edges_df = edges_df.astype("float")
        
        # Modifying the "Target" values to be spots ID.
        edges_df = "ID" + edges_df.astype(int).astype("str")
        
        # Merging the 2 dataframes.
        tracks = pd.concat([tracks_df, edges_df], axis = 1)
        
        ## Rescaling the coordinates in case we need to.
        tracks["COORD"] = [arr*np.array(rescaling) for arr in tracks["COORD"]]
        
        return tracks
    
    def load_from_trackmate(dirpath, rescaling = [1, 1, 1], filtering = False):
        """
        Load the csv files that have been generated by Trackmate.
        The expected files are the tracks>spots csv and tracks>edges csv.
        Returns a pandas dataframe containing coordinates, 
        displacements vectors as well as clustering information if filtering
        have been set to True.

        Parameters
        ----------
        dirpath : str
            Path to the folder containing csv files.
        rescaling : list, optional
            Rescaling factors by axis : [x, y, z]. The default is [1, 1, 1].
        filtering : bool, optional
            If True, cluster the spots to remove those that are not part of the
            organoid. 
            The default is False.

        Returns
        -------
        df : pandas.DataFrame
            Dataframe containing merged informations of both csv files.
            Can be used for the next part of the pipeline.

        """
        ## Loading vectors.
        step_time = time.time()
        print("Opening files ...", end = " ")
        df = vectors.read_tracks(dirpath)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        
        ## Computing vectors.
        step_time = time.time()
        print("Computing displacement vectors ...", end = " ")
        df = compute.vectors(df, filtering = filtering)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        
        return df
    
    def load_from_quickPIV(dirpath):
        
        filepaths = filemanager.search_file(dirpath, "vtk")
        file_to_idx = {filepaths[k]: k for k in range(len(filepaths))}
        
        df = pd.DataFrame(dtype = "object")
        
        for filepath in filepaths:
            """
            Code provided by @Marc-3d.
            """
            stream = open(filepath, 'r')
    
            ## reading the size of the vector field
            dims, length = [ 0, 0, 0 ], 0
            
            ## Reading the first 8 lines of the vtk file (metadata) 
            for lidx in range(9):
                line = stream.readline()
                
                if "DIMENSIONS" in line:
                    dims = [ int(x) for x in line.split(" ")[1:4] ]
                    
                elif "POINT_DATA" in line:
                    length = int( line.split(" ")[1] )
                    
            ## Creating 3xN arrays (3 dimensions x N vectors) to store the 
            ## vectors of the vector fields
            displacements = np.zeros( (3, length) )
            positions     = np.zeros( (3, length) )
            
            ## Reading the rest of the file. Each line contains one vector.
            row, col, zet = 0, 1, 1;
            for lidx in range(0,length):
    
                line  = stream.readline()
                displ = [ float(x) for x in line.split(" ") ]
                displacements[:,lidx] = displ
    
                ## Transforming from linear index to a 3D coordinate, 
                ## ( row, col, zet ) or ( y, x, z )
                row = row + 1
                if ( row > dims[0] ):
                    row = 1
                    col += 1
                if ( col > dims[1] ):
                    col = 1
                    row = 1
                    zet += 1
                positions[:,lidx] = [ row, col, zet ]

            stream.close()
        
            ## Removing null vectors while creating a dataframe merging both
            ## position and displacements.
            disp = pd.Series(dtype = "object", name = "DISP_VECT")
            pos = pd.Series(dtype = "object", name = "COORD")
            
            for idx in range(displacements.shape[1]):
                if not np.array_equal(displacements[:, idx], np.zeros(3)) :
                    disp.loc[idx] = displacements[:, idx]
                    pos.loc[idx] = positions[:, idx]
            
            
            tp = pd.Series([file_to_idx[filepath]]*disp.shape[0], name = "TP",
                           index = disp.index)
            
            subdf = pd.concat([pos, disp, tp], axis = 1)
            df = pd.concat([df, subdf], axis = 0, ignore_index = True)
            
        return df
    
    def load_from_csv(dirpath = None, filepath = None, coord_prefix = "COORD_"):
        
        if dirpath is not None:
            filepaths = filemanager.search_file(dirpath, csv)
        elif filepath is not None:
            filepaths = [filepath]
        elif dirpath is not None and filepath is not None:
            raise ValueError("Too many inputs : choose between directory and specific file")
        else :
            raise ValueError("No inputs")
            
        final_df = pd.DataFrame(dtype = "object", 
                                columns = ["TP", "TRACK_ID", "COORD"])
            
        for file_id in range(len(filepaths)) :
            stream = pd.read_csv(file)
            
            if not "TP" in stream.columns :
                temp_df = stream["TRACK_ID"].to_frame()
                temp_df = pd.concat([temp_df, 
                                     pd.Series([file_id]*len(temp_df.index))],
                                    axis = 1)
            else :
                temp_df = stream[["TP", "TRACK_ID"]]
            
            coord = pd.Series(dtype = "object")
            
            for idx in stream.index:
                coord.loc[idx] = np.array(
                    stream.loc[idx, [coord_prefix+k for k in list("XYZ")]].tolist()
                    )
                
            temp_df = pd.concat([temp_df, coord], axis = 1)
            
            final_df = pd.concat([final_df, temp_df], axis = 0)
            
        return final_df
            
    
    def full_analysis(df):
        
        start_time = time.time()
        
        ## Computing drift.
        step_time = time.time()
        print("Computing drift ...", end = " ")
        data = compute.drift(df)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        
        ## Computing volume and radius.
        step_time = time.time()
        print("Computing volume and radius ...", end = " ")
        if df.shape[0]/len(df["TP"].unique()) >= 4 :
            data = pd.concat([data, compute.volume(df)], axis = 1)
        else : 
            data = pd.concat([data, pd.Series([0]*df.shape[0], index = df.index,
                                              name = "volume")], axis = 1)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        
        ## Translating coords to the center ([0, 0, 0])
        step_time = time.time()
        print("Translating coordinates to the center ...", end = " ")
        df = compute.translation(df)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        
        ## Computing the axis of rotation vectors.
        step_time = time.time()
        print("Computing the axis of rotations ...", end = " ")
        df, ndata = compute.rotation_axis(df)
        data = pd.concat([data, ndata], axis = 1)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        
        ## Aligning axis of rotation with the Z axis.
        step_time = time.time()
        print("Aligning rotation axis and Z axis ...", end = " ")
        df, data = compute.alignment(df, data)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")

        ## Computing angular velocity.
        step_time = time.time()
        print("Computing angular velocity ...", end = " ")
        df, data = compute.angular_velocity(df, data)
        print("Done !", "("+str(round(time.time()-step_time, 2))+"s)")
        print("")
        print("Analysis done ! Total time :", 
              str(round(time.time()-start_time, 2))+"s")
        
        ## Returning datasets.
        return df, data
    
    
        