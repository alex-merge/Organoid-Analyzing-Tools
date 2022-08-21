# -*- coding: utf-8 -*-
"""
Export methods for OAT.

@author: alex-merge
@version: 0.8
"""
import pandas as pd
import numpy as np

from modules.utils.filemanager import filemanager


class import_data():
    """
    Set of methods to import data for OAT.
    """
    
    def read_spots(dirpath, rescaling = [1, 1, 1]):
        """
        Load segmentation result of Trackmate detection as a dataframe.

        Parameters
        ----------
        dirpath : str
            Path to the directory containing files.
        rescaling : list, optional
            Rescaling factors by axis : [x, y, z]. The default is [1, 1, 1].
        Returns
        -------
        df : pd.DataFrame
            Dataframe where index are spots and columns are 
            ("QUALITY", "COORD", "TP").

        """
        ## Importing the files.
        filespath = filemanager.search_file(dirpath, "csv")

        df = pd.DataFrame(columns=["QUALITY", "X", "Y", "Z"],
                          dtype = "float")
        
        ## For the moment, files are sorted and imported as if the time point 
        ## starts at 0.
        for tp in range(len(filespath)):

            
            ## Reading the CSV.
            stream = pd.read_csv(filespath[tp], index_col = "LABEL")
            
            # Renaming columns to have smaller names.
            stream.rename(columns = {"POSITION_"+axis: axis for axis in 
                                     ["X", "Y", "Z"]},
                          inplace = True)            
            
            # Keeping useful columns.
            stream = stream.loc[:,["QUALITY", "X", "Y", "Z"]]
            
            ## Setting every values to float type.
            stream = stream.astype("float")
            
            ## Adding the time point.
            stream["TP"] = stream.shape[0]*[tp]
            
            ## Merging the dataframe to the main one.
            df = pd.concat([df, stream])
        
        coords = pd.Series(dtype = "object", name = "COORD")
        for index in df.index :

            coords[index] = np.array( df.loc[index, list("XYZ")].tolist() )
        
        df = pd.concat([df, coords], axis = 1)
        df.drop(columns = list("XYZ"), inplace = True)
        
        df["TP"] = df["TP"].astype("int")    
        
        ## Rescaling coordinates
        df["COORD"] = [arr*np.array(rescaling) for arr in df["COORD"]]
        
        return df
    
    
    def read_tracks(dirpath, rescaling = [1, 1, 1]):
        """
        Load tracks file (tracks.csv and edges.csv) in the 
        \\data\\tracks directory as a DataFrame called tracks. 
        
        tracks.csv correspond to the .csv you can get by saving the dataset 
        found in tracks>spots.
        
        tracks.csv correspond to the .csv you can get by saving the dataset 
        found in tracks>edges.
        
        Parameters
        ----------
        dirpath : str
            Path to the directory containing files.
        rescaling : list, optional
            Rescaling factors by axis : [x, y, z]. The default is [1, 1, 1].

        Returns
        -------
        tracks : pd.DataFrame
            Dataframe where index are spots and columns are 
            ("QUALITY", "COORD", "TP").

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