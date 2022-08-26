# -*- coding: utf-8 -*-
"""
organoid_tracking_tools (OAT) is a set of methods that integrates FIJI's 
Trackmate csv files output to process cell displacement within an organoid.

@author: alex-merge
@version: 0.8
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from modules.utils.filemanager import filemanager
from modules.export import export
from modules.utils.figures import figures
from modules.vectors import vectors
from modules.preprocessing import preprocessing

class OAT():
    
    def __init__(self):
        """
        Initialize the sample analysis by creating the directories needed.

        Parameters
        ----------
        fiji_dir : str
            Directory path for Fiji.app folder.
        wrk_dir : str, optional
            Working directory i.e. the dir. where the tree will be built. 
            The default is the script directory.

        """
        self.version = "0.8"
        
    def info(self):
        """
        Simply print the version of OAT.

        """
        print("Organoid Analyzing Tools")
        print("Version : "+self.version)
        
    def load_spot_data(self, dirpath, rescaling = [1, 1, 1]):
        """
        Load the segmentation data for the timelapse cleaning pipeline.
        Data is loaded into a pandas dataframe called self.spots. 

        Parameters
        ----------
        dirpath : str
            Path to the folder containing csv segmentation data from executing
            trackmate_script.py.
        rescaling : list of float, optional
            List with rescaling factors for each axis. 
            The default is [1, 1, 1].

        """
        ## Checking if dirpath exist
        if not isinstance(dirpath, str) or not filemanager.check_folder(dirpath):
            raise ValueError("dirpath must be a path")
        
        self.spot_dirpath = dirpath
        
        ## Loading files
        try :
            self.spots = preprocessing.load_from_trackmate(dirpath, rescaling)
        except :
            raise ImportError("Unable to load files")
        
    def load_vector_data(self, dirpath, datatype = "auto", 
                         rescaling = [1, 1, 1], filtering = False, 
                         prefixes = ["COORD_", "VECT_"]):
        """
        Load tracking data for the vector analysis pipeline.
        Data is loaded into a pandas dataframe called self.tracks. 

        Parameters
        ----------
        dirpath : str
            Path to the folder containing tracking data.
        datatype : str, optional
            Type of data to import. It can be one of this type :
                - tackmate : Load the tracks>spots csv and tracks>edges csv 
                             from Trackmate.
                - quickPIV : Load vtk files from quickPIV.
                - legacy : Load a csv file.
                - auto : (default) try to assume the type and load the file(s).
        rescaling : list of float, optional
            List with rescaling factors for each axis. 
            The default is [1, 1, 1].
        filtering : bool, optional
            If True, cluster the spots to remove those that are not part of the
            organoid. 
            The default is False.
        prefixes : list of str, optional
            In case of legacy type of import, prefixes contains the prefix name
            for the coordinates and displacement vectors.
            For example, the prefix of [C_X, C_Y, C_Z] is "C_".
            The default is ["COORD_", "VECT_"].

        """
 
        
        if not isinstance(dirpath, str) or not filemanager.check_folder(dirpath):
            raise ValueError("dirpath must be a path")
            
        self.tracks_dirpath = dirpath
        
        if not isinstance(datatype, str):
            raise TypeError("data_type must be string type")   
            
        elif datatype == "trackmate":
            self.tracks = vectors.load_from_trackmate(dirpath, rescaling, 
                                                      filtering)
            
        elif datatype == "quickPIV":
            self.tracks = vectors.load_from_quickPIV(dirpath)
            
        elif datatype == "legacy":
            self.tracks = vectors.load_from_csv(dirpath, prefixes = prefixes)
            
        elif datatype == "auto":
            ## Getting the list of csv and vtk files.
            vtk_files = filemanager.search_file(dirpath, "vtk")
            csv_files = filemanager.search_file(dirpath, "csv")
            
            ## If there are only vtk, it must be quickPIV.
            if len(vtk_files) != 0 and len(csv_files) == 0:
                self.tracks = vectors.load_from_quickPIV(dirpath)
            
            ## If there are only csv, trying trackmate then legacy.
            elif len(vtk_files) == 0 and len(csv_files) != 0:
                try :
                    self.tracks = vectors.load_from_trackmate(dirpath, rescaling, 
                                                              filtering)
                except :
                    try : 
                        self.tracks = vectors.load_from_csv(dirpath, 
                                                            prefixes = prefixes)
                    except : 
                        raise ImportError("Unable to assess the datatype")
            else :
                raise ImportError("Unable to assess the datatype")         
        else :
            raise ValueError("Unrecognized datatype")

    def vectors_analysis(self, savepath = None):
        """
        Compute the full OAT analysis on the imported data.

        Parameters
        ----------
        savepath : str, optional
            Path to the folder where figures any export file must be saved. 
            The default is "export" folder created in the input data folder.

        """
        ## Create an export folder in the input data folder if not specified
        if savepath is None:
            savedir = self.tracks_dirpath+"\\export\\"
            filemanager.check_folder(savedir, create = True)
            
            savepath = savedir+"\\Analysis_over_time.png"
        
        ## Check if the data have been imported
        if not hasattr(self, "tracks"):
            raise ImportError("No data has been loaded")
        
        ## Compute the analysis
        self.tracks, self.data = vectors.full_analysis(self.tracks)
        
        ## Draw the final figure
        figures.show_angular_velocity(self.tracks, self.data, 
                                      savepath = savepath)
    
    def timelapse_cleaning(self, img_path, savepath = None):
        """
        Denoise and create a timelapse from tif images.
        Each images is expected to show the organoid for 1 time point.

        Parameters
        ----------
        img_path : str
            Path to the folder containing tif images.
        savepath : str, optional
            Path to the folder where figures any export file must be saved. 
            The default is "export" folder created in the input data folder.

        """
        ## Create an export folder in the input data folder if not specified
        if savepath is None:
            savedir = self.spot_dirpath+"\\export\\"
            filemanager.check_folder(savedir, create = True)
            
            savepath = savedir+"\\Preprocessed_timelapse.tif"
         
        ## Check if the data have been imported    
        if not hasattr(self, "spots"):
            raise ImportError("No data has been loaded")
        
        ## Create the timelapse
        self.spots = preprocessing.denoise_timelapse(self.spots, img_path,
                                                     savepath = savepath)
        
    def export(self, savedir = None):
        """
        Export all dataframes to a csv in savedir.

        Parameters
        ----------
        savedir : str, optional
            Directory path. The default is None.

        """
        ## Indexing all available dataframes
        df = {}
        if hasattr(self, "spots"):
            df["spots"] = self.spots
        if hasattr(self, "data"):
            df["data"] = self.data
        if hasattr(self, "tracks"):
            df["tracks"] = self.tracks
        
        ## Setting the save directory if not given
        if savedir is None:
            if hasattr(self, "spot_dirpath"):
                savedir = self.spot_dirpath
            if hasattr(self, "tracks_dirpath"):
                savedir = self.tracks_dirpath
        
        ## Exporting dataframes using export.to_csv method
        for label in df:
            export.to_csv(df[label], 
                          savepath = savedir+"\\{}.csv".format(label))