# -*- coding: utf-8 -*-
"""
organoid_tracking_tools (OAT) is a set of methods that integrates FIJI's 
Trackmate csv files output to process cell displacement within an organoid.

@author: alex-merge
@version: 0.7
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from skimage import io
import tifffile
import cv2
import time

from modules.utils.clustering import clustering
from modules.utils.tools import tools
from modules.utils.compute import compute
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
            
        self.version = "0.7"
        
    def info(self):
        print("Organoid Analyzing Tools")
        print("Version : "+self.version)
        
    def load_spot_data(self, dirpath):
        if not isinstance(dirpath, str) or not filemanager.check_folder(dirpath):
            raise ValueError("dirpath must be a path")
        
        self.dirpath = dirpath
        
        
        
    def load_vector_data(self, dirpath, datatype = "auto", 
                         rescaling = [1, 1, 1], filtering = False):
 
        
        if not isinstance(dirpath, str) or not filemanager.check_folder(dirpath):
            raise ValueError("dirpath must be a path")
            
        self.dirpath = dirpath
        
        if not isinstance(datatype, str):
            raise TypeError("data_type must be string type")   
            
        elif datatype == "trackmate":
            self.tracks = vectors.load_from_trackmate(dirpath, rescaling, 
                                                      filtering)
            
        elif datatype == "quickPIV":
            self.tracks = vectors.load_from_quickPIV(dirpath)
            
        elif datatype == "auto":
            vtk_files = filemanager.search_file(dirpath, "vtk")
            csv_files = filemanager.search_file(dirpath, "csv")
            
            if len(vtk_files) != 0 and len(csv_files) == 0:
                self.tracks = vectors.load_from_quickPIV(dirpath)
                
            elif len(vtk_files) == 0 and len(csv_files) != 0:
                self.tracks = vectors.load_from_trackmate(dirpath, rescaling, 
                                                          filtering)
            
            else :
                raise ImportError("Unable to assess the datatype")
                
        else :
            raise ValueError("Unrecognized datatype")

    def vectors_analysis(self, savepath = None):
        if savepath is None:
            savedir = self.dirpath+"\\figures\\"
            filemanager.check_folder(savedir, create = True)
            
            savepath = savedir+"\\Analysis_over_time.png"
        
        if not hasattr(self, "tracks"):
            raise ImportError("No data has been loaded")
            
        self.tracks, self.data = vectors.full_analysis(self.tracks)
        
        figures.show_angular_velocity(self.tracks, self.data, 
                                      savepath = savepath)
    
    def timeplapse_cleaning(self):
        pass
    
                 
if __name__ == "__main__":
    pass
        