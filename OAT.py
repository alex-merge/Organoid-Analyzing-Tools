# -*- coding: utf-8 -*-
"""
organoid_tracking_tools (OAT) is a set of methods that integrates FIJI's 
Trackmate csv files output to process cell displacement within an organoid.

@author: Alex-932
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
    
    def __init__(self, fiji_dir = r"C:\Apps\Fiji.app", wrk_dir = None):
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
        # Creating the directory table
        self.dir = pd.Series(dtype="str")
        
        # Adding the fiji directory
        if not os.path.exists(fiji_dir):
            raise FileNotFoundError("No such fiji directory")
        self.dir["fiji"] = fiji_dir
        
        # Adding the working directory
        if wrk_dir == None :
            self.dir["root"] = os.getcwd()
        else :
            if not os.path.exists(wrk_dir):
                os.makedirs(wrk_dir)
            self.dir["root"] = wrk_dir
            
        # Creating the file table    
        self.files = pd.DataFrame(dtype="str")
        
        # Building the directories tree
        self.dir = filemanager.buildTree(self.dir["root"], fiji_dir)
        
        self.version = "0.7"

    def vectors_analysis(self):
        self.tracks, self.data = vectors.full_analysis(self.dir["tracks"])
        figures.show_angular_velocity(self.tracks, self.data, 
                                      savepath = self.dir["avFigs"])
    
    def timeplapse_cleaning(self):
        pass
    
                 
if __name__ == "__main__":
    T = OAT(fiji_dir = r"C:\Apps\Fiji.app", wrk_dir = r"D:\Wrk\Datasets\4")
    #T.loadTif()
    #T.getROI()
    #T.getVectors(filtering = False, rescaling = [1, 1, 4])
    #T.computeStats()
    # T.showData()
    # T.animVectors()
    S = OAT(wrk_dir = r"D:\Wrk\Datasets\S5")
    #S.getVectors()
    #S.alignRotAxis()
    #S.computeAngularVelocity()
    #S.loadTif()
    #T.getROI()
    #S.getVectors(filtering = False)
        