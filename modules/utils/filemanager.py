# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 09:53:34 2022

@author: Alex-932
"""

import pandas as pd
import os
import re
from modules.utils.compute import *

class filemanager():
    
    def search_file(dirpath, extension, fullpath = True):
        """
        Return the list of files that have the given extension, in the given 
        directory.

        Parameters
        ----------
        dirpath : str
            Directory path.
        extension : str
            File extension searched.

        Returns
        -------
        list
            List of filenames.

        """
        if fullpath :
            return [dirpath+'\\'+file for file in os.listdir(dirpath) 
                    if re.split("\.", file)[-1] == extension]
        
        else :
            return [file for file in os.listdir(dirpath) 
                    if re.split("\.", file)[-1] == extension]
        
    def check_folder(dirpath, create = False):
        ## Checking if the folder exist.
        if not os.path.exists(dirpath):
            
            ## If no, proceed to create it if set to True
            if create:
                os.makedirs(dirpath)
                return True
            
            return False
            
        
        else :
            return True
        
    def check_requirements(dataframe, columns):
        """
        Check if the given columns are in the dataframe.
        Raise an error if not.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Dataframe whose columns have to be tested.
        columns : list of str.
            Columns names that have to be in the dataframe.

        Returns
        -------
        bool.
            True if requirements are met, raise an error otherwise. 

        """
        not_in = [column for column in columns\
                  if column not in dataframe.columns]
            
        if len(not_in) != 0:
            col_names = ", ".join(not_in)
            raise AttributeError("Missing informations in the dataframe : "+col_names)
            
        return True
        
    
    def buildTree(wrkdir, fiji_dir):
        """
        Create the directories tree in the working directory.

        Parameters
        ----------
        wrkdir : str
            Path to the dataset root folder.
        fiji_dir : str
            Path to the fiji directory.

        Returns
        -------
        pd.Series.
            Series containing all relevant paths.

        """
        ## Create a pd.Series to save all relevant paths.
        dirpath = pd.Series(name = "Directories", dtype = "str")
        
        ## Setting the paths for the diverse components of the pipeline.
        dirpath["root"] = wrkdir
        dirpath["tifs"] = dirpath["root"]+'\\data\\organoid_images'
        dirpath["spots"] = dirpath["root"]+'\\data\\spots'
        dirpath["tracks"] = dirpath["root"]+'\\data\\tracks'
        dirpath["out"] = dirpath["root"]+'\\output'
        dirpath["figs"] = dirpath["root"]+'\\output\\figs'
        dirpath["spotsFigs"] = dirpath["root"]+'\\output\\figs\\spots'
        dirpath["vectorsFigs"] = dirpath["root"]+'\\output\\figs\\vectors'
        dirpath["distFigs"] = dirpath["root"]+'\\output\\figs\\distance'
        dirpath["avFigs"] = dirpath["root"]+'\\output\\figs\\angularVelocity'
        dirpath["anim"] = dirpath["root"]+'\\output\\animation'
        dirpath["vtk"] = dirpath["root"]+'\\output\\vtk_export'
        dirpath["mat"] = dirpath["root"]+'\\output\\matlab_export'
        
        ## Creating the directories if they don't already exist.
        for path in dirpath:
            if not os.path.exists(path):
                os.makedirs(path)
                
        ## Adding fiji path
        dirpath["fiji"] = fiji_dir
                
        return dirpath
    
    
        