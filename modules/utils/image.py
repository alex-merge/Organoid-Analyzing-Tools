# -*- coding: utf-8 -*-
"""
Methods for OAT to work with images.

@author: Alex-932
@version: 0.7
"""

from skimage import io
import tifffile
import cv2
import numpy as np
import os

class image():
    
    def load_tif(filepath):
        """
        Load tifs in the directory or the specific given file and returns it
        as an array.

        Parameters
        ----------
        filepath : str
            Path to a file or a folder containing images to import.

        Raises
        ------
        IOError
            Path does not exist.

        Returns
        -------
        array
            Array containing the or all images.

        """
        ## Checking what is the target of the path and getting the full 
        ## file path in each case.
        if os.path.isdir(filepath) :
            files = [filepath+"\\"+filename 
                     for filename in os.listdir(filepath)]
            
        elif os.path.isfile(filepath) :
            files = [filepath]
        
        else :
            raise IOError("Path does not exist")
            
        fullArray = []
        
        ## Loading and returning the images as arrays.
        for file in files :
            stream = io.imread(file)
            imarray = np.array(stream)
            
            fullArray.append(imarray)
            
        if len(fullArray) > 1 :
            return np.array(fullArray)
        else :
            return fullArray[0]
                
        
    
    def unstack_tif(filepath, suffix, savedir):
        """
        Unstack an image in the given save directory.
    
        Parameters
        ----------
        filepath : str
            Path to the image file.
        suffix : str
            Suffix name for the output images.
        savedir : str
            Save directory.
    
        """
        ## Loading the image array.
        imarray = image.loadTif(filepath)
        
        ## Browsing the image through the time axis
        for tp in range(imarray.shape[0]):
            image.saveTif(imarray[tp], savedir+"\\"+suffix+"_"+str(tp)+".tif")
            
    def stack_tif(filepath, savepath):
        """
        Stack multiple tif into one.

        Parameters
        ----------
        filespath : str
            Folder path to the images to stack.
        savepath : str
            Saving path containing the name of the stacked tif file.

        """
            
        imarray = image.loadTif(filepath)
        image.saveTif(imarray, savepath)
        
    def denoise(ROI, imarray):
        """
        Load a tif image and set all pixels that are not in the ROI to 0.
        Used in the cleanImage method.

        Parameters
        ----------
        ROI : pd.Series
            Formatting is the same as self.ROI : Index are 
            ["X_min", "X_max", "Y_min", "Y_max", "Z_min", "Z_max"].
        filepath : str
            Path to the image.

        Returns
        -------
        imarray : np.array
            Denoised array of the image.

        """
        ## For each axis, we get the coordinate (1D) of the pixels that needs  
        ## to be set to 0. 
        X_values = [X for X in range(imarray.shape[2]) if
                     X < ROI["X_min"] or X > ROI["X_max"]]
        Y_values = [Y for Y in range(imarray.shape[1]) if
                     Y < ROI["Y_min"] or Y > ROI["Y_max"]]
        Z_values = [Z for Z in range(imarray.shape[0]) if
                     Z < ROI["Z_min"] or Z > ROI["Z_max"]]
        
        ## Setting the given pixel to 0 on each axis.
        imarray[Z_values,:,:] = 0
        imarray[:,Y_values,:] = 0
        imarray[:,:,X_values] = 0
        
        return imarray
    
    def save_tif(imarray, savepath):
        """
        Save the given array as a tif file in the given directory.

        Parameters
        ----------
        imarray : numpy.array
            Image array.
        savepath : str
            Full path of the output file.

        """
        if len(imarray.shape) == 3 :
            tifffile.imwrite(savepath, imarray, imagej = True,
                             metadata={'axes': "ZYX"})
            
        elif len(imarray.shape) == 4 :
            tifffile.imwrite(savepath, imarray, imagej = True,
                             metadata={'axes': "TZYX"})
            
    def get_shape(dirpath):
        """
        Return the shapes of the 3D tifs in the given directory.
        Basically, it loads every 3D tifs into a 4D tifs and return the 
        3 wanted dimensions. 

        Parameters
        ----------
        dirpath : str
            Directory of the images to get the shape from.

        Returns
        -------
        list.
            Shapes for the given dimensions : [X, Y, Z]

        """
        imarray = image.load_tif(dirpath)
        return [imarray.shape[-3], imarray.shape[-2], imarray.shape[-1]]
    
    def get_center(dirpath):
        """
        Return the center point of the 3D tifs in the given directory.

        Parameters
        ----------
        dirpath : str
            Directory of the images to get the center point from.

        Returns
        -------
        list.
            Coordinates of the center [X, Y, Z]

        """
        shapes = image.get_shape(dirpath)
        return [dim/2 for dim in shapes]
        