# -*- coding: utf-8 -*-
"""
Compute methods for OAT.

@author: Alex-932
@version: 0.7
"""

import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import cv2
from skimage import io
import numpy as np
from modules.utils.tools import tools
from modules.utils.clustering import *

class compute():
    
    def links(df):
        """
        Generate a Series containing , for each track (row), the list of 
        spots ID that they are composed of, in chronological order.
        The Series is called self.spotsLinks.

        """
        
        # Getting the target for each spots.
        links = df["TARGET"]
        
        # Every Nan means that the spots has no target, so it is the last one.
        # We will build the lists by going backward.
        enders = links[links.isnull()].index
        
        # Creating the final list and creating the sublists with the enders ID.
        spotsLinks = [[ID] for ID in enders]
        
        # Looping the spots until no backward link is established.
        unfinished = True
        
        while unfinished:
            unfinished = False
            
            for track in spotsLinks:
                # Trying to add the ID of the previous spots from the last spot
                # in the sublist.
                try : 
                    track.append(links[links == track[-1]].index[0])
                    # If it works, then there is a connection.
                    unfinished = True
                except :
                    pass
        
        # Reversing each sublist.
        for track in spotsLinks :
            track.reverse()
        
        # Saving the info.
        spotsLinks = pd.Series(spotsLinks, 
                               index = [df.loc[idx[0]]["TRACK_ID"] 
                                        for idx in spotsLinks])
    
        return spotsLinks
    
    def vectors(df, filtering = False, center = None, 
                rescaling = [1, 1, 1], inplace = True):
        """
        Compute displacement vectors for every spots in the tracks dataframe 
        and add them to it.
        
        Vectors are computed based on the sequence of the track they're in.
        That means that vectors are computed between 2 following spots of a 
        given track.
        
        They are saved in the same line as the origin spot of the vector.
        
        Parameters
        ----------
        filtering : bool, optional
            If True, use computeClusters() on tracks dataframe and keep the 
            selected ones (F_SELECT = True).
        aligned : bool, optional
            If True, compute displacement vectors for all aligned coordinates.
            See self.alignRotAxis().
        rescaling : list, optional
            Scaling factor for the coordinates. List must be as follow :
            [Xfactor, Yfactor, Zfactor].
    
        """
        # Creating a dataframe to store vectors as they are computed.
        # Each row is a vector with its index being the ID of its origin spot.
        vectors = pd.DataFrame(columns = ["uX", "vY", "wZ"], 
                               dtype = "float")
        
        spotsLinks = compute.links(df)
        
        # Computing vectors, track by track.
        for trackID in spotsLinks.index :
            for spot in range(len(spotsLinks[trackID])-1) :
                
                # Retrieving spot and target spot IDs.
                spotID = spotsLinks[trackID][spot]
                nextSpotID = spotsLinks[trackID][spot+1]

                # Retrieving the spot and target spot coordinates                    
                spotInfo = df.loc[spotID, ["X", "Y", "Z"]]
                nextSpotInfo = df.loc[nextSpotID, ["X", "Y", "Z"]]

                # Adding the computed vectors to the vectos dataframe.
                vectors.loc[spotID] = tools.displacement_vector(spotInfo, 
                                                                nextSpotInfo,
                                                                toList = True)
                
            # Adding a null vector as the last spot don't have any vector.
            vectors.loc[spotsLinks[trackID][-1]] = 3*[np.nan]
            
        ## Clustering and removing bad spots in df dataframe if it 
        ## hasn't already been done.     
        if filtering and center is not None:
            clustering.computeClusters(df, center)
            selected = df[df["F_SELECT"]].index
            vectors = vectors.loc[selected]
        
        if inplace :
            return pd.concat([df, vectors], axis = 1)
        
        else :
            return vectors
    
    def voxels(df, filepath, offset = 10, outerThres = 0.9):
        """
        Compute voxels of cells. For each time point, the corresponding image
        is loaded as an array. We get a threshold by getting the minimal pixel
        value for the pixels that are at spots coordinates, for a given 
        time point. 

        Parameters
        ----------
        TP : int or list, optional
            Time points to compute. The default is "all".
        offset : int, optional
            Added value to the min and max to make sure everything is inside. 
            The default is 10.
        outerThres : float, optional
            Threshold determining that a pixels is inside the voxel and need to
            be set to 0. Used to improve plotting speed. The default is 0.9.

        """
        # Opening the image.
        image = io.imread(filepath)
        
        # Converting into a Numpy array. The shape is in this order : Z, Y, X.
        imarray = np.array(image)
        
        # Getting the minimal value of pixels at spots coordinates.
        df = df.astype("int")
        values = imarray[df["Z"], df["Y"], df["X"]].tolist()
        minimal = min(values)
        
        # Setting the outside as 0 and the inside as 1.
        imarray[imarray < (minimal-offset)] = 0
        imarray[imarray >= (minimal-offset)] = 1
        
        # Transposing the array that way it is oriented the same as the 
        # other objects (spots, vectors).
        imarray = np.transpose(imarray, (2, 1, 0))
        
        # Setting the inner pixels to 0 as well to only get the outer 
        # shell.
        toChange = []
        for x in range(1, imarray.shape[0]-1):
            for y in range(1, imarray.shape[1]-1):
                for z in range(1, imarray.shape[2]-1):
                    
                    # Getting the 3*3 square array centered on (x,y,z). 
                    neighbors = imarray[x-1:x+2, y-1:y+2, z-1:z+2]
                    
                    # Summing the values and if the number of ones is 
                    # greater than the threshold, saving the pixel coord.
                    if neighbors.sum()/27 >= outerThres :
                        toChange.append([x, y, z])
        
        # Setting the values of the selected pixels to 0.
        for coord in toChange:
            imarray[coord[0], coord[1], coord[2]] = 0
    
        return imarray
    
    def drift(df):
        """
        Compute the drift of the organoid between time points.

        """
        
        data = pd.DataFrame(index = df["TP"].unique().tolist())
        
        # Creating DataFrames to hold computation results. 
        centroids = pd.DataFrame(dtype = "float", 
                                 columns = ["Cent_X", "Cent_Y", "Cent_Z"])
        
        drift = pd.DataFrame(dtype = "float", columns = ["drift_distance"])
        
        vectors = pd.DataFrame(dtype = "float", 
                               columns = ["drift_uX", "drift_vY", "drift_wZ"])
        
        # Iterating over time point.
        for tp in df["TP"].unique().tolist():
            
            # Extracting the subdataframe containing the information for a 
            # given file.
            subdf = df[df["TP"] == tp]
            
            # Getting the centroid for this tp.
            centroids.loc[tp] = tools.getCentroid(subdf.loc[:,["X", "Y", "Z"]])
            
            # If we're not at the first file, we can compute the drift vector 
            # between the centroids from the n-1 and n time point.
            # The drift vector is saved with the n-1 time point index. 
            if tp >= 1 :
                vectors.loc[tp-1] = tools.computeVectors(centroids.iloc[tp-1],
                                                       centroids.iloc[tp],
                                                       toList = True)
                
                drift.loc[tp] = tools.euclidDist(centroids.iloc[tp], 
                                                       centroids.iloc[tp-1])
            
        # Merging the several dataframes to data.
        data = pd.concat([data, centroids], axis = 1)
        data = pd.concat([data, vectors], axis = 1)
        data = pd.concat([data, drift], axis = 1)

        return data
        
    def volume(df):
        """
        Use the Convex Hull algorithm of scipy to get the cells that are 
        forming the outershell as well as the volume of the organoid. 

        """       
        data = pd.DataFrame(index = df["TP"].unique().tolist())
        
        # Creating 2 buffer lists.
        volume, spots = [], []
        
        # Iterating over files.
        for tp in df["TP"].unique().tolist():
            
            # Getting the sub dataframe.
            subdf = df[df["TP"] == tp]
            
            # Using the Convex Hull algorithm.
            hull = ConvexHull(subdf.loc[:,["X", "Y", "Z"]])
            
            # Saving the volume and the spots that are the outershell.
            volume.append(hull.volume) 
            spots += list(subdf.iloc[hull.vertices.tolist()].index)
            
        # Setting the bool value for the question : is this spot part of the 
        # outershell ?
        isHull = pd.Series(name = "isHull", dtype="bool")
        for idx in df.index :
            if idx in spots:
                isHull[idx] = True
            else :
                isHull[idx] = False
                
        # Merging the bool isHull to df.
        df = pd.concat([df, isHull], axis = 1)
        
        # Converting the volume list to a Series to add time point informations
        volume = pd.Series(volume, index = df["TP"].unique().tolist(), 
                           name = "volume", dtype = "float")
        
        # Adding volume and mean radius to data
        data = pd.concat([data, volume], axis = 1)
        data["radius"] = [(3*V/(4*np.pi))**(1/3) for V in data["volume"]]
        
        return data
        
    def translation(df, inplace = True):
        """
        Translate the coordinates of all points within df to get the
        centroid of the organoid at [0, 0, 0].
        The translated coordinates are added to df.

        """
        # Setting the center coordinates.
        center = pd.Series([0, 0, 0], index = ["X", "Y", "Z"])
        
        # Creating a DataFrame to store the translated coordinates.
        coords = pd.DataFrame(columns = ["Trans_X", "Trans_Y", "Trans_Z"], 
                              dtype = "float")
        
        # Iterating over time points.
        for tp in df["TP"].unique().tolist():
            
            # Getting the wanted rows.
            subdf = df[df["TP"] == tp]
            
            # Computing the centroid as well as the translation between the 
            # centroid and the center coordinates.
            centroid = tools.getCentroid(subdf.loc[:,["X", "Y", "Z"]])
            translation = tools.displacement_vector(center, centroid)
            
            # Creating a buffer list to store new coordinates.
            new_coords = []
            
            # Iterating over spot IDs and computing the translated coordinates.
            for spot in subdf.index:
                new_coords.append([subdf.loc[spot, "X"]-translation["X"],
                                  subdf.loc[spot, "Y"]-translation["Y"],
                                  subdf.loc[spot, "Z"]-translation["Z"]])
            
            # Adding the translated coordinates to the DataFrame.
            coords = pd.concat([coords, 
                                pd.DataFrame(new_coords, index = subdf.index, 
                                             columns = ["Trans_X", "Trans_Y", 
                                                        "Trans_Z"], 
                                             dtype = "float")])
        if inplace :
            return pd.concat([df, coords], axis = 1)
        
        else : 
            return coords
    
    def rotation_axis(df):
        """
        Compute the rotation axis of the dataset, at each time point.
        Update data with the colinear vectors of the rotation axis.

        """
        
        data = pd.DataFrame(index = df["TP"].unique().tolist())
            
        # Creating a dataframe to store both vectors forming the PCA plane as
        # well as the crossproduct of those 2.
        componentVectors = pd.DataFrame(columns = ["V1_uX", "V1_vY", "V1_wZ",
                                                   "V2_uX", "V2_vY", "V2_wZ",
                                                   "RA_uX", "RA_vY", "RA_wZ"], 
                                        dtype = "float")
        
        # Iterating over time points.
        for tp in df["TP"].unique().tolist():
            subdf = df[df["TP"] == tp].loc[:, ["uX", "vY", "wZ"]]
            subdf = subdf.dropna()
            
            # Checking if the dataframe is empty meaning we can't compute the 
            # axis.
            if not subdf.empty:   
            
                pca = PCA(n_components = 2)
                pca.fit(subdf.loc[:, ["uX", "vY", "wZ"]])
                
                V1 = pca.components_[0]
                V2 = pca.components_[1]
                
                # Creating a temporary df for crossProduct().
                tempDF = pd.DataFrame(pca.components_, 
                                      columns = ["uX", "vY", "wZ"])
                
                # Computingthe crossproduct.
                RA = list(tools.cross_product(tempDF))
                
                # Saving coordinates to th dataframe.
                componentVectors.loc[tp] = [V1[0], V1[1], V1[2], 
                                            V2[0], V2[1], V2[2],
                                            RA[0], RA[1], RA[2]]
        
        # Merging componentVectors with data.
        data = pd.concat([data, componentVectors], axis = 1)
        
        return data
        
    def alignment(df, data, inplace = True):
        """
        Rotate the points of df to get the axis of rotation aligned 
        with the Z axis. New coordinates are saved in df in 
        "Aligned_..." columns.

        """
        
        # # Running required functions
        # if not "Cent_X" in data.columns :
        #     compute.computeDrift()
        # if not "RA_uX" in data.columns :
        #     compute.computeRotationAxis()
        # if not "Trans_X" in df.columns :
        #     compute.translateCoord()
            
        # Trying to align all rotation axis vectors with Z.
        # First aligning with X to get 0 on the Y axis.
        newCoords = df.loc[:, ["Trans_X", "Trans_Y", "Trans_Z"]]
        newCoords.columns = ["X", "Y", "Z"]
        
        newRA = data.loc[:, ["RA_uX", "RA_vY", "RA_wZ"]]
        
        transAngles = pd.DataFrame(columns = ["Theta_X", "Theta_Y"],
                                   dtype = "float")
        
        for tp in df["TP"].unique().tolist() :
            data = newRA.loc[tp]
            
            coord = [data["RA_uX"],
                     data["RA_vY"],
                     data["RA_wZ"]]
            
            theta_x = abs(np.arctan2(coord[1], coord[2]))%np.pi
            transAngles.loc[tp, "Theta_X"] = theta_x
            
            # Applying X rotation
            ycoord = coord[1].copy()
            coord[1] = coord[1]*np.cos(theta_x)-coord[2]*np.sin(theta_x)
            coord[2] = ycoord*np.sin(theta_x)+coord[2]*np.cos(theta_x)
            
            theta_y = abs(np.arctan2(-coord[0], coord[2]))%np.pi
            transAngles.loc[tp, "Theta_Y"] = theta_y
            
            # Applying Y rotation
            xcoord = coord[0].copy()
            coord[0] = coord[0]*np.cos(theta_y)+coord[2]*np.sin(theta_y)
            coord[2] = -xcoord*np.sin(theta_y)+coord[2]*np.cos(theta_y)
            
            newRA.loc[tp] = coord
        
        newRA.columns = ["Aligned_RA_uX", "Aligned_RA_vY", "Aligned_RA_wZ"]
        
        for ID in newCoords.index :
            
            coord = [newCoords.loc[ID, "X"],
                     newCoords.loc[ID, "Y"],
                     newCoords.loc[ID, "Z"]]
            
            tp = df.loc[ID, "TP"]
            
            theta_x = transAngles.loc[tp, "Theta_X"]
            
            ycoord = coord[1].copy()
            coord[1] = coord[1]*np.cos(theta_x)-coord[2]*np.sin(theta_x)
            coord[2] = ycoord*np.sin(theta_x)+coord[2]*np.cos(theta_x)
            
            theta_y = transAngles.loc[tp, "Theta_Y"]
            
            xcoord = coord[0].copy()
            coord[0] = coord[0]*np.cos(theta_y)+coord[2]*np.sin(theta_y)
            coord[2] = -xcoord*np.sin(theta_y)+coord[2]*np.cos(theta_y)
            
            newCoords.loc[ID] = coord
        
        newCoords.columns = ["Aligned_X", "Aligned_Y", "Aligned_Z"]
        
        if inplace :
            return (pd.concat([df, newCoords], axis = 1), 
                    pd.concat([data, newRA], axis = 1))
        
        else :
            return newCoords, newRA
        
    def angular_velocity(df, data, inplace = True):
        
        subdf = df.copy()
        
        angularVelocity = pd.Series(dtype = "float", name = "Angular_Velocity")
        distance = pd.Series(dtype= "float", name = "Distance_rotAxis")
        
        for ID in subdf.index:
            
            distance[ID] = tools.euclid_distance([0, 0], 
                                          list(subdf.loc[ID, ["Aligned_X",
                                                              "Aligned_Y"]]))
            
            RAvect = data.loc[subdf.loc[ID, "TP"], ["Aligned_RA_uX",
                                                         "Aligned_RA_vY",
                                                         "Aligned_RA_wZ"]]
            RAvect.index = ["uX", "vY", "wZ"]
            
            dispVector = subdf.loc[ID, ["Aligned_uX", "Aligned_vY",
                                        "Aligned_wZ"]]
            
            dispVector.index = ["uX", "vY", "wZ"]
            
            vectors = pd.concat([dispVector.to_frame().T, RAvect.to_frame().T])
            
            velocity = abs((tools.cross_product(vectors)**2).sum())**(1/2)/(
                       distance[ID])
            
            angularVelocity[ID] = velocity
        
        velocityByCell = pd.concat([angularVelocity.to_frame(),
                                    distance.to_frame(), df["TP"].to_frame()],
                                   axis = 1)
        
        velocityByTP = pd.DataFrame(columns = ["Mean_AV", "Std_AV"],
                              index = data.index, dtype = "float")
        
        for tp in velocityByTP.index:
            subdf = velocityByCell[velocityByCell["TP"] == tp].copy()
            mean = subdf["Angular_Velocity"].mean()
            std = subdf["Angular_Velocity"].std()
            
            velocityByTP.loc[tp] = [mean, std]
            
        velocityByCell.drop(columns = "TP", inplace = True)
        
        if inplace :
            return pd.concat([data, velocityByTP], axis = 1),\
                    pd.concat([df, velocityByCell], axis = 1)
        
        else :
            return velocityByTP, velocityByCell