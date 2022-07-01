"""
Trackmate automated script.

@author: Alex-932
@version: 1.3 (15/06/22)
@python: 2.7
"""

import sys
import os
import re
 
from ij import IJ
from ij import WindowManager
 
from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import SelectionModel
from fiji.plugin.trackmate import Logger
from fiji.plugin.trackmate.detection import LogDetectorFactory
from fiji.plugin.trackmate.tracking import LAPUtils
from fiji.plugin.trackmate.tracking.sparselap import SparseLAPTrackerFactory
from fiji.plugin.trackmate.gui.displaysettings import DisplaySettingsIO
import fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer as HyperStackDisplayer
import fiji.plugin.trackmate.features.FeatureFilter as FeatureFilter

reload(sys)
sys.setdefaultencoding('utf-8')

# Reading the instruction file
_file = open("OAT_instructions.txt")
instruc_list = _file.readlines()
#We remove the "\n" at the end of each line
instruc_list = [k[:-1] for k in instruc_list] 
#The formatting of the instructions are as bellow:
#   Each row correspond to an image file.
#   The row is seperated in half by a comma.
#   Before the comma is the input image file path.
#   After the comma is the output .csv file path. 

for file in range(len(instruc_list)):
    _split = re.split(",",instruc_list[file])
    _img = _split[0]
    _dir = _split[1]
    
    imp = IJ.openImage(_img)
    #imp.show()
     
     
    #----------------------------
    # Create the model object now
    #----------------------------
     
    # Some of the parameters we configure below need to have
    # a reference to the model at creation. So we create an
    # empty model now.
     
    model = Model()
     
    # Send all messages to ImageJ log window.
    #model.setLogger(Logger.IJ_LOGGER)
     
     
     
    #------------------------
    # Prepare settings object
    #------------------------
     
    settings = Settings(imp)
     
    # Configure detector - We use the Strings for the keys
    settings.detectorFactory = LogDetectorFactory()
    settings.detectorSettings = {
        'DO_SUBPIXEL_LOCALIZATION' : False,
        'RADIUS' : 8.,
        'TARGET_CHANNEL' : 1,
        'THRESHOLD' : 500.,
        'DO_MEDIAN_FILTERING' : True,
    }  
    
    # Following configurations doesn't matter because the tracker will not 
    # track sttic cells (given the file is just a picture at one timepoint).
    # Yet we need to set it to make sure trackmate do its things correctly.
    # Configure tracker - We want to allow merges and fusions
    settings.trackerFactory = SparseLAPTrackerFactory()
    settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap() # almost good enough
    settings.trackerSettings['ALLOW_TRACK_SPLITTING'] = True
    settings.trackerSettings['ALLOW_TRACK_MERGING'] = True
     
    # Add ALL the feature analyzers known to TrackMate. They will 
    # yield numerical features for the results, such as speed, mean intensity etc.
    settings.addAllAnalyzers()
     
    #-------------------
    # Instantiate plugin
    #-------------------
     
    trackmate = TrackMate(model, settings)
     
    #--------
    # Process
    #--------
     
    ok = trackmate.checkInput()
    if not ok:
        sys.exit(str(trackmate.getErrorMessage()))
     
    ok = trackmate.process()
    if not ok:
        sys.exit(str(trackmate.getErrorMessage()))
    
    ## Saving the results in the given file in its given path.
    # Iterate over all the spots that are visible.
    # spots is a SpotCollection, it is an object that contains the spots.
    spots = model.getSpots()
    _list = []
    #To access every datas from every spots, we use the iterator.
    for spot in spots.iterator(True):
    	x = spot.getFeature('POSITION_X')
    	y = spot.getFeature('POSITION_Y')
    	z = spot.getFeature('POSITION_Z')
    	t = spot.getFeature('FRAME')
    	q = spot.getFeature('QUALITY')
    	_list.append(str(spot)+","+str(x)+","+str(y)+","+str(z)+","+\
                  str(q)+"\n")
    #We save it just as a txt file given writing .csv using the csv package is
    #garbage in python 2.7.
    header= "LABEL,POSITION_X,POSITION_Y,POSITION_Z,QUALITY\n"
    _file = open(_dir, 'w')
    _file.write(header)
    #Adding 3 empty lines to match the manual output file which have 3 repeated
    #lines at the beginning (Easing up post processing).
    _file.write("\n")
    _file.write("\n")
    _file.write("\n")
    _file.writelines(_list)
    _file.close()

print("Job finished !")