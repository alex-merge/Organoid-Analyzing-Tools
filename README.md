# Organoid Analyzing Tools

## Overview
OAT is a python 3 package that allows you to analyze organoids movements and rotation using tracking data.  
For the moment, OAT is capable of computing :  
	- The drift between time points  
	- The volume for each time points  
	- The main axis of rotation axis for each time points  
	- The alignment of the rotation axis with the Z-axis  
	- The angular velocity for each cell  
	
This package, while able to give some figures of the results, also save all computed data into 2 pandas dataframes for downstream analysis.

The main purpose of this package is to analyze displacement vectors but,
it also gives you to preprocess your 3D data with some tools :  
	- Filtering of the points for each time points  
	- Creating a "denoised" stacked timelapse  
	
## Requirements

python 3.8.x  
numpy  
pandas  
scikit-learn  
scipy  
matplotlib  
seaborn  

## Utilization

Some jupyter notebooks showing how OAT is used are in /demo.  
Visit /method for more informations regarding the methods used :)
