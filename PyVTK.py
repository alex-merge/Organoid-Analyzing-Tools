# -*- coding: utf-8 -*-
"""
Python to VTK
Translated to python from Julia based on Marc-3d script.

@author: Alex-932
@version: 0.5.2
"""

class PyVTK():
    
    def __init__(self, filename, Xcoords, Ycoords, Zcoords, path, mode):
        """
        Convert coordinates of 3D points from python to a .vtk to be visualized
        in Paraview.

        Parameters
        ----------
        filename : str
            Name of the output file.
        Xcoords : pandas.DataFrame or pandas.Series depending what mode you 
            want. DF : Rows are the time points and columns are the tracks.
                  Series : Rows are the points in whatever order.
                  Each cell is a unique coordinate, X in this case.
        Ycoords : same as Xcoords.
        Zcoords : same as Ycoords.
        path : str
            Path for the output file to be saved in.
        mode : str
            "polydata" : export coordinates into a polydata type file (used for
                         displacement trajectories).
            "points" : export into a points type file which just show dots.

        """
        self.filename = filename
        self.X = Xcoords
        self.Y = Ycoords
        self.Z = Zcoords
        self.path = path
        self.version = "0.5.2"
        if mode == "polydata":
            self.polydataWriter()
        elif mode == "points":
            self.pointsWriter()               
        
    def polydataWriter(self):
        """
        Writer to convert a series of coordinates into a polydata .vtk file.

        """
        #Setting the name of the file and opening it.
        self.filepath = self.path+"\\"+self.filename+".vtk"
        file = open(self.filepath, "w")
        
        # Removing columns (tracks) containing nan values.
        self.X.dropna(axis = "columns", inplace = True)
        self.Y.dropna(axis = "columns", inplace = True)
        self.Z.dropna(axis = "columns", inplace = True)
        
        #Getting the number of columns i.e. the number of different tracks.
        points_nb = len(self.X.columns)
        
        #Getting the number of points per tracks.
        tp_nb = len(self.X.index)
        
        #Writing the file specifications
        file.write("# vtk DataFile Version 2.0\n")
        file.write("PIV3D Trajectories\n")
        file.write("ASCII\n")
        file.write("DATASET POLYDATA\n")
        file.write("POINTS "+str(points_nb*tp_nb)+" double\n")
        
        #For every tracks and every timepoint (row), we write the coordinates 
        #of the given point in the file.
        for pt in range(points_nb):
            for tp in range(tp_nb):
                X = self.X.iloc[tp, pt] 
                Y = self.Y.iloc[tp, pt]                                
                Z = self.Z.iloc[tp, pt]
                file.write(str(X)+" "+str(Y)+" "+str(Z)+"\n")
                    
        #Writing the number of lines and the number of points.
        file.write("LINES "+str(points_nb)+" "+str((tp_nb+1)*points_nb)+"\n" )
        file.write("\n")
        
        #Writing some more informations
        idx = 0;
        for pt in range(points_nb):
            file.write(str(tp_nb)+" \n")
            for tp in range(tp_nb):
                file.write(str(idx)+" \n")
                idx += 1
            file.write("\n")

        file.write("POINT_DATA "+str(points_nb*tp_nb)+"\n")
        file.write("SCALARS index int 1\n")
        file.write("LOOKUP_TABLE default\n")
        file.write("\n")
        for pt in range(points_nb):
             for tp in range(tp_nb):
                file.write(str(tp)+" \n")
                
        #Closing the file and saving it.        
        file.close()
        
    def pointsWriter(self):
        """
        Writer to convert a series of coordinates into a points .vtk file.

        """
        #Setting the name of the file and opening it.
        self.filepath = self.path+"\\"+self.filename+".vtk"
        file = open(self.filepath, "w")
        #Getting the number of columns i.e. the number of different tracks.
        points_nb = len(self.X.index)
        file.write("# vtk DataFile Version 2.0\n")
        file.write("PIV3D Trajectories\n")
        file.write("ASCII\n")
        file.write("DATASET POLYDATA\n")
        file.write("POINTS "+str(points_nb)+" double\n")
        
        #For every tracks and every timepoint (row), we write the coordinates 
        #of the given point in the file.
        for pt in range(points_nb):
            file.write(str(self.X.iloc[pt])+" "+\
                       str(self.Y.iloc[pt])+" "+\
                       str(self.Z.iloc[pt])+"\n")
        #Closing the file and saving it.
        file.close()
        
        