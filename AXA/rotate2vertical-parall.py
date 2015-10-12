import os, sys
import numpy as np
import math as m
from joblib import Parallel, delayed

def F_get_matrix(angle):
    return np.matrix( ((m.cos(angle),-m.sin(angle)), (m.sin(angle), m.cos(angle))) )

def F_getcoordoffirstdirection(coords):
    for coord in coords:
        if coord[0] or coord[1] != 0.:
            firstnonzero = coord
            break
    return firstnonzero

def F_rotateallindriverfolder(driverfolder):
    files = [f for f in os.listdir(driverfolder) if f.endswith(".csv-nrep")] #-nrep
    for filename in files:
        tripfname = os.path.join(driverfolder,filename)
        print 'tripfname',tripfname
        coords = np.genfromtxt(tripfname, delimiter=',', skip_header=1)
        coordoffirstdirection = F_getcoordoffirstdirection(coords)
	angle = np.arctan2(coordoffirstdirection[1],coordoffirstdirection[0]) - np.pi/2.
        rotmatrix = F_get_matrix(angle)
        newcoords = coords * rotmatrix
        np.savetxt(tripfname+'rot', newcoords, fmt='%.2f', delimiter=',', header="x,y")

#### MAIN ####

MyPath = os.path.join(os.path.dirname(os.path.realpath(__file__)))
DataPath = os.path.join(MyPath, "..", "data", "drivers")

driversfolders = [os.path.join(DataPath, f) for f in os.listdir(DataPath) if os.path.isdir(os.path.join(DataPath, f))]

print 'num drivers:',len(driversfolders)

Parallel(n_jobs=32)(delayed(F_rotateallindriverfolder)(folder) for folder in driversfolders)
