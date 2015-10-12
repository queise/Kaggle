import os, sys
import numpy as np
import math as m

def F_get_matrix(angle):
    return np.matrix( ((m.cos(angle),-m.sin(angle)), (m.sin(angle), m.cos(angle))) )

def F_anglevects(v1, v2):
    cosangle = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
    return np.arccos(np.clip(cosangle,-1,1)) # np.clip to force cos to be in range [-1,1], to deal w errors (parallel vectors)

MyPath = os.path.join(os.path.dirname(os.path.realpath(__file__)))
DataPath = os.path.join(MyPath, "..", "data", "drivers")

driversfolders = [os.path.join(DataPath, f) for f in os.listdir(DataPath) if os.path.isdir(os.path.join(DataPath, f))]

print 'num drivers:',len(driversfolders)

numnotedited = 0
for driverfolder in driversfolders:
    files = [f for f in os.listdir(driverfolder) if f.endswith(".csv")]
    for filename in files:
        tripfname = os.path.join(driverfolder,filename)
#        tripfname = '/data/Kaggle/AXA/code-kam/../data/drivers/1424/101.csv'	
	print 'tripfname',tripfname
        coords = np.genfromtxt(tripfname, delimiter=',', skip_header=1)
        newcoords = np.zeros((1,2))
	for i in range(1,len(coords)):
            if not np.array_equal(coords[i], coords[i-1]):
                newcoords = np.append(newcoords, np.reshape(coords[i], (1,2)), axis=0)
        if np.shape(newcoords)[0] == np.shape(coords)[0]:
            numnotedited += 1 
        np.savetxt(tripfname+'-nrep', newcoords, fmt='%.1f', delimiter=',', header="x,y")

    print 'temp num of numnotedited=',numnotedited

print 'total of not edited: ',numnotedited

#        np.savetxt(tripfname+'-rot', newcoords, fmt='%.1f', delimiter=',', header="x,y")
