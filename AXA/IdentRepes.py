from datetime import datetime
from TripDriver import TripDriver
from DriverSelect import DriverSelect
from RegressionDriverRFR import RegressionDriver
import os
import sys
from random import sample, seed
from joblib import Parallel, delayed
import csv
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import itemfreq

def F_writeTripFeaturesfile4onedriver(folder, exten, tripfeatsfile):
    print "Writing features for {0}".format(folder)
    sys.stdout.flush()
    STAND = False
    temp = Driver(folder, exten, STAND)
    with open(tripfeatsfile, 'a') as featsfile:
        csvwriter = csv.writer(featsfile, delimiter=',')
        for i in xrange(len(temp.generate_tripdata_model)):
            csvwriter.writerow(temp.generate_tripdata_model[i])

def F_TripFeatures4onedriver(folder, exten):
    #print "Writing features for {0}".format(folder)
    sys.stdout.flush()
    STAND = False
    temp = TripDriver(folder, exten, STAND)
    return temp.generate_tripdata_model

def F_getTripFeaturesfromfile(tripfeatsfile):
    return np.genfromtxt(tripfeatsfile, delimiter=',')


def checksimilaritiesin1driver(driverfolder, epsi, numminmatch):
        driverID = int(os.path.basename(driverfolder))
        print 'checking similarities in traces from driver:',driverID
        feats = np.array(F_TripFeatures4onedriver(driverfolder, exten))

        sims = {}
        drivsims = []
#        print 'feats = ',feats
        for i in xrange(len(feats)):
#            print 'i:',i,' traceID: ',feats[i][0],' feats:',feats[i][1:]
	    traceID = feats[i][0]
            sims[traceID] = []
            for j in xrange(len(feats)):
                if i < j:
                    mask = np.allclose(feats[i][1:], feats[j][1:], atol=epsi)
                    if mask:
                        sims[traceID].append(feats[j][0])
            if len(sims[traceID]) > (numminmatch-1):
#                print traceID,' is similar to',sims[traceID]
	        drivsims.append((driverID, traceID, sims[traceID]))

	return drivsims

if __name__ == '__main__':
#    n_estimators = int(sys.argv[1])

    exten = ".csv-nreprot"

    driver = "1"
    MyPath = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    DataPath = os.path.join(MyPath, "..", "data", "drivers")
    folder = os.path.join(DataPath, driver)

    folders = [os.path.join(DataPath, f) for f in os.listdir(DataPath) if os.path.isdir(os.path.join(DataPath, f))]
#    folders = ['/local/tmp/jordi/dat/AXA/code-kam/../data/drivers/1704']

#    global allsims
    epsi = 2.e-2
    numminmatch = 1
    allsims = []
    allsims = Parallel(n_jobs=60)(delayed(checksimilaritiesin1driver)(driverfolder, epsi, numminmatch) for driverfolder in folders)

    print '***FINISHED, these are the similarities we found:'

    listofmatches = []
    for drivsims in allsims:
        if drivsims:	# if not empty
            for driver, traceor, tracesim in drivsims:
                listofmatches.append(str(driver)+'_'+str(int(traceor)))
                for matches in tracesim:
                    listofmatches.append(str(driver)+'_'+str(int(matches)))


    listofmatches = list(set(listofmatches))
    print '*** We found ',len(listofmatches),' similarities:'

#    print listofmatches

#    print '*** We found ',len(listofmatches),' similarities:'

    originsubmisfile = 'GBR13-nrro-R0.5-spacbrtrpal-std-e100-l0.010000-d2.csv'
    print ' *** reading original file:',originsubmisfile
    drtrID = [] ; drtrprob = []
    with open(originsubmisfile, "r") as trainfile:
        trainfile.readline()  # skip header
        for line in trainfile:
            items = line.split(",", 2)
            drtrID.append(items[0])
            drtrprob.append(items[1])


    newsubmisfile = originsubmisfile[:-4] + ('-TM-e%.2f-nm%i.csv' % (epsi, numminmatch))
    print ' *** writing new submission file:',newsubmisfile
    with open(newsubmisfile, 'w') as writefile:
        writefile.write("driver_trip,prob\n")
        for i in xrange(len(drtrID)):
            if drtrID[i] in listofmatches:
                writefile.write("%s,1.0\n" % drtrID[i])
            else:
                writefile.write("%s,%s" % (drtrID[i],drtrprob[i]))
