"""Main module for Kaggle AXA Competition

Uses the logistic regression idea described by Stephane Soulier: https://www.kaggle.com/c/axa-driver-telematics-analysis/forums/t/11299/score-0-66-with-logistic-regression
Hence, we use the traces from every driver as positive examples and build a set of references that we use as negative examples. Note that our set is larger by one driver, in case the reference set includes the driver that we are currently using as positive.
"""

from datetime import datetime
from Driver import Driver
from DriverSelect import DriverSelect
from RegressionDriverGBR import RegressionDriver
import os
import sys
from random import sample, seed
from joblib import Parallel, delayed
import csv
import numpy as np

REFERENCE_DATA = {}


def generatedata(drivers):
    """
    Generates reference data for regression

    Input: List of driver folders that are read.
    Returns: Nothing, since this data is stored in global variable ReferenceData
    """
    global REFERENCE_DATA
    for driver in drivers:
        REFERENCE_DATA[driver.identifier] = driver.generate_data_model


def F_refdata4onedriver(referencefolder, exten, STAND, means, stds):
    referencedrivers.append(DriverSelect(referencefolder, exten, STAND, means=means, stds=stds))


def perform_analysis(folder, exten, STAND, means, stds, n_estimators=10, learning_rate=0.1, max_depth=None):
    print "Working on {0}".format(folder)
    sys.stdout.flush()
    temp = Driver(folder, exten, STAND, means=means, stds=stds)
    cls = RegressionDriver(temp, REFERENCE_DATA)
    cls.classify(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    return cls.toKaggle()


def F_writeFeaturesfile4onedriver(folder, exten, nonstandfeatfile):
    print "Writing features for {0}".format(folder)
    sys.stdout.flush()
    STAND = False
    temp = Driver(folder, exten, STAND)
    with open(nonstandfeatfile, 'a') as featsfile:
        csvwriter = csv.writer(featsfile, delimiter=',')
        for i in xrange(len(temp.generate_data_model)):
            csvwriter.writerow(temp.generate_data_model[i])


def F_Features4onedriver(folder, exten):
    print "Writing features for {0}".format(folder)
    sys.stdout.flush()
    STAND = False
    temp = Driver(folder, exten, STAND)
    return temp.generate_data_model


def F_calcmeanstsds(nonstandfeatfile):
    # gets means and stds from file of features non-standarized:
    allfeats = np.genfromtxt(nonstandfeatfile, delimiter=',')
    print ' F_calcmeanstsds... np.shape(allfeats):',np.shape(allfeats)
    means = np.mean(allfeats, axis=0)
    stds = np.std(allfeats, axis=0)
    return means, stds


def analysis(foldername, outdir, referencenum, exten, n_estimators, learning_rate, max_depth):
    """
    Start the analysis

    Input:
        1) Path to the driver directory
        2) Path where the submission file should be written
        3) Number of drivers to compare against
    """
    start = datetime.now()
    submission_id = datetime.now().strftime("%H_%M_%B_%d_%Y")

    folders = [os.path.join(foldername, f) for f in os.listdir(foldername) if os.path.isdir(os.path.join(foldername, f))]

    nonstandfeatfile = 'Features66-NOSTAND-nreprot.csv'
    # generates csv file with NON STANDARIZED features to calculate means and standards afterwards:
    if os.path.exists(nonstandfeatfile):
        print 'initial calculation of all features for standarizing purposes will be skipped because file exists:',nonstandfeatfile
        pass
    else:
        allfeats = Parallel(n_jobs=60)(delayed(F_Features4onedriver)(folder, exten) for folder in folders)
        with open(nonstandfeatfile, 'a') as featsfile:
            csvwriter = csv.writer(featsfile, delimiter=',')
            for item in allfeats:
                for i in xrange(len(item)):
                    csvwriter.writerow(item[i])

    ## Choose between one of the following two lines:
#    STAND = False
    STAND = True
    if STAND:
 	# calculates means and standard deviations in features:
        means, stds = F_calcmeanstsds(nonstandfeatfile)
    else:
        means = None
        stds = None

    # sample drivers to compare individual ones:
    seed(13)
#    referencefolders = [folders[i] for i in sorted(sample(xrange(len(folders)), referencenum))]
    referencefolders = [folders[i] for i in sorted(sample(xrange(len(folders)), int(len(folders)/4)))]
    print 'Generating refdata not in parallel, please wait some minutes...'
    referencedrivers = []
    for referencefolder in referencefolders:
#        referencedrivers.append(Driver(referencefolder, exten, STAND, means=means, stds=stds))
         referencedrivers.append(DriverSelect(referencefolder, exten, STAND, means=means, stds=stds))
    generatedata(referencedrivers)

    results = Parallel(n_jobs=60)(delayed(perform_analysis)(folder, exten, STAND, means, stds, n_estimators, learning_rate, max_depth) for folder in folders)

    namesubmisfile = "GBR13-nrro-R0.25-spacbrtrpalxyab-std-e%i-l%f-d%i.csv" % (n_estimators, learning_rate, max_depth)
    with open(os.path.join(outdir, namesubmisfile), 'w') as writefile:
        writefile.write("driver_trip,prob\n")
        for item in results:
            writefile.write("%s\n" % item)
    print 'submission file ',namesubmisfile,' written'
    print 'Done, elapsed time: %s' % str(datetime.now() - start)


if __name__ == '__main__':
    # default is 100 0.01 2
    n_estimators = int(sys.argv[1])
    learning_rate = float(sys.argv[2])
    max_depth = int(sys.argv[3])

    exten = ".csv-nreprot"

    MyPath = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    DataPath = os.path.join(MyPath, "..", "data", "drivers")
    analysis(DataPath, MyPath, 5, exten, n_estimators, learning_rate, max_depth)

