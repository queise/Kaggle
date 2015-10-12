if __name__ == '__main__':
#    n_estimators = int(sys.argv[1])

    originsubmisfileA = 'GBR13-nrro-R0.5-spacbrtrpal-std-e100-l0.010000-d2.csv'
    print ' *** reading original file A:',originsubmisfileA
    drtrIDA = [] ; drtrprobA = []
    with open(originsubmisfileA, "r") as trainfile:
        trainfile.readline()  # skip header
        for line in trainfile:
            items = line.split(",", 2)
            drtrIDA.append(items[0])
            drtrprobA.append(items[1])


    originsubmisfileB = 'RFR13-nrro-R0.3-spacbrtrpalxyab-std-e60-s3-d3.csv'
    print ' *** reading original file B:',originsubmisfileB
    drtrIDB = [] ; drtrprobB = []
    with open(originsubmisfileB, "r") as trainfile:
        trainfile.readline()  # skip header
        for line in trainfile:
            items = line.split(",", 2)
            drtrIDB.append(items[0])
            drtrprobB.append(items[1])


    newsubmisfile = 'combined.csv'
    print ' *** writing new submission file:',newsubmisfile
    with open(newsubmisfile, 'w') as writefile:
        writefile.write("driver_trip,prob\n")
        for i in xrange(len(drtrIDA)):
            newprob = (float(drtrprobA[i])+float(drtrprobB[i])) / 2.
            writefile.write("%s,%s\n" % (drtrIDA[i],str(newprob)))
