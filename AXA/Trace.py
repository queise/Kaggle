import os, sys
from math import hypot
import numpy as np

def smooth(x, y, steps):
    """
    Returns moving average using steps samples to generate the new trace

    Input: x-coordinates and y-coordinates as lists as well as an integer to indicate the size of the window (in steps)
    Output: list for smoothed x-coordinates and y-coordinates
    """
    xnew = []
    ynew = []
    for i in xrange(steps, len(x)):
        xnew.append(sum(x[i-steps:i]) / float(steps))
        ynew.append(sum(y[i-steps:i]) / float(steps))
    if steps==1:		# to avoid only one point in xnew if len(x)==2, and adds last point anyway for len(x)<10
        xnew.append(x[-1])
        ynew.append(y[-1])
    return xnew, ynew


def distance(x0, y0, x1, y1):
    """Computes 2D euclidean distance"""
    return hypot((x1 - x0), (y1 - y0))


def velocities_and_distance_covered(x, y):
    """
    Returns velocities just using difference in distance between coordinates as well as accumulated distances

    Input: x-coordinates and y-coordinates as lists
    Output: list of velocities
    """
    v = []
    distancesum = 0.0
    vy = [] ; vx = []
    for i in xrange(1, len(x)):
        dist = distance(x[i-1], y[i-1], x[i], y[i])
        if dist<250.:
            v.append(dist)
            distancesum += dist
            vy.append(y[i]-y[i-1])
            vx.append(x[i]-x[i-1])    
    return v, vx, vy, distancesum


def F_calcaccelsbreaks(vels):
#    accandbre = [] ; accels = [] ; breaks = []
    accandbre = [] ; accels = [] ; breaks = [0.] # to avoid breaks being empty
    accels.append(vels[0])      # first velocity corresponds to first acceleration always
    accandbre.append(vels[0])
    for i in xrange(1,len(vels)):
        diff = vels[i]-vels[i-1]
        accandbre.append(diff)
        if diff > 0.:
            accels.append(diff)
        elif diff < 0.:
            breaks.append(abs(diff))
    sign_accandbre = np.sign(accandbre)  
    sign_accandbre[sign_accandbre==0] = -1     # replace zeros with -1  
    nchanges_accbre = len(np.where(np.diff(sign_accandbre))[0])  
    return accels, breaks, nchanges_accbre


def F_calctripfeatures(x, y):
    deltaX = 0.0 ; deltaY = 0.0 ; distXsum = 0.0 ; distYsum = 0.0
    nturnsX = 0 ; nturnsY = 0
    for i in xrange(1, len(x)):
        # trip features
        displX = x[i] - x[i-1]
        displY = y[i] - y[i-1]
        if displX>250 or displY>250:	# to eliminate outliers
            continue
        deltaX += displX
        deltaY += displY
        distXsum += abs(displX)
        distYsum += abs(displY)
        if i >= 2:
            displXant = x[i-1] - x[i-2]
            displYant = y[i-1] - y[i-2]
            if (displXant*displX)<0.:
                nturnsX += 1
            if (displYant*displY)<0.:
                nturnsY += 1
    return distXsum, distYsum, abs(deltaX), abs(deltaY), nturnsX, nturnsY


def F_find_straights(data, thresh=20.0,verbose=False):
        FLAGERR = False
        lend = len(data) # x)
        lines = []
        start_ends = []
        i = 0
        trials = 0
        res = 0.0
        rev_data = data[::-1]
        while i < lend:
            #Loop through all possible end points
            found_linear = False
            for j in range(i+20,lend,5):
                if verbose:
                    print i, j
                coeffs, res, rot_points, FLAGERR = fit_poly(rev_data,i,j)
                if FLAGERR:
                    return 1., FLAGERR
                if res/(j-i) > thresh:
                    if found_linear:
                        lines.append(rot_points_longest)
                        start_ends.append(s)
                        i = j
                    break
                else:
                    found_linear = True
                    rot_points_longest = rot_points.copy()
                    s = [lend-j-1,lend-i-1]
            if res/(lend-i-1) < thresh and len(start_ends) == 0:
                start_ends = [[0,0]]
                lines = [[0.0,0.0]]
                break
            i += 4
        #Determine each straights length for whats it's worth
        straights_lengths = []
        for l in lines:
            straights_lengths.append(np.linalg.norm(l[-1]-l[0]))
       
        return straights_lengths, FLAGERR 


def fit_poly(data,start,end):
    FLAGERR = False
    #Rotate it to be parallel with x axis
    offset = data[start]
    theta = -np.arctan2(data[end,1]-data[start,1],data[end,0]-data[start,0])
    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]])
    mod = np.array([np.dot(rotMatrix,v-offset) for v in data[start:end]])
    x = mod[:,0]
    y = mod[:,1]
    try:
        coeffs, res, rank, svd, r = np.polyfit(x,y,1,full=True)
        p = np.poly1d(coeffs)
        rot_points = np.array([np.dot(rotMatrix.T,[t,p(t)])+offset for t in x])
    except:
        FLAGERR = True
        coeffs = np.array([0., 0.]) ; res = np.array([0., 0.]) ; rot_points = np.array([0., 0.])

    return coeffs, res[0], rot_points, FLAGERR


def F_remove_idle(data):
    " removes repetitions "
    mask = [True for i in data] # just initialization, no sense
    lenold = len(data)
    while np.any(mask):
        x = data[:,0]
        y = data[:,1]
        mask = ((np.diff(x) == 0) & (np.diff(y) == 0))
        data = np.delete(data,np.argwhere(mask),axis=0)
#       print "Removed {0} of {1} points".format(lenold-len(data),lenold)
    return data


def get_total_length(data):
    return np.sum([np.linalg.norm(u-v) for u,v in zip(data[:-1],data[1:])])


class Trace(object):
    """"
    Trace class reads a trace file and computes features of that trace.
    """

    def __init__(self, filename, STAND, means=None, stds=None, filtering=10):
        """Input: path and name of the file of a trace; how many filtering steps should be used for sliding window filtering"""
        self.__id = int(os.path.basename(filename).split(".")[0])
        x = []
        y = []
        with open(filename, "r") as trainfile:
            trainfile.readline()  # skip header
            for line in trainfile:
                items = line.split(",", 2)
                x.append(float(items[0]))
                y.append(float(items[1]))
	filtering = min(max(int(len(x)/10),1), filtering) # too avoid to short paths (useful with nrep)
        self.__xn, self.__yn = smooth(x, y, filtering)
        v, vx, vy, self.distancecovered = velocities_and_distance_covered(self.__xn, self.__yn)
        self.triptime = len(x)
#	self.triplength = distance(x[0], y[0], x[-1], y[-1])
        self.triplength = min(distance(x[0], y[0], x[-1], y[-1]), 40000)
        self.maxspeed = max(v)
	# new feats
	# sp
	self.meanspeed = np.mean(v)
	self.stdspeed = np.std(v)
	self.fstQspeed = np.percentile(v,25)
	self.secQspeed = np.percentile(v,50)
        self.thrQspeed = np.percentile(v,75)
	# acc
	accels, breaks, self.nchangesaccbre = F_calcaccelsbreaks(v)
        self.maxaccel = np.max(accels)
        self.meanaccel = np.mean(accels)
        self.stdaccel = np.std(accels)
        self.fstQaccel = np.percentile(accels,25)
        self.secQaccel = np.percentile(accels,50)
        self.thrQaccel = np.percentile(accels,75)
	# breack
        self.maxbreak = np.max(breaks)
        self.meanbreak = np.mean(breaks)
        self.stdbreak = np.std(breaks)
        self.fstQbreak = np.percentile(breaks,25)
        self.secQbreak = np.percentile(breaks,50)
        self.thrQbreak = np.percentile(breaks,75)
	# trip
	self.distXsum, self.distYsum, self.deltaX, self.deltaY, self.nturnsX, self.nturnsY = F_calctripfeatures(self.__xn, self.__yn)
        # pat feats
        exten = os.path.basename(filename).split(".")[1]
        originfname = filename[:-len(exten)]+'csv'
        dataorig = np.genfromtxt(originfname, delimiter=",",skip_header=1)
        data = F_remove_idle(dataorig)
        l = get_total_length(data)
        if self.distancecovered < 1.0:
            self.straightsoverl = 0.
        else:
            lenstraights, FLAGERR = F_find_straights(data)
            if FLAGERR:
                self.straightsoverl = 0.57 	# the mean, previously calculated
            else:
                self.straightsoverl = np.sum(lenstraights) / l
        idle_time = len(dataorig)-len(data)
        self.idle_fraction = idle_time/float(len(dataorig))
        # sp x   
        self.maxvx = max(vx)
        self.meanvx = np.mean(vx)
        self.stdvx = np.std(vx)
        self.fstQvx = np.percentile(vx,25)
        self.secQvx = np.percentile(vx,50)
        self.thrQvx = np.percentile(vx,75)
        # sp y
        self.maxvy = max(vy)
        self.meanvy = np.mean(vy)
        self.stdvy = np.std(vy)
        self.fstQvy = np.percentile(vy,25)
        self.secQvy = np.percentile(vy,50)
        self.thrQvy = np.percentile(vy,75)
        # acc x
        accelsx, breaksx, temp = F_calcaccelsbreaks(vx)
        self.maxaccelx = np.max(accelsx)
        self.meanaccelx = np.mean(accelsx)
        self.stdaccelx = np.std(accelsx)
        self.fstQaccelx = np.percentile(accelsx,25)
        self.secQaccelx = np.percentile(accelsx,50)
        self.thrQaccelx = np.percentile(accelsx,75)
        # breack x
        self.maxbreakx = np.max(breaksx)
        self.meanbreakx = np.mean(breaksx)
        self.stdbreakx = np.std(breaksx)
        self.fstQbreakx = np.percentile(breaksx,25)
        self.secQbreakx = np.percentile(breaksx,50)
        self.thrQbreakx = np.percentile(breaksx,75)
        # acc y
        accelsy, breaksy, temp = F_calcaccelsbreaks(vy)
        self.maxaccely = np.max(accelsy)
        self.meanaccely = np.mean(accelsy)
        self.stdaccely = np.std(accelsy)
        self.fstQaccely = np.percentile(accelsy,25)
        self.secQaccely = np.percentile(accelsy,50)
        self.thrQaccely = np.percentile(accelsy,75)
        # breack y
        self.maxbreaky = np.max(breaksy)
        self.meanbreaky = np.mean(breaksy)
        self.stdbreaky = np.std(breaksy)
        self.fstQbreaky = np.percentile(breaksy,25)
        self.secQbreaky = np.percentile(breaksy,50)
        self.thrQbreaky = np.percentile(breaksy,75)


        # feature scaling and mean normalization:
	if STAND:
            self.F_scalandnorm(means, stds)


    def F_scalandnorm(self, means, stds):
        self.triplength = (self.triplength - means[0]) / stds[0]
        self.triptime = (self.triptime - means[1]) / stds[1]
	self.distancecovered = (self.distancecovered - means[2]) / stds[2]
	self.maxspeed = (self.maxspeed - means[3]) / stds[3]
	self.meanspeed = (self.meanspeed - means[4]) / stds[4]
	self.stdspeed = (self.stdspeed - means[5]) / stds[5]
        self.fstQspeed = (self.fstQspeed - means[6]) / stds[6]
        self.secQspeed = (self.secQspeed - means[7]) / stds[7]
        self.thrQspeed = (self.thrQspeed - means[8]) / stds[8]
        self.maxaccel = (self.maxaccel - means[9]) / stds[9]
        self.meanaccel = (self.meanaccel - means[10]) / stds[10]
        self.stdaccel = (self.stdaccel - means[11]) / stds[11]
        self.fstQaccel = (self.fstQaccel - means[12]) / stds[12]
        self.secQaccel = (self.secQaccel - means[13]) / stds[13]
        self.thrQaccel = (self.thrQaccel - means[14]) / stds[14]
        self.maxbreak = (self.maxbreak - means[15]) / stds[15]
        self.meanbreak = (self.meanbreak - means[16]) / stds[16]
        self.stdbreak = (self.stdbreak - means[17]) / stds[17]
        self.fstQbreak = (self.fstQbreak - means[18]) / stds[18]
        self.secQbreak = (self.secQbreak - means[19]) / stds[19]
        self.thrQbreak = (self.thrQbreak - means[20]) / stds[20]
	self.nchangesaccbre = (self.nchangesaccbre - means[21]) / stds[21]	     
	self.distXsum = (self.distXsum - means[22]) / stds[22]
	self.distYsum = (self.distYsum - means[23]) / stds[23]
	self.deltaX = (self.deltaX - means[24]) / stds[24]
	self.deltaY = (self.deltaY - means[25]) / stds[25]
	self.nturnsX = (self.nturnsX - means[26]) / stds[26]
	self.nturnsY = (self.nturnsY - means[27]) / stds[27]
	self.straightsoverl = (self.straightsoverl - means[28]) / stds[28]
	self.idle_fraction = (self.idle_fraction - means[29]) / stds[29]
        self.maxvx = (self.maxvx - means[30]) / stds[30]
        self.meanvx = (self.meanvx - means[31]) / stds[31]
        self.stdvx = (self.stdvx - means[32]) / stds[32]
        self.fstQvx = (self.fstQvx - means[33]) / stds[33]
        self.secQvx = (self.secQvx - means[34]) / stds[34]
        self.thrQvx = (self.thrQvx - means[35]) / stds[35]
        self.maxvy = (self.maxvy - means[36]) / stds[36]
        self.meanvy = (self.meanvy - means[37]) / stds[37]
        self.stdvy = (self.stdvy - means[38]) / stds[38]
        self.fstQvy = (self.fstQvy - means[39]) / stds[39]
        self.secQvy = (self.secQvy - means[40]) / stds[40]
        self.thrQvy = (self.thrQvy - means[41]) / stds[41]
        self.maxaccelx = (self.maxaccelx - means[42]) / stds[42]
        self.meanaccelx = (self.meanaccelx - means[43]) / stds[43]
        self.stdaccelx = (self.stdaccelx - means[44]) / stds[44]
        self.fstQaccelx = (self.fstQaccelx - means[45]) / stds[45]
        self.secQaccelx = (self.secQaccelx - means[46]) / stds[46]
        self.thrQaccelx = (self.thrQaccelx - means[47]) / stds[47]
        self.maxbreakx = (self.maxbreakx - means[48]) / stds[48]
        self.meanbreakx = (self.meanbreakx - means[49]) / stds[49]
        self.stdbreakx = (self.stdbreakx - means[50]) / stds[50]
        self.fstQbreakx = (self.fstQbreakx - means[51]) / stds[51]
        self.secQbreakx = (self.secQbreakx - means[52]) / stds[52]
        self.thrQbreakx = (self.thrQbreakx - means[53]) / stds[53]
        self.maxaccely = (self.maxaccely - means[54]) / stds[54]
        self.meanaccely = (self.meanaccely - means[55]) / stds[55]
        self.stdaccely = (self.stdaccely - means[56]) / stds[56]
        self.fstQaccely = (self.fstQaccely - means[57]) / stds[57]
        self.secQaccely = (self.secQaccely - means[58]) / stds[58]
        self.thrQaccely = (self.thrQaccely - means[59]) / stds[59]
        self.maxbreaky = (self.maxbreaky - means[60]) / stds[60]
        self.meanbreaky = (self.meanbreaky - means[61]) / stds[61]
        self.stdbreaky = (self.stdbreaky - means[62]) / stds[62]
        self.fstQbreaky = (self.fstQbreaky - means[63]) / stds[63]
        self.secQbreaky = (self.secQbreaky - means[64]) / stds[64]
        self.thrQbreaky = (self.thrQbreaky - means[65]) / stds[65]

        assert len(means)==66

    @property
    def features(self):
        """Returns a list that comprises all computed features of this trace."""
        features = []
        features.append(self.triplength)
        features.append(self.triptime)
        features.append(self.distancecovered)
          # speed
        features.append(self.maxspeed)
        # new feats
          # speed
	features.append(self.meanspeed)
        features.append(self.stdspeed)
        features.append(self.fstQspeed)
        features.append(self.secQspeed)
        features.append(self.thrQspeed)
          # accel
        features.append(self.maxaccel)
        features.append(self.meanaccel)
        features.append(self.stdaccel)
        features.append(self.fstQaccel)
        features.append(self.secQaccel)
        features.append(self.thrQaccel)
          # break
        features.append(self.maxbreak)
        features.append(self.meanbreak)
        features.append(self.stdbreak)
        features.append(self.fstQbreak)
        features.append(self.secQbreak)
        features.append(self.thrQbreak)
          # num changes acc/breaks
        features.append(self.nchangesaccbre)
	  # trip
	features.append(self.distXsum)
        features.append(self.distYsum)
        features.append(self.deltaX)
        features.append(self.deltaY)
        features.append(self.nturnsX)
        features.append(self.nturnsY)
          # pat
        features.append(self.straightsoverl)
	features.append(self.idle_fraction)
          # sp x y
        features.append(self.maxvx)
        features.append(self.meanvx)
        features.append(self.stdvx)
        features.append(self.fstQvx)
        features.append(self.secQvx)
        features.append(self.thrQvx)
        features.append(self.maxvy)
        features.append(self.meanvy)
        features.append(self.stdvy)
        features.append(self.fstQvy)
        features.append(self.secQvy)
        features.append(self.thrQvy)
	  # acc and breaks x and y
          # accel
        features.append(self.maxaccelx)
        features.append(self.meanaccelx)
        features.append(self.stdaccelx)
        features.append(self.fstQaccelx)
        features.append(self.secQaccelx)
        features.append(self.thrQaccelx)
        features.append(self.maxbreakx)
        features.append(self.meanbreakx)
        features.append(self.stdbreakx)
        features.append(self.fstQbreakx)
        features.append(self.secQbreakx)
        features.append(self.thrQbreakx)
        features.append(self.maxaccely)
        features.append(self.meanaccely)
        features.append(self.stdaccely)
        features.append(self.fstQaccely)
        features.append(self.secQaccely)
        features.append(self.thrQaccely)
        features.append(self.maxbreaky)
        features.append(self.meanbreaky)
        features.append(self.stdbreaky)
        features.append(self.fstQbreaky)
        features.append(self.secQbreaky)
        features.append(self.thrQbreaky)

        assert len(features)==66

        return features

    @property
    def tripfeatures(self):
        """Returns a list that comprises the TRIP features of this trace."""
        tripfeatures = []
        tripfeatures.append(self.distXsum)
        tripfeatures.append(self.distYsum)
        tripfeatures.append(self.deltaX)
        tripfeatures.append(self.deltaY)
        tripfeatures.append(self.nturnsX)
        tripfeatures.append(self.nturnsY)
        return tripfeatures

    def __str__(self):
        return "Trace {0} has this many positions: \n {1}".format(self.__id, self.triptime)

    @property
    def identifier(self):
        """Driver identifier is its filename"""
        return self.__id
