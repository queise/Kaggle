from sklearn import preprocessing
import numpy as np

alltraces = np.genfromtxt('Features28NOSTAND-nreprot.csv', delimiter=',')

#print 'alltraces\n',alltraces

#np.set_printoptions(suppress=True)

means = np.mean(alltraces, axis=0)
print len(means)
str_means = 'means = [ '
for mean in means:
    str_means += str(mean) + ', '
str_means = str_means[:-2] + ']'
print str_means

print ''

stds = np.std(alltraces, axis=0)
print len(stds)
str_stds = 'stds = [ '
for std in stds:
    str_stds += str(std) + ', '
str_stds = str_stds[:-2] + ']'
print str_stds

print 'maxs:'
print np.max(alltraces, axis=0)
