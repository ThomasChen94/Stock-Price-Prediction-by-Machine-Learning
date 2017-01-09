import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from numpy import *
import math
from mpl_toolkits.mplot3d import Axes3D

def normalData(data, ind):
    normData = zeros( (data.shape[0], ind.shape[0]) )
    for i in range(data.shape[0]):
        sum = np.sum(data[i, ind])
        normData[i, :] = data[i, ind] / sum
    return normData

dataAll = np.load('data/article/dataAll.npy')

total = dataAll.shape[0]

useFeat = [0, 1, 2, 3, 4]
useFeat = np.asarray(useFeat)
normFeat = np.asarray([0, 1, 3, 4])

feat = dataAll[:, useFeat]
#feat = normalData(dataAll, normFeat)    # normalize the data
label1 = dataAll[:, 5] > 0
label2 = dataAll[:, 6] > 0

ax=plt.subplot(111,projection='3d')

for i in range(math.floor(total * 0.7)):
    if label1[i] == 0:
        ax.scatter(feat[i,0], feat[i,2], feat[i,3], marker = 'x', color = 'c')
    else:
        ax.scatter(feat[i,0], feat[i,2], feat[i,3], marker = 'o', color = 'r')

plt._show()