import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from numpy import *
import math
from mpl_toolkits.mplot3d import Axes3D

dataAll = np.load('data/report/dataAll.npy')

total = dataAll.shape[0]

useFeat = [0, 1, 2, 3, 4]
useFeat = np.asarray(useFeat)

feat = dataAll[:, useFeat]

label1 = dataAll[:, 5] > 0
label2 = dataAll[:, 6] > 0

ax=plt.subplot(111,projection='3d')

for i in range(130, 149):
    if label1[i] == 0:
        ax.scatter(feat[i,0], feat[i,1], feat[i,2], marker = 'x', color = 'c')
    else:
        ax.scatter(feat[i,0], feat[i,1], feat[i,2], marker = 'o', color = 'r')

plt.xlim(-0.5, 0.5)
plt.ylim(-1, 1)

plt._show()