
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from pylab import *
from numpy import *
from scipy.interpolate import spline
import math


def normalData(data, ind):
    normData = zeros( (data.shape[0], ind.shape[0]) )
    for i in range(data.shape[0]):
        sum = np.sum(data[i, ind])
        normData[i, :] = data[i, ind] / sum
    return normData

def svmDiffTao(trainFeat, testFeat, trainLabel, testLabel):
    x = np.arange(-1, 5, 0.2)
    y = np.arange(-1, 5, 0.2)
    for i in range(x.shape[0]):
        tau = pow(30, x[i])
        clf = svm.SVC(gamma = tau)
        clf.fit(trainFeat, trainLabel)
        pred = clf.predict(testFeat)
        y[i] = (accuPred(pred, testLabel))
    xSmooth = np.linspace(x.min(), x.max(), 200)
    ySmooth = spline(x.tolist(), y, xSmooth)
    plt.plot(xSmooth, ySmooth, linewidth = 2)

def accuPred(pred, real):
    equalNum = np.sum( (pred == real) )
    return equalNum / pred.shape[0]

dataAll = np.load('data/report/dataAll.npy')

def getTrainData(dataAll, useFeat, normFeat, beginInd, endInd):
    # this function is to generate training set from the data pool
    trainData = dataAll[0 : beginInd, :]
    #trainData = np.concatenate(trainData, dataAll[endInd : dataAll.shape[0] - 1, :])
    trainData = np.concatenate((trainData, dataAll[endInd : dataAll.shape[0] - 1, :]))
    trainFeat = trainData[:, useFeat]
    #trainFeat = normalData(trainFeat, normFeat)  # normalize the data
    trainLabel1 = trainData[:, 5] > 0
    trainLabel2 = trainData[:, 6] > 0

    return (trainFeat, trainLabel1, trainLabel2)

def getTestData(dataAll, useFeat, normFeat, beginInd, endInd):
    # this function is to generate testing set from the data pool
    testData = dataAll[beginInd : endInd, :]
    testFeat = testData[:, useFeat]
    #testFeat = normalData(testFeat, normFeat)    # normalize the data
    testLabel1 = testData[:, 5] > 0
    testLabel2 = testData[:, 6] > 0
    return (testFeat, testLabel1, testLabel2)

testNum = dataAll.shape[0] / 8

useFeat = [0, 1, 2, 3, 4]
useFeat = np.asarray(useFeat)
normFeat = np.asarray([0, 1, 3, 4])
for i in range(5):
    beginInd = i * testNum
    endInd = (i + 1) * testNum if (i + 1) * testNum < dataAll.shape[0] else dataAll.shape[0]
    (trainFeat, trainLabel1, trainLabel2) = getTrainData(dataAll, useFeat, normFeat, beginInd, endInd)
    (testFeat, testLabel1, testLabel2)    = getTestData (dataAll, useFeat, normFeat, beginInd, endInd)
   # trainFeat = np.asarray(trainFeat)
   # trainlabel1 = np.asarray(trainLabel1)
   # trainLabel2 = np.asarray(trainLabel2)
   # testFeat = np.asarray(testFeat)
   # testLabel1 = np.asarray(testLabel1)
   # testLabel2 = np.asarray(testLabel2)
    svmDiffTao(trainFeat, testFeat, trainLabel1, testLabel1)
plt.xlim(0, 5)
plt.show()






