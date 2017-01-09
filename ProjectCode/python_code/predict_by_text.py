
print(__doc__)

import xdrlib, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from pylab import *
from numpy import *
import math
from scipy import io

def signFunc(x):
    if x >= 0:
        return 1
    else:
        return 0

def accuPred(pred, real):
    equalNum = np.sum( (pred == real) )
    return equalNum / pred.shape[0]

def normalData(data, ind):
    normData = zeros( (data.shape[0], ind.shape[0]) )
    for i in range(data.shape[0]):
        sum = np.sum(data[i, ind])
        normData[i, :] = data[i, ind] / sum
    return normData

def svmDiffTao(trainFeat, testFeat, trainLabel, testLabel):
    x = np.arange(-1, 5, 0.05)
    y = np.arange(-1, 5, 0.05)
    for i in range(x.shape[0]):
        tau = pow(10, x[i])
        clf = svm.SVC(gamma = tau)
        clf.fit(trainFeat, trainLabel)
        pred = clf.predict(testFeat)
        y[i] = (accuPred(pred, testLabel))
    print(x)
    print(y)
    plt.plot(x, y)
    #plt.show()

trainData = np.load('data/article/trainData.npy')
testData  = np.load('data/article/testData.npy')

useFeat = [0, 1, 2, 3, 4]
useFeat = np.asarray(useFeat)
normFeat = np.asarray([0, 1, 3, 4])

# the we split the data into trainig set and testing set
trainFeat = trainData[:, useFeat]
trainFeat = normalData(trainData, normFeat)    # normalize the data
trainLabel1 = trainData[:, 5]
trainLabel2 = trainData[:, 6]

testFeat = testData[:, useFeat]
testFeat = normalData(testData, normFeat)
testLabel1 = testData[:, 5]
testLabel2 = testData[:, 6]


#------------------------------------------------
#             fit the SVM model
#------------------------------------------------
clf1 = svm.SVC(gamma = 1000)
clf1.fit(trainFeat, trainLabel1)   # fit the model

clf2 = svm.SVC(gamma = 1000)
clf2.fit(trainFeat, trainLabel2)


svmDiffTao(trainFeat, testFeat, trainLabel1, testLabel1)

# do predictions
pred1 = clf1.predict(testFeat)    # make predictoins

#plt.plot(pred1, testLabel1)
#plt.show()
print("The accuracy of predicting after-market trade using SVM: %f" %accuPred(pred1, testLabel1))

pred2 = clf2.predict(testFeat)
print("The accuracy of predicting next-day trade using SVM: %f" %accuPred(pred2, testLabel2))

#------------------------------------------------
#           fit the Naive Bayesian model
#------------------------------------------------
gnb1 = GaussianNB()
gnb1.fit(trainFeat, trainLabel1)

gnb2 = GaussianNB()
gnb2.fit(trainFeat, trainLabel2)

# do predictions
predNB1 = gnb1.predict(testFeat)    # make predictoins

print("The accuracy of predicting after-market trade using Naive Bayesian: %f" %accuPred(predNB1, testLabel1))

predNB2 = gnb2.predict(testFeat)
print("The accuracy of predicting next-day trade using Naive Bayesian: %f" %accuPred(predNB2, testLabel2))

#------------------------------------------------
#           fit the Boosting model
#------------------------------------------------

boost1 = GradientBoostingClassifier(n_estimators=70, learning_rate=1.0, max_depth=1, random_state=0)
boost1.fit(trainFeat, trainLabel1)
print("The accuracy of predicting after-market trade using Boosting: %f" %boost1.score(testFeat, testLabel1))

boost2 = GradientBoostingClassifier(n_estimators=70, learning_rate=1.0, max_depth=1, random_state=0)
boost2.fit(trainFeat, trainLabel2)
print("The accuracy of predicting next-day trade using Boosting: %f" %boost1.score(testFeat, testLabel2))