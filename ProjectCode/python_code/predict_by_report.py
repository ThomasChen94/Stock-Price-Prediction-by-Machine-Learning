
print(__doc__)

import xdrlib, sys
import xlrd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from numpy import *
from scipy import io
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

workExcel = xlrd.open_workbook('Company_Data_Trend.xlsx')
tableMix = []

numRows = [26, 26, 26, 26, 26, 19, 26, 13, 25, 20, 26] # the number of useful rows in each table
dataInd = [7, 14, 15, 16, 17, 8, 9]  # the index of feature and label columns, the first 5 are features, the left 2 are labels
numRows = np.asarray(numRows)
dataInd = np.asarray(dataInd)

def getFeature(table, rowNum):
    feat = zeros( ( rowNum - 2, len(dataInd) ))
    for i in range(2, rowNum):
        rowData = np.asarray(table.row_values(i))
        feat[i-2,:] = rowData[dataInd]
    return feat

def signFunc(x):
    if x >= 0:
        return 1
    else:
        return 0

def accuPred(pred, real):
    equalNum = np.sum( (pred == real) )
    return equalNum / pred.shape[0]

dataAll = zeros((1,7))  # the header of data

for i in range(11):
    # read in all tables
    table = workExcel.sheets()[i]
    data = getFeature(table, numRows[i])
    dataAll = np.concatenate((dataAll, data))

# before training, we have to shuffle the data

ind = np.arange(1, dataAll.shape[0])
#np.random.shuffle(ind)
dataAll = dataAll[ind, :]
#np.save('dataAll.npy', dataAll)
#io.savemat('dataAll.mat', {'array': dataAll})

total = dataAll.shape[0]
trainNum = math.floor(total * 0.75)

total = 237
trainNum = 232

useFeat = [0, 1, 2, 3 , 4]
np.asarray(useFeat)

# the we split the data into trainig set and testing set
trainFeat = dataAll[215 : trainNum, useFeat]
print(dataAll.shape)
trainLabel1 = dataAll[215 : trainNum, 5]
trainLabel2 = dataAll[215 : trainNum, 6]

testFeat = dataAll[trainNum : total, useFeat]
testLabel1 = dataAll[trainNum : total, 5]
testLabel2 = dataAll[trainNum : total, 6]

# set label as 0 or 1
trainLabel1 = (trainLabel1 > 0)
trainLabel2 = (trainLabel2 > 0)
testLabel1 = (testLabel1 > 0)
testLabel2 = (testLabel2 > 0)

#------------------------------------------------
#           fit the SVM model
#------------------------------------------------
# fit the SVM model
clf1 = svm.NuSVC()
clf1.fit(trainFeat, trainLabel1)

clf2 = svm.NuSVC()
clf2.fit(trainFeat, trainLabel2)

# do predictions
#pred1 = clf1.decision_function(testData) > 0
pred1 = clf1.predict(testFeat)
print(pred1)
print("The accuracy of predicting after-market trade: %f" %accuPred(pred1, testLabel1))

pred2 = clf2.predict(testFeat)
print("The accuracy of predicting next-day trade: %f" %accuPred(pred2, testLabel2))

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