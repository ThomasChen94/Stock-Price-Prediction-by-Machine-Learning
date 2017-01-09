import xdrlib, sys
import xlrd
import numpy as np
from numpy import *
import math
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from scipy import io
from sklearn.ensemble import GradientBoostingClassifier

workExcel = xlrd.open_workbook('Company_Data_Text.xlsx')
tableMix = []

beginRow = [51, 51, 51, 51, 51, 31, 51, 31, 37] # the begining row of text data
endRow   = [78, 79, 80, 79, 78, 62, 77, 53, 75] # the ending row of text data
dataInd  = [1, 2, 3, 4, 5, 6, 7]  # the index of feature and label columns, the first 5 are features, the left 2 are labels
beginRow = np.asarray(beginRow)
endRow = np.asarray(endRow)
dataInd = np.asarray(dataInd)

def getFeature(table, begin, end):
    leng = end - begin + 1
    feat = zeros(( leng, len(dataInd) ))
    for i in range(begin - 1, end):
        rowData = np.asarray(table.row_values(i))
        feat[i - begin + 1,:] = rowData[dataInd]
    return feat

dataAll = zeros((1,7))  # the header of data

for i in range(9):
    # read in all tables
    table = workExcel.sheets()[i]
    data = getFeature(table, beginRow[i], endRow[i])
    dataAll = np.concatenate((dataAll, data))

# before training, we have to shuffle the data
ind = np.arange(1, dataAll.shape[0])
np.random.shuffle(ind)
dataAll = dataAll[ind, :]

#io.savemat('dataAll.mat', {'array': dataAll})

total = dataAll.shape[0]
print(total)
trainNum = math.floor(total * 0.8 )

# the we split the data into trainig set and testing set
trainData = dataAll[1 : trainNum, :]
trainData[:, 5 : 7] = trainData[:, 5 : 7] > 0

testData = dataAll[trainNum : total, :]
testData[:, 5 : 7] = testData[:, 5 : 7] > 0

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

useFeat = [0, 1,  3, 4]
useFeat = np.asarray(useFeat)
#normFeat = np.asarray([0, 1,  3, 4])

# the we split the data into trainig set and testing set
trainFeat = trainData[:, useFeat]
#trainFeat = normalData(trainData, normFeat)    # normalize the data
trainLabel1 = trainData[:, 5]
trainLabel2 = trainData[:, 6]

testFeat = testData[:, useFeat]
#testFeat = normalData(testData, normFeat)
testLabel1 = testData[:, 5]
testLabel2 = testData[:, 6]



#------------------------------------------------
#             fit the SVM model
#------------------------------------------------
clf1 = svm.SVC()
clf1.fit(trainFeat, trainLabel1)   # fit the model

clf2 = svm.SVC()
clf2.fit(trainFeat, trainLabel2)


# do predictions
pred1 = clf1.predict(testFeat)    # make predictoins
print(testFeat.shape)

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
predNB1 = gnb1.predict(testFeat)    # make predictions
print(testFeat.shape)

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



#------------------------------------------------
#       write prediction result into a txt file
#------------------------------------------------
f = open("unnormalized_article_result.txt", "a")
f.write("%f  " %accuPred(pred1, testLabel1) )
f.write("%f  " %accuPred(pred2, testLabel2) )
f.write("%f  " %accuPred(predNB1, testLabel1) )
f.write("%f  " %accuPred(predNB2, testLabel2) )
f.write("%f  " %boost1.score(testFeat, testLabel1) )
f.write("%f  \n" %boost1.score(testFeat, testLabel2) )
f.close()