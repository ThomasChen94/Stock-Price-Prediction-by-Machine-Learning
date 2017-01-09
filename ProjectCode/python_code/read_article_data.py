import xdrlib, sys
import xlrd
import numpy as np
from numpy import *
import math
from scipy import io


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

np.save('data/article/dataAll.npy', dataAll)
io.savemat('data/article/dataAll.mat', {'array': dataAll})

total = dataAll.shape[0]
trainNum = math.floor(total * 0.8 )

# the we split the data into trainig set and testing set
trainData = dataAll[1 : trainNum, :]
trainData[:, 5 : 7] = trainData[:, 5 : 7] > 0

testData = dataAll[trainNum : total, :]
testData[:, 5 : 7] = testData[:, 5 : 7] > 0

print(testData.shape)

np.save('data/article/trainData.npy', trainData)
np.save('data/article/testData.npy', testData)