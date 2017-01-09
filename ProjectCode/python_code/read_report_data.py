import xdrlib, sys
import xlrd
import numpy as np
from numpy import *
import math
from scipy import io


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


dataAll = zeros((1,7))  # the header of data

for i in range(11):
    # read in all tables
    table = workExcel.sheets()[i]
    data = getFeature(table, numRows[i])
    dataAll = np.concatenate((dataAll, data))

# before training, we have to shuffle the data
ind = np.arange(1, dataAll.shape[0])
#np.random.shuffle(ind)   # shuffle or not
dataAll = dataAll[ind, :]

np.save('data/report/dataAll.npy', dataAll)
io.savemat('data/report/dataAll.mat', {'array': dataAll})