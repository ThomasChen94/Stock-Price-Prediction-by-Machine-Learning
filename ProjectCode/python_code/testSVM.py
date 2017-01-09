import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

def accuPred(pred, real):
    equalNum = np.sum( (pred == real) )
    return equalNum / pred.shape[0]


trainFeat = sio.loadmat('data/test/trainFeat.mat')
testFeat = sio.loadmat('data/test/testFeat.mat')
trainLabel = sio.loadmat('data/test/trainLabel.mat')
testLabel = sio.loadmat('data/test/testLabel.mat')

trainFeat = trainFeat['Xtrain']
testFeat = testFeat['Xtest']
trainLabel = trainLabel['ytrain']
testLabel= testLabel['ytest']

print(type(trainFeat))


clf = svm.SVC()
clf.fit(trainFeat, trainLabel)
pred = clf.predict(testFeat)
print(pred)
print(accuPred(pred, testLabel))


