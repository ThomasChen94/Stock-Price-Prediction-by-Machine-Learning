load ('dataAll.mat');
dataAll = array;

ytest = dataAll(170:195,6) > 0;
Xtest = dataAll(170:195,1:5);

numTestDocs = size(Xtest, 1);
numTokens = size(Xtest, 2);

predictions = zeros(numTestDocs, 1);

%---------------
% YOUR CODE HERE
testFeat = full(Xtest);
testLabel = full(ytest);

kerTest = zeros(numTestDocs, m);

for i = 1 : numTestDocs
    for j = 1:m
        kerTest(i,j) = Ker(trainFeat(j,:), testFeat(i,:), tao);
    end
end

predictions = kerTest * alphaOutput; 


%---------------
% Compute the error on the test set
error = sum(ytest .* predictions <= 0) / numTestDocs;
fprintf(1, 'Test error for SVM: %1.4f\n', error);
