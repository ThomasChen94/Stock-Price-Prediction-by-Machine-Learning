
load ('dataAll.mat');
dataAll = array;

ytrain = dataAll(1:170,6) > 0;
Xtrain = dataAll(1:170,1:5);

numTrainDocs = size(Xtrain, 1);
numTokens = size(Xtrain, 2);

average_alpha = zeros(numTrainDocs, 1);

trainFeat = full(Xtrain);
trainLabel = full(ytrain);

m = numTrainDocs;
alpha = zeros(m, 1);
tao = 8;
lambda = 1/ (64 * m);
iterStepTotal = 40 * m;
alphaOutput = zeros(m, 1);

K = zeros(m,m);
% first compute the kernel matrix
for i = 1 : m
    for j = 1 : m
        K(i,j) = Ker(trainFeat(i,:), trainFeat(j,:), tao);
    end
end

for s = 1:iterStepTotal
    loss = 1/ m * sum( 1 - trainLabel.* (K * alpha));
    i = unidrnd(m);
    %i = mod(s, m) + 1;
    deri = (trainLabel(i) * K(i,:) * alpha < 1) * (- trainLabel(i) * K(:,i));  % the derivative
    deri = deri + m * lambda * K(:,i) * alpha(i);
    stepSize = 1/(sqrt(s));
    alpha = alpha - stepSize * deri;
    alphaOutput = alphaOutput + 1 / iterStepTotal * alpha;
end




%---------------
