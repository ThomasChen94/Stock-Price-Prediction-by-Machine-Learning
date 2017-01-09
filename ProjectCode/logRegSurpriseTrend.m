% This file uses the differene between earning per share and expected value
% to predict the movement of stock price
% Algorithm Used: logistic regression

loadStockFile;   % first load the data

% we use first 8 companies to predict oracle

dataTrain = [appleData(:,8); googleData(:,8); microData(:,8); faceData(:,8); ...
             amazonData(:,8); twitData(:,8); yahooData(:,8); oracleData(:,8)];
         
labelTrain = [appleMove; googleMove; microMove; faceMove; ...
             amazonMove; twitMove; yahooMove; oracleMove];
    
dataScaleTrain = sigmf(dataTrain, [20 0]);  % scale the data
theta = glmfit(dataScaleTrain, [labelTrain ones(size(labelTrain, 1), 1)], 'binomial');

dataTest = sigmf(teslaData(:,8), [20 0]);

pred = (sigmf ( theta(2) *  dataTest + theta(1), [1 0]) > 0.5);
accu = (sum(pred == teslaMove)) / size(teslaMove,1);
