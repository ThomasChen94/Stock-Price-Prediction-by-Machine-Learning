% this file is to lead data from the excel

appleData  =  xlsread('CompanyData2.xls', 'Apple');
googleData =  xlsread('CompanyData2.xls', 'Google');
teslaData  =  xlsread('CompanyData2.xls', 'Tesla Motors');
microData  =  xlsread('CompanyData2.xls', 'Microsoft');
amazonData =  xlsread('CompanyData2.xls', 'Amazon');
faceData   =  xlsread('CompanyData2.xls', 'Facebook');
yahooData  =  xlsread('CompanyData2.xls', 'Yahoo');
twitData   =  xlsread('CompanyData2.xls', 'Twitter');
oracleData =  xlsread('CompanyData2.xls', 'Oracle');

appleMove  = zeros(size(appleData, 1), 1);
googleMove = zeros(size(googleData, 1), 1);
teslaMove  = zeros(size(teslaData, 1), 1);
microMove  = zeros(size(microData, 1), 1);
amazonMove = zeros(size(amazonData, 1), 1);
faceMove   = zeros(size(faceData, 1), 1);
yahooMove  = zeros(size(yahooData, 1), 1);
twitMove   = zeros(size(twitData, 1), 1);
oracleMove = zeros(size(oracleData, 1), 1);

appleMove   = appleData(:,9) > 0;
googleoMove = googleData(:,9) > 0;
teslaMove   = teslaData(:,9) > 0;
microMove   = microData(:,9) > 0;
amazonMove  = amazonData(:,9) > 0;
faceMove    = faceData(:,9) > 0;
yahooMove   = yahooData(:,9) > 0;
twitMove    = twitData(:,9) > 0;
oracleMove  = oracleData(:,9) > 0;