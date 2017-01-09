% visualize the data

load('dataAllArt.mat')
dataAll = array;

feat = dataAll(:, 1 : 5);
label1 = dataAll(:, 6) > 0;
label2 = dataAll(:, 7) > 0;

featNorm = dataAll(:, [1 2 4 5]);
featNorm = featNorm ./ repmat(sum(featNorm, 2), 1 ,4);

for i = 1 : 128
    if(label1(i) == 1) 
        scatter3(featNorm(i, 1), featNorm(i, 2), featNorm(i, 4), 50, 'r', 'o','MarkerFaceColor','r' );
        hold on
    else
        scatter3(featNorm(i, 1), featNorm(i, 2), featNorm(i, 4), 50, 'b', '*','MarkerFaceColor','b' );
        hold on
    end
end