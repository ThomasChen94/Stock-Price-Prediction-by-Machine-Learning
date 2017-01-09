% visualize the data

load('dataAll.mat')
dataAll = array;

feat = dataAll(:, 1 : 5);
label1 = dataAll(:, 6) > 0;
label2 = dataAll(:, 7) > 0;


for i = 1 : 128
    if(label1(i) == 1) 
        scatter3(feat(i, 1), feat(i, 2), feat(i, 3), 50, 'r', 'o','MarkerFaceColor','r' );
        hold on
    else
        scatter3(feat(i, 1), feat(i, 2), feat(i, 3), 50, 'b', '*','MarkerFaceColor','b' );
        hold on
    end
end