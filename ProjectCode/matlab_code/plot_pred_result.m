load unnormalized_article_result.txt
pred_result = unnormalized_article_result(1:20, :);
x = 1 : size (pred_result, 1);
type = cellstr(['r-o'; 'r-d';'g-o'; 'g-d'; 'b-o'; 'b-d']);
for i = 1 :  size(pred_result, 2)
    plot(x, pred_result(:, i), type{i}, 'linewidth', 1.5)
    hold on;
end

ave = mean(pred_result)
vari =  mean ((pred_result - repmat(ave, size(pred_result, 1), 1) ).* ...
        (pred_result - repmat(ave, size(pred_result, 1), 1) ))