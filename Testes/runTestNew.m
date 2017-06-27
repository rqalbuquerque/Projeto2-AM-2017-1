%% load
load('results_new_knn.mat');
knnResults = struct('accuracy', accuracyKnn, ...
                   'minTimeTrain', minTimeTrain, ...
                   'minTimePredict', minTimePredict);

       
%% intervalo de confiança
conf = 0.95;
[pd,ci] = funcGenConfidenceInterval(knnResults.accuracy, conf)
