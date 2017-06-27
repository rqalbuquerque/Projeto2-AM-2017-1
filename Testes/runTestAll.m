%% loads
load('results_decision_tree.mat');
dtResults = struct('accuracy', accuracyTree, ...
                   'minTimeTrain', minTimeTrain, ...
                   'minTimePredict', minTimePredict);
clear minTimeTrain minTimePredict;


load('results_knn.mat');
knnResults = struct('accuracy', accuracyKnn, ...
                    'minTimeTrain', minTimeTrain, ...
                    'minTimePredict', minTimePredict);
clear minTimeTrain minTimePredict;


load('results_svm.mat');
svmResults = struct('accuracy', accuracySvm, ...
                    'minTimeTrain', minTimeTrain, ...
                    'minTimePredict', minTimePredict);
clear minTimeTrain minTimePredict;


load('results_rna.mat');
rnaResults = struct('accuracy', accuracyRna, ...
                    'minTimeTrain', minTimeTrain, ...
                    'minTimePredict', minTimePredict);
clear minTimeTrain minTimePredict;


%% intervalo de confiança
conf = 0.95;

[pd_dt,ci_dt] = funcGenConfidenceInterval(dtResults.accuracy, conf);
[pd_knn,ci_knn] = funcGenConfidenceInterval(knnResults.accuracy, conf);
[pd_svm,ci_svm] = funcGenConfidenceInterval(svmResults.accuracy, conf);
[pd_rna,ci_rna] = funcGenConfidenceInterval(rnaResults.accuracy, conf);


%% teste de Friedman
allAccuracies = [dtResults.accuracy' ...
                 knnResults.accuracy' ...
                 svmResults.accuracy' ... 
                 rnaResults.accuracy'];
[hypothesis, stt, cValue, meanRanks] = funcApplyFriedmanTest(allAccuracies, conf);


%% teste de Nemenyi
N = size(dtResults.accuracy',1);
k = 4;
[ntest,qa] = funcApplyNemenyiTest(N, k, meanRanks, conf)
