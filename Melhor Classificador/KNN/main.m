%Distance : mahalanobis
%DistanceWeight : inverse
%NumNeighbors : 30

%Script to test knn
TEST_KNN_PARAMETERS = 1;

%Lendo os dados das views
%Dados das planilhas foram colocados em ordem de acordo com a classes
dados_order = '..\\..\\Base de dados\\abalone_Data_Order_3_Class.csv';
delimiterIn = ';';
headerlinesIn = 1;
dataBase = importdata(dados_order,delimiterIn,headerlinesIn);

%Dados da coluna das classes
X = dataBase.data(1:4177,1:8);
Y = dataBase.data(1:4177,9);
tabulate(Y);

%Declaração dos vetores
clear accuracyKnn
accuracyKnn = zeros(1,30);

%Num. de folds utilizados
kfolds = 10;
count = 0;

if TEST_KNN_PARAMETERS == 1

    %Parametros de teste
    k = 30;
    distanceName = 'mahalanobis';
    distanceWeight = 'inverse';
    normOption = 1;

    % knn
    mdlKnn = Knn(k,distanceName,distanceWeight,normOption);  
    
    % Para gerar sempre os folds certos
    rng('default');

    minTimeTrain = Inf;
    minTimePredict = Inf;
    
    %Loop para 30 repeated 10-fold cross validation
    for i = 1:30
        
        % Para gerar sempre os mesmos folds
        rng(i);

        %Gera uma partição
        cv = cvpartition(length(X),'kfold',kfolds);
        for j=1:kfolds
            trainIdxs{j} = find(training(cv,j));
            testIdxs{j}  = find(test(cv,j));
        end

        minTimeTrainCount = 0;
        minTimePredictCount = 0;
        
        totalAcc = 0;
        for j=1:kfolds   
            
            %Start Train 
            tstart = tic;
            mdlKnn = mdlKnn.train(X(trainIdxs{j},:),Y(trainIdxs{j},:));
            telapsed = toc(tstart);
            minTimeTrainCount = minTimeTrainCount + telapsed;
            %Finish Train
            
            %Start Predict 
            tstart = tic;
            knnResult = mdlKnn.predict(X(testIdxs{j},:));
            telapsed = toc(tstart);
            minTimePredictCount = minTimePredictCount + telapsed;
            %Finish Predict 
            
            oriResult = Y(testIdxs{j},:);
            accuracy = mean(knnResult==oriResult);
            totalAcc = totalAcc + accuracy;
        end
        
        minTimeTrain = min(minTimeTrainCount,minTimeTrain);
        minTimePredict = min(minTimePredictCount,minTimePredict);
        accuracyKnn(i) = totalAcc/kfolds;

        count = count + 1

    end
end


% %Média da acurácia após o 30 repeated 10-fold cross validation
accuracyKnnAfterkfold = sum(accuracyKnn)/30;
fprintf('Accuracy: %f%% \n',accuracyKnnAfterkfold );
fprintf('Train Time: %f s\n',minTimeTrain );
fprintf('Predict Time: %f s\n',minTimePredict );
save results_knn accuracyKnn accuracyKnnAfterkfold minTimeTrain minTimePredict