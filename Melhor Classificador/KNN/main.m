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
accuracyKnn = 1:30;

%Num. de folds utilizados
kfolds = 10;
timesToRun = 30;
count = 0;

if TEST_KNN_PARAMETERS

    %Parametros de teste
    k = 30;
    distanceName = 'euclidean';
    distanceWeight = 'inverse';
    normOption = 1;
    
    % knn
    mdlKnn = Knn(k,distanceName,distanceWeight,normOption);  

    for i=1:timesToRun

        %Gera uma partição
        cv = cvpartition(length(X),'kfold',kfolds);
        for j=1:kfolds
            trainIdxs{j} = find(training(cv,j));
            testIdxs{j}  = find(test(cv,j));
        end

        totalAcc = 0;
        for j=1:kfolds    
            mdlKnn = mdlKnn.train(X(trainIdxs{j},:),Y(trainIdxs{j},:));
            knnResult = mdlKnn.predict(X(testIdxs{j},:));
            oriResult = Y(testIdxs{j},:);
            accuracy = mean(knnResult==oriResult);
            totalAcc = totalAcc + accuracy;
        end
        accuracy = totalAcc/kfolds

        count = count + 1
        
        accuracyKnn(i) = accuracy;
    end
end

disp('Knn Accuracy');
mean(accuracyKnn)