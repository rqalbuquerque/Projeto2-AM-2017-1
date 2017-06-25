%Lendo os dados das views
%Dados das planilhas foram colocados em ordem de acordo com a classes
dados_order = '..\\Base de dados\\abalone_Data_Order_3_Class.csv';
delimiterIn = ';';
headerlinesIn = 1;
dados = importdata(dados_order,delimiterIn,headerlinesIn);

%Dados da coluna das classes
Y = dados.data(1:4177,9);
tabulate(Y);

%Declara��o das classes na ordem que aparecem nos dados
classNames = 1:3;
%Declara��es dos preditores
predictorNames = {'Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight'};
%Nome da coluna das classes
responseName = 'Rings';

%Declara��o dos vetores
clear accuracyKnn
accuracyKnn = 1:30;


%Obtendo os melhores par�metros da Knn
% Mdl = fitcknn(dados.data(1:4177,1:8),Y,'OptimizeHyperparameters',{'Distance','NumNeighbors','DistanceWeight','Standardize'},...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%     'expected-improvement-plus','Optimizer',...
%     'gridsearch','NumGridDivisions',...
%     10))

knnModel = fitcknn(dados.data(1:4177,2:8),Y,'ClassNames',classNames,'Distance','euclidean','ResponseName',responseName,'PredictorNames',predictorNames,'Standardize',1,'NumNeighbors',31); %22 melhor acur�cia

minTimeTrain = Inf;
minTimePredict = Inf;
%Loop para 30 repeated 10-fold cross validation
for i = 1:30
    %Particionamento com estratifica��o dos dados
    stratifiedKfold = cvpartition(Y,'KFold',10);
    
    %Start Train 
    tstart = tic;
    % Cross validation 10-fold knn
    knnModelCV = crossval(knnModel,'cvpartition',stratifiedKfold);
    telapsed = toc(tstart);
    minTimeTrain = min(telapsed,minTimeTrain);
    %Finish Train
    
    %Start Predict 
    tstart = tic;
    predict = kfoldPredict(knnModelCV);
    telapsed = toc(tstart);
    minTimePredict = min(telapsed,minTimePredict);
    %Finish Predict 
    
    accuracyKnn(i) = sum(Y==predict)/ numel(Y);
end

% %M�dia da acur�cia ap�s o 30 repeated 10-fold cross validation
accuracyKnnAfterkfold = sum(accuracyKnn)/30;
fprintf('Accuracy: %f%% \n',accuracyKnnAfterkfold );
fprintf('Train Time: %f s\n',minTimeTrain );
fprintf('Predict Time: %f s\n',minTimePredict );