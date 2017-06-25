%Lendo os dados das views
%Dados das planilhas foram colocados em ordem de acordo com a classes
dados_order = '..\\Base de dados\\abalone_Data_Order_3_Class.csv';
delimiterIn = ';';
headerlinesIn = 1;
dados = importdata(dados_order,delimiterIn,headerlinesIn);

%Dados da coluna das classes
Y = dados.data(1:4177,9);
tabulate(Y);

%Declaração das classes na ordem que aparecem nos dados
classNames = 1:3;
%Declarações dos preditores
predictorNames = {'Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight'};
%Nome da coluna das classes
responseName = 'Rings';

%Declaração dos vetores
clear accuracyKnn
accuracyKnn = 1:30;
clear accuracySvm
accuracySvm = 1:30;
clear accuracyTree
accuracyTree = 1:30;
%clear accuracyKnnAfterkfold
%accuracyKnnAfterkfold = 1:64;

%{
%Loop para escolher o valor de vizinhos do knn
for n = 1:64
    %Função KNN
    knnModel = fitcknn(dados.data(1:4177,1:8),Y,'ClassNames',classNames,'Distance','euclidean','ResponseName',responseName,'PredictorNames',predictorNames,'Standardize',1,'NumNeighbors',n); 
    
    %30 repeated 10-fold cross validation
    for i = 1:30
        stratifiedKfold = cvpartition(Y,'KFold',10);
        knnModelCV = crossval(knnModel,'cvpartition',stratifiedKfold);
        accuracyKnn(i) = sum(Y==kfoldPredict(knnModelCV))/ numel(Y);
    end
    
    %Acuracy after 30 repeated 10-fold
    accuracyKnnAfterkfold(n) = sum(accuracyKnn)/30; %30
end
%}

% Modelo KNN

% knnModel = fitcknn(dados.data(1:4177,1:8),Y,'ClassNames',classNames,'Distance','euclidean','ResponseName',responseName,'PredictorNames',predictorNames,'Standardize',1,'NumNeighbors',31); %22 melhor acurácia
%Obtendo os melhores parâmetros da Knn
% Mdl = fitcknn(dados.data(1:4177,1:8),Y,'OptimizeHyperparameters',{'Distance','NumNeighbors','DistanceWeight','Standardize'},...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%     'expected-improvement-plus','Optimizer',...
%     'gridsearch','NumGridDivisions',...
%     10))

% Modelo SVM multiclass (one-versus-one)
% t is a template object that contains options for SVM classification. 
% t = templateSVM('Standardize',1,'SaveSupportVectors',true);
% svmModel = fitcecoc(dados.data(1:4177,1:8), Y,'Learners',t,'ResponseName',responseName,'PredictorNames',predictorNames,'ClassNames',classNames);

%Obtendo os melhores parâmetros da SVM
% Mdl = fitcecoc(dados.data(1:4177,1:8), Y,'OptimizeHyperparameters',{'BoxConstraint','KernelFunction','Standardize'},...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%     'expected-improvement-plus','Optimizer',...
%     'gridsearch','NumGridDivisions',...
%     10))
% % Modelo árvore de decisão
% treeModel = fitctree(dados.data(1:4177,1:8),Y,'ResponseName',responseName,'PredictorNames',predictorNames,'ClassNames',classNames);


%Loop para 30 repeated 10-fold cross validation
% for i = 1:30
%     %Particionamento com estratificação dos dados
%     stratifiedKfold = cvpartition(Y,'KFold',10);
%     % Cross validation 10-fold knn
%     knnModelCV = crossval(knnModel,'cvpartition',stratifiedKfold);
%     accuracyKnn(i) = sum(Y==kfoldPredict(knnModelCV))/ numel(Y);
%     % Cross validation 10-fold svm
%     svmModelCV = crossval(svmModel,'cvpartition',stratifiedKfold);
%     accuracySvm(i) = sum(Y==kfoldPredict(svmModelCV))/ numel(Y);
%     % Cross validation 10-fold árvore de decisão
%     treeModelCV = crossval(treeModel,'cvpartition',stratifiedKfold);
%     accuracyTree(i) = sum(Y==kfoldPredict(treeModelCV))/ numel(Y);
% end
% 
% %Média da acurácia após o 30 repeated 10-fold cross validation
% accuracyKnnAfterkfold = sum(accuracyKnn)/30; 
% accuracySvmAfterkfold = sum(accuracySvm)/30;
% accuracyTreeAfterkfold = sum(accuracyTree)/30;
