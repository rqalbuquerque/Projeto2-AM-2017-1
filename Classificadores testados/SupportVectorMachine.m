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
clear accuracySvm
accuracySvm = 1:30;


%Obtendo os melhores parâmetros da Knn
% Mdl = fitcecoc(dados.data(1:4177,1:8), Y,'OptimizeHyperparameters',{'BoxConstraint','KernelFunction','Standardize'},...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%     'expected-improvement-plus'))

% Modelo SVM multiclass (one-versus-one)
%t is a template object that contains options for SVM classification. 
t = templateSVM('Standardize',1,'SaveSupportVectors',true,'KernelFunction','polynomial','BoxConstraint',0.021387);
svmModel = fitcecoc(dados.data(1:4177,1:8), Y,'Learners',t,'ResponseName',responseName,'PredictorNames',predictorNames,'ClassNames',classNames);

minTimeTrain = Inf;
minTimePredict = Inf;
%Loop para 30 repeated 10-fold cross validation
for i = 1:30
    %Particionamento com estratificação dos dados
    stratifiedKfold = cvpartition(Y,'KFold',10);
    
    %Start Train 
    tstart = tic;
    % Cross validation 10-fold knn
    svmModelCV = crossval(svmModel,'cvpartition',stratifiedKfold);
    telapsed = toc(tstart);
    minTimeTrain = min(telapsed,minTimeTrain);
    %Finish Train
    
    %Start Predict 
    tstart = tic;
    predict = kfoldPredict(svmModelCV);
    telapsed = toc(tstart);
    minTimePredict = min(telapsed,minTimePredict);
    %Finish Predict 
    fprintf('Ite: %d \n',i );
    accuracySvm(i) = sum(Y==predict)/ numel(Y);
end

% %Média da acurácia após o 30 repeated 10-fold cross validation
 accuracySvmAfterkfold = sum(accuracySvm)/30;
fprintf('Accuracy: %f%% \n',accuracySvmAfterkfold );
fprintf('Train Time: %f s\n',minTimeTrain );
fprintf('Predict Time: %f s\n',minTimePredict );