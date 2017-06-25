%Lendo os dados das views
%Dados das planilhas foram colocados em ordem de acordo com a classes
dados_order = '..\\Base de dados\\abalone_Data_Order_3_Class.csv';
delimiterIn = ';';
headerlinesIn = 1;
dados = importdata(dados_order,delimiterIn,headerlinesIn);

% Dados da coluna das classes para rodar na patternnet
% Declaração da matriz
mat = zeros(3, 4177);
t = dados.data(1:4177,9)';
% Loop para converter a coluna de classes em um vetor de matriz de 0 e 1
for i = 1:4177
   if t(i)==1
       mat(1,i)=1;
   elseif t(i)==2
       mat(2,i)=1;
   else
       mat(3,i)=1;
   end
end

%Dados das colunas dos preditores
x2 = dados.data(1:4177,1:8);

%Normalizando os dados;
x = zscore(x2,1)';

%Treina o modelo
k = 10;
cv = cvpartition(length(x),'kfold',k);
for i=1:k
    trainIdxs{i} = find(training(cv,i));
    testIdxs{i}  = find(test(cv,i));
end

%Parametros de teste
paramLenLayer1 = [2 5 10];
paramLenLayer2 = [0 2 5 10];
paramEpochs = [1000 2000 4000];
paramLearnRate = [0.01 0.05 0.1];
paramTransFcn = {'logsig' 'tansig'};

count = 0;
bestAcc = 0;
bestLayers = [];
bestNEpochas = 0;
bestLearnRat = 0;
bestTransfFunc = [];
for l1 = 1:length(paramLenLayer1)
    for l2 = 1:length(paramLenLayer2)
        for ep = 1:length(paramEpochs)
            for lr = 1:length(paramLearnRate)
                for tf = 1:1
                    nLayers1 = paramLenLayer1(l1);
                    nLayers = nLayers1;
                    nLayers2 = paramLenLayer2(l2);  
                    if(nLayers2 > 0)
                        nLayers = [nLayers1 nLayers2];
                    end
                    nEpochas = paramEpochs(ep);
                    learnRat = paramLearnRate(lr);
                    transfFunc = char(paramTransFcn(tf));
                        
                    totalAcc = 0;
                    for i=1:k    
                        netModel = patternnet(nLayers, 'traingd');
                        netModel.trainParam.epochs = nEpochas;
                        netModel.trainParam.lr = learnRat;
                        netModel.layers{1}.transferFcn = transfFunc; 

                        net = train(netModel, x(:,trainIdxs{i}), mat(:,trainIdxs{i}));

                        nnResult = vec2ind(net(x(:,testIdxs{i})));
                        orResult = vec2ind(mat(:,testIdxs{i}));
                        accuracy = sum(nnResult==orResult)/ numel(nnResult);
                        totalAcc = totalAcc + accuracy;
                    end
                    accuracyRna = totalAcc/k;
                    
                    count = count + 1
                    
                    if(accuracyRna > bestAcc)
                        bestAcc = accuracyRna;
                        bestLayers = [nLayers1 nLayers2];
                        bestNEpochas = nEpochas;
                        bestLearnRat = learnRat;
                        bestTransfFunc = transfFunc;
                    end
                end
            end
        end
    end
end

bestAcc = accuracyRna;
bestLayers = [nLayers1 nLayers2];
bestNEpochas = nEpochas;
bestLearnRat = learnRat;
bestTransfFunc = transfFunc;

save bestParamNN bestAcc bestLayers bestLearnRat bestNEpochas bestTransfFunc