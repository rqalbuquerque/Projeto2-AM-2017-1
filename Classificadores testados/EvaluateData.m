%Lendo os dados das views
%Dados das planilhas foram colocados em ordem de acordo com a classes
dados_order = '..\\Base de dados\\abalone_Data_Order_3_Class.csv';
delimiterIn = ';';
headerlinesIn = 1;
dados = importdata(dados_order,delimiterIn,headerlinesIn);

%Dados da coluna das classes
Y = dados.data(1:4177,9);

%Declarações dos preditores
predictorNames = {'S.','L.','D.','H.','W. W.','S. W.','V. W.','S. W.','R.'};
%predictorNames = {'S.','L.','D.','H.','W. W.','S. W.','V. W.','S. W.'};

fprintf('----------- Correlações -------------\n');
fprintf('C.  Sex: %f | max: %f min: %f | var: %f\n',corr2(dados.data(1:4177,1),Y),max(dados.data(1:4177,1)), min(dados.data(1:4177,1)),var(dados.data(1:4177,1)) );
fprintf('C.  Length: %f | max: %f min: %f | var: %f\n',corr2(dados.data(1:4177,2),Y),max(dados.data(1:4177,2)), min(dados.data(1:4177,2)),var(dados.data(1:4177,2)));
fprintf('C.  Diameter: %f | max: %f min: %f | var: %f\n',corr2(dados.data(1:4177,3),Y),max(dados.data(1:4177,3)), min(dados.data(1:4177,3)),var(dados.data(1:4177,3)));
fprintf('C.  Height: %f | max: %f min: %f | var: %f\n',corr2(dados.data(1:4177,4),Y),max(dados.data(1:4177,4)), min(dados.data(1:4177,4)),var(dados.data(1:4177,4)));
fprintf('C.  Whole weight: %f | max: %f min: %f | var: %f\n',corr2(dados.data(1:4177,5),Y),max(dados.data(1:4177,5)), min(dados.data(1:4177,5)),var(dados.data(1:4177,5)));
fprintf('C.  Shucked weight: %f | max: %f min: %f | var: %f\n',corr2(dados.data(1:4177,6),Y),max(dados.data(1:4177,6)), min(dados.data(1:4177,6)),var(dados.data(1:4177,6)));
fprintf('C.  Viscera weight: %f | max: %f min: %f | var: %f\n',corr2(dados.data(1:4177,7),Y),max(dados.data(1:4177,7)), min(dados.data(1:4177,7)),var(dados.data(1:4177,7)));
fprintf('C.  Shell weight: %f | max: %f min: %f | var: %f\n',corr2(dados.data(1:4177,8),Y),max(dados.data(1:4177,8)), min(dados.data(1:4177,8)),var(dados.data(1:4177,8)));

corrplot(dados.data,'varNames',predictorNames)