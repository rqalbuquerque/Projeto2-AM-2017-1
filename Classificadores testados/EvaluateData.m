%Lendo os dados das views
%Dados das planilhas foram colocados em ordem de acordo com a classes
dados_order = '..\\Base de dados\\abalone_Data_Order_3_Class.csv';
delimiterIn = ';';
headerlinesIn = 1;
dados = importdata(dados_order,delimiterIn,headerlinesIn);

%Dados da coluna das classes
Y = dados.data(1:4177,9);

%Declaração das classes na ordem que aparecem nos dados
classNames = 1:3;
%Declarações dos preditores
predictorNames = {'Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight'};
%Nome da coluna das classes
responseName = 'Rings';

fprintf('----------- Correlações -------------\n');
fprintf('C.  Sex: %f \n',corr2(dados.data(1:4177,1),Y) );
fprintf('C.  Length: %f \n',corr2(dados.data(1:4177,2),Y) );
fprintf('C.  Diameter: %f \n',corr2(dados.data(1:4177,3),Y) );
fprintf('C.  Height: %f \n',corr2(dados.data(1:4177,4),Y) );
fprintf('C.  Whole weight: %f \n',corr2(dados.data(1:4177,5),Y) );
fprintf('C.  Shucked weight: %f \n',corr2(dados.data(1:4177,6),Y) );
fprintf('C.  Viscera weight: %f \n',corr2(dados.data(1:4177,7),Y) );
fprintf('C.  Shell weight: %f \n',corr2(dados.data(1:4177,8),Y) );

