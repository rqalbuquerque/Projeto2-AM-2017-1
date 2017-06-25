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

% Divisão dos caonjuntos de treinamento 90% e teste 10%
Q = size(x, 2);
Q1 = floor(Q * 0.90);
Q2 = Q - Q1;
ind = randperm(Q);
ind1 = ind(1:Q1);
ind2 = ind(Q1 + (1:Q2));
x1 = x(:, ind1);
t1 = mat(:, ind1);
x2 = x(:, ind2);
t2 = mat(:, ind2);

net = patternnet(10);
numNN = 10;
NN = cell(1, numNN);
perfs = zeros(1, numNN);

% Treino da RNA
NN = train(net, x1, t1);
y2 = NN(x2); 

% Conversão de matriz nas classes
classes = vec2ind(y2); 
classesRef = vec2ind(t2);

% Acurácia
accuracyRna = sum(classes==classesRef)/ numel(classes);