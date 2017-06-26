%Lendo os dados das views
%Dados das planilhas foram colocados em ordem de acordo com a classes
dados_order = '..\\Base de dados\\abalone_Data_Order_3_Class.csv';
delimiterIn = ';';
headerlinesIn = 1;
dados = importdata(dados_order,delimiterIn,headerlinesIn);

% Dados da coluna das classes para rodar na patternnet
% Declara��o da matriz
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
X = zscore(x2,1);

%Dados da coluna das classes
Y = dados.data(1:4177,9);
tabulate(Y);

%Declara��o das classes na ordem que aparecem nos dados
classNames = 1:3;
%Declara��es dos preditores
predictorNames = {'Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight'};
%Nome da coluna das classes
responseName = 'Rings';

%% Setup the parameters
input_layer_size  = 8;    % 8 Input Images of Digits
hidden_layer_size = 10;   % 10 hidden units
num_labels = 3;           % 3 labels, from 1 to 3   
                          % (note that we have mapped "0" to label 1)

%% ================ Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:); initial_Theta2(:)];

%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 200);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, Y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == Y)) * 100);




                          