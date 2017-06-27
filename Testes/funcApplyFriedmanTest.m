%%
%Fun��o que aplica o Friedman test sobre o conjunto de elementos.
%Entrada:
% x     - matriz com o conjunto de valores para aplica��o do teste.
%         dimens�es: N x k 
%         Tipicamente: N - tamanhos do conjunto de dados
%                      k - n�mero de avalia��es
% conf  - n�vel de confian�a para a estat�stica de teste.
%Sa�da:
% hypothesis    - hipotese escolhida (0 - nula, 1 - alternativa)
% fEst          - estat�stica de teste
% cValue        - valor cr�tico da distribui��o F com k-1 e (k-1)(N-1) graus
%             de liberdade

%%
function [hypothesis, stt, cValue, meanRanks] = funcApplyFriedmanTest(x, conf)
    
    [N, k] = size(x);
    
    % ranks
    [R,~] = tiedrank(x');
    ranks = (k+1)-R'
    
    % avg. ranks
    meanRanks = mean(ranks);
    
    % test estat�stic
    X2_F = ((12*N)/(k*k+k))*(sum(meanRanks.^2) - (k*(k+1)^2)/4);
    
    if N>10 || k>5
        %Use Chi-square 
        cValue = chi2inv(conf,k-1);
        hypothesis = X2_F > cValue;
        stt = X2_F;
    else
        %Use Ff
        fEst = ((N-1)*X2_F) / (N*(k-1)-X2_F + eps);
        cValue = finv(conf,k-1,(k-1)*(N-1));
        hypothesis = fEst > cValue;
        stt = fEst;
    end
end






