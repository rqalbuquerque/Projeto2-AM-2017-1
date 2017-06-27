%%
%Função que gera uma estimativa pontual e um intervalo de confiança
%a cerca do conjunto de valores de entrada.
%Entrada:
% x     - conjunto de valores (taxas de acerto/erro de um classificador)
% conf  - valor de confiança para estimativa do intevalo (0.0 < conf <= 1.0)
%Saída:
% pd    - estimativa pontual da distribuição de probabilidade normal
% ci    - intervalo de confiança para os valores de estimativa pontual

%%
function [pd,ci] = funcGenConfidenceInterval(x, conf)

    if size(x,2) > size(x,1)
       x = x'; 
    end

    if conf <= 0.0
       pd = [0.0, 0.0];
       ci = [0.0 0.0; 0.0 0.0];
    else
        if conf > 1.0
           conf = conf/100.0;
        end

        alpha = 1.0001-conf;
        probDist = fitdist(x,'Normal');
        ci = paramci(probDist,'Alpha',alpha);
        pd = [probDist.mu, probDist.sigma];
    end
end