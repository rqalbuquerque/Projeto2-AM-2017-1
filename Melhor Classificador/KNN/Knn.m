classdef Knn
    
   properties
        % parameters
        k = 3;
        distanceName = 'euclidean';
        distanceWeight = 'none';
        normOption = 0;
        
        % train examples
        X = [];
        Y = [];
        maxValues = [];
        minValues = [];
        nSamples = 0;
   end 
   
   methods
       
        function obj = Knn(K,distName,distWeight,normOption)
            obj.k = K;
            obj.distanceName = distName;
            obj.distanceWeight = distWeight;
            obj.normOption = normOption;
        end
        
        function obj = train(obj,x,y)
            
            if size(x,1) ~= size(y,1)
               error('The number of examples and labels dont match!');
            end
            
            obj.nSamples = size(x,1);
            
            obj.maxValues = max(x);
            obj.minValues = min(x);
            
            obj.Y = y;
            obj.X = x;
            
            if obj.normOption
                normX = x - ones(obj.nSamples,1)*obj.minValues;
                obj.X = normX ./ (ones(obj.nSamples,1)*(obj.maxValues-obj.minValues));
            end
        end
        
        function d = calcDist(obj, x1, x2)

            if strcmp(obj.distanceName,'mahalanobis')
                %d = mahal(x1,x2);
            elseif strcmp(obj.distanceName,'euclidean')
                d = sqrt(sum((x1-x2).^2,2));
            end

        end
        
        function pred = predict(obj, samples)
                       
            if size(samples,2) ~= size(obj.X,2)
               error('Wrong number of sample attributes!');
            end
            
            pred = zeros(size(samples,1),1);
            
            for i=1:length(samples)
                
                normSample = samples(i,:);
                
                if obj.normOption
                    normSample = (samples(i,:)-obj.minValues) ./ (obj.maxValues-obj.minValues);
                end
                
                % get k nearest neighbor
                fun = @(a,b) obj.calcDist(a,b);
                dists = bsxfun(fun,ones(obj.nSamples,1)*normSample,obj.X);
                
                % compute weight and classify
                [sortedDists,idxs] = sort(dists);
                classNames = obj.Y(idxs(1:obj.k));
                classFind = unique(classNames);
                dists = sortedDists(1:obj.k);
                
                if strcmp(obj.distanceWeight, 'none')
                    dists = ones(length(dists),1);
                elseif strcmp(obj.distanceWeight,'inverse')
                    dists = 1.0 ./ dists;
                elseif strcmp(obj.distanceWeight,'inverse_squared')
                    dists = 1.0 ./ (dists.^2);
                end
                
                maxAccDist = -inf;
                maxIdxClass = 0;
                for c=1:length(classFind)
                    sumDist = sum(dists(classNames == classFind(c)));
                    if sumDist > maxAccDist
                       maxAccDist = sumDist;
                       maxIdxClass = classFind(c);
                    end
                end
                
                pred(i) = maxIdxClass;
            end
            
        end
        
   end
   
end