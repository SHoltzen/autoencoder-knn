function [ newY ] = knn( xTrain, yTrain, newX, k )
    D = squareform(pdist([xTrain; newX]));
    [m,n] = size(D);
    distances = D(n,1:n-1);
    
    [min, minIndex] = sort(distances);
    if(min(1)==0)
        newY = yTrain(minIndex(1));
    else
        lowest_k = minIndex(1:k);
        labels = yTrain(lowest_k);
        newY = mode(labels);
    end
end

