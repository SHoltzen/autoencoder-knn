%% Display the output of a hidden layer
% W: The weight matrix, where column i corresponds to the weights from 
%       input node i to hidden units
% j: The j'th hidden layer 
% I: The output image that maximizes the response for this hidden node
function [ I ] = displayHidden( W, j )
    ai = W(j, :)';
    I = zeros(size(ai, 1), 1);
    total = norm(ai);
    for i=1:size(ai, 1)
        I(i) = ai(i) / total;
    end
end

