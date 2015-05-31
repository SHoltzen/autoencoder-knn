%% Shows a bar plot of overall activation of each hidden node for all the
% images in I.
% I: Vector of images of the same class in each column
function [  ] = visualizeActivationByClass( theta, hiddenSize, visibleSize, I, W )
activations = zeros(hiddenSize, size(I, 2));
for i=1:size(I, 2)
    activations(:, i) = feedForwardAutoencoder(theta, hiddenSize, visibleSize, I(:, i)');
end
avg = mean(activations'); % each column is 1 feature, take mean of features

bar(1:200, avg);
title('Average detector response');

end

