%% Display a grid of images visualizing the maximal activation of hidden 
% layers for a particular input image

function [ ] = visualizeMaxResponses( theta, hiddenSize, visibleSize, I, W )
activation = feedForwardAutoencoder(theta, hiddenSize, visibleSize, I);
[B,idx] = sort(activation); % idx holds the max indices that we want to visualize

subplot(5,1,1);
imshow(reshape(I, [28 28]));
title('Original Image');
subplot(5,1,2);
imagesc(reshape(displayHidden(W, idx(200)), [28 28]));
title(strcat('Highest Response: Filter ', num2str(idx(200))));
subplot(5,1,3);
imagesc(reshape(displayHidden(W, idx(199)), [28 28]));
title(strcat('Second Highest Response: Filter ', num2str(idx(199))));
subplot(5,1,4);
imagesc(reshape(displayHidden(W, idx(198)), [28 28]));
title(strcat('Third Highest Response: Filter ', num2str(idx(198))));
subplot(5,1,5);
bar(1:200, activation);
title('Overall Response Profile');

end

