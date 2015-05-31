%% ======================================================================
%  Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

inputSize  = 28 * 28;
numLabels  = 5;
hiddenSize = 200;
sparsityParam = 0.1; % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		             %  in the lecture notes). 
lambda = 3e-3;       % weight decay parameter       
beta = 3;            % weight of sparsity penalty term   
maxIter = 400;
rng(123);

%% ======================================================================
%  Load data from the MNIST database
%
%  This loads our training and test data from the MNIST database files.
%  We have sorted the data for you in this so that you will not have to
%  change it.

% Load MNIST database files
% load mnist/mnist_train.amat % loads mnist_train
% first shufle it

% Use the unlabeled test images for pre-training
% load mnist/mnist_test.amat 


N = size(mnist_test,1);
shuffled = mnist_test(randperm(N), :);
trainData = shuffled(1:floor(N/2), 1:784);
trainLabel = shuffled(1:floor(N/2), 785);
testData = shuffled(floor(N/2):N, 1:784);
testLabel = shuffled(floor(N/2):N, 785);


% split this into train-test split

%% ======================================================================
%  STEP 2: Train the sparse autoencoder
%  This trains the sparse autoencoder on the unlabeled training
%  images. 

%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, inputSize);

%  Find opttheta by running the sparse autoencoder on
%  unlabeledTrainingImages

opttheta = theta; 

addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, mnist_test(1:30000, 1:784)'), ...
                              theta, options);


%% -----------------------------------------------------
                          
% Visualize weights
W1 = reshape(opttheta(1:hiddenSize * inputSize), hiddenSize, inputSize);
display_network(W1');

visualizeMaxResponses(opttheta, hiddenSize, inputSize, testData(1, :)', W1);
visualizeMaxResponses(opttheta, hiddenSize, inputSize, testData(2000, :)', W1);
visualizeMaxResponses(opttheta, hiddenSize, inputSize, testData(4000, :)', W1);
visualizeMaxResponses(opttheta, hiddenSize, inputSize, testData(4500, :)', W1);


%%======================================================================
%% Testing K-NN Classifier using the neural network features as inputs

% 3 VS. 7
classA = trainData(trainLabel == 3, :);
classB = trainData(trainLabel == 7, :);


classAFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, classA')';
classBFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, classB')';

combinedFeatures = [classAFeatures; classBFeatures];
combinedLabels = [zeros(size(classAFeatures, 1), 1); ones(size(classBFeatures, 1), 1)];
N = size(combinedLabels, 1);
% now shuffle them up
perm = randperm(N);

combinedLabels = combinedLabels(perm, :);
combinedFeatures = combinedFeatures(perm, :);


trainAData = combinedFeatures(1:floor(N*0.9), :);
testAData = combinedFeatures(floor(N*0.1):N, :);
trainALabel = combinedLabels(1:floor(N*0.9), :);
testALabel = combinedLabels(floor(N*0.1):N, :);

mdl = fitcknn(trainAData, trainALabel);
pred = predict(mdl,testAData);

acc = sum(pred == testALabel) / size(testALabel,1); % 99.99% (only 3 misclassifications out of 5136 data points)

% find some misclassified samples
wrong = pred ~= testALabel;
wrongid = find(wrong); % finds the indices that have non-zero entries
wrongid = perm(wrongid); % reverse the permutation to get back to the original images
subplot(3, 1, 1)
imshow(reshape(classA(wrongid(1), :), [28 28]))
subplot(3, 1, 2);
imshow(reshape(classA(wrongid(2), :), [28 28]))
subplot(3, 1, 3);
imshow(reshape(classA(wrongid(3), :), [28 28]))



%%%%%%%%%% ALL 
N = size(mnist_test, 1);
perm = randperm(N);
shuffled = mnist_test(perm, :);
trainImageData = shuffled(1:floor(N*0.9), 1:784);
trainData = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, shuffled(1:floor(N*0.9), 1:784)');
trainLabel = shuffled(1:floor(N*0.9), 785);
testImageData = shuffled(floor(N*0.9):N, :);
testData = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, shuffled(floor(N*0.9):N, 1:784)');
testLabel = shuffled(floor(N*0.9):N, 785);

% test with neural network features
mdl = fitcknn(trainData', trainLabel);
pred = predict(mdl,testData');
acc = sum(pred == testLabel) / size(testLabel,1); % 0.9658

% try it with various numbers of features to see how accuracy improves
acc = zeros(200, 1);
for i = 10:10:200
    mdl = fitcknn(trainData(1:i, :)', trainLabel); % only first i features
    pred = predict(mdl,testData(1:i, :)');
    acc(i) = sum(pred == testLabel) / size(testLabel,1); 
end

% test with just 768 pixel features (veeery slow)
pixelmdl = fitcknn(trainImageData, trainLabel);
pixelpred = predict(pixelmdl,testImageData(:, 1:784));
acc = sum(pixelpred == testLabel) / size(testLabel,1); % 0.9698


% find classified and misclassified examples
wrong = find(pred ~= testLabel);
right = find(pred == testLabel);

cnt = 1;
for i=1:6
    % show some wrong
    for j=1:3
        subplot(6, 6, 6*(i-1)+j);
        
        imshow(flipdim(rot90(reshape(testImageData(wrong(cnt), 1:784)', ...
            [28 28])), 1));
        title(strcat('(', num2str(6*(i-1)+j), ')', 'L=', ... 
            num2str(testImageData(wrong(cnt), 785)), ...
            ', Got L=', num2str(pred(wrong(cnt)))));

        cnt = cnt + 1;
    end
    % show some right
    for j=4:6
        subplot(6, 6,6*(i-1)+j);
        imshow(flipdim(rot90(reshape(testImageData(right(cnt), 1:784)', [28 28])), 1));
        title(strcat('(', num2str(6*(i-1)+j), ')', 'Correct, L=', ...
            num2str(testImageData(right(cnt), 785))));
        cnt = cnt + 1;
    end
end

