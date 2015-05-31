%perform PCA whitening on the image
% see http://ufldl.stanford.edu/wiki/index.php/Implementing_PCA/Whitening
% I: A vector of input images (NxNxNxK, RGB with index into image)
% k: The number of dimensions to reduce using the principle components
% Iout: The outputted reduced image
function [ Iout ] = preprocess_image( I, k )
    imgmat = zeros(size(I,1)*size(I,1), size(I,4));
    for i=1:size(I, 4)
        imgmat(:, i) = reshape(rgb2gray(I(:,:,:,i)), [32*32 1]);
    end
    epsilon = 10^-5;
    c = cov(imgmat);
    [U S V] = svd(c);
    Iout =  I;
end

