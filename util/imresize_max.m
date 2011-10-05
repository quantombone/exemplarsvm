function I = imresize_max(I, maxdim)
%Resize image such that maximum size is maxdim

I = im2double(I);
scaler = maxdim/max(size(I,1),size(I,2));
I = max(0.0,min(1.0,imresize(I,scaler)));
