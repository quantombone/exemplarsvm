% Compile the image resize function
mex -O resize.cc

% Compile Felzenszwalb's 31D features
mex -O features_pedro.cc

%Compile a variant of Felzenszwalb's features which doesn't do normalization
mex -O features_raw.cc

% mulththreaded convolution without blas (see voc-release-4)
mex -O fconvblas.cc -lmwblas -o fconvblas
