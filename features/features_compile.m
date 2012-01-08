% Compile the image resize function
%mex -O resize.cc
mex CXX=gcc CXXOPTIMFLAGS='-O3 -DNDEBUG -funroll-all-loops' resize.cc


% Compile Felzenszwalb's 31D features
mex -O features_pedro.cc

%Compile a variant of Felzenszwalb's features which doesn't do normalization
mex -O features_raw.cc

% mulththreaded convolution without blas (see voc-release-4)
%mex -O fconvblas.cc -lmwblas -o fconvblas
%mex CC=gcc LD=gcc COPTIMFLAGS='-O3 -DNDEBUG' fconvblas.cc -lmwblas ...
%          -o fconvblas

mex CXX=gcc CXXOPTIMFLAGS='-O3 -DNDEBUG -funroll-all-loops' fconvblas.cc ...
        -lmwblas
