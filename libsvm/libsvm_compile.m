% add -largeArrayDims on 64-bit machines

mex -largeArrayDims -O -c svm.cpp
mex -largeArrayDims -O -c svm_model_matlab.c
mex -largeArrayDims -O libsvmtrain.c svm.o svm_model_matlab.o

%These files are not used
%mex -O svmpredict.c svm.obj svm_model_matlab.obj
%mex -O libsvmread.c
%mex -O libsvmwrite.c
