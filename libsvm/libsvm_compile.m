% add -largeArrayDims on 64-bit machines

mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims -O -c svm.cpp
mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims -O -c svm_model_matlab.c
mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims -O libsvmtrain.c svm.o svm_model_matlab.o

mex CFLAGS="\$CFLAGS -std=c99" -O libsvmpredict.c svm.o ...
    svm_model_matlab.o
%These files are not used
%mex -O libsvmread.c
%mex -O libsvmwrite.c
