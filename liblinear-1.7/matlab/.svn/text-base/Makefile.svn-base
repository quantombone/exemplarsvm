# This Makefile is used under Linux

MATLABDIR ?= /afs/cs.cmu.edu/local/matlab/amd64_f7/7.11/lib/matlab7/
CXX ?= g++
#CXX = g++-3.3
CC ?= gcc
CFLAGS = -Wall -Wconversion -O3 -fPIC -I$(MATLABDIR)/extern/include -I..

MEX = $(MATLABDIR)/bin/mex
MEX_OPTION = CC\#$(CXX) CXX\#$(CXX) CFLAGS\#"$(CFLAGS)" CXXFLAGS\#"$(CFLAGS)"
# comment the following line if you use MATLAB on a 32-bit computer
MEX_OPTION += -largeArrayDims
MEX_EXT = $(shell $(MATLABDIR)/bin/mexext)

OCTAVEDIR ?= /usr/include/octave
OCTAVE_MEX = env CC=$(CXX) mkoctfile
OCTAVE_MEX_OPTION = --mex
OCTAVE_MEX_EXT = mex
OCTAVE_CFLAGS = -Wall -O3 -fPIC -I$(OCTAVEDIR) -I..

all:	matlab

matlab:	binary

octave:
	@make MEX="$(OCTAVE_MEX)" MEX_OPTION="$(OCTAVE_MEX_OPTION)" \
	MEX_EXT="$(OCTAVE_MEX_EXT)" CFLAGS="$(OCTAVE_CFLAGS)" \
	binary

binary: liblinear_train.$(MEX_EXT) predict.$(MEX_EXT) libsvmread.$(MEX_EXT) libsvmwrite.$(MEX_EXT)

liblinear_train.$(MEX_EXT): liblinear_train.c ../linear.h tron.o linear.o linear_model_matlab.o ../blas/blas.a
	$(MEX) $(MEX_OPTION) liblinear_train.c tron.o linear.o linear_model_matlab.o ../blas/blas.a

predict.$(MEX_EXT): predict.c ../linear.h tron.o linear.o linear_model_matlab.o ../blas/blas.a
	$(MEX) $(MEX_OPTION) predict.c tron.o linear.o linear_model_matlab.o ../blas/blas.a

libsvmread.$(MEX_EXT):	libsvmread.c
	$(MEX) $(MEX_OPTION) libsvmread.c

libsvmwrite.$(MEX_EXT):	libsvmwrite.c
	$(MEX) $(MEX_OPTION) libsvmwrite.c

linear_model_matlab.o: linear_model_matlab.c ../linear.h
	$(CXX) $(CFLAGS) -c linear_model_matlab.c

linear.o: ../linear.cpp ../linear.h
	$(CXX) $(CFLAGS) -c ../linear.cpp

tron.o: ../tron.cpp ../tron.h
	$(CXX) $(CFLAGS) -c ../tron.cpp

../blas/blas.a:
	cd ../blas; make OPTFLAGS='$(CFLAGS)' CC='$(CC)';

clean:
	cd ../blas;	make clean
	rm -f *~ *.o *.mex* *.obj
