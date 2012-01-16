function esvm_compile
%Compiles everything compile-able, must be in exemplarsvm parent directory
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).


addpath(genpath(pwd));

status = test_compiled;
if status == 1
  fprintf(1,'All functions already compiled\n');
  return;
else
  cwd = pwd;
  cd features
  features_compile;
  cd ../util/
  util_compile;
  cd ../libsvm/
  libsvm_compile
  status2 = test_compiled;

  if status2 == 1
    fprintf(1,'All functions just compiled\n');
    cd(cwd);
    return;
  else
    fprintf(1,'Compilation failed\n');
    cd(cwd);
    error('Compilation failed');
  end
end
  

function status = test_compiled
%Test the C++ mex functions
%Return 1 if all function calls were successful, 0 otherwise

status = 1;

I = rand(20,20,3);

try
  I2 = resize(I,.34);
catch
  fprintf(1,'resize.cc not compiled\n');
  status = 0;
  return;
end

try
  f = features_pedro(I,8);
  f = features_raw(I,8);
catch
  fprintf(1,'features_pedro.cc not compiled\n');
  status = 0;
  return;
end

try
  f = rand(10,10,31);
  w = rand(2,2,31);  
  res = fconvblas(f,{w},1,1);
catch
  fprintf(1,'fconvblas.cc not compiled\n');
  status = 0;
  return;
end

try
  numbers = (1:10)';
  [a,b] = psort(numbers,3);
catch
  fprintf(1,'psort.cpp not compiled\n');
  status = 0;
  return;
end

try
  Y = [-1 -1 -1 1 1 1]';
  X = randn(4,6);
  svm_c = .01;
  svm_model = libsvmtrain(Y, X',sprintf(['-s 0 -t 0 -c' ...
                    ' %f -q'], svm_c));
catch
  fprintf(1,'libsvmtrain.cpp not compiled\n');
  status = 0;
  return;
end
