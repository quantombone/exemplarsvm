function [models,M,test_set] = esvm_download_models(cls)
%Download a pre-trained PASCAL VOC2007 model from my MIT CSAIL
%homepage (http://people.csail.mit.edu/tomasz/)
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm


f = sprintf('%s.mat',cls);
models = [];
M = [];
test_set = {};
if ~exist(f,'file')
  unix(sprintf('wget http://people.csail.mit.edu/tomasz/exemplarsvm/models/%s',f));
else
  fprintf(1,'Found %s, not downloading\n',f);
end

if nargout == 0
  return;
end

load(f);

