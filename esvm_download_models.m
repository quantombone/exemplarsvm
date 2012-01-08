function [models,M,test_set] = esvm_download_models(cls)
%Download a pre-trained PASCAL VOC2007 model from my MIT CSAIL
%homepage (http://people.csail.mit.edu/tomasz/)

f = sprintf('voc2007-%s.mat',cls);
models = [];
M = [];
test_set = {};
if ~exist(f,'file')
  unix(sprintf('wget http://people.csail.mit.edu/tomasz/exemplarsvm/models/%s',f));
end

load(f);

