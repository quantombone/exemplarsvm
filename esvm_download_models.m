function [models,betas,M] = esvm_download_models(cls)
%Download a pre-trained PASCAL VOC2007 model from my CSAIL webspace

f = sprintf('voc2007-%s.mat',cls);
models = [];
betas = [];
M = [];
if ~exist(f,'file')
  unix(sprintf('wget http://people.csail.mit.edu/tomasz/exemplarsvm/models/%s',f));
end

load(f);