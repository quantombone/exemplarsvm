function [models,betas,M] = esvm_download_models(cls)
f=sprintf('voc2007-%s.mat',cls);
models = [];
betas = [];
M = [];
if ~exist(f,'file')
  unix(sprintf('wget http://people.csail.mit.edu/tomasz/exemplarsvm/models/%s'));
  
end

load(f);