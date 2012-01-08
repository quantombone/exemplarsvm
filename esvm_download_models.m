function [models,M,test_set] = esvm_download_models(cls)
%Download a pre-trained PASCAL VOC2007 model from my MIT CSAIL homepage

f = sprintf('voc2007-%s.mat',cls);
models = [];
M = [];
test_set = {};
if ~exist(f,'file')
  unix(sprintf('wget http://people.csail.mit.edu/tomasz/exemplarsvm/models/%s',f));
end

load(f);

% if isfield(models{1},'I') && isstr(models{1}.I) && length(models{1}.I)>=7 ...
%       && strcmp(models{1}.I(1:7),'http://')
%   fprintf(1,'Warning: Models have images as URLs\n -- Use [models]=esvm_update_voc_models(models,local_dir);\n');
% end
  
