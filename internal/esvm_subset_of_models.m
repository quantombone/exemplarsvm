function [models,M] = esvm_subset_of_models(models,M,subset)
%Choose a subset of models and a subset of the calibration matrix
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

if ~exist('subset','var')
  s1 = cellfun(@(x)size(x.model.w,1),models);
  s2 = cellfun(@(x)size(x.model.w,2),models);
  urows = unique([s1; s2]','rows');
  [c,d] = ismember([s1; s2]',urows,'rows');
  [aa,bb] = max(hist(d,1:size(urows,1)));
  target = urows(bb,:);
  subset = find(s1==target(1) & s2==target(2));
end
hits = repmat(1:length(models),2,1);

models = models(subset);

subset2 = find(ismember(hits(:),subset));
M.w = cellfun2(@(x)x(subset2),M.w);
M.w = M.w(subset2);
M.b = M.b(subset2);
M.C = M.C(subset2,subset2);