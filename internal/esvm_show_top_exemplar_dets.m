function I = esvm_show_top_exemplar_dets(test_struct, test_set, ...
                                         models, index, A, B)
% function I = esvm_show_top_exemplar_dets(test_struct, test_set, ...
%                                          models, index)
% Given the results of a detector, show the top exemplar detectiosn
% for exemplar id=[index]. Needs test_set to be able to load images
% for icons
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

bbs = cat(1,test_struct.unclipped_boxes{:});
[aa,bb] = sort(bbs(:,end),'descend');
bbs = bbs(bb,:);

m = models{index};
m.model.svbbs = bbs;

m.model.svbbs = m.model.svbbs(m.model.svbbs(:,6)==index,:);

try
  m.model = rmfield(m.model,'svxs');
catch
end
m.train_set = test_set;

if ~exist('A','var')
  A = 4;
end
if ~exist('B','var')
  B = 4;
end

I = esvm_show_det_stack(m,A,B);

