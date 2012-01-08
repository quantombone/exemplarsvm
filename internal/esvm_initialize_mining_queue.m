function mining_queue = esvm_initialize_mining_queue(imageset, ordering)
%Initialize the mining queue with ordering (random by default)
%function mining_queue = esvm_initialize_mining_queue(imageset, ordering)
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).

if ~exist('ordering','var')
  fprintf(1,'Randomizing mining queue\n');
  myRandomize;
  ordering = randperm(length(imageset));
end

mining_queue = cell(0,1);
for zzz = 1:length(ordering)
  mining_queue{zzz}.index = ordering(zzz);
  mining_queue{zzz}.num_visited = 0;
end

