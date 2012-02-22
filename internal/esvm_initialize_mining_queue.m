function mining_queue = esvm_initialize_mining_queue(m, cls)
%Initialize the mining queue with ordering (random by default)
%function mining_queue = esvm_initialize_mining_queue(data_set, ordering)
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

indices = 1:length(m.data_set);
negatives = find(cellfun(@(x)length(x.objects)==0, m.data_set));
myRandomize;
r = randperm(length(negatives));
negatives = negatives(r(1:min(length(negatives), ...
                            m.params.train_max_negative_images)));

positives = find(cellfun(@(x)length(x.objects)~=0, m.data_set));

if m.params.mine_from_negatives == 0
  negatives = [];
end

if m.params.mine_from_positives == 0
  positives = [];
end


indices = indices(cat(1,negatives(:), positives(:)));

mining_queue = cell(0,1);
for zzz = 1:length(indices)
  mining_queue{zzz}.index = indices(zzz);
  mining_queue{zzz}.num_visited = 0;
end

fprintf(1,'Randomizing mining queue with %d images\n',...
        length(mining_queue));
myRandomize;
mining_queue = mining_queue(randperm(length(mining_queue)));
L = min(length(mining_queue),m.params.train_max_mined_images);
mining_queue = mining_queue(1:L);