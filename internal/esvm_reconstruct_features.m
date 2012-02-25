function [target_id, target_x] = esvm_reconstruct_features(cb, model, ...
                                                  data_set, N)
% Extract top N detection feature vectors from bounding boxes and the
% to be-computed feature pyramid. This allows us to only store
% (flip,scale,offset) information instead of the 8*8*31 numbers for
% the feature vector.  To reconstruct the feature, we load the image,
% and follow the exact step of flip,scale,offset operations and only
% extract features from a single level of the pyramid (the target
% level)
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

if ~exist('N','var')
  N = size(cb,1);
end
%[~,bb] = sort(cb(:,end),'descend');
N = min(N,size(cb,1));

fprintf(1,'Obtaining top N=%d feature vectors\n',N);
hg_size = model.models{1}.hg_size;

if nargout == 2
  target_x = zeros(prod(hg_size), N);
end
target_id = cell(1, N);

starter = tic;
for i = 1:N
  fprintf(1,'.');

  target_id{i}.scale = cb(i,8);
  target_id{i}.offset = cb(i,9:10);
  target_id{i}.flip = cb(i,7);
  target_id{i}.bb = cb(i,1:4);
  target_id{i}.curid = cb(i,11);
  
  %only do the feature extraction (which takes time!) when two
  %outputs are requested
  if nargout == 1
    continue
  end
  
  I = toI(data_set{cb(i,11)});

  if (target_id{i}.flip == 1)
    I = flip_image(I);
  end

  I = resize(I,target_id{i}.scale);
  
  %%NOTE: both sbin=8 and padder=5 are hard-coded here
  %fprintf(1,'Note: hardcoded esvm_features\n');
  
  full = model.params.init_params.features(I, ...
                                           model.params ...
                                           .init_params.sbin);
  
  p = model.params.detect_pyramid_padding;
  f = padarray(full,[p+1 p+1 0]);

  %rootmatch = cellfun2(@(x)(x-model.models{1}.b),fconvblas(f, {model.models{1}.w}, 1, 1));  
  % s = model.models{1}.w(:)'*reshape(f(target_id{i}.offset(1)-1+(1:(hg_size(1))),...
  %                                     target_id{i}.offset(2)-1+(1:(hg_size(2))),:), ...
  %                                   [],1)-model.models{1}.b
   
  f = f(target_id{i}.offset(1)-1+(1:(hg_size(1))), ...
        target_id{i}.offset(2)-1+(1:(hg_size(2))),:);

  target_x(:,i) = f(:);
end
fprintf(1,'esvm_reconstruct_features took: %.3fsec\n',toc(starter));

function id = get_file_id(filer)
[tmp,curid,tmp] = fileparts(filer);
id = str2num(curid);
