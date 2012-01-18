function esvm_show_M(models, M)
% Show the M calibration matrix which, as the co-activation matrix, is
% a type of visual memex matrix.
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

for i = 1:length(models)
  
  [aa,bb] = sort(M.w{i},'descend');
  
  figure(1)
  clf
  subplot(4,4,1)
  imagesc(esvm_get_exemplar_icon(models,i))
  axis image
  axis off
  title(sprintf('Source exemplar %d',i))
  
  for q = 1:15
    %if aa(q) <= 1
    %  break
    %end
    subplot(4,4,q+1)

    flipper = 0;
    if bb(q) > length(models)
      bb(q) = bb(q) - length(models);
      flipper = 1;
      I=flip_image(esvm_get_exemplar_icon(models,bb(q)));
    else
      I=esvm_get_exemplar_icon(models,bb(q));
    end
    imagesc(I)
    axis image
    axis off
    title(sprintf('%d F=%d %.2f',bb(q),flipper,aa(q)))
  end
  
  pause
end