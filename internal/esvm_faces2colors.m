function seg2 = esvm_faces2colors(seg)
% Given a segmentation, fill it in with colors.
% Used for buslabeling to show facades as colors. Hardcoded color
% scheme based on Matlab's jet.
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

sizer = size(seg);
colors = jet(5);
colors(end+1,:) = 0;
colors = colors(end:-1:1,:);
seg2 = colors(seg(:)+1,:);
seg2 = reshape(seg2,[sizer(1) sizer(2) 3]);
