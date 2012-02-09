function [data_set] = esvm_generate_sketchup_dataset(image_set, cls)
% Generate a dataset from one object per frame from Google sketchup data

% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

if ~exist('cls','var')
  cls = 'object';
end

draw = 0;
tic
N = length(image_set);
for i = 1:N  
  
  %estimate the bg color
  bg_color = estimate_bg_color(image_set{i});
  
  %get foreground segmentation
  [mask,bb] = estimate_fg(image_set{i}, bg_color);
  
  data_set{i}.I = image_set{i};
  object.class = cls;
  object.view = '';
  object.truncated = 0;
  object.occluded = 0;
  object.difficult = 0;
  object.bbox = bb;
  object.polygon = [];    
  data_set{i}.objects = [object];
  
  if draw == 1
   figure(1)
   clf
   imagesc(I)
   plot_bbox(bb);
   drawnow
  end
end
toc

figure(1)
clf
vis_box_tiles(data_set);
title('Synthetic Data Set','FontSize',20)
axis image
axis off

function bg_color = estimate_bg_color(I)
%Estimate the median color at the border of the image
I = toI(I);
bg_color = zeros(3,1);
P1 = ceil(size(I,1)*.05);
P2 = ceil(size(I,2)*.05);
for i = 1:3
  bg_color(i) = ...
      median(cat(1,...
                 reshape(I(:,1:P2,i),[],1),...
                 reshape(I(:,end-P2:end,i),[],1),...
                 reshape(I(1:P1,:,1),[],1),...
                 reshape(I(end-P1:end,:,i),[],1)));
end

function [fg, bb] = estimate_fg(I, bg_color)
I = toI(I);
P1 = ceil(size(I,1)*.05);
P2 = ceil(size(I,2)*.05);

Irow = reshape(I,[],3)';
d = distSqr_fast(Irow, bg_color);
d = reshape(d,size(I,1),size(I,2));
d(1:P1,:) = 0;
d(end-P1:P1,:) = 0;
d(:,1:P2) = 0;
d(:,end-P2:end) = 0;

fg = (d>.01);

if 0
  %If this is enabled, we re-crop the images to contain the object
  %centered without too much white space around it
  [u,v] = find(fg);
  ddd = ((max(u)-min(u)) + (max(v)-min(v)))/2;
  goods = (bwdist(fg)<ddd);
  [u,v] = find(goods);
  fg = fg(min(u):max(u),min(v):max(v),:);
  I = I(min(u):max(u),min(v):max(v),:);
end

[u,v] = find(fg);
bb = [min(v) min(u) max(v) max(u)];

