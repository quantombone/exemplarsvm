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
data_set = cell(N,1);
for i = 1:N  
  fprintf(1,'.');
  %estimate the bg color
  bg_color = estimate_bg_color(image_set{i});
  
  %get foreground segmentation
  [mask,bb] = estimate_fg(image_set{i}, bg_color);
  data_set{i}.I = image_set{i};

  [baser,filer,exter] = fileparts(data_set{i}.I);
  newdir = 'seg/';
  data_set{i}.segmentation = double(mask);

  if ~exist([baser newdir],'dir')
    try
      mkdir([baser newdir]);
    end
  end
  
  data_set{i}.segmentation = [baser newdir filer exter];
  if ~fileexists(data_set{i}.segmentation)
    try
      imwrite(mask,data_set{i}.segmentation);
    catch
    end
  end
  
  figure(1)
  clf
  imagesc(segImage(toI(data_set{i}.I),mask))
  drawnow  
  
  %data_set{i}.segmentation = mask;
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

% figure(2)
% clf
% vis_box_tiles(data_set);
% title('Synthetic Data Set','FontSize',20)
% axis image
% axis off

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

function [fg, bb] = estimate_fg(Iname, bg_color)
I = toI(Iname);
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

[a]=bwlabel(1-fg);
fg = double(a~=a(1));


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

fg2 = bwmorph(fg,'dilate');
outside = find(fg2==0);
I1 = I(:,:,1);
I2 = I(:,:,2);
I3 = I(:,:,3);
I1(outside) = bg_color(1);
I2(outside) = bg_color(2);
I3(outside) = bg_color(3);

I = cat(3,I1,I2,I3);
imwrite(I,Iname);

