function [Iex,Iexmask] = esvm_get_seg_icon(models,index,flip,subind,VOCopts)
% get the segmentation icon, used to transfer segmentation 
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

model = models{index};

STEAL_SEGMENTATION = 1;


filer = sprintf('%s/%s/SegmentationObject/%s.png',VOCopts.datadir, ...
                VOCopts.dataset,model.curid);

filer_class = sprintf('%s/%s/SegmentationClass/%s.png',VOCopts.datadir, ...
                      VOCopts.dataset,model.curid);


cmap=VOClabelcolormap;

if ~fileexists(filer)
  Iex = [];
  Iexmask = [];
  return;
end
  
classes = {'aeroplane','bicycle','bird','boat','bottle','bus', ...
           'car','cat','chair','cow','diningtable','dog','horse', ...
           'motorbike','person','pottedplant','sheep','sofa', ...
           'train','tvmonitor'};

clsid = find(ismember(classes,model.cls));

has_seg = 1;
res = imread(filer);
res_class = imread(filer_class);
res_class = reshape(cmap(res_class(:)+1,:),size(res_class,1), ...
                    size(res_class,2),3);

Iexmask = double((res==models{index}.objectid));
Iex = res_class;

if flip == 1
  Iex = flip_image(Iex);
  Iexmask = flip_image(Iexmask);
end
