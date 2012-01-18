function [Iex,Iexmask,Icb,Icbmask] = esvm_get_exemplar_icon(models, ...
                                                  index,flip,subind, ...
                                                  loadseg, VOCopts)
% Extract an exemplar visualization image (one from gt box, one from
% cb box) [and does flip if specified]
% Allows allows for the loading of a segmentation too
% NOTE(TJM): this function works, but needs cleanup
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

if ~isfield(models{index}.model,'bb')
  Iex = ones(10,10,3);
  Iexmask = ones(10,10,1);
  Icb = ones(10,10,3);
  Icbmask = ones(10,10,1);
  return;
end

%Subind indicates which window to show (defaults to the base)
if ~exist('subind','var')
  subind = 1;
%else
%  flip = models{index}.model.bb(subind,7);
end

if ~exist('flip','var')
  flip = 0;
end

if ~exist('loadseg','var')
  loadseg = 1;
end

%cb = models{index}.model.bb(subind,1:4);
%TJM: changed
cb = models{index}.gt_box;    
d1 = max(0,1 - cb(1));
d2 = max(0,1 - cb(2));
d3 = max(0,cb(3) - models{index}.sizeI(2));
d4 = max(0,cb(4) - models{index}.sizeI(1));
mypad = max([d1,d2,d3,d4]);
PADDER = round(mypad)+2;

if isfield(models{index},'I')
  I = convert_to_I(models{index}.I);
else
  I = ...
      convert_to_I(models{index}.train_set{models{index}.model.bb(subind,11)});
end
%pre-pad mask because GT region can be outside image
mask = pad_image(zeros(size(I,1),size(I,2)),PADDER);
g = models{index}.gt_box;
g2 = g + PADDER;
mask(g2(2):g2(4),g2(1):g2(3)) = 1;

if loadseg == 1 && exist('VOCopts','var')
  try
    %NOTE(TJM): this is voc-only and should not be here in the most
    %general scenario
    [I2, mask2] = load_seg(VOCopts,models{index});
    if numel(I2) > 0
      I = I2;
      mask = mask2;
    end  
  catch
  end
end

cb = models{index}.gt_box;    
Iex = pad_image(I, PADDER);
Iexmask = pad_image(mask,PADDER);
cb = round(cb + PADDER);

Iex = Iex(cb(2):cb(4),cb(1):cb(3),:);
Iexmask = Iexmask(cb(2):cb(4),cb(1):cb(3));

cb = models{index}.model.bb(subind,1:4);
Icb = pad_image(I, PADDER);
cb = round(cb + PADDER);
Icb = Icb(cb(2):cb(4),cb(1):cb(3),:);
Icbmask = mask(cb(2):cb(4),cb(1):cb(3));

if flip == 1
  Iex = flip_image(Iex);
  Icb = flip_image(Icb);
  Iexmask = flip_image(Iexmask);
  Icbmask = flip_image(Icbmask);
end

function [I, mask] = load_seg(VOCopts, model)

filer = sprintf('%s/%s/SegmentationObject/%s.png',VOCopts.datadir, ...
                VOCopts.dataset,model.curid);

filer_class = sprintf('%s/%s/SegmentationClass/%s.png',VOCopts.datadir, ...
                      VOCopts.dataset,model.curid);


cmap=VOClabelcolormap;

if ~fileexists(filer)
  I = [];
  mask = [];
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

mask = double((res==model.objectid));
I= res_class;

