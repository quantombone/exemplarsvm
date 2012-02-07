function model = esvm_initialize_goalsize_exemplar(I, bbox, init_params)
%% Initialize the exemplar (or scene) such that the representation
% which tries to choose a region which overlaps best with the given
% bbox and contains roughly init_params.goal_ncells cells, with a
% maximum dimension of init_params.MAXDIM
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

if ~exist('init_params','var')
  init_params.sbin = 8;
  init_params.hg_size = [8 8];
  init_params.MAXDIM = 10;
end

if ~isfield(init_params,'MAXDIM')
  init_params.MAXDIM = 10;
  fprintf(1,'Default MAXDIM is %d\n',init_params.MAXDIM);
end

%Expand the bbox to have some minimum and maximum aspect ratio
%constraints (if it it too horizontal, expand vertically, etc)
bbox = expand_bbox(bbox,I);
bbox = max(bbox,1);
bbox([1 3]) = min(size(I,2),bbox([1 3]));
bbox([2 4]) = min(size(I,1),bbox([2 4]));

%Create a blank image with the exemplar inside
Ibox = zeros(size(I,1), size(I,2));    

Ibox(bbox(2):bbox(4), bbox(1):bbox(3)) = 1;

%% NOTE: why was I padding this at some point and now I'm not???
%% ANSWER: doing the pad will create artifical gradients
ARTPAD = 0;
I_real_pad = pad_image(I, ARTPAD);

%Get the hog feature pyramid for the entire image
clear params;
params.detect_levels_per_octave = 10;
params.init_params = init_params;
[f_real,scales] = esvm_pyramid(I_real_pad, params);

%Extract the regions most overlapping with Ibox from each level in the pyramid
[masker,sizer] = get_matching_masks(f_real, Ibox);

%Now choose the mask which is closest to N cells
[targetlvl, mask] = get_ncell_mask(init_params, masker, ...
                                                sizer);
[uu,vv] = find(mask);
curfeats = f_real{targetlvl}(min(uu):max(uu),min(vv):max(vv),:);

model.init_params = init_params;
model.hg_size = size(curfeats);
model.mask = logical(ones(model.hg_size(1),model.hg_size(2)));

fprintf(1,'initialized with HOG_size = [%d %d]\n',model.hg_size(1),model.hg_size(2));
model.w = curfeats - mean(curfeats(:));
model.b = 0;
model.x = curfeats(:);;
model.hg_size = size(model.w);

%Fire inside self-image to get detection location
[model.bb, model.x] = get_target_bb(model, I, init_params);

%Normalized-HOG initialization
model.w = reshape(model.x,size(model.w)) - mean(model.x(:));

if isfield(init_params,'wiggle_number') && ...
      (init_params.wiggle_number > 1)
  savemodel = model;
  model = esvm_get_model_wiggles(I, model, init_params.wiggle_number);
end


function [targetlvl,mask] = get_ncell_mask(init_params, masker, ...
                                                        sizer)
%Get a the mask and features, where mask is closest to NCELL cells
%as possible
for i = 1:size(masker)
  [uu,vv] = find(masker{i});
  if ((max(uu)-min(uu)+1) <= init_params.MAXDIM) && ...
        ((max(vv)-min(vv)+1) <= init_params.MAXDIM)
    targetlvl = i;
    mask = masker{targetlvl};
    return;
  end
end
fprintf(1,'didnt find a match\n');
%Default to older strategy
ncells = prod(sizer,2);
[aa,targetlvl] = min(abs(ncells-init_params.goal_ncells));
mask = masker{targetlvl};

function [masker,sizer] = get_matching_masks(f_real, Ibox)
%Given a feature pyramid, and a segmentation mask inside Ibox, find
%the best matching region per level in the feature pyramid

masker = cell(length(f_real),1);
sizer = zeros(length(f_real),2);

for a = 1:length(f_real)
  goods = double(sum(f_real{a}.^2,3)>0);
  
  masker{a} = max(0.0,min(1.0,imresize(Ibox,[size(f_real{a},1) size(f_real{a}, ...
                                                  2)])));
  [tmpval,ind] = max(masker{a}(:));
  masker{a} = (masker{a}>.1) & goods;

  if sum(masker{a}(:))==0
    [aa,bb] = ind2sub(size(masker{a}),ind);
    masker{a}(aa,bb) = 1;
  end
  [uu,vv] = find(masker{a});
  masker{a}(min(uu):max(uu),min(vv):max(vv))=1;
  sizer(a,:) = [range(uu)+1 range(vv)+1];
end

function bbox = expand_bbox(bbox,I)
%Expand region such that is still within image and tries to satisfy
%these constraints best
%requirements: each dimension is at least 50 pixels, and max aspect
%ratio os (.25,4)
for expandloop = 1:10000
  % Get initial dimensions
  w = bbox(3)-bbox(1)+1;
  h = bbox(4)-bbox(2)+1;
  
  if h > w*4 || w < 50
    %% make wider
    bbox(3) = bbox(3) + 1;
    bbox(1) = bbox(1) - 1;
  elseif w > h*4 || h < 50
    %make taller
    bbox(4) = bbox(4) + 1;
    bbox(2) = bbox(2) - 1;
  else
    break;
  end
  
  bbox([1 3]) = cap_range(bbox([1 3]), 1, size(I,2));
  bbox([2 4]) = cap_range(bbox([2 4]), 1, size(I,1));      
end


function [target_bb,target_x] = get_target_bb(model, I, init_params)
%Get the bounding box of the top detection

params.detect_keep_threshold = -100000.0;
params.detect_max_windows_per_exemplar = 1;
params.detect_levels_per_octave = 10;
params.detect_save_features = 1;
params.detect_add_flip = 0;
params.detect_pyramid_padding = 5;
params.dfun = 0;
params.nnmode = 0;
params.init_params = init_params;

[rs,t] = esvm_detect(I,{model},params);
target_bb = rs.bbs{1}(1,:);
target_x = rs.xs{1}{1};
