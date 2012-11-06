function model = esvm_initialize_dt(data_set, cls, params)
% DalalTriggs model creation which creates an initial positive set
% by warping positives into a single canonical position, where the
% canonical position is obtained from statistics of bounding box
% aspect ratios. The variable params.init_params defines the
% initialization function.
%
% INPUTS:
% data_set: the training set of objects
% cls: the target category from which we extract positives and
%   use remaining images to define negatives
% params [optional]: ESVM parameters
%
% OUTPUTS:
% model: A single DalalTriggs model for the category cls
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

%save with dt as the model name
model_name = [cls '-dt'];

if length(params.localdir)>0
  CACHE_FILE = 1;
else
  CACHE_FILE = 0;
  params.localdir = '';
end

if ~exist('model_name','var')
  model_name = '';
end

cache_dir =  ...
    sprintf('%s/models/',params.localdir);

cache_file = ...
    sprintf('%s/%s.mat',cache_dir,model_name);

if CACHE_FILE ==1 && fileexists(cache_file)
  model = load(cache_file,'model');
  model = model.model;
  return;
end

results_directory = ...
    sprintf('%s/models/',params.localdir);

if CACHE_FILE==1 && ~exist(results_directory,'dir')
  fprintf(1,'Making directory %s\n',results_directory);
  mkdir(results_directory);
end

filer = sprintf('%s/%s.mat', results_directory, model_name);
filerlock = [filer '.lock'];

if CACHE_FILE == 1
  if fileexists(filer) 
    model = load(filer,'model');
    model = model.model;
    return;
  end
end

[cur_pos_set, cur_neg_set, pos_subset] = split_sets(data_set, cls);
save_data_set = data_set;
% cur_pos_set = cur_pos_set(1:min(length(cur_pos_set),...
%                                 params.max_number_of_positives));

if length(cur_pos_set) == 0
  fprintf(sprintf('No positives of class "%s" found',cls));
  model = [];
  return;
end

data_set = cat(1,cur_pos_set(:),cur_neg_set(:));


if isfield(params,'hg_size')
  hg_size = params.hg_size;
elseif isfield(params.init_params,'hg_size')
  hg_size = params.init_params.hg_size;
else
  [hg_size] = get_hg_size(cur_pos_set, params.init_params.sbin);
  hg_size = hg_size * min(1,params.init_params.MAXDIM/max(hg_size));
  hg_size = max(1,round(hg_size));
end



curfeats = cell(0,1);
bbs = cell(0,1);
fprintf(1,['esvm_initialize_dt(%s): warping to size [%d x %d]\n'],...
        cls,hg_size(1),hg_size(2));
allwarps = cell(0,1);

minsize = .9 * (hg_size(1)*hg_size(2)*params.init_params.sbin* ...
                params.init_params.sbin);

minsize = 1000;

total = 0;
num_keepers = 0;
total_negatives = 0;
badfeats = cell(0,1);
badboxes = cell(0,1);

TOTALNEGS = 30000;

for j = 1:length(data_set)  
  %Skip positive generation if there are no objects
  if ~isfield(data_set{j},'objects') || length(data_set{j}.objects) == 0
    continue
  end
  
  obj = data_set{j}.objects;
  I = toI(data_set{j}.I);
  
  if params.dt_initialize_with_flips == 1
    flipI = flip_image(I);
  end

  gt_boxes = cat(1,obj.bbox);
  
  for k = 1:length(obj)    
    % Warp original bounding box
    bbox = obj(k).bbox;    
    UUU = bbox(3)-bbox(1)+1;
    VVV = bbox(4)-bbox(2)+1;
    
    factor = params.dt_pad_factor;
    bbox([1]) = bbox([1]) - UUU*factor;
    bbox([3]) = bbox([3]) + UUU*factor;
    
    bbox([2]) = bbox([2]) - VVV*factor;
    bbox([4]) = bbox([4]) + VVV*factor;

    bbox([1 2]) = max(1,bbox([1 2]));
    bbox(3) = min(size(I,2),bbox(3));
    bbox(4) = min(size(I,1),bbox(4));
    
    
    UUU = bbox(3)-bbox(1)+1;
    VVV = bbox(4)-bbox(2)+1;
    

    total = total + 1 + (params.dt_initialize_with_flips == 1);
    
    % skip small examples
    % if  (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1) < minsize
    %   fprintf(1,'S');
    %   continue
    % end    
    
    % aspect = UUU/VVV;
    % aspect2 = hg_size(1)/hg_size(2);
    
    % aspect/aspect2

    

    angle = abs((atan2(VVV,UUU) - atan2(hg_size(2),hg_size(1))));
    ANGLETHRESH = pi/6;
    %this turns this off
    ANGLETHRESH = 10000;
    %fprintf(1,'Angle is %f, thresh is %f\n',angle*180/pi,ANGLETHRESH*180/pi);
    if (angle < ANGLETHRESH) && (obj(k).truncated==0) && (obj(k).difficult==0)
      warped1 = mywarppos(hg_size, I, params.init_params.sbin, ...
                          bbox);

      allwarps{end+1} = warped1;
      curfeats{end+1} = params.init_params.features(warped1, ...
                                                    params ...
                                                    .init_params ...
                                                    .sbin);
      bbox(11) = j;
      bbox(12) = 0;
      bbs{end+1} = bbox;
      
      % [badfeats{end+1},badboxes{end+1}] ...
      %     = get_bad_warps(I, bbox, gt_boxes, hg_size, minsize, ...
      %                        params);
      % badboxes{end}(:,11) = j;
      % badboxes{end}(:,7) = 0;
      

      
      %figure(2)
      %imagesc(I)
      %plot_bbox(bbox)
      %drawnow
      %pause

      if params.dt_initialize_with_flips == 1

        % Warp LR flipped version
        bbox2 = flip_box(bbox,size(I));
        warped2 = mywarppos(hg_size, flipI, params.init_params.sbin, ...
                            bbox2);
        
        allwarps{end+1} = warped2;
        curfeats{end+1} = params.init_params.features(warped2, ...
                                                      params.init_params ...
                                                      .sbin);
        bbox2 = flip_box(bbox2,size(I));
        bbox2(11) = j;
        bbox2(12) = 0;
        bbox2(7) = 1; %indicate the flip
        bbs{end+1} = bbox2;
        
        % [badfeats{end+1},badboxes{end+1}] ...
        %   = get_bad_warps(flipI, bbox2, flip_box(gt_boxes,size(I)), ...
        %                          hg_size, minsize, ...
        %                          params);
        % badboxes{end}(:,11) = j;
        % badboxes{end}(:,7) = 1;
        
        
        %figure(2)
        %imagesc(flipI)
        %plot_bbox(bbox2)
        %drawnow
        %pause
      end
      fprintf(1,'+');
    else
      fprintf(1,'-');
    end
  end
end

fprintf(1,'\nesvm_initialize_dt(%s): %d out of %d positives\n',...
        cls,length(curfeats), total);
curfeats = cellfun2(@(x)reshape(x,[],1),curfeats);
curfeats = cat(2,curfeats{:});


model.data_set = save_data_set;
model.cls = cls;
model.model_name = model_name;
model.params = params;

m.hg_size = [hg_size params.init_params.features()];
%m.mask = ones(m.hg_size(1),m.hg_size(2));

%positive features: x
m.x = curfeats;

%positive windows: bb
m.bb = cat(1,bbs{:});

m.bb(:,11) = pos_subset(m.bb(:,11));

%m.svxs = cat(2,badfeats{:});
%negative features: svxs
%m.svxs = zeros(prod(m.hg_size),0);

%negative windows: svbbs
%m.svbbs = zeros(0,12);
%m.svbbs = cat(1,badboxes{:});

%create an initial classifier
m.w = mean(curfeats,2);
m.w = m.w - mean(m.w(:));
m.w = reshape(m.w, m.hg_size(1), m.hg_size(2),[]);
m.b = 0;
m.params = params;

%m = esvm_update_svm(m);

%m.name = sprintf('dt-%s',m.cls);
%m.curid = m.cls;
%m.objectid = -1;
%m.data_set = data_set;

%sort by initial score
[~,order] = sort(m.w(:)'*m.x,'descend');

%Take at most a finite number of positives
%order = order(1:min(length(order),params.max_number_of_positives));
allwarps = allwarps(order);
m.x = m.x(:,order);
m.bb = m.bb(order,:);

% Is = cat(4,allwarps{:});
% Imean = mean(Is,4);
% m.icon = Imean;

model.models{1} = m;

if params.display == 1
  figure(1)
  clf
  show_model_data(model,10);
end

if CACHE_FILE == 1
  save(filer,'model');
  if fileexists(filerlock)
    rmdir(filerlock);
  end
end


function warped = mywarppos(hg_size, I, sbin, bbox)
% warped = mywarppos(name, model, c, pos)
% Warp positive examples to fit model dimensions.
% Used for training root filters from positive bounding boxes.
% Taken from Felzenszwalb et al's code

pixels = hg_size * sbin;
h = bbox(4) - bbox(2) + 1;
w = bbox(3) - bbox(1) + 1;

cropsize = (hg_size+2) * sbin;

padx = sbin * w / pixels(2);
pady = sbin * h / pixels(1);
x1 = round(bbox(1)-padx);
x2 = round(bbox(3)+padx);
y1 = round(bbox(2)-pady);
y2 = round(bbox(4)+pady);

window = subarray(I, y1, y2, x1, x2, 1);
warped = imresize(window, cropsize(1:2), 'bilinear');

function [badfeats,bbox_bad] = get_bad_warps(I, bbox, gt_boxes, hg_size, minsize, ...
                                                params)

bbox_bad = repmat(bbox(1,1:4),1000,1);
bbox_bad = bbox_bad + 200*randn(size(bbox_bad));
bbox_bad = clip_to_image(bbox_bad,[1 1 size(I,2) size(I,1)]);
curA = (bbox_bad(:,3)-bbox_bad(:,1)+1) .* (bbox_bad(:,4)- ...
                                           bbox_bad(:,2)+1);
UUU = bbox_bad(:,3)-bbox_bad(:,1)+1;
VVV = bbox_bad(:,4)-bbox_bad(:,2)+1;

angle = abs((atan2(VVV,UUU) - atan2(hg_size(1),hg_size(2))));
bbox_bad = bbox_bad(curA>=minsize & (angle)<pi/6,:);
bbox_bad = bbox_bad(max(getosmatrix_bb(bbox_bad,gt_boxes),[], ...
                        2)<=.01,:);
bbox_bad(:,12) = 0;%randn(size(bbox_bad,1),1);
%bbox_bad = esvm_nms(bbox_bad,.5);

badfeats = zeros(hg_size(1)*hg_size(2)*params.init_params.features(),size(bbox_bad,1));
for j = 1:size(bbox_bad,1)       
  bf = params.init_params.features(mywarppos(hg_size,I, ...
                                                params ...
                                                .init_params.sbin,bbox_bad(j,:)),params.init_params.sbin);
  badfeats(:,j) = bf(:);
end


