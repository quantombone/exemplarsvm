function models = esvm_initialize_exemplars_dt(e_set, params, ...
                                            models_name)

% Initialize script which writes out initial model files for all
% exemplars in an exemplar stream e_set (see get_pascal_stream)
% NOTE: this function is parallelizable (and dalalizable!)  
% 
% INPUTS:
% params.dataset_params: the parameters of the current dataset
% e_set: the exemplar stream set which contains
%   e_set{i}.I, e_set{i}.cls, e_set{i}.objectid, e_set{i}.bbox
% models_name: model name
% init_params: a structure of initialization parameters
% init_params.init_function: a function which takes as input (I,bbox,params)
%   and returns a model structure [if not specified, then just dump
%   out names of resulting files]
%
% OUTPUTS:
% allfiles: The names of all outputs (which are .mat model files
%   containing the initialized exemplars)
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

if isfield(params,'dataset_params') && ...
      isfield(params.dataset_params,'localdir') && ...
      length(params.dataset_params.localdir)>0
  CACHE_FILE = 1;
else
  CACHE_FILE = 0;
  params.dataset_params.localdir = '';
end

if ~exist('models_name','var')
  models_name = '';
end

cache_dir =  ...
    sprintf('%s/models/',params.dataset_params.localdir);

cache_file = ...
    sprintf('%s/%s.mat',cache_dir,models_name);

if CACHE_FILE ==1 && fileexists(cache_file)
  models = load(cache_file);
  models = models.models;
  return;
end

results_directory = ...
    sprintf('%s/models/%s/',params.dataset_params.localdir, ...
            models_name);

if CACHE_FILE==1 && ~exist(results_directory,'dir')
  fprintf(1,'Making directory %s\n',results_directory);
  mkdir(results_directory);
end

filer = sprintf('%s/%s-dt.mat',results_directory, e_set{1}.cls');
filerlock = [filer '.lock'];


if CACHE_FILE == 1
  if fileexists(filer) 
    m = load(filer);
    models{1} = m.m;
    return;
  end
end

hg_size = get_hg_size(e_set, params.init_params.sbin);

curfeats = cell(0,1);
fprintf(1,['esvm_initialize_exemplars_dt: initializing features by' ...
           ' warping to a canonical size\n']);
for j = 1:length(e_set)  
  bbox = e_set{j}.bbox;  
  I = convert_to_I(e_set{j}.I);
  warped = mywarppos(hg_size, I, params.init_params.sbin, bbox);
  curfeats{end+1} = params.init_params.features(warped, ...
                                                params.init_params.sbin);

  warped = mywarppos(hg_size, flip_image(I), params.init_params.sbin, ...
                     flip_box(bbox,size(I)));
  curfeats{end+1} = params.init_params.features(warped, ...
                                                params.init_params.sbin);
  fprintf(1,'.');
end  
fprintf(1,'esvm_initialize_exemplars: finished with %d windows\n',length(curfeats));
curfeats = cellfun2(@(x)reshape(x,[],1),curfeats);
curfeats = cat(2,curfeats{:});
m.model.init_params = params.init_params;
m.model.hg_size = [hg_size params.init_params.features()];
m.model.mask = ones(hg_size(1),hg_size(2));
m.model.w = mean(curfeats,2);
m.model.w = m.model.w - mean(m.model.w(:));
m.model.w = reshape(m.model.w, m.model.hg_size);
m.model.b = 0;
m.model.x = curfeats;
m.model.bb = [];
m.model.svxs = [];
m.model.svbbs = [];
m.cls = e_set{1}.cls;
m.models_name = models_name;
m.name = sprintf('dt-%s',m.cls);
m.curid = m.cls;
m.objectid = -1;
models{1} = m;
save(filer,'m');
if fileexists(filerlock)
  rmdir(filerlock);
end

function [hg_size] = get_hg_size(e_set, sbin)
%% Load ids of all images in trainval that contain cls

bbs = cellfun2(@(x)x.bbox,e_set);
bbs = cat(1,bbs{:});

W = bbs(:,3)-bbs(:,1)+1;
H = bbs(:,4)-bbs(:,2)+1;

[hg_size] = get_bb_stats(H, W, sbin);

function modelsize = get_bb_stats(h,w, sbin)

xx = -2:.02:2;
filter = exp(-[-100:100].^2/400);
aspects = hist(log(h./w), xx);
aspects = convn(aspects, filter, 'same');
[peak, I] = max(aspects);
aspect = exp(xx(I));

% pick 20 percentile area
areas = sort(h.*w);
%TJM: make sure we index into first element if not enough are
%present to take the 20 percentile area
area = areas(max(1,floor(length(areas) * 0.2)));
area = max(min(area, 5000), 3000);

% pick dimensions
w = sqrt(area/aspect);
h = w*aspect;

modelsize = [round(h/sbin) round(w/sbin)];

function warped = mywarppos(hg_size, I, sbin, bbox)

% warped = warppos(name, model, c, pos)
% Warp positive examples to fit model dimensions.
% Used for training root filters from positive bounding boxes.

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
