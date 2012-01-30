function models = esvm_update_exemplars_dt(e_set, params, ...
                                               models_name, models)

% Updates positives with latent placements which pass os threshold test
% 
% INPUTS:
% params.dataset_params: the parameters of the current dataset
% e_set: the exemplar stream set which contains
%   e_set{i}.I, e_set{i}.cls, e_set{i}.objectid, e_set{i}.bbox
% models_name: model name
% models: last iteration's model whose xs and bbs will be updated
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

%NOTE BUG (TJM): no filerlock locking here


if CACHE_FILE == 1
  if fileexists(filer) 
    m = load(filer);
    models{1} = m.m;
    return;
  end
end

hg_size = get_hg_size(e_set, params.init_params.sbin);
p = esvm_get_default_params;
p.detect_save_features = 1;
p.detect_keep_threshold = -1.5;
p.detect_exemplar_nms_os_threshold = 1.0;
p.detect_max_windows_per_exemplar = 100;


curx = cell(0,1);
curbb = cell(0,1);

for j = 1:length(e_set)  
  bbox = e_set{j}.bbox;  
  I = convert_to_I(e_set{j}.I);

  warped = mywarppos(hg_size, I, params.init_params.sbin, bbox);
  curfeats{j} = params.init_params.features(warped, ...
                                            params.init_params ...
                                            .sbin);
if 0
  xd = bbox(3)-bbox(1)+1;
  yd = bbox(4)-bbox(2)+1;
  KKK = 3;
  diffx = round(linspace(-xd*.2,xd*.2,KKK));
  diffy = round(linspace(-yd*.2,yd*.2,KKK));

  clear cf;
  clear bs;
  counter = 1;
  for a = 1:KKK
    for b = 1:KKK
      for c = 1:KKK
        for d = 1:KKK
fprintf(1,'?');
          bbox2 = bbox + [diffx(a) diffy(b) diffx(c) diffy(d)];
          warped = mywarppos(hg_size, I, params.init_params.sbin, bbox2);
          curf = params.init_params.features(warped, ...
                                                    params.init_params ...
                                                    .sbin);
          cf{counter} = curf;
          bs{counter} = bbox2;
          counter = counter + 1;
        end
      end
    end
  end

  res = emap(@(x)models{1}.model.w(:)'*x(:),cf);
  res = cellfun(@(x)x,res);
  [aa,bb] = max(res);

  figure(123)
  clf
  imagesc(I)
  plot_bbox(bbox,'original')
  plot_bbox(bs{bb},'new',[1 0 0]);
  drawnow
end


  rs = esvm_detect(I,models,p);

  % figure(123)
  % clf
  % imagesc(I)
  % plot_bbox(bbox,'original')
  
  %get overlaps, and keep ones above threshold

  if size(rs.bbs{1})~=0

    os_thresh = 0.5;
    os = getosmatrix_bb(rs.bbs{1},bbox);
    goods = find(os >= os_thresh);

    if length(goods) >=1
      curx{end+1} = rs.xs{1}{goods(1)};
      curbb{end+1} = rs.bbs{1}(goods(1),:);
      %plot_bbox(curbb{j},'latent update',[1 0 0])
  fprintf(1,'+');
    else
      fprintf(1,'-');
      %title('no update');
      %curx{j} = curfeats{j}(:);
      %curbb{j} = zeros(1,12);
    end
  end
  %drawnow

end  


models{1}.model.x = cat(2,curx{:});
models{1}.model.bb = cat(1,curbb{:});
models{1}.models_name = models_name;
m = models{1};

fprintf(1,'Got latent updates for %d examples\n',size(models{1}.model.bb,1));

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
