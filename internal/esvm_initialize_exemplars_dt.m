function allfiles = esvm_initialize_exemplars_dt(dataset_params, e_set, ...
                                        models_name, init_params)
error('Deprecated Function, needs fixing')
% Initialize script which writes out initial model files for all
% exemplars in an exemplar stream e_set (see get_pascal_stream)
% NOTE: this function is parallelizable (and dalalizable!)  
% 
% INPUTS:
% dataset_params: the parameters of the current dataset
% e_set: the exemplar stream set which contains
%   e_set{i}.I, e_set{i}.cls, e_set{i}.objectid, e_set{i}.bbox
% models_name: model name
% init_params: a structure of initialization parameters
% init_params.init_function: a function which takes as input (I,bbox,params)
%   and returns a model structure [if not specified, then just dump
%   out names of resulting files]

% OUTPUTS:
% allfiles: The names of all outputs (which are .mat model files
%   containing the initialized exemplars)
%
% Tomasz Malisiewicz (tomasz@cmu.edu)

% DTstring = '';
% if dalalmode == 1
%   %Find the best window size from taking statistics over all
%   %training instances of matching class
%   hg_size = get_hg_size(cls);
%   DTstring = '-dt';
% elseif dalalmode == 2
%   hg_size = [8 8];
%   DTstring = '-gt';
% end
  
%if (dalalmode == 1) || (dalalmode == 2)
%  %Do the dalal-triggs anisotropic warping initialization
%  model = initialize_model_dt(I,bbox,SBIN,hg_size);
%else

results_directory = ...
    sprintf('%s/models/%s-%s/',dataset_params.localdir, e_set{1}.cls, ...
            models_name);

if ~exist(results_directory,'dir')
  fprintf(1,'Making directory %s\n',results_directory);
  mkdir(results_directory);
end

filer = sprintf('%s/%s-dt.mat',results_directory, e_set{1}.cls);
%filerlock = [filer '.lock'];

if fileexists(filer) 
  allfiles{1} = filer;
  return;
end
%  || (mymkdir_dist(filerlock)==0)
%  return;
%end

hg_size = get_hg_size(e_set, init_params.sbin);

for i = 1:length(e_set)  
  bbox = e_set{i}.bbox;  
  I = convert_to_I(e_set{i}.I);

  warped = mywarppos(hg_size, I, init_params.sbin, bbox);
  curfeats{i} = init_params.features(warped, init_params);
  fprintf(1,'.');
end  

curfeats = cellfun2(@(x)reshape(x,[],1),curfeats);
curfeats = cat(2,curfeats{:});
m.model.init_params = init_params;
m.model.hg_size = [hg_size features];
m.model.mask = ones(hg_size(1),hg_size(2));
m.model.w = mean(curfeats,2);
m.model.w = m.model.w - mean(m.model.w(:));
m.model.w = reshape(m.model.w, m.model.hg_size);
m.model.b = 0;
m.model.x = curfeats;
m.model.bb = [];
m.model.svxs = [];
m.cls = e_set{1}.cls;
m.models_name = models_name;
m.name = sprintf('dt-%s',m.cls);

save(filer,'m');
%if exist(filerlock,'dir')
%  rmdir(filerlock);
%end
allfiles{1} = filer;
  
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
