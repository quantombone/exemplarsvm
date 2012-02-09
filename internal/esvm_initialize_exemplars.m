function model = esvm_initialize_exemplars(data_set, cls, params)
% Initialize examplars for training by setting initial features and
% initial detectors
% 
% INPUTS:
% data_set: some sort of dataset
% cls: we treat all instances in data_set which are of this class
%      as positives, and everything else is negatives
% params: a structure of initialization parameters [optional]
%
% OUTPUTS:
% model: The initialized ExemplarSVMs
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

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
  model = load(cache_file);
  model = model.model;
  return;
end

results_directory = ...
    sprintf('%s/models/%s/',params.localdir, ...
            model_name);

if CACHE_FILE==1 && ~exist(results_directory,'dir')
  fprintf(1,'Making directory %s\n',results_directory);
  mkdir(results_directory);
end

%Randomize creation order
if CACHE_FILE == 1
  myRandomize;
  rrr = randperm(length(data_set));
  if (params.display == 1)
    rrr = 1:length(rrr);
  end
  data_set = data_set(rrr);
end

[cur_pos_set, cur_neg_set] = get_positive_negative_sets(data_set, cls);
data_set = [cur_pos_set cur_neg_set];

model.data_set = data_set;
model.cls = cls;
model.model_name = model_name;
model.params = params;

%Create an array of all final file names
allfiles = cell(1,length(data_set));

for j = 1:length(data_set)  
  curid = j;
  
  %Skip positive generation if there are no objects
  if ~isfield(data_set{j},'objects') || length(data_set{j}.objects) == 0
    continue
  end

  obj = {data_set{j}.objects};
  I = toI(data_set{j}.I);
  
  for k = 1:length(obj)  
    objectid = k;
    % Warp original bounding box
    bbox = obj{k}.bbox;    

    filer = sprintf('%s/%d.%d.%s.mat',...
                    results_directory, curid, objectid, cls);
    
    allfiles{j} = filer;
    if ~isfield(params,'init_params')
      error('Warning, cannot initialize without params.init_params\n');
    end
    filerlock = [filer '.lock'];

    if CACHE_FILE == 1
      if fileexists(filer) || (mymkdir_dist(filerlock)==0)
        continue
      end
    end
    gt_box = bbox;
    fprintf(1,'.');
    
    I = toI(data_set{j});
    
    %Call the init function which is a mapping from (I,bbox) to (model)
    m = params.init_params.init_function(I, bbox, params.init_params);

    %m.model = model;        
    %Save filename (or original image)
    m.I = data_set{j}.I;
    m.curid = curid;
    m.objectid = objectid;
    m.cls = cls;    
    m.gt_box = gt_box;
    
    m.sizeI = size(I);
    m.model_name = model_name;
    m.name = sprintf('%s.%d.%s',m.curid,m.objectid,m.cls);

    if CACHE_FILE == 1
      save(filer,'m');
      if exist(filerlock,'dir')
        rmdir(filerlock);
      end
    else
      allfiles{j} = m;
    end
    
    % %Print the bounding box overlap between the initial window and
    % %the final window
    % finalos = getosmatrix_bb(m.gt_box, m.model.bb(1,:));
    % fprintf(1,'Final OS is %.3f\n', ...
    %         finalos);
    
    % fprintf(1,'Final hg_size is %d %d\n',...
    %         m.model.hg_size(1), m.model.hg_size(2));
    
    %Show the initialized exemplars
    if params.display == 1
      esvm_show_exemplar_frames({m}, 1, params);
      drawnow
      snapnow;
    end
  end
end

if CACHE_FILE == 0
  model.models = allfiles;
  return;
end

%sort files so they are in alphabetical order
[allfiles, bb] = sort(allfiles);

%Load all of the initialized exemplars
CACHE_FILE = 1;

if length(model_name) == 0
  CACHE_FILE = 0;
end

STRIP_FILE = 0;
DELETE_INITIAL = 0;

model = esvm_load_models(params, model_name, allfiles, ...
                          CACHE_FILE, STRIP_FILE, DELETE_INITIAL);


fprintf(1,'\n   --- Done initializing %d exemplars\n',length(e_set));
