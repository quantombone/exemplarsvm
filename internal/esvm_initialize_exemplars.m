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

[cur_pos_set, cur_neg_set] = split_sets(data_set, cls);

data_set = cat(1,cur_pos_set(:),cur_neg_set(:));

model.data_set = data_set;
model.cls = cls;
model.model_name = model_name;

SKIP_TRUNC = 1;

if ~isfield('params','hg_size')

  [hg_size,N,Ntrunc] = get_hg_size(cur_pos_set, ...
                                   params.init_params.sbin);
  

  if (N == Ntrunc)
    fprintf(1,'warning, no skipping any truncated examples\n');
    SKIP_TRUNC = 0;
  end
  hg_size = hg_size * min(1,params.init_params.MAXDIM/max(hg_size));
  hg_size = max(1,round(hg_size));
  %fprintf(1,'Found a HOG template size of [%d x %d] from %d examples\n',hg_size(1),hg_size(2),N);

  %params.init_params.hg_size = hg_size;
  params.init_params.N = N;
  
  if isfield(params.init_params,'MAX_POS_CACHE')
    params.init_params.K = round((params.init_params.MAX_POS_CACHE/ ...
                                  (params.init_params.N)));
    %NOTE: deprecated
    %fprintf(1,'Saving %d wiggles per exemplar for Positive Cache\n',params.init_params.K);
  end
end


model.params = params;

%Create an array of all final file names
allfiles = cell(0,1);%1,length(data_set));
numskip = 0;
counter = 1;
for j = 1:length(data_set)  
  curid = j;
  
  %Skip positive generation if there are no objects
  if ~isfield(data_set{j},'objects') || length(data_set{j}.objects) == 0
    continue
  end

  obj = data_set{j}.objects;
  I = toI(data_set{j}.I);
  
  for k = 1:length(obj)  
    objectid = k;
    % Warp original bounding box
    bbox = obj(k).bbox; 
    
    if (obj(k).truncated==1) && SKIP_TRUNC == 1
      numskip = numskip + 1;
      continue
    end

    filer = sprintf('%s/%d.%d.%s.mat',...
                    results_directory, curid, objectid, cls);
    
    allfiles{end+1} = filer;
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
    m = params.init_params.init_function(I, bbox, params);

    %Save filename (or original image)
    m.I = data_set{j}.I;
    m.curid = curid;
    m.objectid = objectid;
    m.cls = cls;    
    m.gt_box = gt_box;
    m.bb(:,11) = j;

    for q = 1:length(m.models)
      m.models{q}.bb(11) = j;
    end
    m.sizeI = size(I);
    m.name = sprintf('%d.%d.%s',m.curid,m.objectid,m.cls);
    m.counter = counter;
    counter = counter + 1;
    if CACHE_FILE == 1
      save(filer,'m');
      if exist(filerlock,'dir')
        rmdir(filerlock);
      end
    else
      allfiles{end} = m;
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

fprintf(1,'\nSkipped %d truncated objects\n',numskip);

if CACHE_FILE == 0
  model.models = allfiles;
  return;
else
  %sort files so they are in alphabetical order
  [allfiles, ~] = sort(allfiles);
end

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
