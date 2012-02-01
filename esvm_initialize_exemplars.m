function models = esvm_initialize_exemplars(e_set, params, ...
                                            models_name)
% Function needs update
error('needs update');

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

%fprintf(1,'Writing %d Exemplars to directory %s\n',length(e_set),...
%        results_directory);

%Randomize creation order
if CACHE_FILE == 1
  myRandomize;
  rrr = randperm(length(e_set));
  
  if (params.dataset_params.display == 1)
    rrr = 1:length(rrr);
  end
  e_set = e_set(rrr);
end

%Create an array of all final file names
allfiles = cell(length(e_set), 1);

for i = 1:length(e_set)
  cls = e_set{i}.cls;
  objectid = e_set{i}.objectid;
  bbox = e_set{i}.bbox;
  curid = e_set{i}.curid;

  %[tmp,curid,tmp] = fileparts(e_set{i}.I);    
  %filer = sprintf('%s/%s.%d.%s.mat',...
  %                results_directory, curid, objectid, cls);
  
  filer = sprintf('%s/%s',results_directory, e_set{i}.filer);
  
  allfiles{i} = filer;
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
  
  I = convert_to_I(e_set{i}.I);

  %Call the init function which is a mapping from (I,bbox) to (model)
  [model] = params.init_params.init_function(I, bbox, params.init_params);
  
  clear m
  m.model = model;    

  %Save filename (or original image)
  m.I = e_set{i}.I;
  m.curid = curid;
  m.objectid = objectid;
  m.cls = cls;    
  m.gt_box = gt_box;
  
  m.sizeI = size(I);
  m.models_name = models_name;
  m.name = sprintf('%s.%d.%s',m.curid,m.objectid,m.cls);


  if CACHE_FILE == 1
    save(filer,'m');
    if exist(filerlock,'dir')
      rmdir(filerlock);
    end
  else
    allfiles{i} = m;
  end

  % %Print the bounding box overlap between the initial window and
  % %the final window
  % finalos = getosmatrix_bb(m.gt_box, m.model.bb(1,:));
  % fprintf(1,'Final OS is %.3f\n', ...
  %         finalos);
  
  % fprintf(1,'Final hg_size is %d %d\n',...
  %         m.model.hg_size(1), m.model.hg_size(2));

  %Show the initialized exemplars
  if params.dataset_params.display == 1
    esvm_show_exemplar_frames({m}, 1, params.dataset_params);
    drawnow
    snapnow;
  end
end  

if CACHE_FILE == 0
  models = allfiles;
  return;
end

%sort files so they are in alphabetical order
[allfiles, bb] = sort(allfiles);

%Load all of the initialized exemplars
CACHE_FILE = 1;

if length(models_name) == 0
  CACHE_FILE = 0;
end

STRIP_FILE = 0;
DELETE_INITIAL = 0;

models = esvm_load_models(params.dataset_params, models_name, allfiles, ...
                          CACHE_FILE, STRIP_FILE, DELETE_INITIAL);


fprintf(1,'\n   --- Done initializing %d exemplars\n',length(e_set));
