function allfiles = esvm_initialize(dataset_params, e_set, ...
                                        models_name, init_params)
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

%fprintf(1,'Writing %d Exemplars to directory %s\n',length(e_set),...
%        results_directory);

%Randomize creation order
myRandomize;
rrr = randperm(length(e_set));
e_set = e_set(rrr);

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
  if ~exist('init_params','var')
    continue
  end
  filerlock = [filer '.lock'];
  
  if fileexists(filer) || (mymkdir_dist(filerlock)==0)
    continue
  end
  gt_box = bbox;
  fprintf(1,'.');
  
  I = convert_to_I(e_set{i}.I);

  %Call the init function which is a mapping from (I,bbox) to (model)
  [model] = init_params.init_function(I, bbox, init_params);
  
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

  save(filer,'m');
  if exist(filerlock,'dir')
    rmdir(filerlock);
  end

  % %Print the bounding box overlap between the initial window and
  % %the final window
  % finalos = getosmatrix_bb(m.gt_box, m.model.bb(1,:));
  % fprintf(1,'Final OS is %.3f\n', ...
  %         finalos);
  
  % fprintf(1,'Final hg_size is %d %d\n',...
  %         m.model.hg_size(1), m.model.hg_size(2));

  %Show the initialized exemplars
  if dataset_params.display == 1
    show_exemplar_frames({m}, 1, dataset_params);
  end
end  

%sort files so they are in alphabetical order
[allfiles, bb] = sort(allfiles);

fprintf(1,'\n   --- Done initializing %d exemplars\n',length(e_set));
