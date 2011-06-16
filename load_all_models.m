function [models] = load_all_models(cls, DET_TYPE, dataset_params, ...
                                    CACHE_FILE, STRIP_FILE)
%Load all trained models of a specified class 'cls' and
%type 'DET_TYPE' from a models directory.  If CACHE_FILE is enabled
%(defaults to turned off), then will try to load/save a cached
%file. if STRIP_FILE is enabled, then save a stripped file (raw
%detectors only saved)

%Tomasz Malisiewicz (tomasz@cmu.edu)
% if ~exist('cls','var') || length(cls) == 0
%   [cls,DET_TYPE] = load_default_class;
% end

%Define the type of classifiers we want to load (exemplars, dalals)
% if ~exist('DET_TYPE','var')
%   DET_TYPE = 'exemplars';
% else
%   if sum(ismember(DET_TYPE,{'dalals','exemplars','exemplars/mined'})) == 0
%     fprintf(1,'Warning DET_TYPE=%s unrecognized\n',DET_TYPE);
%   end
% end

%This is the prefix we look for when loading files (which usually
%denotes the maximum number of mining iterations)
% if ~exist('FINAL_PREFIX','var')
%   FINAL_PREFIX = '100';
% end

%if enabled, we cache result on disk to facilitate loading at a
%later stage (NOTE: these files might have to be removed manually)
if ~exist('CACHE_FILE','var')
  CACHE_FILE = 0;
end

if ~exist('STRIP_FILE','var')
  STRIP_FILE = 0;
end

if CACHE_FILE == 1
  cache_dir =  ...
      sprintf('%s/models/',dataset_params.localdir);
  
  if ~exist(cache_dir,'dir')
    mkdir(cache_dir);
  end
  
  cache_file = ...
      sprintf('%s/%s-%s.mat',cache_dir,cls,DET_TYPE);
  
  cache_file_stripped = ...
      sprintf('%s/%s-%s.mat',cache_dir,cls,[DET_TYPE '-stripped']);

  filerlock = [cache_file '.lock'];
    
  %keep trying to open model file
  while 1
      
    if fileexists(cache_file)
      fprintf(1,'Loading CACHED file: %s\n', cache_file);
      load(cache_file);
      return;
    end
        
    %If file is not present, then we will do the work to create
    %one, but let's first check to see if another process is doing
    %the same thing.. if another process is doing this, then there
    %will be a lock file present (and we can't create a new one).
    %in this case, we will just pause and try to read main file again
    %try to make a lock directory
    if (mymkdir_dist(filerlock) == 0)
      break;
    end
    pause(3);
  end
  
end

results_directory = ...
    sprintf('%s/models/%s/',dataset_params.localdir,DET_TYPE);

dirstring = [results_directory '*' cls '*.mat'];
files = dir(dirstring);
fprintf(1,'Pattern of files to load: %s\n',dirstring);
fprintf(1,'Length of files to load: %d\n',length(files));

models = cell(1,length(files));
for i = 1:length(files)
  fprintf(1,'.');
  
  %if a load fails, maybe we are loading a partial result DURING
  %mining, so we only exit
  try
    m = load([results_directory files(i).name]);
  catch
    models{i} = [];
    fprintf(1,'#');
    if nargout==0
      fprintf(1,'ERROR: could not load model %s, mining not complete?\n',files(i).name);
      return;
    end
  end

  try
    models{i} = m.m;
  catch
    models{i} = m.models{1};
  end
  
  models{i}.models_name = DET_TYPE;
  if max(models{i}.model.hg_size(1:2)) >= 25 %| length(models{i}.model.x)==0
    fprintf(1,'Truncating very large exemplar id=%d\n', i);
    models{i}.model.w = zeros(1,1,31);
    models{i}.hg_size = [1 1 31];
    models{i}.model.b = 1000;
  end
  
  %models{i}.model.x = [];
  %models{i}.model.allx = [];
  %models{i}.model.wtrace = [];
  %models{i}.model.btrace = [];

  %disable negative support vectors to save space
  models{i}.model.svxs = [];
  
  if ~isfield(models{i},'models_name')
    models{i}.models_name=strrep(DET_TYPE,'/','_');
  end
end

if length(files) == 0
  fprintf(1,'WARNING NO MODELS LOADED FOR %s\n',cls);
  models = cell(0,1);
  return;
end

fprintf(1,'\n');

%This part is not necessary, as the notion of a dalal window is
%generalized to a method for framing windows
% if length(files) == 1
%   %% if here then we have a dalal file, so we create a nn file
%   NM = size(m.m.model.x, 2);
%   for i = 1:NM
%     curm = models{1};
%     curm.model.w = reshape(m.m.model.x(:,i),size(curm.model.w));
%     curm.model.w = curm.model.w - mean(curm.model.w(:));
%     curm.model.b = -100;
%     models{i} = curm;
%     models{i}.models_name = 'nn';
%   end
%   fprintf(1,'finished making nn file\n');
% end

if CACHE_FILE==1
  fprintf(1,'Loaded models, saving to %s\n',cache_file);
  save(cache_file,'models');
  
  if STRIP_FILE == 1
    models_save = models;
    models = strip_model(models);
    fprintf(1,'Saving stripped to %s\n',cache_file_stripped);
    save(cache_file_stripped,'models');
    models = models_save;
  end
  
  %delete the lock file for the file write
  delete(filerlock);
end
