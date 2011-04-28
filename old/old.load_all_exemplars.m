function [models] = load_all_exemplars(cls)
%Load all trained models from disk
%Tomasz Malisiewicz (tomasz@cmu.edu)

VOCinit;

%if enabled, we cache group file to facilitate loading
CACHE_FILE = 1;

if ~exist('cls','var')
  cls = 'venice';
  cls = 'colliseum';
  cls = 'notre_dame';
  cls = 'james.78';
  cls = 'train';
  cls = 'bicycle';
  cls = 'cow';
end

if ~exist('keep_images','var')
  keep_images = 0;
end

final_dir =  ...
    sprintf('%s/exemplars/models',VOCopts.localdir);
if ~exist(final_dir,'dir')
  mkdir(final_dir);
end

final_file = ...
    sprintf('%s/exemplars/models/%s.mat',VOCopts.localdir,cls);

if CACHE_FILE==1 && fileexists(final_file)
  fprintf(1,'Loading final file %s\n',final_file);
  load(final_file);
  return;
end

results_directory = ...
    sprintf('%s/exemplars/mined/',VOCopts.localdir);

%use this for james files
%files = dir([results_directory '10.' cls '-*.mat']);

%files = dir([results_directory '10.*' cls '*.mat']);

files = dir([results_directory '10.*' cls '*.mat']);
fprintf(1,'Length of files is %d\n',length(files));
if length(files) == 0
  fprintf(1,'WARNING NO MODELS LOADED FOR %s\n',cls);
  models = cell(0,1);
  return;
end

for i = 1:length(files)
  fprintf(1,'.');
  models{i} = load([results_directory files(i).name]);
  models{i} = models{i}.m;

  if 0
    fprintf(1,'HACK: killing boundary hogs\n');
    models{i}.model.w(:,1,:) = 0;
    models{i}.model.w(:,end,:) = 0;
    models{i}.model.w(1,:,:) = 0;
    models{i}.model.w(end,:,:) = 0;
  end
  
  if 0
    fprintf(1,['HACK stealing from wtrace and doing proper hole-based' ...
               ' normalization\n']);
    
    x2 = mean(models{i}.model.x,2);
    x2 = reshape(x2,models{i}.model.hg_size);
    %m = mean(x2(find(repmat(models{i}.model.mask,[1 1 31]))));
    m = mean(x2(:));
    x2 = x2 - m;
    %x2(find(repmat(1-models{i}.model.mask,[1 1 31])))=0;
    models{i}.model.w = x2;
    models{i}.model.b = -10;
  end
  
  %models{i}.model.w = ...
  %    reshape(models{i}.model.wtrace(:,1),models{i}.model.hg_size);
  
  %models{i}.model.w = models{i}.model.w - mean(models{i}.model.w(:));
  
  %keyboard
  if 0
    fprintf(1,'HACK: padding boundary\n');
    w = models{i}.model.w;
    w2 = zeros(size(w,1)+2,size(w,2)+2,31);
    w2(2:end-1,2:end-1,:) = w;
    models{i}.model.w = w2;
    
    % x = reshape(models{i}.model.x(:,13),models{i}.model.hg_size);
    % e = (sum(x.^2,3));
    % e = pad_image(e,1)==0;
    % models{i}.model.w(find(repmat(e,[1 1 31]))) = 0;
    
    models{i}.model.hg_size(1:2) = models{i}.model.hg_size(1:2)+2;
  end
  
  if max(models{i}.model.hg_size(1:2)) >= 25 | length(models{i}.model.x)==0
    fprintf(1,'Truncating very large exemplar\n');
    models{i}.model.w = zeros(1,1,31);
    models{i}.hg_size = [1 1 31];
    models{i}.model.b = 1000;
  end

  %disable negative support vectors to save space
  models{i}.model.nsv = [];
  
  %models{i}.model.number_of_negative_supports = size(models{i}.model.nsv,2);

  %Keep all x's for now
  %models{i}.model.x = mean(models{i}.model.x,2);

 
  models{i}.cls = cls;  
  %models{i}.I = [];
  %if keep_images == 0
  %  models{i}.subI = [];
  %end  
end
fprintf(1,'\n');

if CACHE_FILE==1
  fprintf(1,'Loaded models, saving to %s\n',final_file);
  save(final_file,'models');
end

%fprintf(1,'HACK: WARNING doing ridge estimation!!\n');
%models = estimate_regress_ws(models,do_ridge);
