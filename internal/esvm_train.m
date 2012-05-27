function model = esvm_train(model, params)
% Train models with hard negatives mined from the data_set with the
% parameters in params
% Usage:
% >> models = esvm_train(model, params);
%
% >> models = esvm_train(model);
%
% [model]: an initialized model
% [params]: localization and training parameters
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

% if length(data_set) == 0
%   fprintf(1,'esvm_train: Empty dataset, not continuing\n');
%   models = [];
%   return;
% end

if ~exist('params','var')
  params = model.params;
else
  model.params = params;
  %models = cellfun(@(x)setfield(x,'params',params),models,'UniformOutput',false);
end

if length(params.localdir)==0
  CACHE_FILE = 0;
else
  CACHE_FILE = 1;
end

model_name = model.model_name;
new_model_name = [model_name params.training_function('')];

cache_dir =  ...
    sprintf('%s/models/',params.localdir);

cache_file = ...
    sprintf('%s/%s.mat',cache_dir,new_model_name);

cache_file_stripped = ...
    sprintf('%s/%s-stripped.mat',cache_dir,new_model_name);

if CACHE_FILE == 1 && fileexists(cache_file)
  load(cache_file,'model');
  return;
end

if CACHE_FILE == 1 && fileexists(cache_file)
  load(cache_file,'model');
  return;
end

%DUMPDIR = sprintf('%s/results/svs/%s/',params.localdir, ...
%                  new_model_name);

%display of SV pdfs disabled
%if CACHE_FILE==1 && params.display ==1 && ~exist(DUMPDIR,'dir')%  mkdir(DUMPDIR);
%end

final_directory = ...
    sprintf('%s/models/%s/',params.localdir,new_model_name);

%make results directory if needed
%if CACHE_FILE == 1 && ~exist(final_directory,'dir')
%  mkdir(final_directory);
%end

%Assign unique identifiers to all of the models
model.models = cellfun(@(x,y)setfield(x,'identifier',y),...
                       model.models,num2cell(1:length(model.models)),...
                       'UniformOutput',false);

% randomize chunk orderings
if CACHE_FILE == 1
  myRandomize;
  ordering = randperm(length(model.models));
else
  ordering = 1:length(model.models);
end
model.models = model.models(ordering);

allfiles = cell(1,length(model.models));
for i = 1:length(model.models)
  
  m = rmfield(model,'models');
  m.models = model.models(i);
  m.models{1}.params = params;
  filer2final = sprintf('%s/%s_%04d.mat', ...
                            final_directory, new_model_name,...
                            m.models{1}.identifier);
  %[basedir, basename, ext] = fileparts(complete_file);
  %filer2final = sprintf('%s/%s.mat',basedir,basename);  
  
  allfiles{i} = filer2final;
  
  % Check if we are ready for an update
  filerlock = [filer2final '.lock'];

  if CACHE_FILE == 1
    if fileexists(filer2final) || (mymkdir_dist(filerlock) == 0)
      continue
    end
  end
  
  % Append '-svm' to the mode to create the models name
  m.model_name = new_model_name;
  model.model_name = new_model_name;
  m.iteration = 1;
  m.total = -1;
  
  for j = 1:length(m.models)
    if isfield(m.models{j},'mining_stats')
      m.models{j} = rmfield(m.models{j},'mining_stats');
    end
  end

  %save a trace of variables during learning
  %m.models{1}.wtrace = cell(0,1);
  %m.models{1}.btrace = {};

  if isfield(m,'svxs') && numel(m.svxs)>0
    fprintf(1,'Pre-SVMing');
    %do the latent updates
    m = params.training_function(m);
  end
  
  %if we are a distance function, initialize to uniform weights
  if isfield(params,'wtype') && ...
        strcmp(params.wtype,'dfun')==1
    fprintf(1,'WARNING: initializing a dfun, not SVM\n');
    m.models{1}.w = m.w*0-1;
    m.models{1}.b = -1000;
  end

  % The mining queue is the ordering in which we process new images  
  keep_going = 1;
  
  % Add training set and training set's mining queue 
  mining_queue = esvm_initialize_mining_queue(m, model.cls);
  mining_queue = repmat(mining_queue(:),ceil(params.train_max_mined_images/length(mining_queue)));
  mining_queue = mining_queue(1:min(length(mining_queue),...
                                    params.train_max_mined_images));

  total_mines = 0;  
  while keep_going == 1
    % if ~isfield(m,'mining_stats')
    %   total_mines = 0;
    % else
    %   total_mines = sum(cellfun(@(x)x.total_mines, m.mining_stats));
    % end
    % m.total_mines = total_mines;

    m.data_set = model.data_set;
    m.params = model.params;
    m.model_name = model.model_name;

    [m,mining_queue] = esvm_mine_train_iteration(m, mining_queue);
    total_mines = sum(cellfun(@(x)x.total_mines, m.models{1}.mining_stats));

    if ((total_mines >= params.train_max_mined_images) || ...
          (isempty(mining_queue))) || ...
          (m.iteration == params.train_max_mine_iterations)
      
      keep_going = 0;      
      m.models{1} = rmfield(m.models{1},'mining_stats');
      %bump up filename to final file
      %filer2 = filer2final;
    end

    % if length(m.models{1}.wtrace)>=2
    %   diffy = norm(m.models{1}.wtrace{end}(:)- ...
    %                m.models{1}.wtrace{end-1}(:));
    %   if diffy < .0001
    %     fprintf(1,['Stopping learning because w failed to change' ...
    %                ' across an iteration\n']);
    %     keep_going = 0;
    %   end
    % end

    %HACK: remove neg_set which causes save issue when it is a
    %cell array of function pointers
    %msave = m;
    %if any(cellfun(@(x)~isstr(x),neg_set))
    %  m = rmfield(m,'train_set');
    %end
    
    %Save the current result
    if CACHE_FILE == 1
      %savem(filer2final,m);
      
      %filerpng = [filer2 '.png'];
      % if ~fileexists(filerpng)
      %   [aa,bb] = sort(m.model.w(:)'*m.model.svxs,'descend');
      %   Icur = esvm_show_det_stack(m.model.svbbs(bb,:),...
      %                              neg_set, ...
      %                              10,10,m);
      %   imwrite(Icur,filerpng);
      % end
    else
      allfiles{i} = m;
    end
    
    %m = msave;
    
    % if params.display == 1

    %   if params.write_after_display == 1
    %     exid = ordering(i);
    %     filer = sprintf('%s/%s.%s.%05d.png', DUMPDIR, 'train', ...
    %                     m.cls,exid);
        
    %     if fileexists(filer)
    %       continue
    %     end
    %   end
      
    %   figure(445);
    %   clf;
    %   showI = esvm_show_det_stack(m.model.svbbs,m.train_set,5,5,m);
    %   imagesc(showI);
    %   title('Exemplar and Top Dets');
    %   drawnow;
      
    %   figure(235)
      
    %   rpos = m.model.w(:)'*m.model.x-m.model.b;
    %   rneg = m.model.w(:)'*m.model.svxs - m.model.b;
    %   clf;
    %   plot(sort(rpos,'descend'),'r.');
    %   hold on;
    %   plot(length(rpos)+(1:length(rneg)),rneg,'b.');
    %   drawnow;

    %   if params.write_after_display == 1
    %     set(gcf,'PaperPosition',[0 0 20 20]);
    %     imwrite(showI,filer);
    %   end
    % end
    
    % %delete old files
    % if m.iteration > 1
    %   for q = 1:m.iteration-1
    %     filer2old = sprintf(filer2fill,num2str(q));
    %     if fileexists(filer2old)
    %       if CACHE_FILE == 1
    %         delete(filer2old);
    %       end
    %     end
    %   end
    % end
    
    if keep_going==0
      fprintf(1,' ### End of training... \n');
      break;
    end
    
    m.iteration = m.iteration + 1;
  end %iteratiion
  
  if length(m.params.localdir) > 0
    savem(filer2final,m);
  end
  
  try
    if CACHE_FILE == 1
      rmdir(filerlock);
    end
  catch
    fprintf(1,'Cannot delete %s\n',filerlock);
  end
end

if CACHE_FILE == 0
  %model = allfiles;
  model.models = cellfun(@(x)x.models,allfiles);
  return;
end

allfiles = sort(allfiles);

%Load all of the initialized exemplars
CACHE_FILE = 1;
STRIP_FILE = 0;

if new_model_name(1) == '-'
  CACHE_FILE = 0;
  STRIP_FILE = 0;
end

DELETE_INITIAL = 1;

m = esvm_load_models(params, new_model_name, allfiles, ...
                     CACHE_FILE, STRIP_FILE, DELETE_INITIAL);
m = m{1};

function savem(filer2,m)
save(filer2,'m');
