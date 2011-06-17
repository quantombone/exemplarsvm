function allfiles = train_all_exemplars(models, train_set, mining_params, ...
                                        dataset_params, training_function)
%% Train models with hard negatives for all exemplars written to
%% exemplar directory (script is parallelizable)
%% Tomasz Malisiewicz (tomasz@cmu.edu)

models_name = models{1}.models_name;
new_models_name = [models_name '-svm'];

%initial_directory = ...
%    sprintf('%s/%s/',dataset_params.localdir,models_name);

final_directory = ...
    sprintf('%s/models/%s/',dataset_params.localdir,new_models_name);

%make results directory if needed
if ~exist(final_directory,'dir')
  mkdir(final_directory);
end

%Find all initial files of the current class/mode
%files = dir([initial_directory '*' cls '*.mat']);
%files = dir([initial_directory '*000540.1*.mat']);

mining_params.final_directory = final_directory;

% randomize chunk orderings
myRandomize;
ordering = randperm(length(models));
models = models(ordering);

allfiles = cell(length(models), 1);

for i = 1:length(models)

  %filer = sprintf('%s/%s',initial_directory, files(ordering(i)).name);
  %m = load(filer);
  %m = m.m;
  m = models{i};
   
  % Create a naming scheme for saving files
  filer2fill = sprintf('%s/%%s.%s.%d.%s.mat',final_directory, ...
                       m.curid, ...
                       m.objectid, ...
                       m.cls);
  
  filer2final = sprintf('%s/%s.%d.%s.mat',final_directory, ...
                        m.curid, ...
                        m.objectid, ...
                        m.cls);

  allfiles{i} = filer2final;
  
  % Check if we are ready for an update
  filerlock = [filer2final '.mining.lock'];
  
  if fileexists(filer2final) || (mymkdir_dist(filerlock) == 0)
    continue
  end
  
  % Add training set and training set's mining queue 
  m.train_set = train_set;
  m.mining_queue = initialize_mining_queue(m.train_set);
  
  % Add mining_params, and dataset_params to this exemplar
  m.mining_params = mining_params;
  m.dataset_params = dataset_params;

  % Append '-svm' to the mode to create the models name
  m.models_name = new_models_name;
  m.iteration = 1;

  % The mining queue is the ordering in which we process new images  
  keep_going = 1;

  while keep_going == 1
  
    %Get the name of the next chunk file to write
    filer2 = sprintf(filer2fill,num2str(m.iteration));
      
    m = mine_train_iteration(m, training_function);

    total_mines = m.mining_stats{end}.total_mines;
    if ((total_mines >= mining_params.MAX_TOTAL_MINED_IMAGES) || ...
          (length(m.mining_queue) == 0))
      fprintf(1,'Mined enough images, rest up\n');
      keep_going = 0;
      
      %bump up filename to final file
      filer2 = filer2final;
    end

    %Save the current result
    save(filer2,'m');
  
    %delete old files
    if m.iteration > 1
      for q = 1:m.iteration-1
        filer2old = sprintf(filer2fill,num2str(q));
        if fileexists(filer2old)
          delete(filer2old);
        end
      end
    end
    
    if keep_going==0
      fprintf(1,' ##Breaking because we reached end\n');
      break;
    end
    
    m.iteration = m.iteration + 1;
  end %iteratiion
    
  rmdir(filerlock);
end


[allfiles,bb] = sort(allfiles);

% function m = prune_svs(m)
% %When saving file, only keep negative support vectors, not
% %the extra ones we save during training
% rs = m.model.w(:)'*m.model.nsv - m.model.b;
% [aa,bb] = sort(rs,'descend');
% goods = bb(aa >= -1.0);
% oldnsv = m.model.nsv;
% oldsvids = m.model.svids;
% m.model.nsv = oldnsv(:,goods);
% m.model.svids = oldsvids(goods);

% function [target_ids,target_xs] = get_top_from_ex(m,am)

% res = cellfun2(@(x)m.model.w(:)'*x.model.target_x-m.model.b,am);
% for i = 1:length(res)
%   [tmp,ind] = max(res{i});
%   am{i}.model.target_x = am{i}.model.target_x(:,ind);
%   am{i}.model.target_id = am{i}.model.target_id(ind);
% end

% target_ids= cellfun2(@(x)x.model.target_id,am);
% target_xs= cellfun2(@(x)x.model.target_x,am);

% target_ids = cat(1,target_ids{:})';
% target_xs = cat(2,target_xs{:});

% %% convert to curid format as integer

% %% HERE we take all string curids, and treat them as literals into
% %the images

% bg = get_pascal_bg('trainval');
% s = cellfun(@(x)isstr(x.curid),target_ids);
% s = find(s);
% if length(s) > 0
%   train_curids = cell(length(bg),1);
%   for i = 1:length(bg)
%     [tmp,train_curids{i},ext] = fileparts(bg{i});
%   end
  
%   test_curids = cellfun2(@(x)x.curid,target_ids);

%   [aa,bb] = ismember(test_curids,train_curids);
%   for i = 1:length(s)
%     target_ids{s(i)}.curid = bb(i);
%   end  
% end