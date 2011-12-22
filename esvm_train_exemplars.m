function [newmodels] = esvm_train_exemplars(dataset_params, ...
                                            models, train_set)
%% Train models with hard negatives for all exemplars written to
%% exemplar directory (script is parallelizable)
%% Tomasz Malisiewicz (tomasz@cmu.edu)

mining_params = dataset_params.mining_params;

models_name = models{1}.models_name;
new_models_name = [models_name mining_params.training_function()];

DUMPDIR = sprintf('%s/www/svs/%s/',dataset_params.localdir, ...
                  new_models_name);

if dataset_params.display ==1 && ~exist(DUMPDIR,'dir')
  mkdir(DUMPDIR);
end

%initial_directory = ...
%    sprintf('%s/%s/',dataset_params.localdir,models_name);

final_directory = ...
    sprintf('%s/models/%s-%s/',dataset_params.localdir,...
            models{1}.cls,...
            new_models_name);

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

%always use random ordering
%if dataset_params.display == 1
%  ordering = 1:length(ordering);
%end
models = models(ordering);

allfiles = cell(length(models), 1);

parfor i = 1:length(models)

  %filer = sprintf('%s/%s',initial_directory, files(ordering(i)).name);
  %m = load(filer);
  %m = m.m;
  m = models{i};
   
  % Create a naming scheme for saving files
  filer2fill = sprintf('%s/%%s.%s.mat',final_directory, ...
                       m.name);
  
  filer2final = sprintf('%s/%s.mat',final_directory, ...
                        m.name);

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
  
  %if we are a distance function, initialize to uniform weights
  if isfield(dataset_params.params,'wtype') && ...
        strcmp(dataset_params.params.wtype,'dfun')==1
    m.model.w = m.model.w*0-1;
    m.model.b = -1000;

  end

  % The mining queue is the ordering in which we process new images  
  keep_going = 1;

  while keep_going == 1
  
    %Get the name of the next chunk file to write
    filer2 = sprintf(filer2fill,num2str(m.iteration));

    if ~isfield(m,'mining_stats')
      total_mines = 0;
    else
      total_mines = sum(cellfun(@(x)x.total_mines,m.mining_stats));
    end
    m.total_mines = total_mines;
    m = esvm_mine_train_iteration(m, mining_params.training_function);

    %total_mines = m.mining_stats{end}.total_mines;

    if ((total_mines >= mining_params.MAX_TOTAL_MINED_IMAGES) || ...
          (length(m.mining_queue) == 0)) || ...
          (m.iteration == mining_params.MAX_MINE_ITERATIONS)

      keep_going = 0;      
      %bump up filename to final file
      filer2 = filer2final;
    end
    
    %HACK: remove train_set which causes save issue when it is a
    %cell array of function pointers
    msave = m;
    m = rmfield(m,'train_set');
    
    %Save the current result
    savem(filer2,m);
    m = msave;
    
    if 0 %%dataset_params.display == 1

      exid = ordering(i);
      filer = sprintf('%s/%s.%s.%05d.png', DUMPDIR, 'train', ...
                      m.cls,exid);
      
      %if fileexists(filer)
      %  continue
      %end
      
      figure(445)
      clf
      showI = get_sv_stack(m,5,5);
      imagesc(showI)
      drawnow
      
      figure(235)
      
      rpos = m.model.w(:)'*m.model.x-m.model.b;
      rneg = m.model.w(:)'*m.model.svxs - m.model.b;
      clf
      plot(sort(rpos,'descend'),'r.')
      hold on;
      plot(length(rpos)+[1:length(rneg)],rneg,'b.')
      drawnow

      %set(gcf,'PaperPosition',[0 0 20 20]);
      %imwrite(showI,filer);
    end
    
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
      fprintf(1,' ### End of training... \n');
      break;
    end
    
    m.iteration = m.iteration + 1;
  end %iteratiion
    
  try
    rmdir(filerlock);
  catch
  end
end

[allfiles,bb] = sort(allfiles);

%Load all of the initialized exemplars
CACHE_FILE = 1;
STRIP_FILE = 0;
DELETE_INITIAL = 1;
newmodels = esvm_load_models(dataset_params, new_models_name, allfiles, ...
                          CACHE_FILE, STRIP_FILE, DELETE_INITIAL);


function savem(filer2,m)
save(filer2,'m');
