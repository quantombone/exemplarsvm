function train_all_exemplars
%% Train models with hard negatives for all exemplars written to
%% exemplar directory (script is parallelizable)
%% Tomasz Malisiewicz (tomasz@cmu.edu)

EX_PER_CHUNK = 1;

VOCinit;

%Get the default mining parameters
mining_params = get_default_mining_params;
mining_params.SKIP_GTS_ABOVE_THIS_OS = 1.0;
mining_params.dump_last_image = 1;

initial_directory = ...
    sprintf('%s/exemplars/',VOCopts.localdir);

final_directory = ...
    sprintf('%s/mined/',initial_directory);

%make results directory if needed
if ~exist(final_directory,'dir')
  mkdir(final_directory);
end

%fprintf(1,'WARNING hardcoded trains\n');
files = dir([initial_directory '*.mat']);

mining_params.final_directory = final_directory;

%% Chunk the data into EX_PER_CHUNK exemplars per chunk so that we
%process several images, then write results for entire chunk
inds = do_partition(1:length(files),EX_PER_CHUNK);

% randomize chunk orderings
myRandomize;
ordering = randperm(length(inds));
%ordering = 1:length(inds);

for i = 1:length(ordering)

  curfiles = inds{ordering(i)};
  clear models;
  for z = 1:length(curfiles)
    filer = sprintf('%s/%s',initial_directory,files(curfiles(z)).name);
    m = load(filer);
    m = m.m;
    m.model.wtrace{1} = m.model.w;
    m.model.btrace{1} = m.model.b;
    %Set the name of this exemplar type
    m.models_name = 'nips11';
    m.iteration = 0;
    
    %Validation support vectors
    m.model.vsv = zeros(prod(m.model.hg_size),0);
    m.model.vsvids = [];
    
    %Friend support vectors
    m.model.fsv = zeros(prod(m.model.hg_size),0);
    m.model.fsvids = [];

    models{z} = m;
  end

  % Create a naming scheme for saving files
  filer2fill = sprintf('%s/%%s.%s.%05d.mat',final_directory, ...
                        models{1}.cls,ordering(i));

  %% this is the final file which we can write to a file
  %filer2final = sprintf(filer2fill,num2str(mining_params.MAXITER));
  % 
  filer2final = sprintf('%s/%s.%05d.mat',final_directory, ...
                        models{1}.cls,ordering(i));
  
  %% check if we are ready for an update
  filerlock = [filer2final '.mining.lock'];
  
  if fileexists(filer2final) || (mymkdir_dist(filerlock) == 0)
    continue
  end

  %Set up the negative set for this exemplars
  %CVPR2011 paper used all train images excluding category images
  %m.bg = sprintf('get_pascal_bg(''trainval'',''%s'')',m.cls);
  bg = get_pascal_bg('trainval',['-' models{1}.cls]);
  mining_params.alternate_validation = 1;
  
  %remove self image (not needed)
  %bg = setdiff(bg,sprintf(VOCopts.imgpath,m.curid));
  
  % if length(m.model.x) == 0
  %   fprintf(1,'Problem with this exemplar\n');
  %   error('quitting');
  %   m.model.w = m.model.w*0;
  %   m.model.b = m.model.b*0;
  %   mining_queue = '';
  %   filer2 = sprintf(filer2fill,num2str(mining_params.MAXITER));
  %   save(filer2,'m','mining_queue');
  %   clear mining_queue
  %   continue
  % end

  mining_queue = initialize_mining_queue(bg);
  
  % The mining queue is the ordering in which we process new images  
  keep_going = 1;

  %% This is the id of the chunk file we are about to write, it is
  %% not the same as the exemplar iteration, since each mine step
  %% won't necessarily update an iteration for every single exemplar
  FILEID = 0;
  
  while keep_going == 1
    FILEID = FILEID + 1;

    %Get the name of the next chunk file to write
    filer2 = sprintf(filer2fill,num2str(FILEID));
    
    %select exemplars which haven't finished training
    goods = find(cellfun(@(x)x.iteration <= mining_params.MAXITER,models));
    
    [models(goods), mining_queue] = ...
        mine_negatives(models(goods), mining_queue, bg, mining_params, ...
                       FILEID);
  
    total_mines = sum(cellfun(@(x)x.num_visited,mining_queue));
    if ((total_mines >= mining_params.MAX_TOTAL_MINED_IMAGES) || ...
          (length(mining_queue) == 0))
      fprintf(1,'Mined enough images, rest up\n');
      keep_going = 0;
      %bump up filename to final file
      filer2 = filer2final;
    end

    %m.iteration = iteration;
    % if (mining_params.dump_images == 1) || ...
    %       (mining_params.dump_last_image == 1 && iteration == mining_params.MAXITER)
    %   for z = 1:EX_PER_CHUNK
    %     figure(z)
    %     set(gcf,'PaperPosition',[0 0 20 10]);
    %     print(gcf,sprintf('%s/%s_z=%03d_iter=%05d.png', ...
    %                       final_directory,files(i).name,z,iteration),'-dpng');
    %   end
    % end
    
    
    models_save = models; 
    for q = 1:length(models)
      models{q} = prune_svs(models{q});
    end
    save(filer2,'models','mining_queue');
    models = models_save;
    
    %delete old files
    if FILEID > 1
      for q = 1:FILEID-1
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
  end %iteratiion
    
  rmdir(filerlock);
end

function m = prune_svs(m)
%When saving file, only keep negative support vectors, not
%the extra ones we save during training
rs = m.model.w(:)'*m.model.nsv - m.model.b;
goods = find(rs >= -1.0);
oldnsv = m.model.nsv;
oldsvids = m.model.svids;
oldx = m.model.x;
m.model.nsv = oldnsv(:,goods);
m.model.svids = oldsvids(goods);
%don't prune the x's
%m.model.x = [];
