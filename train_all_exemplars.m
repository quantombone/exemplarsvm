function train_all_exemplars
%% Train models with hard negatives for all exemplars written to
%% exemplar directory (script is parallelizable)
%% Tomasz Malisiewicz (tomasz@cmu.edu)

EX_PER_CHUNK = 2;

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
  
    %Set the name of this exemplar type
    m.models_name = 'nips11';
    models{z} = m;  
  end

  filer2fill = sprintf('%s/%%s.%s.%05d.mat',final_directory, ...
                        models{1}.cls,ordering(i));

  % %this is the final file which we can write to a file
  filer2final = sprintf(filer2fill,num2str(mining_params.MAXITER));
  
  % %check if we are ready for an update
  filerlock = [filer2final '.mining.lock'];
  
  %fprintf(1,'hack not checking\n');
  if fileexists(filer2final) || (mymkdir_dist(filerlock) == 0)
    continue
  end

  %Set up the negative set for this exemplars
  %CVPR2011 paper used all train images excluding category images
  %m.bg = sprintf('get_pascal_bg(''trainval'',''%s'')',m.cls);
  bg = get_pascal_bg('train',['-' models{1}.cls]);
  
  %remove self image
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
  
  %Save a trace of learned w's (one per iteration)
  for q = 1:length(models)
    models{q}.model.wtrace{1} = models{q}.model.w;
    models{q}.model.btrace{1} = models{q}.model.b;
  end
  % The mining queue is the ordering in which we process new images  
  keep_going = 1;
  
  for iteration = 1:mining_params.MAXITER       
    [models, mining_queue] = ...
        mine_negatives(models, mining_queue, bg, mining_params, iteration);
  
    total_mines = sum(cellfun(@(x)x.num_visited,mining_queue));
    if ((total_mines >= mining_params.MAX_TOTAL_MINED_IMAGES) || ...
          (length(mining_queue) == 0))
      fprintf(1,'Mined enough images, rest up\n');
      keep_going = 0;
      iteration = mining_params.MAXITER;
    end

    %m.iteration = iteration;
    if (mining_params.dump_images == 1) || ...
          (mining_params.dump_last_image == 1 && iteration == mining_params.MAXITER)
      figure(1)
      set(gcf,'PaperPosition',[0 0 15 15]);
      print(gcf,sprintf('%s/%s_%03d.png',final_directory,files(i).name,iteration),'-dpng');
    end
    
    if 1
      filer2 = sprintf(filer2fill,num2str(iteration));
      models_save = models; 
      for q = 1:length(models)
        models{q} = prune_svs(models{q});
      end
      save(filer2,'models','mining_queue');
      models = models_save;
      
      %delete old files
      if iteration > 1
        for q = 1:iteration-1
          filer2old = sprintf(filer2fill,num2str(q));
          if fileexists(filer2old)
            delete(filer2old);
          end
        end
      end
    end
    
    if keep_going==0
      fprintf(1,' ##Breaking because we reached end\n');
      break;
    end
  end %iteration
    
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
