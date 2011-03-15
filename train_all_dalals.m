function train_all_dalals
%% Train models with hard negatives for all exemplars written to
%% exemplar directory (script is parallelizable)
%% Tomasz Malisiewicz (tomasz@cmu.edu)

VOCinit;

%Get the default mining parameters
mining_params = get_default_mining_params;
mining_params.NMS_MINES_OS = 10;
mining_params.MAXITER = 100;
mining_params.SKIP_GTS_ABOVE_THIS_OS = 100;
mining_params.beyond_nsv_multiplier = 3.0;
mining_params.max_negatives = 3000;
mining_params.DOMINANT_GRADIENT_PROJECTION = 0;

initial_directory = ...
    sprintf('%s/dalals/',VOCopts.localdir);

final_directory = ...
    sprintf('%s/mined/',initial_directory);

%make results directory if needed
if ~exist(final_directory,'dir')
  mkdir(final_directory);
end

files = dir([initial_directory '*.mat']);

%randomize file access
myRandomize;
rrr = randperm(length(files));
files = files(rrr);

%keyboard
%files = files(1);
%files(1).name = 'dalal.train.mat';
%files(1).name = '000827.1.cow.mat';
%files(1).name = '001073.1.cow.mat';
%files(1).name = '000540.1.train.mat';
%files(1).name = '004102.1.train.mat';
%files(1).name = 'special.171.train.mat';
for i = 1:length(files)

  filer = sprintf('%s/%s',initial_directory,files(i).name);  
  filer
  filer2fill = sprintf('%s/%%s.%s',final_directory, ...
                       files(i).name);

  %this is the final file which we can write to a file
  filer2final = sprintf(filer2fill,num2str(mining_params.MAXITER));
  
  %check if we are ready for an update
  filerlock = [filer2final '.mining.lock'];

  if fileexists(filer2final) || (mymkdir_dist(filerlock) == 0)
    continue
  end

  m = load(filer);
  m = m.m;

  fprintf(1,'HACK using only 50 exemplars!!!\n');
  m.model.allx = m.model.x;
  m.model.keepx = 100;
  r = (m.model.w(:)'*m.model.x);
  [aa,bb] = sort(r,'descend');
  m.model.x = m.model.x(:,bb(1:100));

  
  %use negative set for mining
  m.bg = sprintf('get_pascal_bg(''train'',''-%s'')',m.cls);
  
  if length(m.model.x) == 0
    m.model.w = m.model.w*0;
    m.model.b = m.model.b*0;
    mining_queue = '';
    filer2 = sprintf(filer2fill,num2str(mining_params.MAXITER));
    save(filer2,'m','mining_queue');
    clear mining_queue
    continue
  end

  %If exemplar has its own bg (encoded as a string)
  if isfield(m,'bg') && isstr(m.bg)
    %TODO: using eval might have its own problems in the future :-/
    bg = eval(m.bg);
  else
    bg = get_pascal_bg('train','',m.curid);
  end
  
  mining_queue = initialize_mining_queue(bg);
  
  %Save a trace of learned w's (one per iteration)
  m.model.wtrace{1} = m.model.w;
  m.model.btrace{1} = m.model.b;
  
  % The mining queue is the ordering in which we process new images  
  keep_going = 1;
  
  for iteration = 1:mining_params.MAXITER       
    [m, mining_queue] = ...
        mine_negatives(m, mining_queue, bg, mining_params, iteration);
        
    % Append new w to trace
    m.model.wtrace{end+1} = m.model.w;
    m.model.btrace{end+1} = m.model.b;
    
    total_mines = sum(cellfun(@(x)x.num_visited,mining_queue));
    if ((total_mines >= mining_params.MAX_TOTAL_MINED_IMAGES) || ...
          (length(mining_queue) == 0))
      fprintf(1,'Mined enough images, rest up\n');
      keep_going = 0;
      iteration = mining_params.MAXITER;
    end

    m.iteration = iteration;
    if mining_params.dump_images == 1
      figure(1)
      set(gcf,'PaperPosition',[0 0 15 15]);
      print(gcf,sprintf('%s/%s_%03d.png',final_directory,files(i).name,iteration),'-dpng');
    end
    
    if 1
      filer2 = sprintf(filer2fill,num2str(iteration));
      
      %When saving file, only keep negative support vectors, not
      %the extra ones we save during training
      rs = m.model.w(:)'*m.model.nsv - m.model.b;
      goods = find(rs >= -1.0);
      oldnsv = m.model.nsv;
      oldsvids = m.model.svids;
      m.model.nsv = oldnsv(:,goods);
      m.model.svids = oldsvids(goods);
      save(filer2,'m','mining_queue');
      
      %re-populate with all negatives
      m.model.nsv = oldnsv;
      m.model.svids = oldsvids;
      
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
