function train_all_chunks
%% Train models with hard negatives for all exemplars written to
%% exemplar directory (script is parallelizable)
%% Tomasz Malisiewicz (tomasz@cmu.edu)

VOCinit;

%Get the default mining parameters
mining_params = get_default_mining_params;
mining_params.SKIP_GTS_ABOVE_THIS_OS = 1.0;
mining_params.dump_last_image = 1;

initial_directory = ...
    sprintf('%s/test_chunks/',VOCopts.localdir);

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

%files = files(1);
%files(1).name = '2008_000343.2.train.mat';
%files(1).name = '000827.1.cow.mat';
%files(1).name = '001073.1.cow.mat';
%files(1).name = '000540.1.train.mat';
%files(1).name = '004102.1.train.mat';
%files(1).name = 'special.171.train.mat';
for i = 1:length(files)

  filer = sprintf('%s/%s',initial_directory,files(i).name);  
  filer
  
  ms = load(filer);
  ms = ms.ms;
 
  
  for qqq = 1:length(ms)

  m = ms{qqq};
  
  
  filer2fill = sprintf('%s/%%s.%d.%s',final_directory, ...
                       qqq,files(i).name);

  %this is the final file which we can write to a file
  filer2final = sprintf(filer2fill,num2str(mining_params.MAXITER));
  
  %check if we are ready for an update
  filerlock = [filer2final '.mining.lock'];

  if fileexists(filer2final) || (mymkdir_dist(filerlock) == 0)
    continue
  end

  
  %Set the name of this exemplar type
  m.models_name = 'nips11';
  
  %Set up the negative set for this exemplars
  %CVPR2011 paper used all train images excluding category images
  %m.bg = sprintf('get_pascal_bg(''trainval'',''%s'')',m.cls);
  bg = get_james_bg(10000);
  rinds = randperm(length(bg));
  bg = bg(rinds(1:50));
  for q = 1:length(bg)
    fprintf(1,'.');
    cur = convert_to_I(bg{q});
    cur = imresize(cur,300/max(size(cur)));
    cur = max(0.0,min(1.0,cur));
    bg{q} = cur;
  end

  %remove self image
  %bg = setdiff(bg,sprintf(VOCopts.imgpath,m.curid));
  
  %bg = bg(1:100);
  
  %bg{1} = '/nfs/hn38/users/sdivvala/Datasets/Pascal_VOC/VOC2009/JPEGImages/2009_001300.jpg'
  
  %m.bg = sprintf('get_pascal_bg(''train'',''-%s'')',m.cls);
  %m.fg = sprintf('get_pascal_bg(''test'')');
  
  % os = getosmatrix_bb(m.gt_box,m.model.coarse_box)
  % if os<.5
    
  %   figure(1)
  %   clf
  %   P = 100;
  %   I = pad_image(im2double(imread(sprintf(VOCopts.imgpath,m.curid))),P);
  %   imagesc(I)
  %   plot_bbox(m.gt_box+P)
  %   plot_bbox(m.model.coarse_box+P,'',[1 0 0]);
  %   pause
  % end
  % continue;
    
  %fprintf(1,'HACK using trains as negatives!!!\n');
  %m.bg = 'get_pascal_bg(''train'',''train'')';

  if length(m.model.x) == 0
    fprintf(1,'Problem with this exemplar\n');
    error('quitting');
    m.model.w = m.model.w*0;
    m.model.b = m.model.b*0;
    mining_queue = '';
    filer2 = sprintf(filer2fill,num2str(mining_params.MAXITER));
    save(filer2,'m','mining_queue');
    clear mining_queue
    continue
  end

  %If exemplar has its own bg (encoded as a string)
  %if isfield(m,'bg') && isstr(m.bg)
  %  %TODO: using eval might have its own problems in the future :-/
  %  bg = eval(m.bg);
  %else
  %  bg = get_pascal_bg('train','',m.curid);
  %end
  
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
    if (mining_params.dump_images == 1) || ...
          (mining_params.dump_last_image == 1 && iteration == mining_params.MAXITER)
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
      oldx = m.model.x;
      m.model.nsv = oldnsv(:,goods);
      m.model.svids = oldsvids(goods);
      m.model.x = [];
      save(filer2,'m','mining_queue');
      
      %re-populate with all negatives
      m.model.nsv = oldnsv;
      m.model.svids = oldsvids;
      m.model.x = oldx;
      
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
  
    

end
