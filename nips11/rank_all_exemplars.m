function rank_all_exemplars(am)
%% Here we do visual direction pre-training where there are 100 pyramid wiggles around
%% each GT object
%% Tomasz Malisiewicz (tomasz@cmu.edu)

%This field is here to allow mining of multiple exemplars
%simultaneously (but it is experimental since i'm still not 100%
%happy with what is happening).. so please keep this at 1
EX_PER_CHUNK = 1;

VOCinit;

%Get the default mining parameters
mining_params = get_default_mining_params;
mining_params.SKIP_GTS_ABOVE_THIS_OS = 1.0;
mining_params.dump_last_image = 1;

%original training doesnt do flips
mining_params.FLIP_LR = 1;

mining_params.max_negatives = 100;

%nms prevents thrashing
mining_params.NMS_MINES_OS = 0.5;
mining_params.MAX_WINDOWS_PER_IMAGE = 10;
mining_params.MAX_WINDOWS_BEFORE_SVM = 100;

%maximum number of mining iterations
%mining_params.MAXITER = 2;
%mining_params.MAX_TOTAL_MINED_IMAGES = 20;


if ~exist('am','var')
  %%% note loading models which contain wiggles
  am = load_all_models;
end

%Extract all ids and svs from all instances (these contain ~50
%wiggles per side, one normal, one LR flip of image for a total of
%~100 wiggles per instance)
target_ids= cellfun2(@(x)x.model.target_id,am);
target_xs= cellfun2(@(x)x.model.target_x,am);
target_ids = cat(1,target_ids{:})';
target_xs = cat(2,target_xs{:});

%% here we should steal all annotationse within the detection
%window
fprintf(1,'getting all gt frames\n');
tic
target_ids = get_gt_frames(target_ids);
toc

initial_directory = ...
    sprintf('%s/exemplars/',VOCopts.localdir);

final_directory = ...
    sprintf('%s/exemplars-rank2/',VOCopts.localdir);

%make results directory if needed
if ~exist(final_directory,'dir')
  mkdir(final_directory);
end

if ~exist('cls','var')
  [cls] = load_default_class;
end

files = dir([initial_directory '*' cls '*.mat']);
%files = dir([initial_directory '*002367.2*.mat']);
%files = dir([initial_directory '*002695.3*.mat']);
%files = dir([initial_directory '*000540.1*.mat']);

mining_params.final_directory = final_directory;

%% memex needs validation nodes
mining_params.extract_negatives = 1;

%% Chunk the data into EX_PER_CHUNK exemplars per chunk so that we
%process several images, then write results for entire chunk
inds = do_partition(1:length(files),EX_PER_CHUNK);

% randomize chunk orderings
myRandomize;
ordering = randperm(length(inds));

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
    m.models_name = 'reg';
    m.iteration = 0;
    
    %Validation support vectors
    m.model.vsv = zeros(prod(m.model.hg_size),0);
    m.model.vsvids = [];
    
    %Friend support vectors
    m.model.fsv = zeros(prod(m.model.hg_size),0);
    m.model.fsvids = [];

    models{z} = m;
  end

  if EX_PER_CHUNK == 1

    % Create a naming scheme for saving files
    filer2fill = sprintf('%s/%%s.%s.%d.%s.mat',final_directory, ...
                         models{1}.curid, ...
                         models{1}.objectid, ...
                         models{1}.cls);
    
    filer2final = sprintf('%s/%s.%d.%s.mat',final_directory, ...
                         models{1}.curid, ...
                         models{1}.objectid, ...
                         models{1}.cls);
  else
    % Create a naming scheme for saving files
    filer2fill = sprintf('%s/%%s.%s.%05d.mat',final_directory, ...
                         models{1}.cls,ordering(i));
   
    filer2final = sprintf('%s/%s.%05d.mat',final_directory, ...
                          models{1}.cls,ordering(i));
  end
    
  %% check if we are ready for an update
  filerlock = [filer2final '.mining.lock'];
  
  if fileexists(filer2final) || (mymkdir_dist(filerlock) == 0)
    continue
  end
  %fprintf(1,'warning no skip\n');

  %Set up the negative set for this exemplar
  %CVPR2011 paper used all train images excluding category images
  %m.bg = sprintf('get_pascal_bg(''trainval'',''%s'')',m.cls);
  
  bg = get_pascal_bg('trainval','');

  for q = 1:length(models)
    models{q}.bg_string1 = 'trainval';
    models{q}.bg_string2 = '';
  end
    
  % bg = get_pascal_bg('train',['-' models{1}.cls]);
  % for q = 1:length(models)
  %   models{q}.bg_string1 = 'train';
  %   models{q}.bg_string2 = ['-' models{1}.cls];
  % end
  
  mining_params.alternate_validation = 0;
  
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
  mining_queue = mining_queue(1:50);
  
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
    goods = find(cellfun(@(x)x.iteration <= mining_params.MAXITER, ...
                         models));
    
    [target_ids,target_xs] = get_top_from_ex(m,am);
    models{goods} = add_new_detections(models{goods},target_xs,target_ids);
    
    [models(goods), mining_queue] = ...
        mine_negatives(models(goods), mining_queue, bg, mining_params, ...
                       FILEID);
    
    total_mines = sum(cellfun(@(x)x.num_visited,mining_queue));
    if ((total_mines >= mining_params.MAX_TOTAL_MINED_IMAGES) || ...
        (length(mining_queue) == 0)) || (mining_params.skip_mine==1)
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
      %models{q}.model.nsv = [];
      %models{q}.model.svids = [];
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
[aa,bb] = sort(rs,'descend');
goods = bb(aa >= -1.0);
oldnsv = m.model.nsv;
oldsvids = m.model.svids;
m.model.nsv = oldnsv(:,goods);
m.model.svids = oldsvids(goods);

function [target_ids,target_xs] = get_top_from_ex(m,am)

res = cellfun2(@(x)m.model.w(:)'*x.model.target_x-m.model.b,am);
for i = 1:length(res)
  %[tmp,ind] = max(res{i});
  %% if we take 1:len, then we are taking all possible window wiggles
  ind = 1:length(res{i});
  am{i}.model.target_x = am{i}.model.target_x(:,ind);
  am{i}.model.target_id = am{i}.model.target_id(ind);
end

target_ids= cellfun2(@(x)x.model.target_id,am);
target_xs= cellfun2(@(x)x.model.target_x,am);

target_ids = cat(1,target_ids{:})';
target_xs = cat(2,target_xs{:});

%% convert to curid format as integer

%% HERE we take all string curids, and treat them as literals into
%the images

bg = get_pascal_bg('trainval');
s = cellfun(@(x)isstr(x.curid),target_ids);
s = find(s);
if length(s) > 0
  train_curids = cell(length(bg),1);
  for i = 1:length(bg)
    [tmp,train_curids{i},ext] = fileparts(bg{i});
  end
  
  test_curids = cellfun2(@(x)x.curid,target_ids);

  [aa,bb] = ismember(test_curids,train_curids);
  for i = 1:length(s)
    target_ids{s(i)}.curid = bb(i);
  end  
end