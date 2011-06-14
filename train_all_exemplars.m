function train_all_exemplars(cls,mode)
%% Train models with hard negatives for all exemplars written to
%% exemplar directory (script is parallelizable)
%% Tomasz Malisiewicz (tomasz@cmu.edu)


VOCinit;

if ~exist('cls','var')
  [cls,mode] = load_default_class;
end

%Get the default mining parameters
mining_params = get_default_mining_params;
mining_params.SKIP_GTS_ABOVE_THIS_OS = 1.0;
mining_params.dump_last_image = 1;
mining_params.dump_images = 1;

mining_params.MAXSCALE = 0.5;

%original training doesnt do flips
mining_params.FLIP_LR = 1;

%nms prevents thrashing, but it is much slower
mining_params.NMS_MINES_OS = 1.0;

initial_directory = ...
    sprintf('%s/%s/',VOCopts.localdir,mode);

final_directory = ...
    sprintf('%s/%s-svm/',VOCopts.localdir,mode);

%make results directory if needed
if ~exist(final_directory,'dir')
  mkdir(final_directory);
end

%Find all initial files of the current class/mode
files = dir([initial_directory '*' cls '*.mat']);
%files = dir([initial_directory '*000540.1*.mat']);

mining_params.final_directory = final_directory;

% Enable this if we need to check mined windows whethere they are
% from validation set or from negative set... (esvm only uses negatives)
mining_params.extract_negatives = 0;

% Chunk the data into EX_PER_CHUNK exemplars per chunk so that we
% process several images, then write results for entire chunk
%inds = do_partition(1:length(files),EX_PER_CHUNK);

% randomize chunk orderings
myRandomize;
ordering = randperm(length(files));

for i = 1:length(ordering)

  filer = sprintf('%s/%s',initial_directory, files(ordering(i)).name);
  m = load(filer);
  m = m.m;
  m.model.wtrace{1} = m.model.w;
  m.model.btrace{1} = m.model.b;
    
  %Set the name of this exemplar type
  m.models_name = sprintf('%s-svm',mode);
  m.iteration = 0;

  % Create a naming scheme for saving files
  filer2fill = sprintf('%s/%%s.%s.%d.%s.mat',final_directory, ...
                       m.curid, ...
                       m.objectid, ...
                       m.cls);
  
  filer2final = sprintf('%s/%s.%d.%s.mat',final_directory, ...
                        m.curid, ...
                        m.objectid, ...
                        m.cls);
    
  %% check if we are ready for an update
  filerlock = [filer2final '.mining.lock'];
  
  if fileexists(filer2final) || (mymkdir_dist(filerlock) == 0)
    continue
  end

  %Set up the negative set for this exemplars
  %CVPR2011 paper used all train images excluding category images
  %m.bg = sprintf('get_pascal_bg(''trainval'',''%s'')',m.cls);
  
  set_string = 'train';
  subset_string = sprintf('-%s',m.cls);
  bg = get_pascal_bg(set_string,subset_string);  
  mining_params.alternate_validation = 0;
  
  %remove self image (not needed)
  %bg = setdiff(bg,sprintf(VOCopts.imgpath,m.curid));
  
  fprintf(1,'nonrandom queue\n');
  mining_queue = initialize_mining_queue(bg,1:length(bg));
  
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
      
    [m, mining_queue] = ...
        mine_negatives(m, mining_queue, bg, mining_params, ...
                       FILEID);
  
    total_mines = sum(cellfun(@(x)x.num_visited,mining_queue));
    if ((total_mines >= mining_params.MAX_TOTAL_MINED_IMAGES) || ...
          (length(mining_queue) == 0))
      fprintf(1,'Mined enough images, rest up\n');
      keep_going = 0;
      %bump up filename to final file
      filer2 = filer2final;
    end

    %Save the current result
    save(filer2,'m','mining_queue');
  
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

function [target_ids,target_xs] = get_top_from_ex(m,am)

res = cellfun2(@(x)m.model.w(:)'*x.model.target_x-m.model.b,am);
for i = 1:length(res)
  [tmp,ind] = max(res{i});
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