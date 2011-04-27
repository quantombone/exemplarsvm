function [hn, models, mining_queue, mining_stats] = ...
    load_hn_fg(models, mining_queue, bg, mining_params)

%Load hard negatives from the set of images (ids)
%according to current classifier (w,b)
%choose windows above threshold (thresh)
%take at most (TOPK) windows from a single image
%operate at (lpo) levels per octave
%process the image at resolution (image_multiplier)

%Returned Data:
%(hn) is the struct containing hard negatives for (w,b)
%(I,rs) is for the last image to get hard negatives from
%(bboxes) optinally returns the actual boxes
%VOCinit;

if ~exist('mining_params','var')
  mining_params = get_default_mining_params;
end

number_of_violating_images = 0;
number_of_windows = zeros(length(models),1);

violating_images = zeros(0,1);
empty_images = zeros(0,1);

for i = 1:length(mining_queue)
  index = mining_queue{i}.index;
  I = convert_to_I(bg{index});
  %HACK ROTATE UPSIDE DOWN
  %fprintf(1,'HACK: rotate upside down negatives\n');
  %I = imrotate(I,180);

  starter = tic;
  
  KEEPSV = 1;
    
  %IF NMS is enabled, get more windows (5 times as many) since
  %nms will prune many away
  TOPK_FINAL = mining_params.MAX_WINDOWS_PER_IMAGE;
  %if mining_params.NMS_MINES_OS < 1
  %  TOPK_FINAL = TOPK_FINAL * 5;
  %end
  
  localizeparams.thresh = mining_params.detection_threshold;
  localizeparams.TOPK = TOPK_FINAL;
  localizeparams.lpo = mining_params.lpo;
  localizeparams.SAVE_SVS = KEEPSV;
  
  [rs,t] = localizemeHOG(I, models, localizeparams);
  

  for aaa = 1:length(rs.id_grid)
    for bbb = 1:length(rs.id_grid{aaa})
      rs.id_grid{aaa}{bbb}.curid = index;
    end
  end

  
  %[rs, newpositives] = prune_gts(rs, bg, index, m, mining_params, I);
  
  %if length(newpositives.id_grid) > 0
  %  newx=cat(2,newpositives.support_grid{:});
  %  m.model.x = cat(2,m.model.x,newx);
  %end
  %rs = prune_nms(rs, mining_params);

  %% Make sure we only keep 3 times the number of violating windows
  clear scores
  for q = 1:length(models)
    nviol = sum(rs.score_grid{q} >= -1);
    [aa,bb] = sort(rs.score_grid{q},'descend');
    bb = bb(1:min(length(bb),ceil(nviol*mining_params.beyond_nsv_multiplier)));
    rs.score_grid{q}=rs.score_grid{q}(bb);
    rs.id_grid{q}=rs.id_grid{q}(bb);
    rs.support_grid{q}=rs.support_grid{q}(bb);    
    scores{q} = cat(2,rs.score_grid{q});
  end
  
  
  % %% if all scores are below -1, remove all scores
  % if max(scores)<-1
  %   scores = [];
  % end

  
  addon ='';
  supersize = sum(cellfun(@(x)length(x),scores));
  if supersize > 0 %length(scores)>0
    addon=sprintf(', max = %.3f',max(cellfun(@(x)max_or_this(x,-1000),scores)));
  end
  total = sum(cellfun(@(x)x.num_visited,mining_queue));
  fprintf(1,'Found %d/%d windows, image:%05d (#seen=%d/%05d%s)\n',...
          supersize, sum(cellfun(@(x)sum(x>=-1),scores)), index, ...
          mining_queue{i}.num_visited, total, addon);

  %increment how many times we processed this image
  mining_queue{i}.num_visited = mining_queue{i}.num_visited + 1;

  number_of_windows = number_of_windows + cellfun(@(x)length(x),scores)';
  
  clear x objid
  for q = 1:length(models)
    x{q} = [];
    objid{q} = [];
    if length(rs.support_grid{q})==0
      continue
    end
    goods = cellfun(@(x)prod(size(x)),rs.support_grid{q})>0;
    %for z = 1:length(rs.support_grid{q})
    x{q} = cat(2,x{q},rs.support_grid{q}{goods});
    objid{q} = cat(2,objid{q},rs.id_grid{q}(goods));
    %end
  end
  
  %concatenate all features
  %x = cat(2,rs.support_grid{:});
  Ndets = cellfun(@(x)size(x,2),x);
  %if no detections, just skip image because there is nothing to store
  if sum(Ndets) == 0
    empty_images(end+1) = index;
    continue
  end
  
  %an image is violating if it contains some violating windows,
  %else it is an empty image
  if max(cellfun(@(x)max_or_this(x,-1000),scores))>=-1
    if (mining_queue{i}.num_visited==1)
      number_of_violating_images = number_of_violating_images + 1;
    end
        
    violating_images(end+1) = index;
  else
    empty_images(end+1) = index;
  end

  %keyboard
  %x = cat(2,x{:});

  %objid = cat(2,rs.id_grid{:});
  %objid = cat(2,objid{:});

  % if exist('w','var')  
  %   newscores = w(:)'*x - b;
  %   goods = find(newscores >= thresh);
  %   x = x(:,goods);
  %   objid = objid(goods);
  %   scores = newscores;  
  % end

  %xs2 = cellfun2(@(x)x',x);
  %xs2=[xs2{:}];
  
  %ob2 = cellfun2(@(x)x',objid);
  %ob2=[ob2{:}];
  
  for a = 1:length(models)
    xs{a}{i} = x{a};
    objids{a}{i} = objid{a};
  end

   
  
  if (max(number_of_windows) >= mining_params.MAX_WINDOWS_BEFORE_SVM) || ...
        (number_of_violating_images >= mining_params.MAX_IMAGES_BEFORE_SVM)
    fprintf(1,['Stopping mining because we have %d windows from' ...
                                                ' %d new violators\n'],...
            max(number_of_windows), number_of_violating_images);
    break;
  end
end

if ~exist('xs','var')
  %If no detections from from anymodels, return an empty matrix
  hn.xs = zeros(prod(size(models{1}.model.w)),0);
  hn.objids = [];
  mining_stats.num_violating = 0;
  mining_stats.num_empty = 0;
  return;
end

%xs = cat(2,xs{:});
%objids = cat(2,objids{:});
hn.xs = cellfun2(@(x)cat(2,x{:}),xs);
hn.objids = cellfun2(@(x)cat(2,x{:}),objids);

fprintf(1,'# Violating images: %d, #Non-violating images: %d\n', ...
        length(violating_images), length(empty_images));

mining_stats.num_empty = length(empty_images);
mining_stats.num_violating = length(violating_images);

%NOTE: there are several different mining scenarios possible here
%a.) dont process already processed images
%b.) place violating images at end of queue, eliminate free ones
%c.) place violating images at start of queue, eliminate free ones

if strcmp(mining_params.queue_mode,'onepass') == 1
  %% MINING QUEUE UPDATE by removing already seen images
  mining_queue = update_mq_onepass(mining_queue, violating_images, ...
                                   empty_images);
elseif strcmp(mining_params.queue_mode,'cycle-violators') == 1
  %% MINING QUEUE update by cycling violators to end of queue
  %mining_queue = update_mq_cycle_violators(mining_queue, violating_images, ...
  %                                 empty_images);
elseif strcmp(mining_params.queue_mode,'front-violators') == 1
  %% MINING QUEUE UPDATE by removing already seen images, and
  %front-placing violators (used in CVPR11)
  %mining_queue = update_mq_front_violators(mining_queue, ...
  %                                         violating_images, ...
  %                                         empty_images);
else
  error(sprintf('Invalid queue mode: %s\n', ...
                mining_params.queue_mode));
end



function mining_queue = update_mq_onepass(mining_queue, violating_images, ...
                                           empty_images)


%% Take the violating images and remove them from queue

mover_ids = find(cellfun(@(x)ismember(x.index,violating_images), ...
                         mining_queue));

%enders = mining_queue(mover_ids);
mining_queue(mover_ids) = [];
%HACK eliminate all seen images from queue
%fprintf(1,'HACK: not adding violators to end of queue\n');
%mining_queue = cat(2,mining_queue,enders);

if 0
    %% We now take the empty images and place them on the end of the queue
    mover_ids = find(cellfun(@(x)ismember((x.index),empty_images), ...
                             mining_queue));
    
    enders = mining_queue(mover_ids);
    mining_queue(mover_ids) = [];
    mining_queue = cat(2,mining_queue,enders);
end

%% We now take the empty images and remove them from queue
mover_ids = find(cellfun(@(x)ismember(x.index,empty_images), ...
                         mining_queue));

%enders = mining_queue(mover_ids);
mining_queue(mover_ids) = [];

if 0  %NOTE: this strategy will decrease variety in the world
  %% We now take the violating images and place them on the start of the queue
  mover_ids = find(cellfun(@(x)ismember((x.index),violating_images), ...
                           mining_queue));
  
  starters = mining_queue(mover_ids);
  mining_queue(mover_ids) = [];
  mining_queue = cat(2,starters,mining_queue);
end


function mining_queue = update_mq_front_violators(mining_queue,...
                                                  violating_images, ...
                                                  empty_images)
%An update procedure where the violating images are pushed to front
%of queue, and empty images are removed

%% We now take the empty images and remove them from queue
mover_ids = find(cellfun(@(x)ismember(x.index,empty_images), ...
                         mining_queue));

%enders = mining_queue(mover_ids);
mining_queue(mover_ids) = [];


%% We now take the violating images and place them on the start of the queue
mover_ids = find(cellfun(@(x)ismember((x.index),violating_images), ...
                         mining_queue));

starters = mining_queue(mover_ids);
mining_queue(mover_ids) = [];
mining_queue = cat(2,starters,mining_queue);


function mining_queue = update_mq_cycle_violators(mining_queue,...
                                                  violating_images, ...
                                                  empty_images)
%An update procedure where the violating images are pushed to front
%of queue, and empty images are pushed to back

%% We now take the violating images and place them on the end of the queue
mover_ids = find(cellfun(@(x)ismember((x.index),violating_images), ...
                         mining_queue));

enders = mining_queue(mover_ids);
mining_queue(mover_ids) = [];
mining_queue = cat(2,mining_queue,enders);

%% We now take the empty images and remove them from the queue
mover_ids = find(cellfun(@(x)ismember(x.index,empty_images), ...
                         mining_queue));

enders = mining_queue(mover_ids);
mining_queue(mover_ids) = [];
%mining_queue = cat(2,mining_queue,enders);

function [rs,newpositives] = prune_gts(rs, bg, index, m, mining_params, ...
                                       I)
%% Here we skip all objects that overlap ANY gt polygons by the
%threshold SKI_GTS_OS
newpositives.id_grid = cell(0,1);
newpositives.support_grid = cell(0,1);
newpositives.score_grid = [];

if (mining_params.SKIP_GTS_ABOVE_THIS_OS >= 1) || ...
      (length(rs.id_grid{1}) == 0)
  return;
end

objids = [rs.id_grid{1}{:}];
overlaps = get_overlaps_with_gt(m, objids, bg, I);

for aaa = 1:length(rs.id_grid{1})
  rs.id_grid{1}{aaa}.maxos = overlaps(aaa);
end
killids = (overlaps >= mining_params.SKIP_GTS_ABOVE_THIS_OS);
%max_killed_score = max(rs.score_grid{1}((overlaps >= ...
%                                         mining_params ...
%                                         .SKIP_GTS_ABOVE_THIS_OS) & ...
%                                        (overlaps < ...
%                                         mining_params.ASSIMILATE_GTS_ABOVE_THIS_OS)));
%if length(max_killed_score) == 0
%  max_killed_score = -1;
%end

%stealids = (overlaps >= mining_params.ASSIMILATE_GTS_ABOVE_THIS_OS) & ...
%    (rs.score_grid{1}' > max_killed_score);%

%NS = sum(stealids==1);
%if NS > 0
%  fprintf(1,'STOLE %d@GT ',NS);
%  newpositives.id_grid = rs.id_grid{1}(stealids);
%  newpositives.support_grid = rs.support_grid{1}(stealids);
%  newpositives.score_grid = rs.score_grid{1}(stealids);
%end

NK = sum(killids==1);
if NK > 0
  fprintf(1,'KILLED %d@GT ',NK);
end
rs.score_grid{1}(killids) = [];
rs.id_grid{1}(killids) = [];
rs.support_grid{1}(killids) = [];

function rs = prune_nms(rs, mining_params)
%Prune via nms to eliminate redundant detections

if (mining_params.NMS_MINES_OS >= 1) || ...
      (length(rs.id_grid{1}) == 0)
  return;
end

bbs=cellfun2(@(x)x.bb,rs.id_grid{1});
bbs = cat(1,bbs{:});
bbs(:,5) = 1:size(bbs,1);
bbs(:,6) = 1;
bbs(:,7) = rs.score_grid{1}';
bbs = nms(bbs, mining_params.NMS_MINES_OS);
bbs = bbs(1:min(size(bbs,1),mining_params.MAX_WINDOWS_PER_IMAGE),:);
ids = bbs(:,5);
rs.score_grid{1} = rs.score_grid{1}(ids);
rs.id_grid{1} = rs.id_grid{1}(ids);
rs.support_grid{1} = rs.support_grid{1}(ids);
