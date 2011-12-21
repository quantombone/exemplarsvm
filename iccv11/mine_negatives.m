function [hn, mining_queue, mining_stats] = ...
    mine_negatives(models, mining_queue, bg, mining_params)
% Compute detections "aka Hard-Negatives" hn for the images in the
% stream/queue [bg/mining_queue] when given K classifiers [models]
% 
% Input Data:
% models: Kx1 cell array of models
% mining_queue: the mining queue create from
%    initialize_mining_queue(bg)
% bg: the source of images (potentially already in pyramid feature
%   format)
% mining_params: the parameters of the mining/localization
% procedure
% 
% Returned Data: 
% hn: Kx1 cell array where hn{i} contains info for model i
% hn contains:
%   hn{:}.xs "features"
%   hn{:}.bbs "bounding boxes"

% Tomasz Malisiewicz (tomasz@cmu.edu)

if ~exist('mining_params','var')
  mining_params = get_default_mining_params;
end

number_of_violating_images = 0;
number_of_windows = zeros(length(models),1);

violating_images = zeros(0,1);
empty_images = zeros(0,1);

mining_params.SAVE_SVS = 1;

numpassed = 0;

for i = 1:length(models)
  if ~isfield(models{i},'total_mines')
    models{i}.total_mines = 0;
  end
end

for i = 1:length(mining_queue)
  index = mining_queue{i}.index;
  I = convert_to_I(bg{index});

  %HACK ROTATE UPSIDE DOWN
  %fprintf(1,'HACK: rotate upside down negatives\n');
  %I = imrotate(I,180);

  %starter = tic;
  % if isfield(mining_params,'wtype') && strcmp(mining_params.wtype, ...
  %                                              'dfun')

  %   [rs,t] = localizemeHOG_dfun(I, models, mining_params);
  
  %   %plot(rs.bbs{1}(:,end))
  %   %keyboard
  % else
  [rs,t] = esvm_detect(I, models, mining_params);

  if isfield(models{1}.mining_params,'SOFT_NEGATIVE_MINING') && ...
        (models{1}.mining_params.SOFT_NEGATIVE_MINING==1)
    for j=1:length(rs.bbs)
      if size(rs.bbs{j},1) > 0
        top_det = rs.bbs{j}(1,:);
        os = getosmatrix_bb(rs.bbs{j},top_det);
        goods = find(os<models{j}.mining_params.SOFT_NEGATIVE_MINING_OS);
        rs.bbs{j} = rs.bbs{j}(goods,:);
        rs.xs{j} = rs.xs{j}(goods);
      end
    end
  end

  numpassed = numpassed + 1;

  curid_integer = index;  
  for q = 1:length(rs.bbs)
    if ~isempty(rs.bbs{q})
      rs.bbs{q}(:,11) = curid_integer;
    end
  end
 
  %% Make sure we only keep 3 times the number of violating windows
  clear scores
  scores{1} = [];
  for q = 1:length(models)
    if ~isempty(rs.bbs{q})
      s = rs.bbs{q}(:,end);
      nviol = sum(s >= -1);
      [aa,bb] = sort(s,'descend');
      bb = bb(1:min(length(bb),...
                    ceil(nviol*mining_params.beyond_nsv_multiplier)));
      

      rs.xs{q} = rs.xs{q}(bb);    
      scores{q} = cat(2,s);
    end
  end
  
  addon ='';
  supersize = sum(cellfun(@(x)length(x),scores));
  if supersize > 0
    addon=sprintf(', max = %.3f',max(cellfun(@(x)max_or_this(x,-1000),scores)));
  end
  total = sum(cellfun(@(x)x.num_visited,mining_queue));
  fprintf(1,'Found %d/%d windows, image:%05d (#seen=%d/%05d%s)\n',...
          supersize, sum(cellfun(@(x)sum(x>=-1),scores)), index, ...
          length(bg)-length(mining_queue)+i, length(bg), addon);

  %increment how many times we processed this image
  mining_queue{i}.num_visited = mining_queue{i}.num_visited + 1;

  number_of_windows = number_of_windows + cellfun(@(x)length(x),scores)';
  
  clear curxs curbbs
  for q = 1:length(models)
    curxs{q} = [];
    curbbs{q} = [];
    if isempty(rs.xs{q})
      continue
    end
    
    goods = cellfun(@(x)prod(size(x)),rs.xs{q})>0;
    curxs{q} = cat(2,curxs{q},rs.xs{q}{goods});
    curbbs{q} = cat(1,curbbs{q},rs.bbs{q}(goods,:));
  end
  
  Ndets = cellfun(@(x)size(x,2),curxs);
  %if no detections, just skip image because there is nothing to store
  if sum(Ndets) == 0
    empty_images(end+1) = index;
    %continue
  end
  
  %an image is violating if it contains some violating windows,
  %else it is an empty image
  if max(cellfun(@(x)max_or_this(x,-1000),scores))>=-1
    if (mining_queue{i}.num_visited==1)
      number_of_violating_images = number_of_violating_images + 1;
    end 
    violating_images(end+1) = index;
  end
  
  for a = 1:length(models)
    xs{a}{i} = curxs{a};
    bbs{a}{i} = curbbs{a};
  end

  
  if (numpassed + models{1}.total_mines >= ...
      mining_params.MAX_TOTAL_MINED_IMAGES) || ...
        (max(number_of_windows) >= mining_params.MAX_WINDOWS_BEFORE_SVM) || ...
        (numpassed >= mining_params.MAX_IMAGES_BEFORE_SVM)
    fprintf(1,['Stopping mining because we have %d windows from' ...
                                                ' %d new violators\n'],...
            max(number_of_windows), number_of_violating_images);
    break;
  end
end

if ~exist('xs','var')
  %If no detections from from anymodels, return an empty matrix
  for i = 1:length(models)
    hn.xs{i} = zeros(prod(size(models{i}.model.w)),0);
    hn.bbs{i} = [];
  end
  mining_stats.num_violating = 0;
  mining_stats.num_empty = 0;
  
  return;
end

hn.xs = xs;
hn.bbs = bbs;

fprintf(1,'# Violating images: %d, #Non-violating images: %d\n', ...
        length(violating_images), length(empty_images));

mining_stats.num_empty = length(empty_images);
mining_stats.num_violating = length(violating_images);
mining_stats.total_mines = mining_stats.num_violating + mining_stats.num_empty;

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

mining_queue(mover_ids) = [];

%% We now take the empty images and remove them from queue
mover_ids = find(cellfun(@(x)ismember(x.index,empty_images), ...
                         mining_queue));

mining_queue(mover_ids) = [];

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
