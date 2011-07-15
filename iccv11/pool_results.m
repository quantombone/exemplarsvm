function final = pool_results(dataset_params, models, grid, M)
%% Perform detection box post-processing and pool detection boxes
%(which will then be ready to go into the PASCAL evaluation code)

%REMOVE FIRINGS ON SELF-IMAGE (these create artificially high scores)
REMOVE_SELF = 0;

if REMOVE_SELF == 1
  curids = cellfun2(@(x)x.curid,models);
end

cls = models{1}.cls;

excurids = cellfun2(@(x)x.curid,models);
bboxes = cell(1,length(grid));
maxos = cell(1,length(grid));

fprintf(1,'Loading bboxes\n');
curcls = find(ismember(dataset_params.classes,models{1}.cls));

for i = 1:length(grid)  
  curid = grid{i}.curid;
  bboxes{i} = grid{i}.bboxes;
  if size(bboxes{i},1) == 0
    continue
  end
  
  if length(grid{i}.extras)>0 && isfield(grid{i}.extras,'maxos')
    maxos{i} = grid{i}.extras.maxos;
    maxos{i}(grid{i}.extras.maxclass~=curcls) = 0;
  end
  
  if REMOVE_SELF == 1
    %% remove self from this detection image!!! LOO stuff!
    exes = bboxes{i}(:,6);
    excurids = curids(exes);
    badex = find(ismember(excurids,{curid}));
    bboxes{i}(badex,:) = [];
    
    if length(grid{i}.extras)>0 && isfield(grid{i}.extras,'maxos')
      if length(maxos{i})>0
        maxos{i}(badex) = [];
      end
    end
  end
end

%raw_boxes = bboxes;

%%%NOTE: the LRs haven't been consolidated
%%NOTE: seems better turned off
if 0 %turned off for nn baseline, since it is done already
  fprintf(1,'applying exemplar nms\n');
  for i = 1:length(bboxes)
    if size(bboxes{i},1) > 0
      bboxes{i}(:,5) = 1:size(bboxes{i},1);
      bboxes{i} = nms_within_exemplars(bboxes{i},.5);
      if length(grid{i}.extras)>0 && isfield(grid{i}.extras,'os')
        maxos{i} = maxos{i}(bboxes{i}(:,5));
      end
    end
  end
end

if exist('M','var') && length(M)>0 && isfield(M,'betas')
  fprintf(1,'Propagating scores onto raw detections\n');
  %% propagate scores onto raw boxes
  for i = 1:length(bboxes)
    %HACK: turn off calibration here
    %ob{i} = bboxes{i};

    if isfield(M,'neighbor_thresh')
      calib_boxes = bboxes{i};
      calib_boxes(:,end) = calib_boxes(:,end)+1;
    else
      calib_boxes = calibrate_boxes(bboxes{i},M.betas); 
    end
    oks = find(calib_boxes(:,end) > dataset_params.params.calibration_threshold);
    calib_boxes = calib_boxes(oks,:);
    bboxes{i} = calib_boxes;
  end
end

if exist('M','var') && length(M)>0 && isfield(M,'neighbor_thresh')
  fprintf(1,'Applying M\n');
  tic
  for i = 1:length(bboxes)
    fprintf(1,'.');
    [xraw,nbrs] = get_box_features(bboxes{i},length(models),M.neighbor_thresh);
    r2 = apply_boost_M(xraw,bboxes{i},M);
    bboxes{i}(:,end) = r2;
    %bboxes{i}(:,end) = bboxes{i}(:,end).*(r2');
  end
  toc
end

fprintf(1,'applying competitive NMS\n');
for i = 1:length(bboxes)
  if size(bboxes{i},1) > 0
    bboxes{i}(:,5) = 1:size(bboxes{i},1);
    bboxes{i} = nms(bboxes{i},.5);
    if length(grid{i}.extras)>0 && isfield(grid{i}.extras,'maxos')
      maxos{i} = maxos{i}(bboxes{i}(:,5));
    end
    bboxes{i}(:,5) = 1:size(bboxes{i},1);
  end
end

%% clip boxes to image dimensions
unclipped_boxes = bboxes;
for i = 1:length(bboxes)
  bboxes{i} = clip_to_image(bboxes{i},grid{i}.imbb);
end

final_boxes = bboxes;

%% return unclipped boxes for transfers
final.final_boxes = final_boxes;
final.final_maxos = maxos;
final.unclipped_boxes = unclipped_boxes;

calib_string = '';
if exist('M','var') && length(M)>0 && isfield(M,'betas')
   calib_string = '-calibrated';
end

if exist('M','var') && length(M)>0 && isfield(M,'betas') && isfield(M,'w')
  calib_string = [calib_string '-M'];
end

final.calib_string = calib_string;
final.imbb = cellfun2(@(x)x.imbb,grid);
