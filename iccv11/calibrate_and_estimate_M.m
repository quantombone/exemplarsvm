function M = calibrate_and_estimate_M(dataset_params, models, grid)
%% Learn a combination matrix M which multiplexes the detection
%% results by compiling co-occurrence statistics on true positives
%% Tomasz Malisiewicz (tomasz@cmu.edu)

if ~exist('M','var')
  betas = perform_calibration(models, grid, dataset_params);
  M.betas = betas;
end

% target_directory = 'trainval';
% %% prune grid to contain only images from target_directory
% [cur_set, gt] = textread(sprintf(VOCopts.imgsetpath,target_directory),['%s' ...
%                     ' %d']);

% gridids = cellfun2(@(x)x.curid,grid);
% goods = ismember(gridids,cur_set);
% grid = grid(goods);

if length(grid) == 0
  error(sprintf('Found no images of type %s\n',results_directory))
end

%REMOVE FIRINGS ON SELF-IMAGE (these create artificially high scores)
REMOVE_SELF = 1;

cls = models{1}.cls;

for i = 1:length(models)
  if ~isfield(models{i},'curid')
    models{i}.curid = '-1';
  end
end

excurids = cellfun2(@(x)x.curid,models);
bboxes = cell(1,length(grid));
maxos = cell(1,length(grid));

if REMOVE_SELF == 0
  fprintf(1,'Warning: Not removing self-hits\n');
end
curcls = find(ismember(dataset_params.classes,models{1}.cls));

fprintf(1,'Loading bboxes\n');
tic
for i = 1:length(grid)
  
  curid = grid{i}.curid;
  bboxes{i} = grid{i}.bboxes;
  
  calib_boxes = calibrate_boxes(bboxes{i},M.betas);
  %Threshold at .1
  oks = find(calib_boxes(:,end)>.1);
  bboxes{i} = calib_boxes(oks,:);
  if length(grid{i}.extras)>0
    %grid{i}.extras.os(:,end+1) = 0;
    %grid{i}.extras.cats{end+1} = models{1}.cls;
    %curhits = find(ismember(grid{i}.extras.cats,{models{1}.cls}));
    %maxos{i} = max(grid{i}.extras.os(:,curhits),[],2)';
    
    maxos{i} = grid{i}.extras.maxos;
    maxos{i}(grid{i}.extras.maxclass~=curcls) = 0;
    maxos{i} = maxos{i}(oks);
  end

  if REMOVE_SELF == 1
    %% remove self from this detection image!!! LOO stuff!
    %fprintf(1,'hack not removing self!\n');
    badex = find(ismember(excurids,curid));
    badones = ismember(bboxes{i}(:,6),badex);
    bboxes{i}(badones,:) = [];
    if length(maxos{i})>0
      maxos{i}(badones) = [];
    end
  end
end
toc

if 0
%% clip boxes to image
fprintf(1,'clipping boxes\n');
for i = 1:length(bboxes)
  bboxes{i} = clip_to_image(bboxes{i},grid{i}.imbb);
end
end

lens = cellfun(@(x)size(x,1),bboxes);
bboxes(lens==0) = [];
maxos(lens==0) = [];

%fprintf(1,'Applyio.mng betas\n');
%bboxes = cellfun2(@(x)calibrate_boxes(x,betas),bboxes);

nthresh = 0.5;
cthresh = 0.5;

betas = M.betas;

[M] = mmht_scores(bboxes, maxos, models, nthresh, ...
                  cthresh);
M.betas = betas;
