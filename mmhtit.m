function [M] = mmhtit(models,grid)
%% Learn a combination matrix M which multiplexes the detection
%% results by compiling co-occurrence statistics on true positives

%% Tomasz Malisiewicz (tomasz@cmu.edu)

% if ~exist('models','var')
%   models = load_all_models('tvmonitor','exemplars','10');
% end


% if ~exist('grid','var')
%   grid = load_result_grid(models, target_directory);
% end

VOCinit;

target_directory = 'trainval';
%% prune grid to contain only images from target_directory
[cur_set, gt] = textread(sprintf(VOCopts.imgsetpath,target_directory),['%s' ...
                    ' %d']);

gridids = cellfun2(@(x)x.curid,grid);
goods = ismember(gridids,cur_set);
grid = grid(goods);

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

fprintf(1,'Loading bboxes\n');
for i = 1:length(grid)
  
  curid = grid{i}.curid;
  bboxes{i} = grid{i}.bboxes;
  
  if length(grid{i}.extras)>0
    grid{i}.extras.os(:,end+1) = 0;
    grid{i}.extras.cats{end+1} = models{1}.cls;
    curhits = find(ismember(grid{i}.extras.cats,{models{1}.cls}));
    maxos{i} = max(grid{i}.extras.os(:,curhits),[],2)';
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


%fprintf(1,'Applying betas\n');
%bboxes = cellfun2(@(x)calibrate_boxes(x,betas),bboxes);

nthresh = 0.5;
cthresh = 0.5;
[M] = mmht_scores(bboxes, maxos, models, nthresh, ...
                  cthresh);

betas = perform_calibration(models, grid);
M.betas = betas;

% return;


% %% below is some code for doing a parameter search on validation data

% nthreshes = .2:.05:.9;
% cthreshes = .2:.05:.9;

% %myRandomize;
% %rrr = randperm(length(nthreshes));
% %nthreshes = nthreshes(rrr);

% %rrr = randperm(length(cthreshes));
% %cthreshes = cthreshes(rrr);
% [uu,vv] = meshgrid(1:length(nthreshes),1:length(cthreshes));

% vals = [uu(:) vv(:)];
% vals(:,1) = nthreshes(vals(:,1));
% vals(:,2) = cthreshes(vals(:,2));

% myRandomize;
% rrr = randperm(size(vals,1));
% vals = vals(rrr,:);

% for i = 1:size(vals,1)
  
%   nthresh = (vals(i,1));
%   cthresh = (vals(i,2));

%   startfile = sprintf('%s/M/%s-%d-%s-%f-%f',VOCopts.localdir,...
%                       models{1}.cls,length(models),target_directory,...
%                       nthresh,cthresh);
%   filerlock = sprintf('%s.lock',startfile);
%   ddd = dir([startfile '*']);
%   if length(ddd)>0 || (mymkdir_dist(filerlock) == 0)
%     continue
%   end
  
%   [M] = mmht_scores(bboxes, maxos, models, nthresh, ...
%                     cthresh);
    
%   [results,selffinal] = evaluate_pascal_voc_grid(models, ...
%                                        grid, target_directory, ...
%                                        M);
  
%   M.results = results;
%   M.selffinal = selffinal;
  
%   endfile = sprintf('%s-%.3f-%.3f.mat',startfile,results.ap, ...
%                     results.apold);
%   save(endfile,'M');
%   try
%     rmdir(filerlock);
%   catch
%     fprintf(1,'Directory %s already gone\n',filerlock);
%   end
  
% end
