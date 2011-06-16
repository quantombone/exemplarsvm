function [results,final] = ...
    evaluate_pascal_voc_grid(VOCopts,models,grid,target_directory,M,CACHE_FILE)
%% Evaluate PASCAL VOC detection task with the models, their output
%% firings grid, on the set target_directory which can be either
%% 'trainval' or 'test'

%In case we want to evaluate a subset of detectors
% targets = [4 5];
% for i = 1:length(grid)
%   goods = find(ismember(grid{i}.bboxes(:,6),targets));
%   grid{i}.coarse_boxes = grid{i}.coarse_boxes(goods,:);
%   grid{i}.bboxes = grid{i}.bboxes(goods,:);
%   grid{i}.extras.os = grid{i}.extras.os(goods,:);
% end

if ~exist('CACHE_FILE','var')
  CACHE_FILE = 0;
end
VOCopts.testset = target_directory;

calib_string = '';
if exist('M','var') && length(M)>0 && isfield(M,'betas')
   calib_string = '-calibrated';
end

resfile = sprintf('%s/%s.%s%s_%s_results.mat',VOCopts.resdir,...
                  models{1}.models_name,...
                  models{1}.cls,calib_string,...
                  target_directory');


if CACHE_FILE && fileexists(resfile) 
  fprintf(1,'Pre loading %s\n',resfile);
  load(resfile);
  return;
end

%% prune grid to contain only images from target_directory
[cur_set, gt] = textread(sprintf(VOCopts.imgsetpath,target_directory),['%s' ...
                    ' %d']);

if 0
  fprintf(1,'HACK: choosing only the subset which contains true positives\n');
  %% prune grid to contain only images from target_directory
  [cur_set, gt] = textread(sprintf(VOCopts.clsimgsetpath,models{1}.cls,target_directory),['%s' ...
                    ' %d']);
  cur_set = cur_set(gt==1);
end

gridids = cellfun2(@(x)x.curid,grid);
goods = ismember(gridids,cur_set);
grid = grid(goods);

%REMOVE FIRINGS ON SELF-IMAGE (these create artificially high scores)
REMOVE_SELF = 1;

cls = models{1}.cls;

excurids = cellfun2(@(x)x.curid,models);
bboxes = cell(1,length(grid));
maxos = cell(1,length(grid));

if REMOVE_SELF == 0
  fprintf(1,'Warning: Not removing self-hits\n');
end

fprintf(1,'Loading bboxes\n');
curcls = find(ismember(VOCopts.classes,models{1}.cls));
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
    badex = find(ismember(excurids,curid));
    badones = ismember(bboxes{i}(:,6),badex);
    bboxes{i}(badones,:) = [];
    if length(grid{i}.extras)>0 && isfield(grid{i}.extras,'maxos')
      if length(maxos{i})>0
        maxos{i}(badones) = [];
      end
    end
  end
end

raw_boxes = bboxes;

%%%NOTE: the LRs haven't been consolidated
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
    calib_boxes = calibrate_boxes(bboxes{i},M.betas);

    %Threshold at .1
    oks = find(calib_boxes(:,end) > .1);
    calib_boxes = calib_boxes(oks,:);
    bboxes{i} = calib_boxes;
  end
end

if exist('M','var') && length(M)>0 && isfield(M,'neighbor_thresh')
  fprintf(1,'Applying M\n');
  tic
  for i = 1:length(bboxes)
    fprintf(1,'.');
    [xraw] = get_box_features(bboxes{i},length(models),M.neighbor_thresh);
    r2 = apply_boost_M(xraw,bboxes{i},M);
    bboxes{i}(:,end) = r2;
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
if 1
  for i = 1:length(bboxes)
    bboxes{i} = clip_to_image(bboxes{i},grid{i}.imbb);
  end
else
  fprintf(1,'NOT clipping boxes!\n');
end

final_boxes = bboxes;
final_maxos = maxos;

%% return unclipped boxes for transfers
final.final_boxes = final_boxes;
final.final_maxos = final_maxos;
final.unclipped_boxes = unclipped_boxes;
final.raw_boxes = raw_boxes;

filer = sprintf(VOCopts.detrespath,'comp3',cls);
%Create directory if it is not present
[aaa,bbb,ccc] = fileparts(filer);
if ~exist(aaa,'dir')
  mkdir(aaa);
end

fprintf(1,'Writing File %s\n',filer);
fid = fopen(filer,'w');
for i = 1:length(bboxes)
  curid = grid{i}.curid;
  for q = 1:size(bboxes{i},1)
    fprintf(fid,'%s %f %f %f %f %f\n',curid,...
            bboxes{i}(q,end), bboxes{i}(q,1:4));
  end
end
fclose(fid);

%fprintf(1,'HACK: changing OVERLAP HERE!\n');
%VOCopts.minoverlap = .4;

figure(2)
clf
[results.recall,results.prec,results.ap,results.apold,results.fp,results.tp,results.npos,results.corr] = VOCevaldet(VOCopts,'comp3',cls,true);

set(gca,'FontSize',16)
set(get(gca,'Title'),'FontSize',16)
set(get(gca,'YLabel'),'FontSize',16)
set(get(gca,'XLabel'),'FontSize',16)
axis([0 1 0 1]);

if ~exist(VOCopts.wwwdir,'dir')
  mkdir(VOCopts.wwwdir);
end

filer = sprintf(['%s/%s-%s%s-on-%s.pdf'], ...
                VOCopts.wwwdir, ...
                models{1}.cls,...
                models{1}.models_name,...
                calib_string, ...
                target_directory);
set(gcf,'PaperPosition',[0 0 8 8])
print(gcf,'-dpdf',filer);
fprintf(1,'Just Wrote %s\n',filer);

%results.recall = recall;
%results.prec = prec;
%results.ap = ap;
%results.apold = apold;
results.cls = models{1}.cls;
drawnow


%fprintf(1,'Saving results to %s\n',resfile);
%if ~exist('M','var')
%  M = [];
%end

%TODO: we are saving really large files for exemplarNN
save(resfile,'results','final','M');
