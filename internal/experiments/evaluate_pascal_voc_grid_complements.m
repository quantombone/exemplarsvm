function [results,final] = ...
    evaluate_pascal_voc_grid_complements(models,grid,target_directory,M)
%% Evaluate PASCAL VOC detection task with the models, their output
%% firings grid, on the set target_directory which can be either
%% 'trainval' or 'test'

% targets = [4 5];
% for i = 1:length(grid)
%   goods = find(ismember(grid{i}.bboxes(:,6),targets));
%   grid{i}.coarse_boxes = grid{i}.coarse_boxes(goods,:);
%   grid{i}.bboxes = grid{i}.bboxes(goods,:);
%   grid{i}.extras.os = grid{i}.extras.os(goods,:);
% end

VOCinit;
VOCopts.testset = target_directory;

% resfile = sprintf('%s/%s.%s_%s_results.mat',VOCopts.resdir,...
%                   models{1}.models_name,...
%                   models{1}.cls,target_directory');

% if fileexists(resfile)
%   %fprintf(1,'Pre loading %s\n',resfile);
%   %load(resfile);
%   %return;
% end


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
  
  if length(grid{i}.extras)>0 && isfield(grid{i}.extras,'os')
    grid{i}.extras.os(:,end+1) = 0;
    grid{i}.extras.cats{end+1} = models{1}.cls;
    curhits = find(ismember(grid{i}.extras.cats,{models{1}.cls}));
    maxos{i} = max(grid{i}.extras.os(:,curhits),[],2)';
  end

  if REMOVE_SELF == 1
    %% remove self from this detection image!!! LOO stuff!
    badex = find(ismember(excurids,curid));
    badones = ismember(bboxes{i}(:,6),badex);
    bboxes{i}(badones,:) = [];
    if length(grid{i}.extras)>0 && isfield(grid{i}.extras,'os')
      if length(maxos{i})>0
        maxos{i}(badones) = [];
      end
    end
  end
end

raw_boxes = bboxes;

if 1
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

if exist('M','var') && length(M)>0
  fprintf(1,'Applying M\n');
  tic
  nbrlist = cell(1,length(bboxes));
  for i = 1:length(bboxes)
    [xraw,nbrlist{i}] = get_box_features(bboxes{i},length(models),M.neighbor_thresh);
    r2 = esvm_apply_M(xraw,bboxes{i},M);
    bboxes{i}(:,end) = r2;
  end
  toc
end

if 0
  fprint(1,'Propagating scores onto raw detections\n');
  %% propagate scores onto raw boxes
  for i = 1:length(bboxes)
    calib_boxes = calibrate_boxes(unclipped_boxes{i},M.betas);
    raw_scores = calib_boxes(:,end);
    
    new_scores = raw_scores;
    for j = 1:length(nbrlist{i})
      new_scores(nbrlist{i}{j}) = max(new_scores(nbrlist{i}{j}),...
                                      raw_scores(nbrlist{i}{j}).*...
                                      bboxes{i}(nbrlist{i}{j},end));
    end
    bboxes{i}(:,end) = new_scores;
  end
end

pre_nms_boxes = bboxes;

fprintf(1,'applying NMS\n');
for i = 1:length(bboxes)
  if size(bboxes{i},1) > 0
    bboxes{i}(:,5) = 1:size(bboxes{i},1);
    bboxes{i} = nms(bboxes{i},.5);
    if length(grid{i}.extras)>0 && isfield(grid{i}.extras,'os')
      maxos{i} = maxos{i}(bboxes{i}(:,5));
    end
    nbrlist{i} = nbrlist{i}(bboxes{i}(:,5));
    bboxes{i}(:,5) = 1:size(bboxes{i},1);
  end
end

%% clip boxes to image dimensions
if 1
  fprintf(1,'clipping boxes\n');
  unclipped_boxes = bboxes;
  for i = 1:length(bboxes)
    bboxes{i} = clip_to_image(bboxes{i},grid{i}.imbb);
  end
end

%if isfield(M,'betas')
%  fprintf(1,'Applying betas\n');
%  pre_nms_boxes = cellfun2(@(x)calibrate_boxes(x,M.betas),pre_nms_boxes);
%end

final_boxes = bboxes;
final_maxos = maxos;
 
%% return unclipped boxes for transfers
final.final_boxes = final_boxes;
final.final_maxos = final_maxos;
final.unclipped_boxes = unclipped_boxes;
final.pre_nms_boxes = pre_nms_boxes;
final.raw_boxes = raw_boxes;
final.nbrlist = nbrlist;
final.M = M;

%%TODO: create the directory programatically instead of doing it manually
filer=sprintf(VOCopts.detrespath,'comp3',cls);
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
try
  [recall,prec,ap,apold] = VOCevaldet(VOCopts,'comp3',cls,true);
catch
  recall = [];
  prec = [];
  ap = -1;
  apold = -1;
end
axis([0 1 0 1]);
extra = '';
if ismember(models{1}.models_name,{'dalal'})
  extra='-dalal';
end
%filer = sprintf(['/nfs/baikal/tmalisie/labelme400/www/voc/tophits/' ...
%                 '%s%s-on-%s-ap=%.5f.png'],models{1}.cls,extra,target_directory,ap);
%set(gcf,'PaperPosition',[0 0 5 5])
%print(gcf,'-dpng',filer);
%fprintf(1,'Just Wrote %s\n',filer);

results.recall = recall;
results.prec = prec;
results.ap = ap;
results.apold = apold;
results.cls = models{1}.cls;

%fprintf(1,'Saving results to %s\n',resfile);
%save(resfile,'results','final','M');
