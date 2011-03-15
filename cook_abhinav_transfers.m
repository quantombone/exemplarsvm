function [results] = cook_abhinav_transfers(models,grid,target_directory)
%% Evaluate PASCAL VOC detection task with the models, their output
%% firings grid, on the set target_directory which can be either
%% 'trainval' or 'test'

VOCinit;
VOCopts.testset = target_directory;

[cur_set, gt] = textread(sprintf(VOCopts.clsimgsetpath,...
                                 models{1}.cls,...
                                 target_directory),['%s' ...
                    ' %d']);
cur_set = cur_set(gt==1);

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

for i = 25%1:length(grid)
  boxes = grid{i}.bboxes;
  
  boxes(:,5) = 1:size(boxes,1);
  
  [aa,bb] = sort(boxes(:,end),'descend');
  boxes = boxes(bb,:);
  boxes = nms_within_exemplars(boxes,.5);
  
  %% ONLY SHOW TOP 5 detections or fewer
  boxes = boxes(1:min(size(boxes,1),5),:);
  cboxes = grid{i}.coarse_boxes(boxes(:,5),:);

  %figure(1)
  %clf
  
  Itest = im2double(imread(sprintf(VOCopts.imgpath, ...
                                   grid{i}.curid)));
  
  for j = 1:size(boxes,1)
    Itrain = im2double(imread(sprintf(VOCopts.imgpath, ...
                                      models{boxes(j,6)}.curid)));
    d = cboxes(j,1:4);
    c = models{boxes(j,6)}.model.coarse_box;
    g2 = boxes(j,1:4);
    g = models{boxes(j,6)}.gt_box;
    filer = sprintf('/nfs/baikal/tmalisie/trainxfers/transfers.%05d.%05d.mat',...
                    i,j);
    save(filer,'Itest','Itrain','d','c','g','g2');    
  end
  
  %show_hits_figure(models,boxes,I);
  %keyboard
end

keyboard

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

fprintf(1,'Applying betas\n');
if exist('betas','var')
  bboxes = cellfun2(@(x)calibrate_boxes(x,betas),bboxes);
end

keyboard
[bboxes] = mmht_scores(bboxes, maxos, models);
fprintf(1,'Applying NMS\n');
bboxes = cellfun2(@(x)nms(x,.5),bboxes);

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
[recall,prec,ap,apold] = VOCevaldet(VOCopts,'comp3',cls,true);
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

resfile = sprintf('%s/%s_%s_results.mat',VOCopts.resdir,...
                  models{1}.cls,target_directory');

fprintf(1,'Saving results to %s\n',resfile);
save(resfile,'results');
