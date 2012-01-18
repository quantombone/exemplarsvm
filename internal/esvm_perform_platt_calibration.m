function [betas] = esvm_perform_platt_calibration(grid, models, ...
                                                  params, CACHE_FILES)
% Perform calibration by learning the sigmoid parameters (linear
% transformation of svm scores) for each model independently. If we
% perform an operation such as NMS, we will now have "comparable"
% scores.  This is performed on the 'trainval' set for PASCAL VOC.
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

if length(grid) == 0 || length(models) == 0
  betas = [];
  ALL_bboxes = [];
  return;
end

if ~exist('CACHE_FILES','var')
  CACHE_FILES = 0;
end

%if enabled, do NMS, if disabled return raw detections
DO_NMS = 0;

% if enabled, display images
display = params.dataset_params.display;

% if display is enabled and dump_images is enabled, then dump images
% into DUMPDIR
dump_images = 0;

models_name = '';
if length(models)>=1 && isfield(models{1},'models_name') && ...
      isstr(models{1}.models_name)
  models_name = models{1}.models_name;
end


DUMPDIR = sprintf('%s/www/calib/',params.dataset_params.localdir, ...
                  models_name);

if dump_images==1 && ~exist(DUMPDIR,'dir')
  mkdir(DUMPDIR);
end
% show A NxN grid of top detections (if display is turned on)
SHOW_TOP_N_SVS = 10;

if nargin < 1
  fprintf(1,'Not enough arguments, need at least the grid\n');
  return;
end

final_dir = ...
    sprintf('%s/models',params.dataset_params.localdir);

if CACHE_FILES == 1 && ~exist(final_dir','dir')
  mkdir(final_dir);
end

final_file = ...
    sprintf('%s/%s-betas.mat',...
            final_dir, ...
            models_name);

if CACHE_FILES == 1 
  lockfile = [final_file '.lock'];
  if fileexists(final_file) || (mymkdir_dist(lockfile)==0)
    
    %wait until lockfiles are gone
    wait_until_all_present({lockfile},5,1);
    fprintf(1,'Loading final file %s\n',final_file);
    res = load_keep_trying(final_file);
    betas = res.betas;
    return;
  end
end

for i = 1:length(models)
  if ~isfield(models{i},'curid')
    models{i}.curid = '-1';
  end
end

model_ids = cellfun2(@(x)x.curid,models);
targets = 1:length(models);

cls = models{1}.cls;

if isfield(params.dataset_params,'classes')
  targetc = find(ismember(params.dataset_params.classes, ...
                          models{1}.cls));
else
  targetc = models{1}.cls;
end


%fprintf(1,'Preparing boxes for calibration\n');
for i = 1:length(grid)    
  if mod(i,100)==0
    fprintf(1,'.');
  end
  cur = grid{i};
  
  %do not process grids with no bboxes
  if size(cur.bboxes,1) == 0
    continue;
  end
  
  if size(cur.bboxes,1) >= 1
    cur.bboxes(:,5) = 1:size(cur.bboxes,1);    
    cur.coarse_boxes(:,5) = 1:size(cur.bboxes,1);    
    if DO_NMS == 1
      cur.bboxes = esvm_nms_within_exemplars(cur.bboxes,.5);
      cur.coarse_boxes = cur.coarse_boxes(cur.bboxes(:,5),:);
    end
    
    if length(cur.extras)>0
      cur.extras.os = cur.extras.maxos(cur.bboxes(:,5));
      
      cur.extras.os = cur.extras.os.* ...
          reshape(double(ismember(cur.extras.maxclass,targetc)),...
                  size(cur.extras.os));
      
      % cur.extras.os = cur.extras.os.* ...
      %     reshape((cur.extras.maxclass(cur.bboxes(:,5)) == ...
      %              targetc),size(cur.extras.os));

    end
  end
  
  cur.bboxes(:,5) = grid{i}.index;
  cur.coarse_boxes(:,5) = grid{i}.index;
  
  coarse_boxes{i} = cur.coarse_boxes;
  bboxes{i} = cur.bboxes;
   
  %if we have overlaps, collect them
  if length(cur.extras) > 0
    
    %use all objects as ground truth
    %goods = 1:length(cur.extras.cats);
    
    %% find the ground truth examples of the right category
    %goods = find(ismember(cur.extras.cats,cls));
    
    exids = cur.bboxes(:,6);
   
    %if length(goods) == 0
    %  os{i} = zeros(size(bboxes{i},1),1);
    %else
    %  curos = cur.extras.os(:,goods);
    os{i} = cur.extras.maxos; %max(curos,[],2);
    %end    
  else
    os{i} = zeros(size(bboxes{i},1),1);    
  end
  
  scores{i} = cur.bboxes(:,7)';
end
  
ALL_bboxes = cat(1,bboxes{:});
ALL_coarse_boxes = cat(1,coarse_boxes{:});
ALL_os = cat(1,os{:});

curids = cellfun2(@(x)x.curid,grid);
% Pre-processing models for calibration

for exid = 1:length(models)
  fprintf(1,'.');
  sourcegrid = find(ismember(curids,models{exid}.curid));
  if length(sourcegrid) == 0
    sourcegrid = -1;
  end
  
  hits = find((ALL_bboxes(:,6)==exid));
  all_scores = ALL_bboxes(hits,end);
  all_os = ALL_os(hits,:);
  
  good_scores = all_scores(all_os>=.5);
  good_os = all_os(all_os>=.5);
  
  bad_scores = all_scores(all_os<.5);
  bad_os = all_os(all_os<.5);

  %add virtual sample at os=1.0, score=1.0
  %good_os = cat(1,good_os,1.0);
  %good_scores = cat(1,good_scores,1.0);

  if length(good_os) <= 1 || (length(bad_os) ==0)
    beta = [.1 100];
  else

    [aa,bb] = sort(bad_scores,'descend');
    curlen = min(length(bb),10000*length(good_scores));
    bb = bb(round(linspace(1,length(bb),curlen)));

    bad_scores = bad_scores(bb);
    bad_os = bad_os(bb);
    all_scores = [good_scores; bad_scores];
    all_os = [good_os; bad_os];
    
    [aaa,bbb] = sort(all_scores, 'descend');
    %bbb = bbb(1:min(1000,length(bbb)));
    %all_scores = all_scores(bbb);
    %all_os = all_os(bbb);
    beta = esvm_learn_sigmoid(all_scores, all_os);
  end

  if beta(1)<.001
    fprintf(1,['warning[esvm_perform_platt_calibration.m]: beta(1)' ...
               ' is low']);
    % beta(1) = .001;
  end

  betas(exid,:) = beta;

  if (sum(ismember(exid,targets))==0)
    continue
  end

  if display == 1
    
    figure(1)
    clf
    subplot(1,2,1)  
    plot(all_scores,all_os,'r.')
    xs = linspace(min(all_scores),max(all_scores),1000);
    fx = @(x)(1./(1+exp(-beta(1)*(x-beta(2)))));
    
    hold on
    plot(xs,fx(xs),'b','LineWidth',2)
    axis([min(xs) max(xs) 0 1])
    xlabel('SVM score')
    ylabel(sprintf('Max Overlap Score with %s',models{exid}.cls))
    
    title(sprintf('Learned Sigmoid \\beta=[%.3f %.3f]',beta(1), ...
                  beta(2)))
    subplot(1,2,2)
    Iex = convert_to_I(models{exid}.I);
    imagesc(Iex)
    plot_bbox(models{exid}.gt_box)
    axis image
    axis off

    title(sprintf('Topdets Calibration Ex %s.%d.%s',...
                  models{exid}.curid,...
                  models{exid}.objectid, ...
                  models{exid}.cls))
    drawnow
    snapnow
    
    bbs=ALL_coarse_boxes(hits,:);
    bbs_os = ALL_os(hits,:);
    [aa,bb] = sort(bbs(:,end),'descend');
    bbs_show = bbs(bb,:);

    %models{exid}.model.svids = {};
    %m = try_reshape(models{exid},bbs_show,100);

    %[models{exid}.model.svids,models{exid}.model.nsv] = ...
    %    esvm_reconstruct_features(bbs_show,100);%,'trainval','');

    models{exid}.model.svbbs = bbs_show;
    m2 = models(exid);
    if isfield(params,'val_set')
      m2{1}.train_set = params.val_set;
      m2{1}.model.svbbs(:,6) = 1;
      m2{1}.model.svxs = [];
      figure(445)
      clf
      imagesc(esvm_show_det_stack(m2{1},8))
      drawnow
      title(sprintf('Calib Ex %s.%d.%s',...
                    models{exid}.curid,...
                    models{exid}.objectid, ...
                    models{exid}.cls))
      drawnow
      snapnow
    end
  end
  
  if (display == 0)
    continue
  end
    
  if dump_images == 1
    figure(2)
    filer = sprintf('%s/result.%d.%s.%s.png', DUMPDIR, ...
                    exid,models{exid}.cls,models{exid}.models_name);
    set(gcf,'PaperPosition',[0 0 20 20]);
    print(gcf,filer,'-dpng');
    
  else
    pause(.01)
  end  
end

if CACHE_FILES == 1
  fprintf(1,['\nLoaded calibration parameters "betas", saving to' ...
             ' %s\n'],final_file);
  save(final_file,'betas');
  rmdir(lockfile);
end
