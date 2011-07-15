function M = estimate_M(dataset_params, models, grid, betas, ...
                        CACHE_FILES)
%Given a bunch of detections, learn the M boosting matrix, which
%makes the final scores multiplexed

neighbor_thresh = dataset_params.params.calibration_neighbor_thresh;
count_thresh    = dataset_params.params.calibration_count_thresh;

if ~exist('CACHE_FILES','var')
  CACHE_FILES = 0;
end

final_file = ...
    sprintf('%s/betas/%s-%s-M.mat',...
            dataset_params.localdir, models{1}.cls, models{1}.models_name);

if CACHE_FILES == 1 
  lockfile = [final_file '.lock'];
  if fileexists(final_file) || (mymkdir_dist(lockfile)==0)    
    %wait until lockfiles are gone
    wait_until_all_present({lockfile},5,1);
    fprintf(1,'Loading final file %s\n',final_file);
    res = load_keep_trying(final_file);
    M = res.M;
    return;
  end
end

if length(grid) == 0
  error(sprintf('Found no images of type %s\n',results_directory))
end

%REMOVE FIRINGS ON SELF-IMAGE (these create artificially high scores)
REMOVE_SELF = 1;

cls = models{1}.cls;

excurids = cellfun2(@(x)x.curid,models);
boxes = cell(1,length(grid));
maxos = cell(1,length(grid));

if REMOVE_SELF == 0
  fprintf(1,'Warning: Not removing self-hits\n');
end
curcls = find(ismember(dataset_params.classes,models{1}.cls));

fprintf(1,'Extracting all boxes\n');
tic
for i = 1:length(grid)
  
  curid = grid{i}.curid;
  boxes{i} = grid{i}.bboxes;

  %new-method: use calibrated scores (doesn't work too well)
  %calib_boxes = calibrate_boxes(boxes{i},betas);

  %old-method: use raw SVM scores + 1 (works better!)
  calib_boxes = boxes{i};
  calib_boxes(:,end) = calib_boxes(:,end)+1;
  
  %Threshold at the target value specified in parameters
  oks = find(calib_boxes(:,end) >= dataset_params.params.calibration_threshold);
  boxes{i} = calib_boxes(oks,:);
  if length(grid{i}.extras)>0
    maxos{i} = grid{i}.extras.maxos;
    maxos{i}(grid{i}.extras.maxclass~=curcls) = 0;
    maxos{i} = maxos{i}(oks);
  end

  if REMOVE_SELF == 1
    %% remove self from this detection image!!! LOO stuff!
    %fprintf(1,'hack not removing self!\n');
    badex = find(ismember(excurids,curid));
    badones = ismember(boxes{i}(:,6),badex);
    boxes{i}(badones,:) = [];
    if length(maxos{i})>0
      maxos{i}(badones) = [];
    end
  end
end
toc

if 0
%% clip boxes to image
fprintf(1,'clipping boxes\n');
for i = 1:length(boxes)
  boxes{i} = clip_to_image(boxes{i},grid{i}.imbb);
end
end

lens = cellfun(@(x)size(x,1),boxes);
boxes(lens==0) = [];
maxos(lens==0) = [];

%already nms-ed within exemplars (but not within LR flips)
%%NOTE: should this be turned on?
if 0
  for i = 1:length(boxes)
    boxes{i}(:,5) = 1:size(boxes{i},1);
    boxes{i} = nms_within_exemplars(boxes{i},.5);
    maxos{i} = maxos{i}(boxes{i}(:,5));
    boxes{i}(:,5) = i;
  end
end

K = length(models);
N = sum(cellfun(@(x)size(x,2),maxos));

y = cat(1,maxos{:});
os = cat(1,maxos{:})';

scores = cellfun2(@(x)x(:,end)',boxes);
scores = [scores{:}];

xraw = cell(length(boxes),1);
allboxes = cat(1,boxes{:});

for i = 1:length(boxes)
  fprintf(1,'.');
  xraw{i} = get_box_features(boxes{i}, K, neighbor_thresh);
end
x = [xraw{:}];

exids = allboxes(:,6);
exids(allboxes(:,7)==1)= exids(allboxes(:,7)==1) + length(models);
imids = allboxes(:,5);

%osmats = cellfun2(@(x)getosmatrix_bb(x,x),boxes);
%thetadiffmats = cellfun2(@(x)getaspectmatrix_bb(x,x),boxes);


fprintf(1,'learning M by counting\n');
tic
%This one works best so far
M = learn_M_counting(x, exids, os, count_thresh);
toc

M.neighbor_thresh = neighbor_thresh;
M.count_thresh = count_thresh;

r = cell(length(xraw),1);
fprintf(1,'applying boost matrix\n');
tic
for j = 1:length(xraw)
  r{j} = apply_boost_M(xraw{j},boxes{j},M);
end

r = [r{:}];
[aa,bb] = sort(r,'descend');
goods = os>.5;

res = (cumsum(goods(bb))./(1:length(bb)));
M.score = mean(res);
toc

figure(4)
subplot(1,2,1)
plot(scores,os,'r.')
xlabel('singleton scores')
ylabel('OS with gt')

subplot(1,2,2)
plot(r,os,'r.')
xlabel('combined score')
ylabel('os')

figure(5)
clf
[aa,bb] = sort(scores,'descend');
plot(cumsum(os(bb)>.5)./(1:length(os)),'r-')
hold on;
[aa,bb] = sort(r,'descend');
plot(cumsum(os(bb)>.5)./(1:length(os)),'b-')
title('Precision-Recall');
legend('singleton','combined')

if CACHE_FILES == 1
  fprintf(1,'Computed M, saving to %s\n',final_file);
  save(final_file,'M');
  rmdir(lockfile);
end

function M = learn_M_counting(x, exids, os, count_thresh)
% function M = learn_M_counting(x, exids, maxos, count_thresh, osmats, ...
%                               thetadiffmats, boxes)
%Learn the matrix by counting activations on positives
N = size(x,2);
K = size(x,1);
C = zeros(K,K);

for i = 1:N
  cur = find(x(:,i)>0);  

  %old way: works better!
  C(cur,exids(i)) = C(cur,exids(i)) + os(i)*(os(i) >= count_thresh) / ...
       length(cur);

  %new way: not so well
  %C(cur,exids(i)) = C(cur,exids(i)) + os(i)*(os(i) >= count_thresh) / ...
  %    length(cur)/sum(x(:,i);
end

for i = 1:K
  M.w{i} = C(:,i);
  M.b{i} = 0;
end

M.C = sparse(C);
