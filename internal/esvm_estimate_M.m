function M = esvm_estimate_M(b, model)
% Given a bunch of detections, learn the M boosting matrix, which
% makes a final boxes's score depend on the co-occurrence of certain
% "friendly" detections
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

neighbor_thresh = model.params.calibration_matrix_neighbor_thresh;
count_thresh    = model.params.calibration_matrix_count_thresh;

%REMOVE FIRINGS ON SELF-IMAGE (these create artificially high scores)
REMOVE_SELF = 1;

cls = model.cls;

N = length(model.data_set);

maxos = cell(1,N);

if REMOVE_SELF == 0
  fprintf(1,'Warning: Not removing self-hits\n');
end

fprintf(1,' -Computing Box Features:');
starter=tic;
for i = 1:N
  curid = i;
  hits = find(b(:,11) == i);
  boxes{i} = b(hits,:);
  
  if size(boxes{i},1) == 0
    maxos{i} = [];
    continue
  else
    hits = ismember({model.data_set{i}.objects.class},model.cls);
    if length(hits) == 0
      maxos{i} = zeros(size(boxes{i},1),1);
    else
      maxos{i} = max(getosmatrix_bb(boxes{i}(:,1:4),cat(1,model.data_set{i}.objects.bbox)),[],2);
    end
  end
  

  %old-method: use raw SVM scores + 1 (works better!)  NOTE: this
  %works better than using calibrated scores (doesn't work too well)
  calib_boxes = boxes{i};
  calib_boxes(:,end) = calib_boxes(:,end)+1;
  boxes{i} = calib_boxes;
  %Threshold at the target value specified in parameters

  if 0 %REMOVE_SELF == 1
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


lens = cellfun(@(x)size(x,1),boxes);
boxes(lens==0) = [];
maxos(lens==0) = [];

K = length(model.models);
N = sum(cellfun(@(x)size(x,2),maxos));

y = cat(1,maxos{:});
os = cat(1,maxos{:})';


scores = cellfun2(@(x)x(:,end)',boxes);
scores = [scores{:}];

xraw = cell(length(boxes),1);
allboxes = cat(1,boxes{:});

for i = 1:length(boxes)
  fprintf(1,'.');
  xraw{i} = esvm_get_M_features(boxes{i}, K, neighbor_thresh);
end
x = [xraw{:}];

exids = allboxes(:,6);
exids(allboxes(:,7)==1)= exids(allboxes(:,7)==1) + length(model.models);
imids = allboxes(:,5);
fprintf(1,'took %.3fsec\n',toc(starter));


fprintf(1,' -Learning M by counting: ');
starter=tic;

%This one works best so far

M = learn_M_counting(x, exids, os, count_thresh);
fprintf(1,'took %.3fsec\n',toc(starter));

M.neighbor_thresh = neighbor_thresh;
M.count_thresh = count_thresh;

r = cell(length(xraw),1);
fprintf(1,' -Applying M to %d images: ',length(xraw));
starter=tic;
for j = 1:length(xraw)
  r{j} = esvm_apply_M(xraw{j},boxes{j},M);
end

r = [r{:}];
[aa,bb] = sort(r,'descend');
goods = os>.5;

res = (cumsum(goods(bb))./(1:length(bb)));
M.score = mean(res);
fprintf(1,'took %.3fsec\n',toc(starter));

if model.params.display == 1
  figure(4)
  subplot(1,2,1)
  plot(scores,os,'r.','MarkerSize',12)
  xlabel('Detection Score')
  ylabel('OS wrt gt')
  title('w/o calibration')
  
  subplot(1,2,2)
  plot(r,os,'r.','MarkerSize',12)
  xlabel('Detection Score')
  ylabel('OS wrt gt')
  title('w/ M-matrix')
  drawnow
  snapnow
  
  figure(5)
  clf
  [aa,bb] = sort(scores,'descend');
  plot(cumsum(os(bb)>.5)./(1:length(os)),'r-','LineWidth',3)
  hold on;
  [aa,bb] = sort(r,'descend');
  plot(cumsum(os(bb)>.5)./(1:length(os)),'b--','LineWidth',3)
  xlabel('#instances Recalled')
  ylabel('Precision')
  title('M-matrix estimation Precision-Recall');
  legend('no matrix','matrix')
  drawnow
  snapnow
end


function M = learn_M_counting(x, exids, os, count_thresh)
%Learn the matrix by counting activations on positives

N = size(x,2);
K = size(x,1);
C = zeros(K,K);

for i = 1:N
  cur = find(x(:,i)>0);  

  %old way: works better!
  C(cur,exids(i)) = C(cur,exids(i)) + os(i)*(os(i) >= count_thresh) / ...
       length(cur);
end

for i = 1:K
  M.w{i} = C(:,i);
  M.b{i} = 0;
end

M.C = sparse(C);
