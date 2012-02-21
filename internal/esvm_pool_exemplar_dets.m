function final = esvm_pool_exemplar_dets(grid, models, M, params)
% Perform detection post-processing and pool detection boxes
% (which will then be ready to go into the PASCAL evaluation code)
% If there are overlap scores associated with boxes, then they are
% also kept track of propertly, even after NMS.
% 
% If M is empty, then just NMS is performed
% If M has neighbor_thresh defined, then we apply the
% calibration-matrix
% If M has betas defined, then do platt-calibration
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

%REMOVE FIRINGS ON SELF-IMAGE (these create artificially high
%scores when evaluating on the training set, but no need to set
%this on the testing set as we don't train on testing data)
REMOVE_SELF = 0;

if REMOVE_SELF == 1
  curids = cellfun2(@(x)x.curid,models);
end

%cls = models{1}.cls;
%excurids = cellfun2(@(x)x.curid,models);
bboxes = cell(1,length(grid));
maxos = cell(1,length(grid));

% try
%   curcls = find(ismember(params.dataset_params.classes, ...
%                          models{1}.cls));
% catch
%   %dataset_params is missing
% end

for i = 1:length(grid)  
  curid = grid{i}.curid;
  bboxes{i} = grid{i}.bboxes;
  if size(bboxes{i},1) == 0
    continue
  end
  
  % if ~isempty(grid{i}.extras) && isfield(grid{i}.extras,'maxos')
  %   maxos{i} = grid{i}.extras.maxos;
  %   maxos{i}(grid{i}.extras.maxclass~=curcls) = 0;
  % end
  
  if REMOVE_SELF == 1
    exes = bboxes{i}(:,6);
    excurids = curids(exes);
    badex = find(ismember(excurids,{curid}));
    bboxes{i}(badex,:) = [];
    
    % if ~isempty(grid{i}.extras) && isfield(grid{i}.extras,'maxos')
    %   if ~isempty(maxos{i})
    %     maxos{i}(badex) = [];
    %   end
    % end
  end
end

raw_boxes = bboxes;

%Perform score rescaling
%1. no scaling
%2. platt's calibration (sigmoid scaling)
%3. raw score + 1

if (exist('M','var') && (~isempty(M)) && isfield(M,'betas') && ...
    ~isfield(M,'neighbor_thresh'))
  fprintf(1,'Applying betas to %d images:',length(bboxes));
  for i = 1:length(bboxes)
    %if neighbor thresh is defined, then we are in M-mode boosting
    if size(bboxes{i},1) == 0
      continue
    end
    calib_boxes = esvm_calibrate_boxes(bboxes{i},M.betas); 
    oks = find(calib_boxes(:,end) > params.calibration_threshold);
    calib_boxes = calib_boxes(oks,:);
    bboxes{i} = calib_boxes;
  end
elseif exist('M','var') && ~isempty(M) && isfield(M,'neighbor_thresh')
  %fprintf(1,'Applying M-matrix to %d images:',length(bboxes));
  %starter=tic;

  nbrlist = cell(length(bboxes),1);
  for i = 1:length(bboxes)
    %fprintf(1,'.');
    if size(bboxes{i},1) == 0
      continue
    end
    
    bboxes{i}(:,end) = bboxes{i}(:,end)+1;

    [xraw,nbrlist{i}] = esvm_get_M_features(bboxes{i},length(models), ...
                                            M.neighbor_thresh);
    r2 = esvm_apply_M(xraw,bboxes{i},M);
    bboxes{i}(:,end) = r2;
  end
  %fprintf(1,'took %.3fsec\n',toc(starter));
else
  %fprintf(1,'No betas, No M-matrix, no calibration\n');
end


if ~isfield(params,'calibrate_nms')
  params.calibrate_nms = 0.3;
end
%fprintf(1, 'Applying NMS (OS thresh=%.3f)\n',os_thresh);
for i = 1:length(bboxes)
  if size(bboxes{i},1) > 0
    bboxes{i}(:,5) = 1:size(bboxes{i},1);
    bboxes{i} = esvm_nms(bboxes{i},params.calibrate_nms);
    if ~isempty(grid{i}.extras) && isfield(grid{i}.extras,'maxos')
      maxos{i} = maxos{i}(bboxes{i}(:,5));
    end
    if exist('nbrlist','var')
      nbrlist{i} = nbrlist{i}(bboxes{i}(:,5));
    end
    bboxes{i}(:,5) = 1:size(bboxes{i},1);
  end
end

if params.calibration_matrix_propagate_onto_raw && ...
      exist('M','var') && length(M)>0 && isfield(M,'betas')
  fprintf(1,'Propagating scores onto raw detections\n');
  %% propagate scores onto raw boxes
  for i = 1:length(bboxes)
    if size(bboxes{i},1) > 0
      allMscores = bboxes{i}(:,end);
      calib_boxes = esvm_calibrate_boxes(raw_boxes{i},M.betas);
      beta_scores = calib_boxes(:,end);
      
      osmat = getosmatrix_bb(bboxes{i},raw_boxes{i});
      for j = 1:size(osmat,1)
        curscores = (osmat(j,:)>.5) .* beta_scores';
        [aa,bb] = max(curscores);
        bboxes{i}(j,:) = raw_boxes{i}(bb,:);
        bboxes{i}(j,end) = aa;
      end
      bboxes{i}(:,end) = allMscores;
      
      % new_scores = beta_scores;
      % for j = 1:length(nbrlist{i})
      %   new_scores(nbrlist{i}{j}) = max(new_scores(nbrlist{i}{j}),...
      %                                   beta_scores(nbrlist{i}{j}).*...
      %                                   bboxes{i}(nbrlist{i}{j},end));
      % end
      % bboxes{i}(:,end) = new_scores;
    end
  end
end

% Clip boxes to image dimensions since VOC testing annotation
% always fall within the image
unclipped_boxes = bboxes;
for i = 1:length(bboxes)
  bboxes{i} = clip_to_image(bboxes{i},grid{i}.imbb);
end

final_boxes = bboxes;

% return unclipped boxes for transfers
final.unclipped_boxes = unclipped_boxes;
final.final_boxes = final_boxes;
final.final_maxos = maxos;

%Create a string which summarizes the pooling type
calib_string = '';
if exist('M','var') && ~isempty(M) && isfield(M,'betas')
   calib_string = '-calibrated';
end

if exist('M','var') && ~isempty(M) && isfield(M,'betas') && isfield(M,'w')
  calib_string = [calib_string '-M'];
end

final.calib_string = calib_string;

%NOTE: is this necessary anymore?
final.imbb = cellfun2(@(x)x.imbb,grid);
