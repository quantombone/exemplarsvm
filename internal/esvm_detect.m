function [resstruct,feat_pyramid] = esvm_detect(I, models, params)
% Localize a set of models in an image.
% function [resstruct,feat_pyramid] = esvm_detect(I, models, params)
%
% If there is a small number of models (such as in per-exemplar
% mining), then fconvblas is used for detection.  If the number is
% large, then the BLOCK feature matrix method (with a single matrix
% multiplication) is used.
%
% NOTE: These local detections can be pooled with esvm_pool_exemplars_dets.m
%
% I: Input image (or already precomputed pyramid)
% models: A cell array of models to localize inside this image
%   models{:}.w: Learned template
%   models{:}.b: Learned template's offset
% params: Localization parameters (see esvm_get_default_params.m)
%
% resstruct: Sliding window output struct with 
%   resstruct.bbs{:}: Detection boxes and pyramid locations
%   resstruct.xs{:}: Detection features
% feat_pyramid: The Feature pyramid output
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

if exist('params','var')
  if isstruct(models)
    models.params = params;
  else
    models = cellfun2(@(x)setfield(x,'params',params),models);
  end
elseif isstruct(models) && isfield(models,'params')
  params = models.params;
elseif iscell(models) && isfield(models{1},'params')
  params = models{1}.params;
else 
  params = esvm_get_default_params;
end


% if isfield(models.models{1},'A')
%   params.max_models_before_block_method = 0;
% end

if isempty(models)
  fprintf(1,'Warning: empty models in esvm_detect\n');
  resstruct.bbs{1} = zeros(0,0);
  resstruct.xs{1} = zeros(0,0);
  feat_pyramid = [];
  return;
end

if isstruct(models) && isfield(models,'models')
  if isfield(models,'basis')
    params.basis = models.basis;
  end
  [resstruct,feat_pyramid] = esvm_detect(I,models.models,params);
  return;
end

if ~iscell(models) && ~isfield(models,'models')
  models = {models};
  [resstruct,feat_pyramid] = esvm_detect(I,models,params);

  %for q = 1:length(feat_pyramid)
  %  feat_pyramid(q).scales = feat_pyramid(q).scales*params.factor_multiplier;
  %end

  return;
end


%params.max_image_size = 500;
%Make sure image is in double format
raw_max_dim = max([size(I,1) size(I,2)]);
raw_factor = 1.0;
Ioriginal = I;
if raw_max_dim > params.max_image_size
  raw_factor = params.max_image_size/raw_max_dim;
  I = imresize_max(Ioriginal, params.max_image_size);
  params.factor_multiplier = raw_factor;
else
  params.factor_multiplier = 1.0;
end

%pad image with mirrors
%params.frac = .2;
%[I,H2,W2] = rotate_pad(I,params.frac);

H2 = 0;
W2 = 0;
%if ~isfield(params,'nnmode')
%  params.nnmode = '';
%end

doflip = params.detect_add_flip;

params.detect_add_flip = 0;
[rs1, t1] = esvm_detectdriver(I, models, params);
%s1 = max(abs(rs1.bbs{1}(:,end)' - (models{1}.w(:)'*cat(2,rs1.xs{1}{:})- ...
%     models{1}.b)))
if size(rs1.bbs{1},1) > 0
  rs1.bbs{1}(:,[1 3]) = rs1.bbs{1}(:,[1 3]) - W2;
  rs1.bbs{1}(:,[2 4]) = rs1.bbs{1}(:,[2 4]) - H2;
end

rs1 = prune_nms(rs1, params);

%s1 = max(abs(rs1.bbs{1}(:,end)' - (models{1}.w(:)'*cat(2,rs1.xs{1}{:})- ...
%     models{1}.b)))


if doflip == 1
  params.detect_add_flip = 1;
  [rs2, t2] = esvm_detectdriver(I, models, params);
  if size(rs2.bbs{1},1) > 0
    rs2.bbs{1}(:,[1 3]) = rs2.bbs{1}(:,[1 3]) - W2;
    rs2.bbs{1}(:,[2 4]) = rs2.bbs{1}(:,[2 4]) - H2;
  end

  %s1 = max(abs(rs2.bbs{1}(:,end)' - (models{1}.w(:)'*cat(2,rs2.xs{1}{:})- ...
  %   models{1}.b)))

  rs2 = prune_nms(rs2, params);
  %s1 = max(abs(rs2.bbs{1}(:,end)' - (models{1}.w(:)'*cat(2,rs2.xs{1}{:})- ...
  %                                   models{1}.b)))

else %If there is no flip, then we are done
  resstruct = rs1;
  feat_pyramid = t1;

  for q = 1:length(feat_pyramid)
    feat_pyramid(q).scales = feat_pyramid(q).scales*params.factor_multiplier;
  end
  
  
  for q = 1:length(resstruct.bbs)
    if numel(resstruct.bbs{q}) > 0
      resstruct.bbs{q}(:,1:4) = resstruct.bbs{q}(:,1:4)/ ...
          params.factor_multiplier;
    end
  end
  
  return;
end

%If we got here, then the flip was turned on and we need to concatenate
%results
for q = 1:length(rs1.bbs)
  rs1.xs{q} = cat(2,rs1.xs{q}, ...
                  rs2.xs{q});


  rs1.bbs{q} = cat(1,rs1.bbs{q},rs2.bbs{q});
end

resstruct = rs1;

%Concatenate normal and LR pyramids
feat_pyramid = cat(1,t1,t2);


for q = 1:length(feat_pyramid)
  feat_pyramid(q).scales = feat_pyramid(q).scales*params.factor_multiplier;
end


for q = 1:length(resstruct.bbs)
  if numel(resstruct.bbs{q}) > 0
    resstruct.bbs{q}(:,1:4) = resstruct.bbs{q}(:,1:4)/ ...
        params.factor_multiplier;
  end
end


I2 = I(:,:,1)*0-1;
bbs = resstruct.bbs{1};
for q = 1:size(bbs,1)
  bb = bbs(q,:);
  bb(1:4) = round(bb(1:4));
  bb(1:4) = clip_to_image(bb(1:4),[1 1 size(I,2) size(I,1)]);
  I2(bb(2):bb(4),bb(1):bb(3)) = max(bb(end),I2(bb(2):bb(4),bb(1):bb(3)));
end

resstruct.I2 = I2;
function [resstruct,t] = esvm_detectdriver(I, models, ...
                                             params)
if ~isfield(params,'max_models_before_block_method')
  params.max_models_before_block_method = 20;
end

if (length(models) > params.max_models_before_block_method) ...
      || (~isempty(params.nnmode) && params.nnmode~=0)

  [resstruct,t] = esvm_detectdriverBLOCK(I, models, ...
                                         params);
  return;
end


N = length(models);
ws = cellfun2(@(x)x.w,models);
bs = cellfun2(@(x)x.b,models);

%NOTE: all exemplars in this set must have the same sbin
luq = 1;

if isfield(models{1}.params,'init_params')
  sbins = cellfun(@(x)x.params.init_params.sbin,models);
  luq = length(unique(sbins));
end

if isfield(models{1}.params,'init_params') && luq == 1
  sbin = models{1}.params.init_params.sbin;
elseif ~isfield(models{1},'init_params')
  if isfield(params,'init_params')
    sbin = params.init_params.sbin;
  else
    fprintf(1,'No hint for sbin!\n');
    error('No sbin provided');
  end
  
else
  fprintf(1,['Warning: not all exemplars have save sbin, using' ...
             ' first]\n']);
  sbin = models{1}.params.init_params.sbin;
end

t = get_pyramid(I, sbin, params);
resstruct.padder = t.padder;
resstruct.bbs = cell(N,1);
xs = cell(N,1);

maxers = cell(N,1);
for q = 1:N
  maxers{q} = -inf;
end


if params.dfun == 1
  wxs = cellfun2(@(x)reshape(x.x(:,1),size(x.w)), ...
                 models);
  ws2 = ws;
  special_offset = zeros(length(ws2),1);
  for q = 1:length(ws2)
    ws2{q} = -2*ws{q}.*wxs{q};
    special_offset(q) = ws{q}(:)'*(models{q}.x(:,1).^2);
  end
end

%start with smallest level first
for level = length(t.hog):-1:1
  featr = t.hog{level};
  
  if params.dfun == 1
    featr_squared = featr.^2;
    
    %Use blas-based fast convolution code
    rootmatch1 = fconvblas(featr_squared, ws, 1, N);
    rootmatch2 = fconvblas(featr, ws2, 1, N);
     
    for z = 1:length(rootmatch1)
      rootmatch{z} = rootmatch1{z} + rootmatch2{z} + special_offset(z);
    end
    
  else  
    %Use blas-based fast convolution code
    rootmatch = fconvblas(featr, ws, 1, N);
  end
  
  rmsizes = cellfun2(@(x)size(x), ...
                     rootmatch);
  for exid = 1:N
    if prod(rmsizes{exid}) == 0
      continue
    end

    cur_scores = rootmatch{exid} - bs{exid};
    [aa,indexes] = sort(cur_scores(:),'descend');
    NKEEP = sum((aa>maxers{exid}) & (aa>=params.detect_keep_threshold));
    aa = aa(1:NKEEP);
    indexes = indexes(1:NKEEP);
    if NKEEP==0
      continue
    end
    sss = size(ws{exid});
    
    [uus,vvs] = ind2sub(rmsizes{exid}(1:2),...
                        indexes);
    
    scale = t.scales(level);
    
    o = [uus vvs] - t.padder;

    bbs = ([o(:,2) o(:,1) o(:,2)+size(ws{exid},2) ...
               o(:,1)+size(ws{exid},1)] - 1) * ...
             sbin/scale + 1 + repmat([0 0 -1 -1],length(uus),1);

    bbs(:,5:12) = 0;
    bbs(:,5) = (1:size(bbs,1));
    bbs(:,6) = exid;
    bbs(:,8) = scale;
    bbs(:,9) = uus;
    bbs(:,10) = vvs;
    bbs(:,12) = aa;
    
    if (params.detect_add_flip == 1)
      bbs = flip_box(bbs,t.size);
      bbs(:,7) = 1;
    end
    
    resstruct.bbs{exid} = cat(1,resstruct.bbs{exid},bbs);
    
    if params.detect_save_features == 1
      for z = 1:NKEEP
        xs{exid}{end+1} = ...
            reshape(t.hog{level}(uus(z)+(1:sss(1))-1, ...
                                 vvs(z)+(1:sss(2))-1,:), ...
                    [],1);
      end
    end
        
    if (NKEEP > 0)
      newtopk = min(params.detect_max_windows_per_exemplar,size(resstruct.bbs{exid},1));
      [aa,bb] = psort(-resstruct.bbs{exid}(:,end),newtopk);
      resstruct.bbs{exid} = resstruct.bbs{exid}(bb,:);
      if params.detect_save_features == 1
        xs{exid} = xs{exid}(:,bb);
      end
      %TJM: changed so that we only maintain 'maxers' when topk
      %elements are filled
      if (newtopk >= params.detect_max_windows_per_exemplar)
        maxers{exid} = min(-aa);
      end
    end    
  end
end

if params.detect_save_features == 1
  resstruct.xs = xs;
else
  resstruct.xs = cell(N,1);
end
%fprintf(1,'\n');

%Concatenate everything to make it nice

goods = find(cellfun(@(x)numel(x),resstruct.xs));
resstruct.xs(goods) = cellfun2(@(x)cat(2,x{:}),resstruct.xs(goods));


function [resstruct,t] = esvm_detectdriverBLOCK(I, models, ...
                                             params)

%%HERE is the chunk version of exemplar localization

N = length(models);
ws = cellfun2(@(x)x.w,models);
bs = cellfun(@(x)x.b,models)';
bs = reshape(bs,[],1);
sizes1 = cellfun(@(x)x.hg_size(1),models);
sizes2 = cellfun(@(x)x.hg_size(2),models);

S = [max(sizes1(:)) max(sizes2(:))];
fsize = params.init_params.features();
templates = zeros(S(1),S(2),fsize,length(models));
templates_x = zeros(S(1),S(2),fsize,length(models));
template_masks = zeros(S(1),S(2),fsize,length(models));

for i = 1:length(models)
  t = zeros(S(1),S(2),fsize);
  t(1:models{i}.hg_size(1),1:models{i}.hg_size(2),:) = ...
      models{i}.w;

  templates(:,:,:,i) = t;
  template_masks(:,:,:,i) = repmat(double(sum(t.^2,3)>0),[1 1 fsize]);

  if (~isempty(params.nnmode)) || ...
        (isfield(params,'wtype') && ...
         strcmp(params.wtype,'dfun')==1)
    x = zeros(S(1),S(2),fsize);
    x(1:models{i}.hg_size(1),1:models{i}.hg_size(2),:) = ...
        reshape(models{i}.x(:,1),models{i}.hg_size);
    templates_x(:,:,:,i) = x;

  end
end

%maskmat = repmat(template_masks,[1 1 1 fsize]);
%maskmat = permute(maskmat,[1 2 4 3]);
%templates_x  = templates_x .* maskmat;

sbin = models{1}.params.init_params.sbin;
t = get_pyramid(I, sbin, params);
resstruct.padder = t.padder;

pyr_N = cellfun(@(x)( max(0,size(x,1)-S(1)+1)*max(0,size(x,2)-S(2)+1)),t.hog);
pyr_N = pyr_N.*(pyr_N>0);
sumN = sum(pyr_N);

X = zeros(S(1)*S(2)*fsize,sumN);
offsets = cell(length(t.hog), 1);
uus = cell(length(t.hog),1);
vvs = cell(length(t.hog),1);

counter = 1;
for i = 1:length(t.hog)
  s = size(t.hog{i});
  NW = s(1)*s(2);
  ppp = reshape(1:NW,s(1),s(2));
  curf = reshape(t.hog{i},[],fsize);
  b = im2col(ppp,[S(1) S(2)]);
  

  offsets{i} = b(1,:);
  offsets{i}(end+1,:) = i;
    
  for j = 1:size(b,2)
   X(:,counter) = reshape (curf(b(:,j),:),[],1);
   counter = counter + 1;
  end
  
  [uus{i},vvs{i}] = ind2sub(s,offsets{i}(1,:));
end

offsets = cat(2,offsets{:});

uus = cat(2,uus{:});
vvs = cat(2,vvs{:});


% m.model.w = zeros(S(1),S(2),fsize);
% m.model.b = 0;
% temp_params = params;
% temp_params.detect_save_features = 1;
% temp_params.detect_exemplar_nms_os_threshold = 1.0;
% temp_params.max_models_before_block_method = 1;
% temp_params.detect_max_windows_per_exemplar = 28000;

% [rs] = esvm_detect(I, {m}, temp_params);
% X2=cat(2,rs.xs{1}{:});
% bbs2 = rs.bbs{1};


exemplar_matrix = reshape(templates,[],size(templates,4));

if isfield(params,'wtype') && ...
      strcmp(params.wtype,'dfun')==1
  W = exemplar_matrix;
  U = reshape(templates_x,[],length(models));
  r2 = repmat(sum(W.*(U.^2),1)',1,size(X,2));
  r =  (W'*(X.^2) - 2*(W.*U)'*X + r2);
  r = bsxfun(@minus, r, bs);
elseif isempty(params.nnmode)
  %nnmode 0: Apply linear classifiers by performing one large matrix
  %multiplication and subtract bias
  
  if ~isfield(params,'basis')

    r = exemplar_matrix' * X;
    r = bsxfun(@minus, r, bs);
    %keyboard
    %s = sum(max(r,-1),1);
    % s=sum(exp(10*(max(r,-1))),1);
    % r(1,:) = s;
    % r(2:end,:) = -1;
    
    if isfield(models{1},'Akernel')
      tic
      coeff = models{1}.Akernel*X;
      coeff = sum(coeff.^2,1);
      toc
      r = r + coeff;
    end


  else
    v = cellfun2(@(x)x.v,models);
    v = cat(2,v{:});
    r = bsxfun(@plus,(v(1:end-1,:)'*(params.basis'*X)),v(end,:)');
  end
elseif strcmp(params.nnmode,'normalizedhog') == 1
  r = exemplar_matrix' * X;
elseif strcmp(params.nnmode,'nndfun') == 1
  %Do euclidean distance (but only over the regions corresponding
  %to the in-mask (non-padded) regions
  W = reshape(template_masks,[],length(models));
  W = W / 100;
  U = reshape(templates_x,[],length(models));
  r2 = repmat(sum(W.*(U.^2),1)',1,size(X,2));
  r = - (W'*(X.^2) - 2*(W.*U)'*X + r2);
else
  error('invalid nnmode=%s\n',params.nnmode);
end

resstruct.bbs = cell(N,1);
resstruct.xs = cell(N,1);

for exid = 1:N

  goods = find(r(exid,:) >= params.detect_keep_threshold);
  
  if isempty(goods)
    continue
  end
  
  [sorted_scores,bb] = ...
      psort(-r(exid,goods)',...
            min(params.detect_max_windows_per_exemplar, ...
                length(goods)));
  bb = goods(bb);

  sorted_scores = -sorted_scores';

  resstruct.xs{exid} = X(:,bb);
  levels = offsets(2,bb);

  scales = t.scales(levels);
  curuus = uus(bb);
  curvvs = vvs(bb);
  o = [curuus' curvvs'] - t.padder;

  bbs = ([o(:,2) o(:,1) o(:,2)+size(ws{exid},2) ...
           o(:,1)+size(ws{exid},1)] - 1) .* ...
             repmat(sbin./scales',1,4) + 1 + repmat([0 0 -1 ...
                    -1],length(scales),1);
  
  bbs(:,5:12) = 0;
  bbs(:,5) = (1:size(bbs,1));
  bbs(:,6) = exid;
  bbs(:,8) = scales;
  bbs(:,9) = uus(bb);
  bbs(:,10) = vvs(bb);
  bbs(:,12) = sorted_scores;
  
  if (params.detect_add_flip == 1)
    bbs = flip_box(bbs,t.size);
    bbs(:,7) = 1;
  end
  
  resstruct.bbs{exid} = bbs;
end


if params.detect_save_features == 0
  resstruct.xs = cell(N,1);
end
%fprintf(1,'\n');


function rs = prune_nms(rs, params)
%Prune via nms to eliminate redundant detections
%NOTE(TJM): there is a bug here whic I found during eccv_mine.. why
%do nms here anyways?
%If the field is missing, or it is set to 1, then we don't need to
%process anything.  If it is zero, we also don't do NMS.
if ~isfield(params,'detect_exemplar_nms_os_threshold') || (params.detect_exemplar_nms_os_threshold >= 1) ...
      || (params.detect_exemplar_nms_os_threshold == 0)
  return;
end

%NOTE: the fifth field must contain elements 1:Nboxes
rs.bbs{1}(:,5) = 1:size(rs.bbs{1},1);
rs.bbs = cellfun2(@(x)esvm_nms(x,params.detect_exemplar_nms_os_threshold),rs.bbs);

if ~isempty(rs.xs)
  for i = 1:length(rs.bbs)
    if ~isempty(rs.xs{i})
      %NOTE: the fifth field must contain elements 1:Nboxes
      rs.xs{i} = rs.xs{i}(:,rs.bbs{i}(:,5));
    end
  end
end

function t = get_pyramid(I, sbin, params)
%Extract feature pyramid from variable I (which could be either an image,
%or already a feature pyramid)

if isnumeric(I)
  if (params.detect_add_flip == 1)
    I = flip_image(I);
  else    
    %take unadulterated "aka" un-flipped image
  end
  
  clear t
  t.size = size(I);

  %Compute pyramid
  [t.hog, t.scales] = esvm_pyramid(I, params);
  t.padder = params.detect_pyramid_padding;
  for level = 1:length(t.hog)
    t.hog{level} = padarray(t.hog{level}, [t.padder t.padder 0], 0);
  end
  
  minsizes = cellfun(@(x)min([size(x,1) size(x,2)]), t.hog);
  t.hog = t.hog(minsizes >= t.padder*2);
  t.scales = t.scales(minsizes >= t.padder*2);  
else
  fprintf(1,'Already found features\n');
  
  if iscell(I)
    if params.detect_add_flip==1
      t = I{2};
    else
      t = I{1};
    end
  elseif length(I)==2 && params.detect_add_flip==1
    t =I(2);
  elseif length(I)==2 && params.detect_add_flip==0
    t = I(1);
  else
    t = I;
  end
end

