function [resstruct,t] = localizemeHOG(I, models, localizeparams)
% Localize set of models in an Image's feature pyramid via sliding
% windows and (dot product + bias recognition score).  If there is a
% small number of models (such as in per-exemplar mining), then
% fconvblas is used for detection.  If the number is large, then the
% BLOCK feature matrix method (with a single matrix multiplication) is
% used.
%
% I: Input image (or already precomputed pyramid)
% models: A cell array of models to localize inside this image
%   models{:}.model.w: Learned template
%   models{:}.model.b: Learned template's offset
% localizeparams: Localization parameters (see get_default_mining_params.m)
%
% resstruct: Sliding window output struct with 
%   resstruct.bbs{:}: Detection boxes and pyramid locations
%   resstruct.xs{:}: Detection features
% t: The Feature pyramid output
%
% Tomasz Malisiewicz (tomasz@cmu.edu)

if ~exist('localizeparams','var')
  localizeparams = get_default_mining_params;
end

if ~isfield(localizeparams,'nnmode')
 localizeparams.nnmode = '';
end

doflip = localizeparams.FLIP_LR;

localizeparams.FLIP_LR = 0;
[rs1, t1] = localizemeHOGdriver(I, models, localizeparams);
rs1 = prune_nms(rs1, localizeparams);

if doflip == 1
  localizeparams.FLIP_LR = 1;
  [rs2, t2] = localizemeHOGdriver(I, models, localizeparams);
  rs2 = prune_nms(rs2, localizeparams);
else %If there is no flip, then we are done
  resstruct = rs1;
  t = t1;
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
t = cell(2,1);
t{1} = t1;
t{2} = t2;

function [resstruct,t] = localizemeHOGdriver(I, models, ...
                                             localizeparams)

if length(models)>20 || length(localizeparams.nnmode)>0
  [resstruct,t] = localizemeHOGdriverBLOCK(I, models, ...
                                         localizeparams);
  return;
end

N = length(models);
ws = cellfun2(@(x)x.model.w,models);
bs = cellfun2(@(x)x.model.b,models);

%NOTE: all exemplars in this set must have the same sbin
sbin = models{1}.model.init_params.sbin;
t = get_pyramid(I, sbin, length(models), localizeparams);

resstruct.padder = t.padder;
resstruct.bbs = cell(N,1);
xs = cell(N,1);

maxers = cell(N,1);
for q = 1:N
  maxers{q} = -inf;
end


if localizeparams.dfun == 1
  wxs = cellfun2(@(x)reshape(x.model.x(:,1),size(x.model.w)), ...
                 models);
  ws2 = ws;
  for q = 1:length(ws2)
    ws2{q} = -2*ws{q}.*wxs{q};
    special_offset(q) = ws{q}(:)'*(models{q}.model.x(:,1).^2);
  end
end

%start with smallest level first
for level = length(t.hog):-1:1
  featr = t.hog{level};
  
  if localizeparams.dfun == 1
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
    NKEEP = sum((aa>maxers{exid}) & (aa>=localizeparams.thresh));
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
    
    if (localizeparams.FLIP_LR == 1)
      bbs = flip_box(bbs,t.size);
      bbs(:,7) = 1;
    end
    
    resstruct.bbs{exid} = cat(1,resstruct.bbs{exid},bbs);
    
    if localizeparams.SAVE_SVS == 1
      for z = 1:NKEEP
        xs{exid}{end+1} = ...
            reshape(t.hog{level}(uus(z)+(1:sss(1))-1, ...
                                 vvs(z)+(1:sss(2))-1,:), ...
                    [],1);
      end
    end
        
    if (NKEEP > 0)
      newtopk = min(localizeparams.TOPK,size(resstruct.bbs{exid},1));
      [aa,bb] = psort(-resstruct.bbs{exid}(:,end),newtopk);
      resstruct.bbs{exid} = resstruct.bbs{exid}(bb,:);
      if localizeparams.SAVE_SVS == 1
        xs{exid} = xs{exid}(:,bb);
      end
      %TJM: changed so that we only maintain 'maxers' when topk
      %elements are filled
      if (newtopk >= localizeparams.TOPK)
        maxers{exid} = min(-aa);
      end
    end    
  end
end

if localizeparams.SAVE_SVS == 1
  resstruct.xs = xs;
else
  resstruct.xs = cell(N,1);
end
fprintf(1,'\n');

function [resstruct,t] = localizemeHOGdriverBLOCK(I, models, ...
                                             localizeparams)

%%HERE is the chunk version of exemplar localization

N = length(models);
ws = cellfun2(@(x)x.model.w,models);
bs = cellfun(@(x)x.model.b,models)';

sizes1 = cellfun(@(x)x.model.hg_size(1),models);
sizes2 = cellfun(@(x)x.model.hg_size(2),models);

S = [max(sizes1(:)) max(sizes2(:))];
templates = zeros(S(1),S(2),features,length(models));
templates_x = zeros(S(1),S(2),features,length(models));
template_masks = zeros(S(1),S(2),features,length(models));

for i = 1:length(models)
  t = zeros(S(1),S(2),features);
  t(1:models{i}.model.hg_size(1),1:models{i}.model.hg_size(2),:) = ...
      models{i}.model.w;

  templates(:,:,:,i) = t;
  template_masks(:,:,:,i) = repmat(double(sum(t.^2,3)>0),[1 1 features]);

  if (length(localizeparams.nnmode) > 0) || ...
        (isfield(localizeparams,'wtype') && ...
         strcmp(localizeparams.wtype,'dfun')==1)
    x = zeros(S(1),S(2),features);
    x(1:models{i}.model.hg_size(1),1:models{i}.model.hg_size(2),:) = ...
        reshape(models{i}.model.x(:,1),models{i}.model.hg_size);
    templates_x(:,:,:,i) = x;

  end
end

%maskmat = repmat(template_masks,[1 1 1 features]);
%maskmat = permute(maskmat,[1 2 4 3]);
%templates_x  = templates_x .* maskmat;

sbin = models{1}.model.init_params.sbin;
t = get_pyramid(I, sbin, length(models), localizeparams);
resstruct.padder = t.padder;

pyr_N = cellfun(@(x)prod([size(x,1) size(x,2)]-S+1),t.hog);
sumN = sum(pyr_N);

X = zeros(S(1)*S(2)*features,sumN);
offsets = cell(length(t.hog), 1);
uus = cell(length(t.hog),1);
vvs = cell(length(t.hog),1);

counter = 1;
for i = 1:length(t.hog)
  s = size(t.hog{i});
  NW = s(1)*s(2);
  ppp = reshape(1:NW,s(1),s(2));
  curf = reshape(t.hog{i},[],features);
  b = im2col(ppp,[S(1) S(2)]);

  offsets{i} = b(1,:);
  offsets{i}(end+1,:) = i;
  
  for j = 1:size(b,2)
    X(:,counter) = reshape(curf(b(:,j),:),[],1);
    counter = counter + 1;
  end
  
  [uus{i},vvs{i}] = ind2sub(s,offsets{i}(1,:));
end

offsets = cat(2,offsets{:});

uus = cat(2,uus{:});
vvs = cat(2,vvs{:});

exemplar_matrix = reshape(templates,[],size(templates,4));

if isfield(localizeparams,'wtype') && ...
      strcmp(localizeparams.wtype,'dfun')==1
  W = exemplar_matrix;
  U = reshape(templates_x,[],length(models));
  r2 = repmat(sum(W.*(U.^2),1)',1,size(X,2));
  r =  (W'*(X.^2) - 2*(W.*U)'*X + r2);
  r = bsxfun(@minus, r, bs);
elseif length(localizeparams.nnmode) == 0
  %nnmode 0: Apply linear classifiers by performing one large matrix
  %multiplication and subtract bias
  r = exemplar_matrix' * X;
  r = bsxfun(@minus, r, bs);
elseif strcmp(localizeparams.nnmode,'normalizedhog') == 1
  r = exemplar_matrix' * X;
elseif strcmp(localizeparams.nnmode,'nndfun') == 1
  %Do euclidean distance (but only over the regions corresponding
  %to the in-mask (non-padded) regions
  W = reshape(template_masks,[],length(models));
  W = W / 100;
  U = reshape(templates_x,[],length(models));
  r2 = repmat(sum(W.*(U.^2),1)',1,size(X,2));
  r = - (W'*(X.^2) - 2*(W.*U)'*X + r2);
elseif strcmp(localizeparams.nnmode,'cosangle') == 1
  %nnmode 1: Apply linear classifiers by performing one large matrix
  %multiplication and subtract bias
  
  %mf = mean(X,1);
  %X = X - repmat(mf,size(X,1),1);
  %r = exemplar_matrix' * X;
  
  exemplar_matrix = reshape(templates_x, [], size(templates_x,4));
  
  % %% do normalization for each window
  % tm = template_masks;
  % tm = reshape(tm,[],length(models));
  % ut = unique(tm','rows');
  % [tmp,utids] = ismember(tm',ut,'rows');
  
  % r = zeros(length(models),size(finalf,2));
  % for j = 1:size(ut,1)
  %   fprintf(1,'!');
  %   curmask = repmat(reshape(ut(j,:),...
  %                     [size(templates_x,1) size(templates_x,2)]),...
  %                    [1 1 features]);
  %   hits = find(utids == j);
    
  %   curf = finalf;
  %   curf = curf.*repmat(curmask(:),1,size(curf,2));
  %   curr = slmetric_pw(exemplar_matrix(:,hits),curf,'nrmcorr');

  %   r(hits,:) = curr;
  % end
  
  r = slmetric_pw(exemplar_matrix, X, 'nrmcorr');

  %% why am I not getting perfect hits for g-mode?
  %% ANSWER: because there is a padding which we cannot enforce on
  %test-windows efficiently...This will affect the normalization of
  %the test windows
  
  % exemplar_x_matrix = reshape(templates_x,[],size(templates_x,4));
  % res = sum(exemplar_matrix .* exemplar_x_matrix,1);
  % exemplar_matrix = exemplar_matrix ./ repmat(res+eps,size(exemplar_matrix,1),1);
  % r = exemplar_matrix' * finalf;

  %keyboard
  
  % rrr = randperm(size(www2,2));
  % rrr = rrr(1:10);
  % r2 = slmetric_pw(www2(:,rrr),finalf,'chisq');
  % r = zeros(size(www2,2),size(finalf,2));  
  % r(rrr,:) = exp(-.001*r2);
  
  %r = exemplar_matrix' * finalf;
  %r = bsxfun(@minus, r, bs);
  
else
  error(sprintf('invalid nnmode=%s\n',localizeparams.nnmode));
end

resstruct.bbs = cell(N,1);
resstruct.xs = cell(N,1);

TOPK = localizeparams.TOPK;
for exid = 1:N

  goods = find(r(exid,:) >= localizeparams.thresh);
  
  if isempty(goods)
    continue
  end
  
  [sorted_scores,bb] = psort(-r(exid,goods)',min(TOPK, ...
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
  
  if (localizeparams.FLIP_LR == 1)
    bbs = flip_box(bbs,t.size);
    bbs(:,7) = 1;
  end
  
  resstruct.bbs{exid} = bbs;
end

if localizeparams.SAVE_SVS == 0
  resstruct.xs = cell(N,1);
end
fprintf(1,'\n');

function rs = prune_nms(rs, params)
%Prune via nms to eliminate redundant detections

%If the field is missing, or it is set to 1, then we don't need to
%process anything.  If it is zero, we also don't do NMS.
if ~isfield(params,'NMS_OS') || (params.NMS_OS >= 1) ...
      || (params.NMS_OS == 0)
  return;
end

rs.bbs = cellfun2(@(x)nms(x,params.NMS_OS),rs.bbs);

if ~isempty(rs.xs)
  for i = 1:length(rs.bbs)
    if ~isempty(rs.xs{i})
      %NOTE: the fifth field must contain elements
      rs.xs{i} = rs.xs{i}(:,rs.bbs{i}(:,5) );
    end
  end
end

function t = get_pyramid(I, sbin, N, localizeparams)
%Extract feature pyramid from variable I (which could be either an image,
%or already a feature pyramid)

if isnumeric(I)

  flipstring = '';
  if (localizeparams.FLIP_LR == 1)
    flipstring = '@F';
    I = flip_image(I);
  else    
    %take unadulterated "aka" un-flipped image
  end
  
  clear t
  t.size = size(I);

  fprintf(1,'Localizing %d in I=[%dx%d@%d%s]',N,...
          t.size(1),t.size(2),localizeparams.lpo,flipstring);

  %Compute pyramid
  [t.hog,t.scales] = featpyramid2(I, sbin, localizeparams);  
  t.padder = localizeparams.pyramid_padder;
  for level = 1:length(t.hog)
    t.hog{level} = padarray(t.hog{level}, [t.padder t.padder 0], 0);
  end
  
  minsizes = cellfun(@(x)min([size(x,1) size(x,2)]), t.hog);
  t.hog = t.hog(minsizes >= t.padder*2);
  t.scales = t.scales(minsizes >= t.padder*2);
    
  %if only_compute_pyramid == 1
  %  resstruct = t;
  %  return;
  %end
  
else
  fprintf(1,'Already found features\n');
  
  if iscell(I)
    if localizeparams.FLIP_LR==1
      t = I{2};
    else
      t = I{1};
    end
  else
    t = I;
  end
  
  fprintf(1,'Localizing %d in I=[%dx%d@%d]',N,...
        t.size(1),t.size(2),localizeparams.lpo);
end

