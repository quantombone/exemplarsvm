function [resstruct,t] = localizemeHOG(I, models, localizeparams)
% Localize object in pyramid via sliding windows and (dot product +
% bias recognition score)
% I: input image (or already precomputed pyramid)
% models: a cell array of models to localize inside this image
% models{.}.model.w: cell array of learned templates
% models{.}.model.b: cell array of corresponding offsets
% localizeparams: localization parameters
% resstruct: sliding window output struct
% t: the feature pyramid output
% Tomasz Malisiewicz (tomasz@cmu.edu)

if ~exist('localizeparams','var')
  localizeparams = get_default_mining_params;
end

doflip = 0;
if localizeparams.FLIP_LR == 1
  doflip = 1;
end

localizeparams.FLIP_LR = 0;
%tic
[rs1,t1] = localizemeHOGdriver(I,models,localizeparams);

%toc
%tic
%[rs2,t2] = localizemeHOGdriver2(I,models,localizeparams);
%toc


if doflip == 1
  localizeparams.FLIP_LR = 1;
  [rs2,t2] = localizemeHOGdriver(I,models,localizeparams);
else
  resstruct = rs1;
  t = t1;
  return;
end

%If we got here, then the flip was turned on
for q = 1:length(rs1.score_grid)
  rs1.score_grid{q} = cat(2,rs1.score_grid{q},rs2.score_grid{q});

  if numel(rs2.support_grid)>0 && ...
        numel(rs1.support_grid)>0 && ...
        numel(rs2.support_grid{q})>0 && ...
        numel(rs1.support_grid{q})>0
    
    rs1.support_grid{q} = cat(2,rs1.support_grid{q}, ...
                              rs2.support_grid{q});

  end

  rs1.id_grid{q} = cat(2,rs1.id_grid{q},rs2.id_grid{q});
end

resstruct = rs1;
t = cell(2,1);
t{1} = t1;
t{2} = t2;

function [resstruct,t] = localizemeHOGdriver(I, models, ...
                                             localizeparams)

if 1
  [resstruct,t] = localizemeHOGdrivernew(I, models, ...
                                         localizeparams);
  return;
end
adjust = 1;
if isfield(models{1},'models_name') ...
      && length(strfind(models{1}.models_name,'-ncc'))>0
  adjust = 1;
end

oldsave = localizeparams.SAVE_SVS;
localizeparams.SAVE_SVS = 1;

N = length(models);
ws = cellfun2(@(x)x.model.w,models);
bs = cellfun2(@(x)x.model.b,models);

%NOTE: all exemplars in this set must have the same sbin
sbin = models{1}.model.params.sbin;
  
% if only one input argument is specified, then just compute the
% pyramid and exit
only_compute_pyramid = 0;
if nargin == 1 && nargout == 1
  only_compute_pyramid = 1;
  ws{1} = [];
  bs{1} = 0;
end

if isnumeric(I)
  starter=tic;

  flipstring = '';
  if isfield(localizeparams,'FLIP_LR') && ...
        (localizeparams.FLIP_LR == 1)
    flipstring = '@F';
    
    %flip image lr here..
    I = flip_image(I);

  else    
    %take unadulterated image
  end
  
  clear t
  t.size = size(I);

  
  fprintf(1,'Localizing %d in I=[%dx%d@%d%s]',N,...
          t.size(1),t.size(2),localizeparams.lpo,flipstring);

  %Compute pyramid
  [t.hog,t.scales] = featpyramid2(I, sbin, localizeparams.lpo);  
  t.padder = 2; 
  for level = 1:length(t.hog)
    t.hog{level} = padarray(t.hog{level}, [t.padder t.padder 0], 0);
  end
  
  minsizes = cellfun(@(x)min([size(x,1) size(x,2)]),t.hog);

  t.hog = t.hog(minsizes >= t.padder*2);
  t.scales = t.scales(minsizes >= t.padder*2);
  resstruct.scales = t.scales;
  
  if only_compute_pyramid == 1
    resstruct = t;
    return;
  end
  
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
  
  resstruct.scales = t.scales;
  fprintf(1,'Localizing %d in I=[%dx%d@%d]',N,...
        t.size(1),t.size(2),localizeparams.lpo);
end

%score grid stores the TOPK scores from each exemplar's firing in
%the image
score_grid = cell(N,1);
id_grid = cell(N,1);
support_grid = cell(N,1);
for q = 1:N
  maxers{q} = -inf;
end
resstruct.padder = t.padder;

normx2 = cellfun(@(x)norm(x(:)).^2, ws);

%start with smallest level first
for level = length(t.hog):-1:1
  featr = t.hog{level};

  %The norm squared integral image is used for euclidean distance computations
  cellnorms2 = sum(featr.^2,3);
  cellnorms2_ii = cumsum(cumsum(cellnorms2,2),1);

  %Use blas-based fast convolution code
  rootmatch = fconvblas(featr, ws, 1, N);
  %rootmatch = fconv(featr, ws, 1, N);

  rmsizes = cellfun2(@(x)size(x), ...
                     rootmatch);

  for exid = 1:N
    if prod(rmsizes{exid}) == 0
      continue
    end

    %% Only do the following stuff if we are in Nearest-Neighbor
    %% Euclidean distance matching mode
    if isfield(models{exid},'NN_MODE') && (models{exid}.NN_MODE ==1)

      if size(ws{exid},1)*size(ws{exid},2) == 1
        curq = cellnorms2;
      else
        shiftedtop = ...
            (circshift2(cellnorms2_ii,[size(ws{exid},1) ...
                    0]));
        
        shiftedleft = ...
            (circshift2(cellnorms2_ii,[0 ...
                    size(ws{exid},2)]));
        
        shiftedboth = ...
            (circshift2(cellnorms2_ii,[size(ws{exid},1) ...
                    size(ws{exid},2)]));
        
        curq = cellnorms2_ii - shiftedtop - shiftedleft + shiftedboth;
        curq = circshift2(curq,-[size(ws{exid},1) size(ws{exid},2)]+1);
        
        curq = curq(1:size(rootmatch{exid},1),...
                    1:size(rootmatch{exid},2));
      end
      cur_scores = -(normx2(exid) + curq - 2*rootmatch{exid});
    else
      cur_scores = rootmatch{exid} - bs{exid};
    end

    hg_size = size(ws{exid});

    [aa,indexes] = sort(cur_scores(:),'descend');
    NKEEP = sum((aa>maxers{exid}) & (aa>=localizeparams.thresh));
    sss = size(ws{exid});
    
    [uus,vvs] = ind2sub(rmsizes{exid}(1:2),...
                        indexes(1:NKEEP));
    
    for z = 1:NKEEP
      score_grid{exid}(end+1) = aa(z);
      ip.level = level;
      ip.scale = t.scales(level);
      ip.offset = [uus(z) vvs(z)] - t.padder;
      ip.bb = [([ip.offset(2) ip.offset(1) ip.offset(2)+size(ws{exid},2) ...
                 ip.offset(1)+size(ws{exid},1)] - 1) * ...
               sbin/ip.scale + 1] + [0 0 -1 -1];
      ip.flip = 0;
      if isfield(localizeparams,'FLIP_LR') && ...
            (localizeparams.FLIP_LR == 1)
        %saver = ip.bb;
        
        %%% NOTE: this is broken because of size(I)
        ip.bb = flip_box(ip.bb,t.size);
        if 0 &&(ip.bb(3) <= 0 || ip.bb(4) <= 0) && size(ws{1},1)*size(ws{1},2)>1
          fprintf(1,'why first one less than 0?\n');
        
          figure(1)
          clf
          imagesc(I)
          plot_bbox(saver,'',[0 1 0])
          plot_bbox(ip.bb,'',[1 0 0])
          axis image
          axis off
          drawnow
        end
         
        ip.flip = 1;
      end
      id_grid{exid}{end+1} = ip; 
      
      uu = uus(z);
      vv = vvs(z);
      if localizeparams.SAVE_SVS == 1
        support_grid{exid}{end+1} = ...
            reshape(t.hog{level}(uu+(1:sss(1))-1, ...
                                 vv+(1:sss(2))-1,:), ...
                    prod(hg_size),1);
      end
    end
    
    if (NKEEP > 0)
      newtopk = min(localizeparams.TOPK,length(score_grid{exid}));
      [aa,bb] = psort(-score_grid{exid}',newtopk);
      score_grid{exid} = score_grid{exid}(bb);
      id_grid{exid} = id_grid{exid}(bb);
      if localizeparams.SAVE_SVS == 1
        support_grid{exid} = support_grid{exid}(bb);
      end
      maxers{exid} = min(-aa);
    end   
  end
end

resstruct.score_grid = score_grid;
resstruct.id_grid = id_grid;
if localizeparams.SAVE_SVS == 1
  resstruct.support_grid = support_grid;
else
  resstruct.support_grid = cell(0,1);
end
fprintf(1,'\n');

if adjust == 1
  %% Here we run an auxilliary distance metric for detections
  for j = 1:length(resstruct.id_grid)
    if length(resstruct.id_grid{j})==0
      continue
    end
    xs = cat(2, ...
             resstruct.support_grid{j}{:});
    norms = sqrt(sum(xs.^2,1));
    xs = xs ./ repmat(norms,size(xs,1),1);
    newd =  xs'* ...
            (models{j}.model.x)/norm(models{j}.model.x);
    [aa,bb] = sort(newd, ...
                   'descend');
    resstruct.score_grid{j} = aa';
    resstruct.id_grid{j} = ...
        resstruct.id_grid{j}(bb);
    resstruct.support_grid{j} = ...
        resstruct.support_grid{j}(bb);
  end
  %% Here we adjust things with a
  %new distance matrix
  if oldsave == 0
    resstruct.support_grid = [];
  end
end

%% Do NMS (nothing happens if the field is turned off, or absent)
resstruct = prune_nms(resstruct,localizeparams);

function rs = prune_nms(rs, params)
%Prune via nms to eliminate redundant detections

%If the field is missing, or it is set to 1, then we don't need to
%process anything
if ~isfield(params,'NMS_MINES_OS') || (params.NMS_MINES_OS >= 1)
  return;
end

%Do NMS independently for each W, not for the combination!
for i = 1:length(rs.id_grid)
  if length(rs.id_grid{i})==0
    continue
  end

  bbs=cellfun2(@(x)x.bb,rs.id_grid{i});
  bbs = cat(1,bbs{:});
  bbs(:,5) = 1:size(bbs,1);
  bbs(:,6) = 1;
  bbs(:,7) = rs.score_grid{i}';
  bbs = nms(bbs, params.NMS_MINES_OS);
  ids = bbs(:,5);
  rs.score_grid{i} = rs.score_grid{i}(ids);
  rs.id_grid{i} = rs.id_grid{i}(ids);
  
  %only access features if they are present
  if length(rs.support_grid) > 0
    %%%TODO: this might break things in older version
    rs.support_grid{i} = rs.support_grid{i}(:,ids);
  end
end

function [resstruct,t] = localizemeHOGdrivernew(I, models, ...
                                             localizeparams)

%%HERE is the chunk version of this
adjust = 1;
if isfield(models{1},'models_name') ...
      && length(strfind(models{1}.models_name,'-ncc'))>0
  adjust = 1;
end

oldsave = localizeparams.SAVE_SVS;
localizeparams.SAVE_SVS = 1;

N = length(models);
ws = cellfun2(@(x)x.model.w,models);
bs = cellfun2(@(x)x.model.b,models);

sizes1 = cellfun(@(x)x.model.hg_size(1),models);
sizes2 = cellfun(@(x)x.model.hg_size(2),models);

S = [max(sizes1(:)) max(sizes2(:))];
templates = zeros(S(1),S(2),features,length(models));
template_masks = zeros(S(1),S(2),length(models));

for i = 1:length(models)
  t = zeros(S(1),S(2),features);
  t(1:models{i}.model.hg_size(1),1:models{i}.model.hg_size(2),:) = ...
      models{i}.model.w;
  templates(:,:,:,i) = t;
  template_masks(:,:,i) = double(sum(t.^2,3)>0);
end

%NOTE: all exemplars in this set must have the same sbin
sbin = models{1}.model.params.sbin;
  
% if only one input argument is specified, then just compute the
% pyramid and exit
only_compute_pyramid = 0;
if nargin == 1 && nargout == 1
  only_compute_pyramid = 1;
  ws{1} = [];
  bs{1} = 0;
end

if isnumeric(I)
  starter=tic;

  flipstring = '';
  if isfield(localizeparams,'FLIP_LR') && ...
        (localizeparams.FLIP_LR == 1)
    flipstring = '@F';
    
    %flip image lr here..
    I = flip_image(I);

  else    
    %take unadulterated image
  end
  
  clear t
  t.size = size(I);

  fprintf(1,'Localizing %d in I=[%dx%d@%d%s]',N,...
          t.size(1),t.size(2),localizeparams.lpo,flipstring);

  %Compute pyramid
  [t.hog,t.scales] = featpyramid2(I, sbin, localizeparams.lpo);  
  t.padder = 5; 
  for level = 1:length(t.hog)
    t.hog{level} = padarray(t.hog{level}, [t.padder t.padder 0], 0);
  end
  
  minsizes = cellfun(@(x)min([size(x,1) size(x,2)]),t.hog);

  t.hog = t.hog(minsizes >= t.padder*2);
  t.scales = t.scales(minsizes >= t.padder*2);
  resstruct.scales = t.scales;
  
  if only_compute_pyramid == 1
    resstruct = t;
    return;
  end
  
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
  
  resstruct.scales = t.scales;
  fprintf(1,'Localizing %d in I=[%dx%d@%d]',N,...
        t.size(1),t.size(2),localizeparams.lpo);
end



finalf  = cell(length(t.hog), 1);
offsets = cell(length(t.hog), 1);

for i = 1:length(t.hog)
  s = size(t.hog{i});
  NW = s(1)*s(2);
  ppp = reshape(1:NW,s(1),s(2));
  curf = reshape(t.hog{i},[],features);
  b = im2col(ppp,[S(1) S(2)]);
  offsets{i} = b(1,:);
  offsets{i}(end+1,:) = i;
  
  finalf{i} = zeros(size(b,1)*features,size(b,2));

  for j = 1:size(b,2)
    finalf{i}(:,j) = reshape(curf(b(:,j),:),[],1);
  end

end

offsets = cat(2,offsets{:});
finalf = cat(2,finalf{:});

www = reshape(templates,[],size(templates,4));

%subtract bias
r = www'*finalf;
r = bsxfun(@minus,r,cellfun(@(x)x.model.b,models)');

% for jjj = 1:size(www,2)
%   masky = template_masks(:,:,jjj);
%   masky = repmat(logical(masky), [1 1 features]);
%   lefty = www(masky,jjj);
%   righty = finalf(masky,:);
%   lefty = lefty ./ norm(lefty);
%   rnorms = sum(righty.^2,1);
%   righty = righty ./ repmat(rnorms,size(righty,1),1);
%   r(jjj,:) = lefty'*righty;
% end
% %% do normalized distance

%score grid stores the TOPK scores from each exemplar's firing in
%the image
score_grid = cell(N,1);
id_grid = cell(N,1);
support_grid = cell(N,1);
resstruct.padder = t.padder;

TOPK = localizeparams.TOPK;
for exid = 1:N
  
  [aa,bb] = sort(r(exid,:),'descend');  
  score_grid{exid} = aa(1:TOPK);
  support_grid{exid} = finalf(:,bb(1:TOPK));

  for j = 1:TOPK

    ip.level = offsets(2,bb(j));
    ip.scale = t.scales(ip.level);
    
    [uu,vv] = ind2sub([size(t.hog{ip.level},1) size(t.hog{ip.level},2)],...
                      offsets(1,bb(j)));
    ip.offset = [uu vv] - t.padder;

    ip.bb = [([ip.offset(2) ip.offset(1) ip.offset(2)+size(ws{exid},2) ...
               ip.offset(1)+size(ws{exid},1)] - 1) * ...
             sbin/ip.scale + 1] + [0 0 -1 -1];

    ip.flip = 0;
    
    if isfield(localizeparams,'FLIP_LR') && ...
          (localizeparams.FLIP_LR == 1)
      ip.bb = flip_box(ip.bb,t.size);
      ip.flip = 1;
    end
    id_grid{exid}{end+1} = ip;
  end
end

resstruct.score_grid = score_grid;
resstruct.id_grid = id_grid;
if localizeparams.SAVE_SVS == 1
  resstruct.support_grid = support_grid;
else
  resstruct.support_grid = cell(0,1);
end
fprintf(1,'\n');

if adjust == 1
  %% Here we run an auxilliary distance metric for detections
  for j = 1:length(resstruct.id_grid)
    if length(resstruct.id_grid{j})==0
      continue
    end
    
    curx = reshape(models{j}.model.x,...
                   models{j}.model.hg_size);
    
    xs = cat(2, ...
             resstruct.support_grid{j});
    xs = reshape(xs,[size(templates,1) size(templates,2) size(templates,3) ...
                     size(xs,2)]);
    xs = xs(1:size(curx,1),1:size(curx,2),:);
    xs = reshape(xs,[],size(resstruct.support_grid{j},2));
    
    norms = sqrt(sum(xs.^2,1));
    xs = xs ./ repmat(norms,size(xs,1),1);

    %newx = templates(:,:,:,1)*0;
    
    
    %newx(1:size(curx,1),1:size(curx,2),:) = curx;

    newd =  xs'* ...
            (curx(:)/norm(curx(:)));

    [aa,bb] = sort(newd, ...
                   'descend');

    resstruct.score_grid{j} = aa';
    resstruct.id_grid{j} = ...
        resstruct.id_grid{j}(bb);
    resstruct.support_grid{j} = ...
        resstruct.support_grid{j}(bb);
  end
  %% Here we adjust things with a
  %new distance matrix
  if oldsave == 0
    resstruct.support_grid = [];
  end
end

%% Do NMS (nothing happens if the field is turned off, or absent)
resstruct = prune_nms(resstruct,localizeparams);
