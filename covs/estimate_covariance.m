function res = estimate_covariance(train_set, cparams, res2)
% Estimates the covariance matrix of all windows of size hg_size
% inside PASCAL VOC set "train_set_name"
% If res2 is provided (optional), the highest probability window is
% diaplayed in each image

%Estimates the covariance matrix by going through all the data, and
%makes sure the data doesn't hit zero "outside-image" cells.

if isfield(cparams,'titler')
  filer = sprintf('/nfs/baikal/tmalisie/cov/final_%s%s_%d_%d.mat',...
                  cparams.titler, cparams.obj, cparams.hg_size(1), ...
                  cparams.hg_size(2));

  if fileexists(filer)
    fprintf(1,'FILE EXISTS, loading from %s\n',filer);
    load(filer)
    return;
  end
end

addpath(genpath(pwd));

if ~exist('data_directory','var')
  data_directory = '/Users/tomasz/projects/pascal/';
end

if ~exist('dataset_directory','var')
  dataset_directory = 'VOC2007';
end

%keep all windows
params = esvm_get_default_params;
params.detect_max_windows_per_exemplar = 1000000;
params.detect_save_features = 1;
params.detect_exemplar_nms_os_threshold = 1;
params.detect_pyramid_padding = 0;
params.detect_add_flip = 1;

model.params = params;
model.models{1}.w = zeros(cparams.hg_size(1),...
                          cparams.hg_size(2),...
                          params.init_params.features());
model.models{1}.b = 0;
model.models{1}.hg_size = cparams.hg_size;

n = 0;
globalN = 0;
fsize = numel(model.models{1}.w);
sums = zeros(fsize, 1);
outers = zeros(fsize, fsize);

%Learn matrix from 1000 random images
r = randperm(length(train_set));
r = r(1:min(length(r),10000));
train_set = train_set(r);

MAX_WINDOWS = 50000000;
maxw = floor(MAX_WINDOWS / length(train_set));


if exist('res2','var')
  tic
  % [u,w,v] = svd(res2.c);
  % [aa,bb] = sort(diag(w),'descend');
  % w(w<aa(10))=0;
  % invcov = v'*1./(w'+eps)*u';  
  %invcov = pinv(res2.c);
  toc
end

masker = zeros(cparams.hg_size(1),cparams.hg_size(2), ...
               esvm_features);
regions = cell(0,1);
for a = 1:cparams.hg_size(1)
  for b = 1:cparams.hg_size(2)
    m = masker;
    m(a,b,:) = 1;
    inds = find(m);
    regions{end+1} = inds;
  end
end

for i = 1:length(train_set)
  fprintf(1,'.');
  if isfield(cparams,'obj') && length(cparams.obj) > 0
    
    if length(train_set{i}.objects) == 0
      continue
    end
    
    curcls = find(ismember({train_set{i}.objects.class}, ...
                           cparams.obj));
    if strcmp(cparams.obj,'all')
      curcls = 1:length(train_set{i}.objects);
    end
    
    if length(curcls) == 0
      continue 
    end
  end
  
  
  I = toI(train_set{i});
  tic
  rs = esvm_detect(I,model);
  toc

  norms = cellfun2(@(x)sum(rs.xs{1}(x,:).^2,1),regions);
  norms = cat(1,norms{:});
  goods = find(sum(norms==0,1)==0);
  rs.bbs{1} = rs.bbs{1}(goods,:);
  rs.xs{1} = rs.xs{1}(:,goods);
  
  r = randperm(length(goods));
  r = r(1:min(maxw,length(r)));
  rs.bbs{1} = rs.bbs{1}(r,:);
  rs.xs{1} = rs.xs{1}(:,r);
  
  %experiment: try adding locations of features into covariance matrix
  %rs.xs{1} = cat(1,rs.xs{1},rs.bbs{1}(:,1:4)');

  if numel(rs.xs{1})==0
    fprintf(1,'tiny image!\n');

    continue
  end

  x = rs.xs{1};
  bbs = rs.bbs{1};
  globalN = globalN + size(x,2);
  
  if isfield(cparams,'obj') && length(cparams.obj) > 0
    
    if length(train_set{i}.objects) == 0
      continue

    end
    
    curcls = find(ismember({train_set{i}.objects.class}, ...
                           cparams.obj));
    if strcmp(cparams.obj,'all')
      curcls = 1:length(train_set{i}.objects);
    end
    
    if length(curcls) == 0
      continue

    end
    
    goodbb = cat(1,train_set{i}.objects.bbox);
    goodbb = goodbb(curcls,:);
    
    if (cparams.obj_os < 0)
      goods = find(max(getosmatrix_bb(bbs,goodbb),[],2) <= ...
                   -cparams.obj_os);
    else
    
      goods = find(max(getosmatrix_bb(bbs,goodbb),[],2) >= ...
                   cparams.obj_os);
    end
    x = x(:,goods);
    bbs = bbs(goods,:);
    % figure(1)
    % clf
    % imagesc(I)

    % plot_bbox(bbs)
    % drawnow
  elseif isfield(cparams,'scene_os')
    scene_bb = [1 1 size(I,2) size(I,1)];
    goods = find(getosmatrix_bb(bbs,scene_bb)>cparams.scene_os);
    x = x(:,goods);
    bbs = bbs(goods,:);
  end

  if exist('res2','var')
    r = res2.w(:)'*x;
    % x2 = -bsxfun(@minus,res2.mean,x);
    % tic;r=diag(x2'*invcov*x2);toc
    
    [aa,bb] = sort(r,'descend');
    for q = 1:1 %length(bb)
      figure(33)
      clf
      imagesc(I)
      plot_bbox(bbs(bb(q),:))
      title(sprintf('rank %d/%d scoree=%.3f',q,length(bb),aa(q)));
      drawnow
    end
  end
  sums = sums + sum(x,2);
  curn = size(x,2);
  
  outers = outers + x*x';
  n = n + curn;
  fprintf(1,sprintf('i=%05d/%05d n = %d/%d\n',i,length(train_set), ...
                    n,globalN));
end

res.c = 1./(n-1)*(outers - sums*sums'/n);
res.n = n;
res.mean = sums/n;
res.params = params;
res.cparams = cparams;
res.hg_size = size(model.models{1}.w);
fprintf(1,'Diagonalizing...\n');
[res.evals,res.evecs] = learn_cov(res);
res.titler = '';

if exist('filer','var')
  res.titler = cparams.titler;
  save(filer,'res');
end
