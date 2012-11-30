function covstruct = estimate_covariance(data_set, cparams)
% Estimates the covariance matrix of all windows of size cparams.hg_size
% Estimates the covariance matrix by going through all the data, and
% makes sure the data doesn't hit zero "outside-image" cells.

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
params.max_image_size = 500;
params.detect_pyramid_padding = 0;
params.detect_add_flip = 0;
params.init_params.features = @esvm_features2;
params.detect_levels_per_octave = 10;
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

%Learn matrix from 10000 random images
r = randperm(length(data_set));
r = r(1:min(length(r),10000));
data_set = data_set(r);

fprintf(1,'Learning covariance from %d images\n',length(r));

MAX_WINDOWS = 10000000;
maxw = floor(MAX_WINDOWS / length(data_set));

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

for i = 1:length(data_set)
  fprintf(1,'.');
  if isfield(cparams,'obj') && length(cparams.obj) > 0
    
    if length(data_set{i}.objects) == 0
      continue
    end
    
    curcls = find(ismember({data_set{i}.objects.class}, ...
                           cparams.obj));
    if strcmp(cparams.obj,'all')
      curcls = 1:length(data_set{i}.objects);
    end
    
    if length(curcls) == 0
      continue 
    end
  end
    
  I = toI(data_set{i});
  figure(1)
  clf
  imagesc(I)
  title(sprintf('Image %d / %d',i,length(data_set)))
  drawnow
  tic
  rs = esvm_detect(I,model);
  toc
  
  if numel(rs.xs{1})==0
    fprintf(1,'tiny image!\n');
    continue
  end
  
  norms = cellfun2(@(x)sum(rs.xs{1}(x,:).^2,1),regions);
  norms = cat(1,norms{:});
  goods = find(sum(norms==0,1)==0);
  
  rs.bbs{1} = rs.bbs{1}(goods,:);
  rs.xs{1} = rs.xs{1}(:,goods);
  
  r = randperm(length(goods));
  r = r(1:min(maxw,length(r)));
  rs.bbs{1} = rs.bbs{1}(r,:);
  rs.xs{1} = rs.xs{1}(:,r);
  
  
  x = rs.xs{1};
  bbs = rs.bbs{1};
  globalN = globalN + size(x,2);
  
  if isfield(cparams,'obj') && length(cparams.obj) > 0
    
    if length(data_set{i}.objects) == 0
      continue
    end
    
    curcls = find(ismember({data_set{i}.objects.class}, ...
                           cparams.obj));
    if strcmp(cparams.obj,'all')
      curcls = 1:length(data_set{i}.objects);
    end
    
    if length(curcls) == 0
      continue

    end
    
    goodbb = cat(1,data_set{i}.objects.bbox);
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
  elseif isfield(cparams,'scene_os')
    scene_bb = [1 1 size(I,2) size(I,1)];
    goods = find(getosmatrix_bb(bbs,scene_bb)>cparams.scene_os);
    x = x(:,goods);
    bbs = bbs(goods,:);
  end

  sums = sums + sum(x,2);
  curn = size(x,2);
  
  outers = outers + x*x';
  n = n + curn;
  fprintf(1,sprintf('i=%05d/%05d n = %d/%d\n',i,length(data_set), ...
                    n,globalN));
end

covstruct.n = n;
covstruct.mean = sums/n;
covstruct.c = 1./(n-1)*(outers - covstruct.mean*covstruct.mean');

covstruct.params = params;
covstruct.cparams = cparams;
 
