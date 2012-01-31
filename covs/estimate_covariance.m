function res = estimate_covariance(train_set_name, hg_size, res2)
% Estimates the covariance matrix of all windows of size hg_size
% inside PASCAL VOC set "train_set_name"
% If res2 is provided (optional), the highest probability window is
% diaplayed in each image

%Estimates the covariance matrix by going through all the data
if ~exist('train_set_name','var')
  train_set_name = 'trainval';
end

if ~exist('hg_size')
  hg_size = [7 11];
end

filer = sprintf('/nfs/baikal/tmalisie/covar_%s_%d_%d.mat',...
                train_set_name,hg_size(1),hg_size(2));
if fileexists(filer)
  load(filer)
  return;
end

addpath(genpath(pwd));

if ~exist('data_directory','var')
  data_directory = '/Users/tomasz/projects/pascal/';
end

if ~exist('dataset_directory','var')
  dataset_directory = 'VOC2007';
end

dataset_params = esvm_get_voc_dataset(dataset_directory, ...
                                      data_directory);
train_set = esvm_get_pascal_set(dataset_params, train_set_name);


%keep all windows
params = esvm_get_default_params;
params.detect_max_windows_per_exemplar = 1000000;
params.detect_save_features = 1;
params.detect_exemplar_nms_os_threshold = 1;
params.detect_exemplar_nms_os_threshold = 1;
params.detect_pyramid_padding = 0;
params.detect_add_flip = 0;

m.model.w = zeros(hg_size(1),hg_size(2),params.init_params.features());
m.model.b = 0;

n = 0;

fsize = numel(m.model.w);
sums = zeros(fsize,1);
outers = zeros(fsize,fsize);
r = randperm(length(train_set));
r = r(1:min(length(r),1000));
train_set = train_set(r);

if exist('res2','var')
tic
  invcov = pinv(res2.c);
  toc
end

for i = 1:length(train_set)
  I = toI(train_set{i});
  rs = esvm_detect(I,m,params);
  if numel(rs.xs{1})==0
    fprintf(1,'tiny image!\n');
    continue
  end

  x = cat(2,rs.xs{1}{:});

  if exist('res2','var')
    x2=bsxfun(@minus,res2.mean,x);
    tic;r=-x2'*invcov*x2;toc
    p = (diag(r));
    [aa,bb] = sort(p,'descend');
    for q = 1:length(bb)
      figure(33)
      clf
      imagesc(I)
      plot_bbox(rs.bbs{1}(bb(q),:))
      pause
    end
  end
  sums = sums + sum(x,2);
  curn = size(x,2);
  outers = outers + cov(x*x');
  n = n + curn;
  fprintf(1,sprintf('i=%05d/%05d\n',i,length(train_set)));
end

res.c = 1./(n-1)*(outers - sums'*sums/n);
res.mean = sums/n;
res.params = params;
res.train_set_name = train_set_name;
res.hg_size = size(m.model.w);
res.evecicons = show_cov(res);
save(filer,'res');

