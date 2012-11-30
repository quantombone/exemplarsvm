%Script which computes the global feature covariance matrix

load /csail/vision-videolabelme/people/tomasz/pascal/VOC2007/trainval.mat 
%load ~/projects/pascal/VOC2007/trainval.mat 
basedir = 'VOC2007';

basedir = sprintf('%s/%s','/csail/vision-torralba6/people/tomasz/covs/',basedir);
if ~exist(basedir,'dir')
  mkdir(basedir);
end

N = length(data_set);
inds = do_partition(1:N, 50);
params.hg_size = [12 12];

myRandomize;
r = randperm(length(inds));
inds = inds(r);

%outers = zeros(12*12*31,12*12*31);
%inners = zeros(12*12*31,1);
%ns = 0;

for i = 1:length(inds)

  filer = sprintf('%s/subcov_%05d.mat',...
                  basedir,r(i));
  
  lockfile = [filer '.lockfile'];
  
  if fileexists(filer) || mymkdir_dist(lockfile) == 0
    fprintf(1,'FILE EXISTS, loading from %s\n',filer);
    % tic
    % res = load(filer);
    % toc
    % res.covstruct.c = res.covstruct.c*(res.covstruct.n-1)/(res.covstruct.n);
    
    % inners = inners + res.covstruct.n*res.covstruct.mean;
    % outers = outers + (res.covstruct.n)*(res.covstruct.c + res.covstruct.mean*res.covstruct.mean');
    % ns = ns + res.covstruct.n;
    continue
  end
  
  covstruct = estimate_covariance(data_set(inds{i}), params);
  save(filer,'covstruct');
  try
    rmdir(lockfile);
  catch
  end
end

%mu = inners/ns;
%c = 1/(ns-1)*(outers-mu*mu');