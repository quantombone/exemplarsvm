%Script which computes the global feature covariance matrix

load ~/projects/pascal/VOC2007/trainval.mat 
basedir = 'VOC2007';

basedir = sprintf('%s/%s','/csail/vision-torralba6/people/tomasz/covs/',basedir);
if ~exist(basedir,'dir')
  mkdir(basedir);
end

N = length(data_set);
inds = do_partition(1:N, 100);
params.hg_size = [12 12];

myRandomize;
r = randperm(length(inds));
inds = inds(r);
for i = 1:length(inds)

  filer = sprintf('%s/subcov_%05d.mat',...
                  basedir,r(i));
  
  lockfile = [filer '.lockfile'];
  
  if fileexists(filer) || mymkdir_dist(lockfile) == 0
    fprintf(1,'FILE EXISTS, loading from %s\n',filer);
    continue
  end
  
  covstruct = estimate_covariance(data_set(inds{i}), params);
  save(filer,'covstruct');
  try
    rmdir(lockfile);
  catch
  end
end
