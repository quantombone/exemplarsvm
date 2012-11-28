function res = load_covstruct
%Load the covariance matrix

[~,a]=unix('echo $HOST');
if strfind(a,'vision')>0
  covs = '/csail/vision-torralba6/people/tomasz/covs/pascal_trainval_12_12.mat';
else
  covs='~/projects/covs/pascal_trainval_12_12.mat';
end
load(covs);