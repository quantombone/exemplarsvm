%Script which computes covariances for a few different datasets
sets = 'trainval'
cls = {'bus','car','train','cow','sheep','chair','sofa'}

hg_size = [8 8];

for i = 1:length(cls)
  s = [sets '+' cls{i}];
  fprintf(1,'Set is %s, hg_size = [%d,%d],\n',s,hg_size(1),hg_size(2));
  [res] = test_covariance(s,[8 8]);
end
