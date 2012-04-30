%Script which computes covariances for a few different datasets
%sets = 'trainval'
if ~exist('data_set','var')
  load ~/projects/pascal/VOC2007/trainval.mat 
end
cls = {'bus','car','train','cow','sheep','chair','sofa'};
cls = {'all'};

for a = 8:12
  for b = 8:12
    params.hg_size = [a b];
    for i = 1:length(cls)  
      %[cur_pos_set, cur_neg_set] = split_sets(data_set, ...
      %                                                cls{i});
      fprintf(1,'Set is %s, hg_size = [%d,%d],\n',...
              cls{i}, params.hg_size(1), params.hg_size(2));
      params.titler = ['all'];
      [res] = estimate_covariance(data_set, params);
      
    end
  end
end
