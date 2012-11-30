%Script which computes covariances for a few different datasets
%sets = 'trainval'

%data_set = dataset.image_files;
%data_set = trainval.data_set;
%if ~exist('data_set','var')
%  load ~/projects/pascal/VOC2007/trainval.mat 
%end


%cls = {'bus','car','train','cow','sheep','chair','sofa'};
cls = {'all'};
%cls = {'cat','bicycle','motorbike','tvmonitor','bottle'};
%cls = {'dog','diningtable','aeroplane','boat'};
%cls = {'person','bird','horse','pottedplant'};

for a = 12
  for b = 12
    params.hg_size = [a b];
    params.obj_os = 0;
    for i = 1:length(cls)  

      fprintf(1,'Set is %s, hg_size = [%d,%d],\n',...
              cls{i}, params.hg_size(1), params.hg_size(2));
      params.titler = [cls{i} '-' num2str(params.obj_os)];;
      %params.obj = cls{i};
      %params.obj_os = .5;
      
      %params.obj = 'cow';
      %params.obj_os = .2;

      [res] = estimate_covariance(data_set, params);
      
    end
  end
end
