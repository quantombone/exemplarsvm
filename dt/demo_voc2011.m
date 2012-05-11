%% A showcase of the ExemplarSVM Library 1.0

addpath(genpath(pwd))
if ~exist('data_set','var')
  load /csail/vision-videolabelme/people/tomasz/pascal/VOC2011/trainval.mat
end

classes = {'person','chair','dog','cat','bus','tvmonitor','diningtable','bird','horse','bicycle','motorbike','car','pottedplant','aeroplane','boat','cow','sheep','bottle','sofa','train'};

% classes = {'sink'};
% classes = {'laptop'};
% classes = {'sofa'};
% classes = {'chair'};
% classes = {'toilet'};
% classes = {'table'};
myRandomize;
classes = classes(randperm(length(classes)));
matlabpool open
parfor i = 1:length(classes)
  model = learnDalalTriggsSaveVOC2011(data_set, classes{i});
end
matlabpool close
%  savemodel(models{i},classes{i});
