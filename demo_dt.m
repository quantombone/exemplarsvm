%% A showcase of the ExemplarSVM Library 1.0

%Generate a training data set with class name 'circle'
%data_set = esvm_generate_dataset(10,10,'circle');

%load ~/projects/pascal/VOC2007/trainval.mat 
addpath(genpath(pwd))
if ~exist('data_set','var')
  load /csail/vision-videolabelme/databases/SUN11/trainval.mat 
end


classes = {'chair','laptop','table',...
           'television','sink','sofa', ...
           'toilet' };

classes = {'door','bottle','bowl','painting','pillow', ...
           'faucet','window','faucet','cushion','curtain','cabinet','floor','ceiling','plant','can','column','pot','lamp','sculpture','book','box','bottle','sign','flower','bookshelf','stair','screen','glass','carpet','towel','outlet','stool','cup'};
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
  model = learnDalalTriggsSave(data_set, classes{i});
end
matlabpool close
%  savemodel(models{i},classes{i});
