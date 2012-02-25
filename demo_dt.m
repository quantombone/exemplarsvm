%% A showcase of the ExemplarSVM Library 1.0

%Generate a training data set with class name 'circle'
%data_set = esvm_generate_dataset(10,10,'circle');

%load ~/projects/pascal/VOC2007/trainval.mat 

if ~exist('data_set','var')
  load /csail/vision-videolabelme/databases/SUN11/trainval.mat 
end

classes = {'chair','laptop','table',...
           'television','sink','sofa', ...
           'toilet' };

classes = {'door','oven','bottle','bowl','painting','pillow', ...
           'faucet','refrigerator','microwave','window','faucet','cushion','curtain','cabinet','floor','ceiling','plant','can','column','pot','lamp','sculpture','book','box','bottle','sign','flower','bookshelf','stair','screen','glass','carpet','towel','outlet','stool','cup'};

% classes = {'sink'};
% classes = {'laptop'};
% classes = {'sofa'};
% classes = {'chair'};
% classes = {'toilet'};
% classes = {'table'};
clear models;
parfor i = 1:length(classes)
  model = learnDalalTriggs(data_set, classes{i});
end
%  savemodel(models{i},classes{i});
