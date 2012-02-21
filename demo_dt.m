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

classes = {'sink'};
classes = {'laptop'};
classes = {'sofa'};
classes = {'chair'};
classes = {'toilet'};
classes = {'table'};
for i = 1:length(classes)
  model = learnDalalTriggs(data_set, classes{i});
  savemodel(model,classes{i});
  %save(sprintf('/csail/vision-videolabelme/databases/SUN11/dt-models/%s.mat',classes{i}),'model');

end

return;

%Create a held-out test-set
test_set = esvm_generate_dataset(20,20,'circle');

%Get detection boxes by applying model to test_set
boxes = applyModel(test_set, model);

%Get the AP curve by comparing against ground_truth
results = evaluateModel(test_set, boxes);

%Show the top detections on the test_set
showTopDetections(test_set, boxes);
