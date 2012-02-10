%% A showcase of the ExemplarSVM Library 1.0

%Generate a training data set with class name 'circle'
%data_set = esvm_generate_dataset(10,10,'circle');

% image_set = list_files_in_directory('~/av/demos/cube/');
% data_set = esvm_generate_sketchup_dataset(image_set, ...
%                                           'cube');
% negative_set = list_files_in_directory(['/projects/pascal/VOC2007/' ...
%                     'JPEGImages/']);
% negative_set = negative_set(1:1000);
% data_set = cat(2,data_set,negative_set);

load ~/projects/pascal/VOC2007/trainval.mat 
model = learnDalalTriggs(data_set,'bottle');

load ~/projects/pascal/VOC2007/test.mat
test_set = data_set;
%Get detection boxes by applying model to test_set
boxes = applyModel(test_set, model);

%Get the AP curve by comparing against ground_truth
results = evaluateModel(test_set, boxes, model);

%Show the top detections on the test_set
showTopDetections(test_set, boxes);

return;
%return;
%Learn an ensemble of ExemplarSVMs1
%model = learnExemplarSVMs(data_set,'circle');
model = learnDalalTriggs(data_set,'cube');
return;

%Create a held-out test-set
test_set = esvm_generate_dataset(20,20,'circle');

%Get detection boxes by applying model to test_set
boxes = applyModel(test_set, model);

%Get the AP curve by comparing against ground_truth
results = evaluateModel(test_set, boxes, model);

%Show the top detections on the test_set
showTopDetections(test_set, boxes);
