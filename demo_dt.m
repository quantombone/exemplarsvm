%% A showcase of the ExemplarSVM Library 1.0

%Generate a training data set with class name 'circle'
%data_set = esvm_generate_dataset(10,10,'circle');

load ~/projects/pascal/VOC2007/trainval.mat 

model = learnDalalTriggs(data_set,'chair');
return;

%Create a held-out test-set
test_set = esvm_generate_dataset(20,20,'circle');

%Get detection boxes by applying model to test_set
boxes = applyModel(test_set, model);

%Get the AP curve by comparing against ground_truth
results = evaluateModel(test_set, boxes);

%Show the top detections on the test_set
showTopDetections(test_set, boxes);
