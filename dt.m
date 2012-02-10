function [model,boxes,results] = dt(cls)
addpath(genpath(pwd))

load('~/projects/pascal/VOC2007/trainval.mat','data_set')
%params.localdir = '/nfs/baikal/tmalisie/ldt/';
model = learnDalalTriggs(data_set,cls);

load('~/projects/pascal/VOC2007/test.mat','data_set')
test_set = data_set;

%Get detection boxes by applying model to test_set
boxes = applyModel(test_set, model);

%Get the AP curve by comparing against ground_truth
results = evaluateModel(test_set, boxes, model);

%Show the top detections on the test_set
showTopDetections(test_set, boxes, model);
