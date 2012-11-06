function [model] = construct_scene_models(data_set, covstruct, N)
%Construct a set of 'scene' models (full image bounding box) from a
%dataset of images and a covariance matrix
% Inputs: 
%   data_set: a cell array of images 
%   covstruct: the HOG covariance matrix 
%   N: the number of equally spaced images to take [defaults to all images]
%
% Tomasz Malisiewicz (tomasz@csail.mit.edu)

if ~exist('N','var')
  inds = 1:length(data_set);
else
  inds = unique(round(linspace(1,length(data_set),N)));
end

fprintf(1,'Construction %d scene models\n',length(inds));

%Place a bounding box as the image
data_set2 = construct_scene_dataset(data_set(inds));

%Learn pre-model just to get features size
pre_model = learnAllEG(data_set2(1), '' , covstruct);

%Learn all EvG models
model = learnAllEG(data_set2, '' , covstruct, pre_model.models{1}.hg_size(1:2));

%Do not do flipping during detection
model.params.detect_add_flip = 0;

%Take at most one detection per image
model.params.detect_max_windows_per_exemplar = 1;

%Maximum image size, so that we get big images
model.params.max_image_size = 200;

%save the subset from big dataset
model.inds = inds;