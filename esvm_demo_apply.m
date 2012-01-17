% Demo: Applying an ensemble of Exemplar-SVMs
% This function can generate a nice HTML page by calling: publish('esvm_quick_demo.m','html')
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

addpath(genpath(pwd));
cls = 'voc2007-bus';

%% Download and load pre-trained VOC2007 bus models
[models, M, test_set] = esvm_download_models(cls);

%If VOC is locally installed, let models point to local images, not URLs
local_dir = '/Users/tomasz/projects/pascal/VOCdevkit/';
if isdir(local_dir)
  [models] = esvm_update_voc_models(models, local_dir);
end

%% Load one image, and apply bus detector
I1 = imread('000858.jpg');
esvm_demo_apply_exemplars(I1, models, M);

%% Create an array of images, and apply bus detector
I2 = imread('009021.jpg');
I3 = imread('009704.jpg');
Iarray = {I2, I3};
esvm_demo_apply_exemplars(Iarray, models, M);

%% Create URL-based set of images, and apply bus detector
%Show image filepaths
cat(1,test_set{1:5})
%Apply detector
esvm_demo_apply_exemplars(test_set(1:5), models, M);

%% Set image path directory, and apply bus detector
Idirectory = '/v2/SUN/Images/b/bus_depot/outdoor/';
if isdir(Idirectory)
  Ilist = get_file_list(Idirectory);
  Ilist = Ilist(1:min(5,length(Ilist)));

  %Virtually resize all images to 200 max dim
  MAXDIM = 500;
  Ilist = cellfun2(@(x)(@()imresize_max(convert_to_I(x),MAXDIM)), ...
                   Ilist);
  esvm_demo_apply_exemplars(Ilist, models, M);
else
  fprintf(1,'Note: not applying because %s is not a directory\n',...
          Idirectory);
end
