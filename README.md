Here is the readme for the ICCV2011 exemplar code 
Author: Tomasz Malisiewicz (tomasz@cmu.edu)
Abhinav was here!`
NOTE: do not distribute, the camera-ready version will be ready
sometime in March, since this might change a lot

It is a combination of:
libsvm-3.0,
fast blas convolution code (from voc-release-4.0), 
hog code (from voc-release-3.1), 
localization within a pyramid code (my variant of pedro's code from voc-release-3.1),
initialize region from user-provided bb,
mining hard-negatives with several mining queues (using linear SVM),
non-maximum suppression to clean up detections,
visualization of detection boxes,
screenshot capture tool

% compiling stuff
% mex -O -lblas fconvblas.cc

%%%% Compilation instructions

%liblinear needs to be compiled, go inside its matlab directory and compile away by editing the Makefile

Quick Start Guide:

%%% VOC README
%1. dump out model files for a VOC category
>> initialize_voc_exemplars('motorcycle');

%2. train all exemplars (all exemplars in exemplar directory)
>> train_all_exemplars;

%3. perform calibration
>> models = load_all_exemplars;
>> grid = load_result_grid(models);
>> betas = perform_calibration(grid,models);

%4. evaluate results on PASCAL VOC comp3 task
>> results = evaluate_pascal_voc_grid(models, grid, 'test', betas);

%5. show top detections on testset
>> show_top_dets(models,grid,'test',betas);

other one: % 1a. Start by setting the workspace path inside file VOCcode/VOCinit.m
devkitroot='~/esvm-local/';

% 1b. Make sure paths are added
>> addpath(genpath(pwd))

% 2. Get exemplars from screenshots;
>> I = capture_screen(10);
>> initialize_some_exemplars(I);

% 3. Learn models
>> train_all_exemplars;

% 4. Apply on all of PASCAL
>> apply_voc_exemplars;

% 5. show results
>> grid = load_results_grid;
>> betas = perform_calibration(models,grid); (this can also visualize the top detections)


%%NOTES:
if you are having problems with things not working, make sure everything is compiled in each directory

if you have an svm function problem, type which svm.m and make sure you aren't using the bioinformatics (built-in) matlab svm function

```ruby
require 'redcarpet'
markdown = Redcarpet.new("Hello World!")
puts markdown.to_html
```