function params = esvm_get_default_params
% Return the default Exemplar-SVM detection/training parameters
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved. 
%
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Sliding window detection parameters 
%(using during training and testing)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Turn on image flips for detection/training. If enabled, processing
%happes on each image as well as its left-right flipped version.
params.detect_add_flip = 1;

%Levels-per-octave defines how many levels between 2x sizes in pyramid
%(denser pyramids will have more windows and thus be slower for
%detection/training)
params.detect_levels_per_octave = 10;

%By default dont save feature vectors of detections (training turns
%this on automatically)
params.detect_save_features = 0;

%Default detection threshold (negative margin makes most sense for
%SVM-trained detectors).  Only keep detections for detection/training
%that fall above this threshold.
params.detect_keep_threshold = -1;

%Maximum #windows per exemplar (per image) to keep
params.detect_max_windows_per_exemplar = 100;

%Determines if NMS (Non-maximum suppression) should be used to
%prune highly overlapping, redundant, detections.
%If less than 1.0, then we apply nms to detections so that we don't have
%too many redundant windows [defaults to 0.5]
%NOTE: mining is much faster if this is turned off!
params.detect_exemplar_nms_os_threshold = 0.5;

%How much we pad the pyramid (to let detections fall outside the image)
params.detect_pyramid_padding = 5;

%The maximum scale to consdider in the feature pyramid
params.detect_max_scale = 1.0;

%The minimum scale to consider in the feature pyramid
params.detect_min_scale = .01;

%Only keep detections that have sufficient overlap with the input's
%global bounding box.  If greater than 0, then only keep detections
%that have this OS with the entire input image.
params.detect_min_scene_os = 0.0;

% Choose the number of images to process in each chunk for detection.
% This parameters tells us how many images each core will process at
% at time before saving results.  A higher number of images per chunk
% means there will be less constant access to hard disk by separate
% processes than if images per chunk was 1.
params.detect_images_per_chunk = 4;

%NOTE: If the number of specified models is greater than 20, use the
%BLOCK-based method
params.max_models_before_block_method = 20;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Training/Mining parameters %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%The maximum number of negatives to keep in the cache while training.
params.train_max_negatives_in_cache = 2000;

%Maximum global number of mining iterations, where an iteration is
%when queue fills with max_windows_before_svm detections or
%max_windows_before_svm images have been processed
params.train_max_mine_iterations = 100;

%Maximum TOTAL number of image accesses from the mining queue
params.train_max_mined_images = 2500;

%Maximum number of negatives to mine before SVM kicks in (this
%defines one iteration)
params.train_max_windows_per_iteration = 1000;

%Maximum number of violating images before SVM is trained with current cache
params.train_max_images_per_iteration = 400;

%NOTE: I don't think these fields are being used since I set the
%global detection threshold to -1.
%when mining, we keep the N negative support vectors as well as
%some more beyond the -1 threshold (alpha*N), but no more than
%1000, where alpha is the "keep nsv multiplier"
params.train_keep_nsv_multiplier = 3;

%ICCV11 constant for SVM learning
params.train_svm_c = .01; %% regularize more with .0001;

%The constant which tells us the weight in front of the positives
%during SVM learning
params.train_positives_constant = 50;

%The svm update equation
params.training_function = @esvm_update_svm;

%experimental flag to skip the mining process
%params.train_skip_mining = 0;

%Mining Queue mode can be one of:
% {'onepass','cycle-violators','front-violators'}
%
% onepass: a single pass through the mining queue
% cycle-violators: discard non-firing images, and place violators
%     (images with detections) at the end of the mining queue
% front-violators: same as above but place violators at front of
% queue
% The last two modes require a termination condition such as
% (train_max_mined_images) so that learning doesn't loop
% indefinitely
params.queue_mode = 'onepass';

% if non-zero, sets weight of positives such that positives and
%negatives are treated equally
%params.BALANCE_POSITIVES = 0;

% % NOTE: this stuff is experimental and currently disabled (see
% % do_svm.m). The goal was to perform dimensionality reduction
% % before the learning process.
% % If non-zero, perform learning in dominant-gradient space
% params.DOMINANT_GRADIENT_PROJECTION = 0;
% % The dimensionality of the local max-gradient descriptor
% params.DOMINANT_GRADIENT_PROJECTION_K = 2;
% % If enabled, do PCA on training data right before SVM (this
% % automatically converts the result to a descriptor in the RAW feature
% % space)
% params.DO_PCA = 0;
% % The degree of the PCA.
% params.PCA_K = 300;
% % If enabled, only do PCA from the positives (so the subspace is what
% % spans the positive examples)
% params.A_FROM_POSITIVES = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Cross-Validation(Calibration) parameters %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%The default calibration threshold (will prune away all windows
%that score below this number)
params.calibration_threshold = -1;

%The M-matrix estimation parameters
params.calibration_count_thresh = .5;
params.calibration_neighbor_thresh = .5;

%If enabled, use M-matrix calibration, but then propagate results
%onto best local "raw" exemplar-based detection
params.calibration_propagate_onto_raw = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Saving and Output parameters %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%if enabled, we dump learning images into results directory
params.dump_images = 0;

%if enabled, we dump the last image
params.dump_last_image = 1;

%NOTE: I'm not sure in which sections should this be defined.  By
%default, NN mode is turned off and we assume per-exemplar SVMs
params.nnmode = '';

%Be default, we use an SVM
params.dfun = 0;

%Set feature-computation function function
init_params.features = @esvm_hog;
init_params.sbin = 8;
init_params.goal_ncells = 100;
init_params.MAXDIM = 12;
init_params.init_function = @esvm_initialize_goalsize_exemplar;

params.init_params = init_params;
params.model_type = 'exemplar';

%Initialize loading/saving directories to being empty (turned off)
params.localdir = '';

% For dalal triggs, we need an update threshold
params.latent_os_thresh = 0.7;
params.latent_iterations = 10;

%Information about where we mine images from
%TODO(TJM): these need to be utilized
params.mine_from_negatives = 0;
params.mine_from_positives = 1;

%If enabled, skips objects durning mining
params.mine_skip_objects = 1;
params.mine_skip_objects_os = .5;

%params.skip_self_mine = 1;

%The threshold for evaluation
params.evaluation_minoverlap = .5;
