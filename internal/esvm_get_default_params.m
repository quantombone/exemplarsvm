function mining_params = esvm_get_default_params
%% Return the default detection/training parameters
% Tomasz Malisiewicz (tomasz@cmu.edu)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Sliding window detection parameters 
%(using during training and testing)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Turn on image flips for detection/training. If enabled, processing
%happes on each image as well as its left-right flipped version.
mining_params.detect_add_flip = 1;

%Levels-per-octave defines how many levels between 2x sizes in pyramid
%(denser pyramids will have more windows and thus be slower for
%detection/training)
mining_params.detect_levels_per_octave = 10;

%By default dont save feature vectors of detections (training turns
%this on automatically)
mining_params.detect_save_features = 0;

%Default detection threshold (negative margin makes most sense for
%SVM-trained detectors).  Only keep detections for detection/training
%that fall above this threshold.
mining_params.detect_keep_threshold = -1;

%Maximum #windows per exemplar (per image) to keep
mining_params.detect_max_windows_per_exemplar = 10;

%Determines if NMS (Non-maximum suppression) should be used to
%prune highly overlapping, redundant, detections.
%If less than 1.0, then we apply nms to detections so that we don't have
%too many redundant windows [defaults to 0.5]
%NOTE: mining is much faster if this is turned off!
mining_params.NMS_OS = 0.5;

%How much we pad the pyramid (to let detections fall outside the image)
mining_params.pyramid_padder = 5;

%The maximum scale to consdider in the feature pyramid
mining_params.MAXSCALE = 1.0;

%The minimum scale to consider in the feature pyramid
mining_params.MINSCALE = .01;

%Only keep detections that have sufficient overlap with the input's
%global bounding box.  If greater than 0, then only keep detections
%that have this OS with the entire input image.
mining_params.MIN_SCENE_OS = 0.0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Training/Mining parameters %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%The maximum number of negatives to keep in the cache while training.
mining_params.max_negatives = 2000;

%Maximum global number of mining iterations, where an iteration is
%when queue fills with max_windows_before_svm detections or
%max_windows_before_svm images have been processed
mining_params.MAX_MINE_ITERATIONS = 100;

%Maximum TOTAL number of image accesses from the mining queue
mining_params.MAX_TOTAL_MINED_IMAGES = 2500;

%Maximum number of negatives to mine before SVM kicks in (this
%defines one iteration)
mining_params.MAX_WINDOWS_BEFORE_SVM = 1000;

%Maximum number of violating images before SVM is trained with current cache
mining_params.MAX_IMAGES_BEFORE_SVM = 400;

%ICCV11 constant
mining_params.SVMC = .01; %% regularize more with .0001;

%NOTE: I don't think these fields are being used since I set the
%global detection threshold to -1.
%when mining, we keep the N negative support vectors as well as
%some more beyond the -1 threshold (alpha*N), but no more than
%1000, where alpha is the beyond_nsv_multiplier
mining_params.beyond_nsv_multiplier = 3;

%Mining Queue mode can be one of:
% {'onepass','cycle-violators','front-violators'}
%
% onepass: a single pass through the mining queue
% cycle-violators: discard non-firing images, and place violators
%     (images with detections) at the end of the mining queue
% front-violators: same as above but place violators at front of
% queue
% The last two modes require a termination condition such as
% (MAX_TOTAL_MINED_IMAGES) so that learning doesn't loop
% indefinitely
mining_params.queue_mode = 'onepass';

%The constant which tells us the weight in front of the positives
%during SVM learning
mining_params.POSITIVE_CONSTANT = 50;

% if non-zero, sets weight of positives such that positives and
%negatives are treated equally
mining_params.BALANCE_POSITIVES = 0;

% NOTE: this stuff is experimental and currently disabled (see
% do_svm.m). The goal was to perform dimensionality reduction
% before the learning process.
% If non-zero, perform learning in dominant-gradient space
mining_params.DOMINANT_GRADIENT_PROJECTION = 0;
% The dimensionality of the local max-gradient descriptor
mining_params.DOMINANT_GRADIENT_PROJECTION_K = 2;
% If enabled, do PCA on training data right before SVM (this
% automatically converts the result to a descriptor in the RAW feature
% space)
mining_params.DO_PCA = 0;
% The degree of the PCA.
mining_params.PCA_K = 300;
% If enabled, only do PCA from the positives (so the subspace is what
% spans the positive examples)
mining_params.A_FROM_POSITIVES = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Cross-Validation(Calibration) parameters %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%The default calibration threshold (will prune away all windows
%that score below this number)
mining_params.calibration_threshold = -1;

%The M-estimation parameters
mining_params.calibration_count_thresh = .5;
mining_params.calibration_neighbor_thresh = .5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Saving Output parameters %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%if enabled, we must extract training negatives from positives and
%held-out positives (not used for iccv11/thesis runs)
mining_params.extract_negatives = 0;

%if enabled, we dump learning images into results directory
mining_params.dump_images = 0;

%if enabled, we dump the last image
mining_params.dump_last_image = 1;

%experimental flag to skip the mining process
mining_params.skip_mine = 0;

%NOTE: I'm not sure in which sections should this be defined.  By
%default, NN mode is turned off and we assume per-exemplar learned
%W's.
mining_params.nnmode = '';

%Be default, we use an SVM
mining_params.dfun = 0;

% Choose the number of images to process in each chunk for detection.
% This parameters tells us how many images each core will process at
% at time before saving results.  A higher number of images per chunk
% means there will be less constant access to hard disk by separate
% processes than if images per chunk was 1.
mining_params.NIMS_PER_CHUNK = 4;
