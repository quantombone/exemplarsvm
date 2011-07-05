function mining_params = get_default_mining_params
%% Return the default detection/training parameters
% Tomasz Malisiewicz (tomasz@cmu.edu)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Sliding window detection parameters 
%(using during training and testing)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Turn on detection on image flips
mining_params.FLIP_LR = 1;

%Levels-per-octave defines how many levels between 2x sizes in pyramid
mining_params.lpo = 10;

%By default dont save feature vectors of detections (training turns
%this on automatically)
mining_params.SAVE_SVS = 0;

%default detection threshold (negative margin makes most sense for
%SVM-trained detectors)
mining_params.thresh = -1;

%Maximum #windows per exemplar (per image) to keep
mining_params.TOPK = 10;

%if less than 1.0, then we apply nms to detections so that we don't have
%too many redundant windows [defaults to 0.5]
%NOTE: mining is much faster if this is turned off!
mining_params.NMS_OS = 0.5;

%How much we pad the pyramid (to let detections fall outside the image)
mining_params.pyramid_padder = 5;

%The maximum scale to consdider in the feature pyramid
mining_params.MAXSCALE = 1.0;

%The minimum scale to consider in the feature pyramid
mining_params.MINSCALE = .01;

%Only keep detections that roughly match the entire image
%if greater than 0, then only keep dets that have this OS with the
%entire testing image
mining_params.MIN_SCENE_OS = 0.0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Training/Mining parameters %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%maximum number of mining iterations
mining_params.MAX_MINE_ITERATIONS = 100;

%maximum number of image-scans during training
mining_params.MAX_TOTAL_MINED_IMAGES = 2500;

%Maximum number of negatives to mine before SVM kicks in (this
%defines one iteration)
mining_params.MAX_WINDOWS_BEFORE_SVM = 1000;

%Maximum number of violating images before SVM is trained with current cache
mining_params.MAX_IMAGES_BEFORE_SVM = 400;

%ICCV11 constant
mining_params.SVMC = .01; %% regularize more with .0001;

%when mining, we keep the N negative support vectors as well as
%some more beyond the -1 threshold (alpha*N), but no more than 1000
mining_params.beyond_nsv_multiplier = 3;
mining_params.max_negatives = 2000;

%Queue mode can be one of: {'onepass','cycle-violators','front-violators'}
mining_params.queue_mode = 'onepass';

% if non-zero, sets weight of positives such that positives and
%negatives are treated equally
mining_params.BALANCE_POSITIVES = 0;

% if non-zero, perform learning in dominant-gradient space
mining_params.DOMINANT_GRADIENT_PROJECTION = 0;
mining_params.DOMINANT_GRADIENT_PROJECTION_K = 2;
mining_params.DO_PCA = 0;
mining_params.PCA_K = 300;
mining_params.A_FROM_POSITIVES = 0;

%if enabled, we must extract training negatives from positives and
%held-out positives
mining_params.extract_negatives = 0;
%mining_params.GET_GT_OS = 0;

%if enabled, we dump learning images into results directory
mining_params.dump_images = 0;

%if enabled, we dump the last image
mining_params.dump_last_image = 1;

%experimental flag to skip the mining process
mining_params.skip_mine = 0;
