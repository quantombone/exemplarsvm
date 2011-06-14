function mining_params = get_default_mining_params
%Return the default mining and SVM training parameters for the
%exemplar svm and the visual memex network
% Tomasz Malisiewicz (tomasz@cmu.edu)

%Turn on detection on image flips
mining_params.FLIP_LR = 1;

%maximum number of mining iterations
mining_params.MAXITER = 100;

%ICCV11 constant
mining_params.SVMC = .01;

%regularize more here
%mining_params.SVMC = .0001;

%if enabled, during mining we use every other image as validation
%where we keep as many top detection from the validation set as are
%considered in the SVM training cache
mining_params.alternate_validation = 0;

%maximum number of image-scans during training
mining_params.MAX_TOTAL_MINED_IMAGES = 2500;

%if enabled, we dump learning images into results directory
mining_params.dump_images = 0;

%if enabled, we dump the last image
mining_params.dump_last_image = 1;

%maximum #windows per image (per exemplar) to mine
%mining_params.MAX_WINDOWS_PER_IMAGE = 100;

%Levels-per-octave defines how many levels between 2x sizes in pyramid
mining_params.lpo = 10;

%By default dont save feature vectors of detections
mining_params.SAVE_SVS = 0;

%default detection threshold
mining_params.thresh = -1;
mining_params.TOPK = 50;

%How much we pad the pyramid (to let detections fall outside the image)
mining_params.pyramid_padder = 5;

%Maximum number of negatives to mine before SVM kicks in (this
%defines one iteration)
mining_params.MAX_WINDOWS_BEFORE_SVM = 1000;

%Maximum number of violating images before SVM kicks in (another
%way to define one iteration)
%NOTE: this variable breaks how I consider iterations, so make it
%really large so it never fires
mining_params.MAX_IMAGES_BEFORE_SVM = 400;

%At this cutoff, we switch thresholds from basically everything to
%a reduced threshold
%mining_params.early_late_cutoff = 3;
%mining_params.early_detection_threshold = -10000;
%mining_params.late_detection_threshold = -1.05;
%mining_params.detection_threshold = mining_params.late_detection_threshold;

%when mining, we keep the N negative support vectors as well as
%some more beyond the -1 threshold (alpha*N), but no more than 1000
mining_params.beyond_nsv_multiplier = 3;
mining_params.max_negatives = 2000;

%%if less than 1.0, then we apply nms to detections so that we don't have
%%too many redundant windows
mining_params.NMS_MINES_OS = 0.5;

%if non-zero, then skip detection at any objects
mining_params.SKIP_GTS_ABOVE_THIS_OS = 10.0;

%if non-zero, add these detections as positives
%mining_params.ASSIMILATE_GTS_ABOVE_THIS_OS = 10;

%Queue mode can be one of: {'onepass','cycle-violators','front-violators'}
mining_params.queue_mode = 'onepass';

%% if non-zero, sets weight of positives such that positives and
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
mining_params.GET_GT_OS = 0;
mining_params.MINE_MODE = 0;

%experimental flag to skip the mining process
mining_params.skip_mine = 0;
