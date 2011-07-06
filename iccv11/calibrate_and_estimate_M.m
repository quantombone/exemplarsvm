function M = calibrate_and_estimate_M(dataset_params, models, grid, ...
                                      cur_set, CACHE_FILES)
%% 1. Perform LABOO calibration procedure and 2. Learn a combination
%matrix M which multiplexes the detection results (by compiling
%co-occurrence statistics on true positives) 
% Tomasz Malisiewicz (tomasz@cmu.edu)

if ~exist('CACHE_FILES','var')
  CACHE_FILES = 0;
end

%% Perform calibration
betas = perform_calibration(dataset_params, models, grid, ...
                            cur_set, CACHE_FILES);

neighbor_thresh = 0.5;
count_thresh = 0.5;

%% Estimate the co-occurrence matrix M
[M] = estimate_M(dataset_params, models, grid, betas, ...
                 neighbor_thresh, ...
                 count_thresh, CACHE_FILES);

M.betas = betas;
