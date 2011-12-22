function M = esvm_perform_calibration(dataset_params, models, grid, ...
                                      cur_set, CACHE_FILES)
%% 1. Perform LABOO calibration procedure and 2. Learn a combination
%matrix M which multiplexes the detection results (by compiling
%co-occurrence statistics on true positives) 
% Tomasz Malisiewicz (tomasz@cmu.edu)

if ~exist('CACHE_FILES','var')
  CACHE_FILES = 0;
end

%% Perform calibration
betas = esvm_perform_platt_calibration(dataset_params, models, grid, ...
                                       cur_set, CACHE_FILES);

if ~(isfield(dataset_params,'SKIP_M') && dataset_params.SKIP_M==1)
  %% Estimate the co-occurrence matrix M
  [M] = esvm_estimate_M(dataset_params, models, grid, betas, CACHE_FILES);
end

M.betas = betas;
