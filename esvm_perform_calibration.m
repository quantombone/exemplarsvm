function M = esvm_perform_calibration(grid, models, params, CACHE_FILES)
%% 1. Perform LABOO calibration procedure and 2. Learn a combination
%matrix M which multiplexes the detection results (by compiling
%co-occurrence statistics on true positives) 

% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).


if ~exist('CACHE_FILES','var')
  CACHE_FILES = 0;
end

%% Perform calibration
betas = esvm_perform_platt_calibration(grid, models, ...
                                       params, CACHE_FILES);

if ~(isfield(params,'SKIP_M') && params.SKIP_M==1)
  %% Estimate the co-occurrence matrix M
  [M] = esvm_estimate_M(grid, models, params, CACHE_FILES);
end

M.betas = betas;
