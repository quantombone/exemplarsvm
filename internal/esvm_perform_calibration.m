function M = esvm_perform_calibration(grid, val_set, models, params)
% 1. Perform LABOO calibration procedure and 2. Learn a combination
% matrix M which multiplexes the detection results (by compiling
% co-occurrence statistics on true positives) 
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

%% Perform calibration
betas = esvm_perform_platt_calibration(grid, val_set, models, ...
                                       params);

%% Estimate the co-occurrence matrix M
if ~(isfield(params,'SKIP_M') && params.SKIP_M==1)
  M = esvm_estimate_M(grid, models, params);
end

%concatenate results
M.betas = betas;
