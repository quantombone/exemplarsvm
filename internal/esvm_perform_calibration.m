function model = esvm_perform_calibration(data_set,model)
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

%Apply model to all of the images in the data set
if ~isfield(model,'boxes')
  model.boxes = applyModel(data_set,model,.01);
end

%% Perform calibration
model = ...
    esvm_perform_platt_calibration(data_set, model);

%% Estimate the co-occurrence matrix M
if ~(isfield(model.params,'SKIP_M'))
  model = esvm_estimate_M(data_set, model);
end

%concatenate results
%M.betas = betas;
%M.aps = aps;

%model.M = M;