function esvm_apply_and_show_exemplars(imageset, models, M, params)
% We apply the ensemble of Exemplar-SVMs represented by [models,M] onto
% the sequence of images [imageset], and display images without
% saving anything.  Takes as input optional [params], which default
% to the default ones.
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

%see: esvm_demo_apply

if nargin < 2
  fprintf(1,'warning: not enough inputs\n');
  return;
end

%Check to see if inside of models is URLs or local file paths
if isfield(models{1},'I') && isstr(models{1}.I) && length(models{1}.I)>=7 ...
      && strcmp(models{1}.I(1:7),'http://')
  fprintf(1,['Warning: Models have images as URLs\n -- If you want' ...
             ' to apply detectors to a LOT of images, download the' ...
              ' PASCAL VOC2007 dataset locally and call :\n' ...
             '[models]=esvm_update_voc_models_to_local(models,local_dir);\n']);
end

%Calibration parameters are optional
if ~exist('M','var')
  M = [];
end

if ~iscell(imageset) 
  if isnumeric(imageset)
    imageset = {imageset};
  end
end

if ~exist('params','var')
  params = esvm_get_default_params;
end

params.calibration_propagate_onto_raw = 1;
for i = 1:length(imageset)
  %Get local detections
  local_detections = esvm_detect_imageset(imageset(i), models, ...
                                          params);  
  %Pool exemplar detections
  result_struct = esvm_pool_exemplar_dets(local_detections, models, M, params);

  %Show maxk top detections in this image
  maxk = 1;
  allbbs = esvm_show_top_dets(result_struct, local_detections, ...
                              imageset(i), models, ...
                              params,  maxk);
  drawnow
  snapnow
end
