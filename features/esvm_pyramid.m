function [feat, scale] = esvm_pyramid(im, params)
% [feat, scale] = esvm_pyramid(im, params);
% Compute a pyramid worth of features by calling resize/features in
% over a set of scales defined inside params
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

if isnumeric(params)
  sbin = params;
elseif isfield(params,'sbin') 
  sbin = params.sbin;
elseif isfield(params,'init_params') && ...
      isfield(params.init_params,'sbin') && ...
      isnumeric(params.init_params.sbin)
  sbin = params.init_params.sbin;
else
  error('esvm_pyramid: cannot find sbin inside params');
end

%Make sure image is in double format
im = double(im);

if isfield(params,'detect_max_scale')
  detect_max_scale = params.detect_max_scale;
else
  detect_max_scale = 1.0;
end

if isfield(params,'detect_min_scale')
  detect_min_scale = params.detect_min_scale;
else
  detect_min_scale = .01;
end

%Hardcoded maximum number of levels in the pyramid
MAXLEVELS = 200;

%Hardcoded minimum dimension of smallest (coarsest) pyramid level
MINDIMENSION = 5;

%Get the levels per octave from the parameters
interval = params.detect_levels_per_octave;

sc = 2 ^(1/interval);

% Start at detect_max_scale, and keep going down by the increment sc, until
% we reach MAXLEVELS or detect_min_scale
scale = zeros(1,MAXLEVELS);
feat = {};
for i = 1:MAXLEVELS
  scaler = detect_max_scale / sc^(i-1);
  
  if scaler < detect_min_scale
    return
  end
  
  scale(i) = scaler;
  scaled = resize(im,scale(i));
  
  %if minimum dimensions is less than or equal to 5, exit
  if min([size(scaled,1) size(scaled,2)])<=MINDIMENSION
    scale = scale(scale>0);
    return;
  end

  feat{i} = params.init_params.features(scaled,sbin);

  %if we get zero size feature, backtrack one, and dont produce any
  %more levels
  if (size(feat{i},1)*size(feat{i},2)) == 0
    feat = feat(1:end-1);
    scale = scale(1:end-1);
    return;
  end

  %recover lost bin!!!
  feat{i} = padarray(feat{i}, [1 1 0], 0);

  %if the max dimensions is less than or equal to 5, dont produce
  %any more levels
  if max([size(feat{i},1) size(feat{i},2)])<=MINDIMENSION
    scale = scale(scale>0);
    return;
  end  
end
