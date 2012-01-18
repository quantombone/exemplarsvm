function mining_params = esvm_get_default_params_scene
%Get the default mining params, but then add some extra scene-specific
%parameters
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm


mining_params = esvm_get_default_params;
mining_params.detect_max_scale = .4;
mining_params.detect_min_scene_os = 0.4;
