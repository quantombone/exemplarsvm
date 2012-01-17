function mining_params = esvm_get_default_params_scene
%Get the default mining params, but then add some extra scene-specific
%parameters

mining_params = esvm_get_default_params;
mining_params.detect_max_scale = .4;
mining_params.detect_min_scene_os = 0.4;
