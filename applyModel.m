function boxes = applyModel(data_set, model, test_set_name)

%% Define test-set parameters
test_params = model.params;
test_params.detect_exemplar_nms_os_threshold = 1.0;
test_params.detect_max_windows_per_exemplar = 500;
test_params.detect_keep_threshold = -2.0;

%% Apply on test set
test_grid = esvm_detect_imageset(data_set, model, test_params);

%% Apply calibration matrix to test-set results
boxes = esvm_pool_exemplar_dets(test_grid, model.models, [], ...
                                      test_params);

