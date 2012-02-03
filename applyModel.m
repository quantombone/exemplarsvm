function test_struct = applyModel(models, data_set, test_set_name)
%% Define test-set
test_params = models{1}.params;
test_params.detect_exemplar_nms_os_threshold = 1.0;
test_params.detect_max_windows_per_exemplar = 500;
test_params.detect_keep_threshold = -2.0;

%% Apply on test set
test_grid = esvm_detect_imageset(data_set, models, test_params);%, test_set_name);

%% Apply calibration matrix to test-set results
test_struct = esvm_pool_exemplar_dets(test_grid, models, [], ...
                                      test_params);

