function [boxes,saveboxes] = applyModel(data_set, model, draw)
% Function to apply a model to a data-set
% Inputs:
%    data_set: a dataset of images
%    model: a model to apply to those images
%    test_set_name: the name of the testset (optional) used to
%       cache results
% Outputs:
%    boxes: a cell array of boxes

%if ~exist('test_set_name','var')
  test_set_name = '';
%end

%% Define test-set parameters
test_params = model.params;
test_params.detect_exemplar_nms_os_threshold = 1.0;
test_params.detect_max_windows_per_exemplar = 200;
test_params.detect_keep_threshold = -1.0;
test_params.calibrate_nms = 1.0;

if length(data_set) == 0
  data_set = get_screenshot_bg(5000,@(x)imresize_max(x,400));
  %data_set = get_screenshot_bg(20);
  test_params.display_detections = 1;

  %test_params.write_top_detection = 1;
  %test_params.detect_max_scale = .1;
end

if ~exist('draw','var')
  draw = 0;
end

if draw > 0
  test_params.display_detections = draw;
end


if isnumeric(data_set) && ...
      size(data_set,4) == 1 && ...
      size(data_set,3) == 3
  data_set = {data_set};
  
  %Given a single image
  boxes = esvm_detect_imageset(data_set, model, test_params, ...
                               test_set_name);

elseif isnumeric(data_set) && ...
      size(data_set,4) > 1 && ...
      size(data_set, 3)==3
  
  %Given a tensor of images
  for i = 1:size(data_set,4)
    curI = data_set(:,:,:,i);
    boxes{i} = esvm_detect_imageset({curI},model,test_params);
    if size(boxes{i},1) > 0
      boxes{i}(:,11) = i;
    end
  end
else
  %% Apply on test set
  boxes = esvm_detect_imageset(data_set, model, test_params, ...
                               test_set_name);  
end

M = [];
if isfield(model,'M')
  M = model.M;
end


%% Apply calibration matrix to test-set results
boxes = esvm_pool_exemplar_dets(boxes, model.models, M, ...
                                test_params);
saveboxes = boxes;
boxes = cat(1,boxes.final_boxes{:});
