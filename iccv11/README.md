Here is code pertinent to the iccv11 exemplar experiments:

Perform calibration and M-estimation:
    function M = calibrate_and_estimate_M(dataset_params, models, grid)

Perform Non-Maximum Suppression:
    function top = nms(boxes, overlap)

Collect detections so they are ready for VOC evaluation:
    function final = pool_results(dataset_params,models,grid,M)

Show the top detections as pdf files:
    function allbbs = show_top_dets(dataset_params, models, grid, fg, set_name, finalstruct, maxk)

