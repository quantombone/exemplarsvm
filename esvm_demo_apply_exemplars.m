function esvm_demo_apply_exemplars(imageset, models, M)
%In this application demo, we simply load the models belonging to
%some class and apply them

if isfield(models{1},'I') && isstr(models{1}.I) && length(models{1}.I)>=7 ...
      && strcmp(models{1}.I(1:7),'http://')
  fprintf(1,'Warning: Models have images as URLs\n -- Use [models]=esvm_update_voc_models(models,local_dir);\n');
end


if ~exist('M','var')
  M = [];
end

if ~iscell(imageset) 
  if isdir(imageset)
    files = dir(imageset);
    isdirs = arrayfun(@(x)x.isdir,files);
    files = files(~isdirs);
  else
    imageset = {imageset};
  end
end

params = esvm_get_default_params;


for i = 1:length(imageset)
  local_detections = esvm_detect_imageset(imageset(i), models, params);
  result_struct = esvm_apply_calibration(local_detections, models, M, params);
  
  maxk = 1;
  allbbs = esvm_show_top_dets(result_struct, local_detections, ...
                              imageset(i), models, ...
                              params,  maxk);
  drawnow
end

return;

data_directory = '/Users/tmalisie/projects/Pascal_VOC/'
dataset_params = get_voc_dataset('VOC2007',...
                                 data_directory);
% imageset = get_pascal_set(dataset_params, 'test' , setname);

%subset = 1;
%models = models(subset);


dataset_params.params = get_default_mining_params;

for i = 1:length(imageset)
  %res =
  %esvm_detect(convert_to_I(imageset{i}),models,models{1}.mining_params);
  tic
  grid = esvm_detect_imageset(imageset(i), models, dataset_params.params);
  toc

  tic
  if ~exist('M','var')
    final = esvm_apply_calibration(dataset_params, models, grid);
  else
    final = esvm_apply_calibration(dataset_params, models, grid, ...
                                     M);
  end
  toc
  


  
  show_top_dets(dataset_params, models, grid, imageset(i), 'noname', ...
                final, 1, 0);
  drawnow
  continue

  I = convert_to_I(imageset{i});
  figure(1)
  clf
  imagesc(I)
  if size(final.final_boxes{1},1) == 0
    drawnow
    continue;
  end
  final.final_boxes{1}(:,end)
  bb = final.final_boxes{1}(1,:);
  plot_bbox(bb)
  axis image
  axis off
  drawnow
end


% calibration_data.betas = betas;
% imageset = imageset(1);
% esvm_detect_set(dataset_params, models, imageset,[], calibration_data);
