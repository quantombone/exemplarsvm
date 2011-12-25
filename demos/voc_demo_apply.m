function voc_demo_apply(imageset, models, M)
%In this application demo, we simply load the models belonging to
%some class and apply them
data_directory = '/Users/tmalisie/projects/Pascal_VOC/'
dataset_params = get_voc_dataset('VOC2007',...
                                 data_directory);
% imageset = get_pascal_set(dataset_params, 'test' , setname);

%subset = 1;
%models = models(subset);

for i = 1:length(models)
  models{i}.I = [data_directory '/' ...
                 models{i}.I(strfind(models{i}.I,'VOC2007/'):end)];
end

dataset_params.params = get_default_mining_params;

for i = 1:length(imageset)
  %res =
  %esvm_detect(convert_to_I(test_set{i}),models,models{1}.mining_params);
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
% test_set = test_set(1);
% esvm_detect_set(dataset_params, models, test_set,[], calibration_data);
