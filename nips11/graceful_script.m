%function graceful_script

%% dump the graceful-degradation files

VOCinit;

cats = {'bus','sofa', 'cow', 'train'};
modes = {'e-svm', 'e-svm-vregmine'};

for i = 1:length(cats)
  for j = 1:length(modes)

    models = load_all_models(cats{i},modes{j});
    thresher = -1.1;
    if strfind(models{1}.models_name,'vreg')
      thresher = .4;
    end
    fprintf(1,'thresher = %.3f\n',thresher);
    grid = load_result_grid(models,'both',thresher);
    M = mmhtit(models,grid);
    [results,final] = evaluate_pascal_voc_grid(models,grid,'test', ...
                                               M);
    
    %good files already saved
    %ofiler = sprintf('%s/nips-%s-%s.mat', VOCopts.resdir, cats{i}, ...
    %                modes{j});
    %save(filer,'results','final');

  end
end

%show_graceful_confusion(models,final)