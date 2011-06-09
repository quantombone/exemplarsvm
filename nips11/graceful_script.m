%function graceful_script

%% dump the graceful-degradation files

VOCinit;

cats = {'bicycle','dog','motorbike','bus','sofa', 'cow', 'train'};
modes = {'e-svm', 'e-svm-vregmine'};

for i = 1:length(cats)
  %show_graceful_confusion(cats{i},1);
  %show_graceful_confusion(cats{i});
  if 1 
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
    filer = sprintf('%s/finalnips-%s-%s.mat', VOCopts.resdir, cats{i}, ...
                    modes{j});
    save(filer,'results','final');

  end
  end
end

