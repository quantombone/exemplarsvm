function [results] = evaluateModel(data_set, test_struct, models)
params = models{1}.params;
results = esvm_evaluate_pascal_voc(test_struct, data_set, models, params);

if params.display
  for mind = 1:length(models)
    I = esvm_show_top_exemplar_dets(test_struct, data_set, ...
                                    models, mind,10,10);
    figure(45)
    imagesc(I)
    title('Top detections','FontSize',18);
    drawnow
    snapnow
    
    %if params.dump_images == 1 && length(params.localdir)>0
    %  filer = sprintf('%s/results/topdet.%s-%04d-%s.png',...
    %                  localdir, models{mind}.models_name, mind, data_set_name);
      
   %   imwrite(I,filer);
   % end 
  end
end
