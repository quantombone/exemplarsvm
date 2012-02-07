function showTopDetections(data_set, boxes, model)
% Show top detections from the model

params = model.params;

if params.display
  for mind = 1:length(model.models)
    I = esvm_show_top_exemplar_dets(boxes, data_set, ...
                                    model.models, mind,10,10);
    figure(45+mind) 
    imagesc(I)
    title('Top detections','FontSize',18);
    drawnow
    snapnow
    
    if params.dump_images == 1 && length(params.localdir)>0
      filer = sprintf('%s/results/topdet.%s-%04d-%s.png',...
                      localdir, models{mind}.models_name, mind, data_set_name);  
       imwrite(I,filer);
     end 
  end
end
