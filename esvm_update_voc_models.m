function models = esvm_update_voc_models(models, data_directory)
%Updates models to internally point to local data

for i = 1:length(models)
  models{i}.I = [data_directory '/' ...
                 models{i}.I(strfind(models{i}.I,'VOC2007/'):end)];
end
