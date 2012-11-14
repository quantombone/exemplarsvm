function [trainval_set, train_set,val_set, test_set] = install_voc_dataset(data_directory, dataset_directory)
% Installs a dataset into a single mat file with pointers to image
% locations and all annotations loaded into a single matlab structure

if nargin == 0
  data_directory = '/Users/tomasz/projects/pascal/';
  dataset_directory = 'VOC2011';

end

trainval_file = sprintf('%s/%s/train.mat',data_directory, ...
                        dataset_directory);


train_file = sprintf('%s/%s/train.mat',data_directory, ...
                        dataset_directory);

val_file = sprintf('%s/%s/val.mat',data_directory, ...
                        dataset_directory);


test_file = sprintf('%s/%s/test.mat',data_directory, ...
                    dataset_directory);

if fileexists(trainval_file)
  trainval_set = load(trainval_file);
  trainval_set = trainval_set.data_set;
  
else
  dataset_params = esvm_get_voc_dataset(dataset_directory, ...
                                        data_directory);
  
  image_set = esvm_get_pascal_set(dataset_params, 'train');
  
  newfiles = cellfun(@(x)strrep(strrep(x,'JPEGImages', ...
                                       'Annotations'),...
                                '.jpg','.xml'),image_set, ...
                     'UniformOutput',false);

  data_set = cellfun(@(x)PASreadrecord(x),newfiles,'UniformOutput', ...
                     false);
  data_set = cellfun(@(x,y)setfield(x,'I',y),data_set,image_set,'UniformOutput',false);
  
  try
    save(trainval_file,'data_set');
  catch
    fprintf(1,'Cannot write to %s\n',trainval_file);
  end
  trainval_set = data_set;
end



if nargout == 1
  return;
end

if fileexists(train_file)
  train_set = load(train_file);
  train_set = train_set.data_set;
  
else
  dataset_params = esvm_get_voc_dataset(dataset_directory, ...
                                        data_directory);
  
  image_set = esvm_get_pascal_set(dataset_params, 'train');
  
  newfiles = cellfun(@(x)strrep(strrep(x,'JPEGImages', ...
                                       'Annotations'),...
                                '.jpg','.xml'),image_set, ...
                     'UniformOutput',false);

  data_set = cellfun(@(x)PASreadrecord(x),newfiles,'UniformOutput', ...
                     false);
  data_set = cellfun(@(x,y)setfield(x,'I',y),data_set,image_set,'UniformOutput',false);
  
  try
    save(train_file,'data_set');
  catch
    fprintf(1,'Cannot write to %s\n',train_file);
  end
  train_set = data_set;
end

if fileexists(val_file)
  val_set = load(val_file);
  val_set =val_set.data_set;
  
else
  dataset_params = esvm_get_voc_dataset(dataset_directory, ...
                                        data_directory);
  
  image_set = esvm_get_pascal_set(dataset_params, 'val');
  
  newfiles = cellfun(@(x)strrep(strrep(x,'JPEGImages', ...
                                       'Annotations'),...
                                '.jpg','.xml'),image_set, ...
                     'UniformOutput',false);

  data_set = cellfun(@(x)PASreadrecord(x),newfiles,'UniformOutput', ...
                     false);
  data_set = cellfun(@(x,y)setfield(x,'I',y),data_set,image_set,'UniformOutput',false);
  
  try
    save(val_file,'data_set');
  catch
    fprintf(1,'Cannot write to %s\n',val_file);
  end
  val_set = data_set;
end


  
if fileexists(test_file)
  test_set = load(test_file);
  test_set = test_set.data_set;
else
  try
    image_set = esvm_get_pascal_set(dataset_params, ['val']);
  catch
    fprintf(1,'No testset to process\n');
    test_set = [];
    return;

  end
  
  newfiles = cellfun(@(x)strrep(strrep(x,'JPEGImages', ...
                                       'Annotations'),...
                                '.jpg','.xml'),image_set, ...
                     'UniformOutput',false);
  
  data_set = cellfun(@(x)PASreadrecord(x),newfiles,'UniformOutput', ...
                     false);
  data_set = cellfun(@(x,y)setfield(x,'I',y),data_set,image_set,'UniformOutput',false);
  
  try
    save(test_file, 'data_set');
  catch
    fprintf(1, 'Cannot write to %s\n', test_file);
  end
  test_set = data_set;
end
