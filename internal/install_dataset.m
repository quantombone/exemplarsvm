function install_dataset

data_directory = '/Users/tomasz/projects/pascal/';
dataset_directory = 'VOC2010';

dataset_params = esvm_get_voc_dataset(dataset_directory, ...
                                      data_directory);

image_set = esvm_get_pascal_set(dataset_params, ['trainval']);

newfiles = cellfun(@(x)strrep(strrep(x,'JPEGImages', ...
                                     'Annotations'),...
                              '.jpg','.xml'),image_set, ...
                   'UniformOutput',false);

tic
data_set = cellfun(@(x)PASreadrecord(x),newfiles,'UniformOutput', ...
                   false);
data_set = cellfun(@(x,y)setfield(x,'I',y),data_set,image_set,'UniformOutput',false);
toc

save(sprintf('%s/%s/trainval.mat',data_directory, ...
             dataset_directory),'data_set');


image_set = esvm_get_pascal_set(dataset_params, ['test']);

newfiles = cellfun(@(x)strrep(strrep(x,'JPEGImages', ...
                                     'Annotations'),...
                              '.jpg','.xml'),image_set, ...
                   'UniformOutput',false);

tic
data_set = cellfun(@(x)PASreadrecord(x),newfiles,'UniformOutput', ...
                   false);
data_set = cellfun(@(x,y)setfield(x,'I',y),data_set,image_set,'UniformOutput',false);
toc

save(sprintf('%s/%s/test.mat',data_directory, ...
             dataset_directory),'data_set');

