%% A showcase of the ExemplarSVM Library 1.0
addpath(genpath(pwd))

%Generate a training data set with class name 'circle'
%data_set = esvm_generate_dataset(10,10,'circle');

image_set = list_files_in_directory('~/av/demos/cube/');
data_set = esvm_generate_sketchup_dataset(image_set, ...
                                           'cube');
negative_set = list_files_in_directory(['/projects/pascal/VOC2007/' ...
                     'JPEGImages/']);
negative_set = negative_set(1:100);
data_set = cat(2,data_set,negative_set);

model = learnDalalTriggs(data_set,'cube');

