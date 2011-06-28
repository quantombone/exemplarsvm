function dataset_params = get_voc_dataset(VOCYEAR)

%%%% SETUP DATASET
%CHOOSE HOW MANY IMAGES WE APPLY PER CHUNK
dataset_params.NIMS_PER_CHUNK = 4;

dataset_params.devkitroot = ['/nfs/baikal/tmalisie/summer11/' ...
                    VOCYEAR];

% change this path to a writable local directory for the example code
dataset_params.localdir = [dataset_params.devkitroot '/local/'];

% change this path to a writable directory for your results
dataset_params.resdir = [dataset_params.devkitroot ['/' ...
                    'results/']];

%This is the directory where we dump visualizations into
[v,r] = unix('hostname');
if strfind(r,'airbone')==1
  dataset_params.datadir = '/projects/Pascal_VOC/';
  dataset_params.display_machine = 'airbone';
else
  dataset_params.datadir = '/nfs/hn38/users/sdivvala/Datasets/Pascal_VOC/';
  dataset_params.display_machine = 'onega';
end

%Some VOC-specific dataset stats
dataset_params.dataset = VOCYEAR;
dataset_params.testset = 'test';
dataset_params = VOCinit(dataset_params);

