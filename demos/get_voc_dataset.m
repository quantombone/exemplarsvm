function dataset_params = get_voc_dataset(VOCYEAR)
%Get the dataset structure for a VOC dataset, given the VOCYEAR
%string which is something like: VOC2007, VOC2010, ...

result_dir = load_results_directory;
datadir = load_data_directory;

if ~exist('VOCYEAR','var')
  fprintf(1,'WARNING: using default VOC2007 dataset\n');
  VOCYEAR = 'VOC2007';
end

% Choose the number of images to process in each chunk for detection.
% This parameters tells us how many images each core will process at
% at time before saving results.  A higher number of images per chunk
% means there will be less constant access to hard disk by separate
% processes than if images per chunk was 1.
dataset_params.NIMS_PER_CHUNK = 4;

% Create a root directory
dataset_params.devkitroot = [result_dir '/' VOCYEAR];

% change this path to a writable local directory for the example code
dataset_params.localdir = [dataset_params.devkitroot '/local/'];

% change this path to a writable directory for your results
dataset_params.resdir = [dataset_params.devkitroot ['/' ...
                    'results/']];

%This is location of the installed VOC datasets
dataset_params.datadir = datadir;

%Some VOC-specific dataset stats
dataset_params.dataset = VOCYEAR;
dataset_params.testset = 'test';

%NOTE: this is if we want the resulting calibrated models to have
%different special characters in their name
dataset_params.subname = '';

%Fill in the params structure with VOC-specific stuff
dataset_params = VOCinit(dataset_params);
