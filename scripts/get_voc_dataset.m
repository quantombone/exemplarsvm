function dataset_params = get_voc_dataset(VOCYEAR, result_dir, datadir)
%Get the dataset structure for a VOC dataset, given the VOCYEAR
%string which is something like: VOC2007, VOC2010, ...

% Issue warning if you are not me (tmalisie), but want default parameters
if nargin ~= 3
  [tmp,user] = unix('whoami');
  user = strtrim(user);
  if strcmp(user,'tmalisie') == 0
    fprintf(1,'Warning user=%s, but defaults only defined for user=%s\n',...
            user,'tmalisie');
  end
end

if ~exist('result_dir','var')
  result_dir = '/nfs/baikal/tmalisie/nn311/';
  fprintf(1,'Using default dataset directory: %s\n', result_dir);
end

% Choose the number of images to process in each chunk (for detections)
dataset_params.NIMS_PER_CHUNK = 4;

% Create a root directory
dataset_params.devkitroot = [result_dir '/' VOCYEAR];

% change this path to a writable local directory for the example code
dataset_params.localdir = [dataset_params.devkitroot '/local/'];

% change this path to a writable directory for your results
dataset_params.resdir = [dataset_params.devkitroot ['/' ...
                    'results/']];

%This is location of the installed VOC datasets
if ~exist('datadir','var')
  [v,r] = unix('hostname');
  if strfind(r,'airbone')==1
    dataset_params.datadir = '/projects/Pascal_VOC/';
  else
    dataset_params.datadir = '/nfs/hn38/users/sdivvala/Datasets/Pascal_VOC/';
  end
else
  dataset_params.datadir = datadir;
end

%Some VOC-specific dataset stats
dataset_params.dataset = VOCYEAR;
dataset_params.testset = 'test';

%Fill in the params structure with VOC-specific stuff
dataset_params = VOCinit(dataset_params);
