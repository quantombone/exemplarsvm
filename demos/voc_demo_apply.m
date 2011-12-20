function voc_demo_apply(cls)
%In this application demo, we simply load the models belonging to
%some class and apply them

if ~exist('cls','var')
  cls = 'bicycle';
end


VOCYEAR = 'VOC2007';
results_directory = load_results_directory;
data_directory = load_data_directory;
dataset_params = get_voc_dataset(VOCYEAR, results_directory, data_directory);
dataset_params.display = 1;

%Do not skip evaluation, unless it is VOC2010
dataset_params.SKIP_EVAL = 0;

test_set = get_pascal_set(dataset_params, 'test' , ['+' cls]);

models = load([results_directory '/' VOCYEAR ['/local/models/' cls '-g-' ...
                    '100-12.exemplar-svm-stripped.mat']]);
betas = load([results_directory '/' VOCYEAR ['/local/betas/' cls '-g-' ...
                    '100-12.exemplar-svm-betas.mat']]);
%M = load([results_directory '/' VOCYEAR ['/local/betas/' cls '-g-' ...
%                    '100-12.exemplar-svm-M.mat']]);

models = models.models;
betas = betas.betas;
%M = M.M;


%models = models(1:10);
%betas = betas(1:10,:);

for i = 1:length(models)
  models{i}.I = [data_directory '/' ...
                 models{i}.I(strfind(models{i}.I,'VOC2007/'):end)];
end

calibration_data.betas = betas;
apply_all_exemplars(dataset_params, models, test_set,[], calibration_data);
