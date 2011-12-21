function voc_demo_apply(cls)
%In this application demo, we simply load the models belonging to
%some class and apply them

if ~exist('cls','var')
  cls = 'bicycle';
end


VOCYEAR = 'VOC2007';
dataset_params = get_voc_dataset(VOCYEAR);
dataset_params.display = 1;

%Do not skip evaluation, unless it is VOC2010
dataset_params.SKIP_EVAL = 0;

test_set = get_pascal_set(dataset_params, 'test' , ['+' cls]);

models = load([load_results_directory '/' VOCYEAR ['/local/models/' cls '-g-' ...
                    '100-12.exemplar-svm-stripped.mat']]);
betas = load([load_results_directory '/' VOCYEAR ['/local/betas/' cls '-g-' ...
                    '100-12.exemplar-svm-betas.mat']]);
%M = load([results_directory '/' VOCYEAR ['/local/betas/' cls '-g-' ...
%                    '100-12.exemplar-svm-M.mat']]);

models = models.models;
betas = betas.betas;
%M = M.M;

subset = 1;
models = models(subset);
betas = betas(subset,:);

for i = 1:length(models)
  models{i}.I = [load_data_directory '/' ...
                 models{i}.I(strfind(models{i}.I,'VOC2007/'):end)];
end

calibration_data.betas = betas;
test_set = test_set(1);
esvm_detect_set(dataset_params, models, test_set,[], calibration_data);
