function [models] = load_all_models_both(cls, DET_TYPE, ...
                                         FINAL_PREFIX)
error('old function, because it duplicates models with lr\n');
%Load all trained models of a specified class 'cls' and specified
%type 'DET_TYPE' from a models directory.

models = load_all_models(cls,DET_TYPE,FINAL_PREFIX);
N = length(models);
models = [models models];
for i = N+1:length(models)
  models{i}.FLIP_LR = 1;
end
