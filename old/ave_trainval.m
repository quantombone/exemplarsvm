function ave_trainval(models,M)
%alias function
if nargin == 0

  [cls,DET_TYPE] = load_default_class;
  models = load_all_models(cls,[DET_TYPE '-stripped']);

  apply_voc_exemplars(models,[],'trainval');
elseif nargin == 1
  apply_voc_exemplars(models,[],'trainval');
else
  apply_voc_exemplars(models,M,'trainval');
end