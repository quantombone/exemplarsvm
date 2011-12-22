function apply_james

for i = 1:78
  models = load_all_exemplars(sprintf('james.%05d',i));
  apply_voc_exemplars(models);
end