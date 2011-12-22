function apply_sketches
VOCinit;
classes = VOCopts.classes;

classes = {'car'};

for i = 1:length(classes)
  models = load_all_exemplars(classes{i});
  
  if length(models) == 0
    continue
  end

  
  apply_voc_exemplars(models);
end