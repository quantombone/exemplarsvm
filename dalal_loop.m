function dalal_loop
VOCinit;

myRandomize; 
classes = VOCopts.classes;
sets = {'trainval','test'};
flips = {0, 1};

r = randperm(length(classes));
classes = classes(r);

r = randperm(length(sets))
sets = sets(r);

r = randperm(length(flips));
flips = flips(r);

for i = 1:length(classes)
  models = load_all_models(classes{i},'dalals','100');
  for j = 1:length(sets)
    s = sets{j};
    for k = 1:length(flips)
      f = flips{k};
      apply_voc_exemplars(models,s,f);
    end
  end
end

