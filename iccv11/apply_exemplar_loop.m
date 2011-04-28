function apply_exemplar_loop
%Apply all of the LR flips here

VOCinit;
classes = setdiff(VOCopts.classes,'person');

%rrr = randperm(length(classes));
%classes = classes(rrr);

for i = 1:length(classes)
  models = load_all_models(classes{i},'exemplars','10');
  curset = 'trainval';
  apply_voc_exemplars(models,curset);
  curset = 'test';
  apply_voc_exemplars(models,curset);  
end
