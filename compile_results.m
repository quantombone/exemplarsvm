%function compile_results
VOCinit;
classes = VOCopts.classes;
classes = setdiff(classes,{'person','car'});

classes = {'bus'};
for i = 1:length(classes)
  load(sprintf(['/nfs/baikal/tmalisie/results/VOC2007/iccv11.' ...
                '%s_test_results.mat'],classes{i}));

  figure(1)
  clf
  plot(results.recall,results.prec,'-');
  grid;
  xlabel 'recall'
  ylabel 'precision'
  title(sprintf('class: %s, subset: %s, AP = %.3f, APold=%.3f', ...
                classes{i},VOCopts.testset,results.ap,results.apold));
  
  sprintf('class: %s, subset: %s, AP = %.3f, APold=%.3f', ...
          classes{i},VOCopts.testset,results.ap,results.apold)
  
  %keyboard
  
end