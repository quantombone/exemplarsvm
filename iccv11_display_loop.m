%function iccv11_display_loop
%Here is a loop to display the results we wanted

VOCinit;

classes = VOCopts.classes;
classes = setdiff(classes,'person');
classes = setdiff(classes,'horse');
classes = setdiff(classes,'bus');
classes = setdiff(classes,'motorbike');
classes = setdiff(classes,'bicycle');
%classes = setdiff(classes,'car');
%classes = setdiff(classes,'chair');
rrr = randperm(length(classes));
classes = classes(rrr);

%classes = {'bird'};

%classes = {'person'};;
%classes = {'cow','diningtable','motorbike','sofa','train'}
%classes = {'cow'};
%classes = {'motorbike'};
%classes = {'bus'};
basedir = sprintf('/nfs/baikal/tmalisie/labelme400/www/voc/iccv11/%s/',...
                  VOCopts.dataset);


for i = 1:length(classes)
  
  %filerlock = sprintf('%s/reslock-2010/%s.lock',VOCopts.localdir, ...
  %                    classes{i});
  %if fileexists(filerlock) || (mymkdir_dist(filerlock) == 0)
  %  continue
  %end
  models = load_all_models_both(classes{i},'exemplars','10');
  gridtest = load_result_grid_both(models,'test');

  load(sprintf('/nfs/baikal/tmalisie/results/VOC2007/iccv11.%s_test_results.mat',models{1}.cls));
  %gridtrainval = load_result_grid_both(models,'trainval');
  
  %M = mmhtit(models,gridtrainval);
  %[results,final] = evaluate_pascal_voc_grid(models,gridtrainval, ...
  %                                           'test',M);
  [results,final] = evaluate_pascal_voc_grid(models,gridtest, ...
                                             'test',M);
  
  %results
  %final.basedir = basedir;

  %figure(2)
  %print(gcf,'-dpng',sprintf('%s/pr.%s.png',basedir,models{1}.cls));
  show_top_dets(models,gridtest,'test',final);
end