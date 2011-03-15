%classes = {'motorbike'};
classes = {'bicycle','motorbike','horse'};%,'diningtable','sofa', ...
%           'chair'};

%classes = {'diningtable'};
%classes = {'sofa'};

%classes = {'horse'};
%classes = {'motorbike'};
classes = classes(end:-1:1);

classes = {'horse','motorbike'};
%classes = {'bicycle'};
%classes = {'diningtable'};
%classes = {'bus'};
%classes = {'chair'};

friendclass = 'person';
%friendclass = 'chair';
%friendclass = 'diningtable';
%clear results;
%clear fracs;
%clear cresults

for i = 1:length(classes)
  
  models = load_all_models_both(classes{i},'exemplars','10');
  gridtest = load_result_grid_both(models,'test');

  load(sprintf('/nfs/baikal/tmalisie/results/VOC2007/iccv11.%s_test_results.mat',models{1}.cls));
  %gridtrainval = load_result_grid_both(models,'trainval');
  
  %[results,final] = evaluate_pascal_voc_grid(models,gridtest, ...
  %                                           'test',M);
  %recs2 = prune_recs(recs,friendclass,models{i}.cls);
  
  final.friendclass = friendclass;
  %person_counts{i} = count_correct_person(models,gridtest,'test', ...
  %                                        final);
  %continue
  %return
  %[cresults{i}]=test_all_complements(models,final, ...
  %                                           gridtest,{friendclass},gtids,recs2);
    

  extras = show_top_dets(models,gridtest,'test',final);
  %M = mmhtit(models,gridtrainval);
  %
  %[results,final] = evaluate_pascal_voc_grid(models,gridtest, ...
  %                                           'test',M);

end
