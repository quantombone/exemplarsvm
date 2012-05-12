function evalDTsave
addpath(genpath(pwd));
load /csail/vision-videolabelme/people/tomasz/pascal/VOC2007/test.mat

classes={...
    'aeroplane'
    'bicycle'
    'bird'
    'boat'
    'bottle'
    'bus'
    'car'
    'cat'
    'chair'
    'cow'
    'diningtable'
    'dog'
    'horse'
    'motorbike'
    'person'
    'pottedplant'
    'sheep'
    'sofa'
    'train'
    'tvmonitor'};

myRandomize;
classes = classes(randperm(length(classes)));
for i = 1:length(classes)
  filer = sprintf(['/csail/vision-videolabelme/people/tomasz/may2012-dt/' ...
                   '%s.mat'],classes{i});


  filer2 = [filer '.fullresults.mat'];
  filer2lock = [filer2 '.lock'];
  if fileexists(filer2) || mymkdir_dist(filer2lock) == 0
    continue
  end
  load(filer);
  pset = data_set;
  %pset = split_sets(data_set,classes{i});
  boxes = applyModel(pset,model);
  boxes2 = esvm_nms(boxes);
  boxes3 = esvm_adjust_boxes(boxes2,model);
  results = VOCevaldet(pset,boxes3,classes{i});
  results.boxes = boxes;
  results.boxes2 = boxes2;
  results.boxes3 = boxes3;
  results.pset = pset;
  I = showBoxes(pset,boxes3);
  save(filer2,'results');

  filer2im = [filer2 '.I.' num2str(results.ap) '.png'];
  imwrite(I,filer2im);
  try
    rmdir(filer2lock);
  catch
  end
end
