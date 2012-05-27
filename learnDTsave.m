function learnDTsave
addpath(genpath(pwd));
load /csail/vision-videolabelme/people/tomasz/pascal/VOC2007/trainval.mat

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
  filer = sprintf(['/csail/vision-videolabelme/people/tomasz/may28/' ...
                   '%s.mat'],classes{i});

  filerlock = [filer '.lock'];
  if fileexists(filer) || mymkdir_dist(filerlock) == 0
    continue
  end
  model = learnDalalTriggs(data_set,classes{i},'',filer);
  save(filer,'model');

  try
    rmdir(filerlock);
  catch
  end
end
