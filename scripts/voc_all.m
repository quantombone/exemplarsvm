
VOCopts.classes={...
    'pottedplant'
    'sheep'
    'sofa'
    'train'
    'tvmonitor'};

for i = 1:length(VOCopts.classes)
  voc_tvmonitor_scene(VOCopts.classes{i});
end
