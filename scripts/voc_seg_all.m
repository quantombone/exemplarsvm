classes = {'train'};
% classes={...
%     'aeroplane'
%     'bicycle'
%     'bird'
%     'boat'
%     'bottle'
%     'bus'
%     'car'
%     'cat'
%     'chair'
%     'cow'
%     'diningtable'
%     'dog'
%     'horse'
%     'motorbike'
%     'person'
%     'pottedplant'
%     'sheep'
%     'sofa'
%     'train'
%     'tvmonitor'};

myRandomize;
r = randperm(length(classes));
classes = classes(r);
for i = 1:length(classes)
  voc_template_exemplar_seg(classes{i},'VOC2010');
  %voc_template_exemplar_seg(classes{i},'VOC2007');
end
