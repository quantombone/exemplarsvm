%classes = {'train'};%'cow','train','bus','boat'};
 classes={...
%     'car'
%     'chair'
%     'diningtable'
     'person'
%     'pottedplant'
};

myRandomize;
r = randperm(length(classes));
classes = classes(r);
for i = 1:length(classes)
  %voc_template_exemplar_seg(classes{i},'VOC2010');
  voc_template_exemplar_seg(classes{i},'VOC2007');
end
