classes = {'car'};

for i = 1:length(classes)
  voc_template_exemplar_seg(classes{i},'VOC2007');
end
