% VOCopts.classes={...
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

%classes = {'tvmonitor','bus','cow','horse','train','tvmonitor'};
%classes = {'chair'};
classes = {'cow'};
%classes = {'bus'};
classes = {'car'};

% for a = 5:12
%   for b = 5:12

%     hg_size = [a b];
    
for i = 1:length(classes)
  cls = classes{i};
  %cls = 'bus+train'
  cls = 'bicycle+motorbike';
  cls = 'cat';
  cls = 'bicycle';
  cls = 'bus';
  cls = 'bicycle';
  data_set2 = replace_class(data_set, cls);
  

  fprintf(1,'Learning gaussian LDA model\n');
  hg_size = [10 8];
  hg_size = [];
  model{i} = learnGaussianTriggs(data_set2,cls,covstruct,hg_size);
  hg_size = model{i}.models{1}.hg_size;
 
  tsets{i} = split_sets(replace_class(test.data_set,cls),cls);
  %tsets{i} = test.data_set;
  dets{i} = applyModel(tsets{i},model{i},.0001);
  figure(1)
  clf
  VOCevaldet(tsets{i},esvm_nms(dets{i}),cls,.5);
  print(gcf,'-dpng',sprintf('/Users/tomasz/Desktop/lda_%s_%d_%d.png',cls,model{i}.models{1}.hg_size(1),model{i}.models{1}.hg_size(2)));
  results{i}  = VOCevaldet(tsets{i},esvm_nms(dets{i}),cls,.5);
  I = showTopDetections(tsets{i},esvm_nms(dets{i}),100, ...
                        results{i}.is_correct);
  imwrite(I,sprintf(['/Users/tomasz/Desktop/' ...
                     'loda_%s_%d_%d_image.png'],cls,model{i}.models{1}.hg_size(1),model{i}.models{1}.hg_size(2)));
  figure(2)
  imagesc(I)
  drawnow
end
%   end
% end