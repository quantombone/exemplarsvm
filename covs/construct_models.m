function model = construct_models(data_set,res)
%Construct a set of 'scene' models from a dataset of images and a
%covariance matrix

inds = round(linspace(1,length(data_set),100));
inds = unique(inds);

x = cell(length(inds),1);
for iii = 1:length(inds)
  i = inds(iii);
  
  I = data_set{i};
  clear s objects
  s.I = I;
  I = toI(I);
  objects(1).class = sprintf('scene=%d',iii);
  objects(1).truncated = 0;
  objects(1).difficult = 0;
  objects(1).bbox = [1 1 size(I,2) size(I,1)];
  s.objects = objects;
  data_set2{iii} = s;
end

model = learnAllEG(data_set2,'',res);

model.params.detect_add_flip = 0;
model.params.detect_max_windows_per_exemplar = 1;
model.params.max_image_size = 200;
