function model = construct_models(data_set,res)

inds = round(linspace(1,length(data_set),100));
inds = unique(inds);

x = cell(length(inds),1);
for iii = 1:length(inds)
  i = inds(iii);
  
  I = data_set{i};
  clear s objects
  s.I = I;
  I = toI(I);
  objects(1).class = sprintf('scene');
  objects(1).truncated = 0;
  objects(1).difficult = 0;
  objects(1).bbox = [1 1 size(I,2) size(I,1)];
  s.objects = objects;
  data_set2{iii} = s;
end

model = learnEG(data_set2,'scene',res);


