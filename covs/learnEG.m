function model = learnEG(data_set,cls,covstruct)
%Learn EvG model

data_set2 = split_sets(data_set,cls);

iii = 1;
for i = 1:length(data_set2)
  saves = data_set2{i};
  for j = 1:length(saves.objects)
    s = saves;
    s.objects = s.objects(j);
    model{iii} = learnGaussianTriggs({s}, cls, covstruct, 0);
    model{iii}.params = model{iii}.params;
    model{iii}.params.detect_add_flip = 1;
    model{iii}.params.detect_max_windows_per_exemplar = 10;
    %model{iii}.models{1}.params = model{iii}.params;
    iii = iii+1;
  end
end

models = cellfun2(@(x)x.models{1},model);
model = model{1};
model.models = models;


